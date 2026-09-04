"""
SenseVoice LoRA 微调版本 (model_lora.py)
========================================
本文件从 model.py fork 而来，专门用于对比学习 LoRA 微调的改动。

与原始 model.py 的核心差异（搜索 "# [LoRA改动]" 快速定位所有改动点）：

1. 新增 import：从 funasr.models.lora 引入 layers 和 mark_only_lora_as_trainable
2. MultiHeadedAttentionSANM.__init__：
   - 原始：linear_out / linear_q_k_v 固定为 nn.Linear（lora_list 参数接收但未使用）
   - 修复：当 lora_list 不为 None 时，替换为 lora.Linear / lora.MergedLinear
3. SenseVoiceEncoderSmall.__init__：
   - 原始：构建 encoder_selfattn_layer_args 时不传 lora 相关参数
   - 修复：显式声明 lora_* 参数，并透传给每个 Attention 层
4. SenseVoiceSmall.__init__：
   - 新增：支持 lora_only 模式，调用 mark_only_lora_as_trainable 冻结非 LoRA 参数
5. 注册名称：
   - SenseVoiceEncoderSmall → SenseVoiceEncoderSmallLoRA
   - SenseVoiceSmall        → SenseVoiceSmallLoRA

训练配置 yaml 示例（在 encoder_conf 下添加）：
    encoder: SenseVoiceEncoderSmallLoRA
    model: SenseVoiceSmallLoRA
    encoder_conf:
        lora_list: "qkvo"   # q/k/v/o 任意组合；设为 null 等同全参微调
        lora_rank: 8
        lora_alpha: 16
        lora_dropout: 0.1
    lora_only: true          # true=只训练 LoRA 参数（推荐）；false=全参微调
    lora_bias: "none"        # none / all / lora_only
"""

import logging
from typing import Iterable, Optional
import types
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch.cuda.amp import autocast

from funasr.metrics.compute_acc import compute_accuracy, th_accuracy
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.train_utils.device_funcs import force_gatherable
from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.ctc.ctc import CTC
from funasr.register import tables
from funasr.models.paraformer.search import Hypothesis
from .utils.ctc_alignment import ctc_forced_align

# [LoRA改动] 1/4 ─ 新增 LoRA 相关 import
# 原始 model.py 中完全没有这两行
from funasr.models.lora import layers as lora
from funasr.models.lora.utils import mark_only_lora_as_trainable


# =============================================================================
# 以下三个类（SinusoidalPositionEncoder / PositionwiseFeedForward / LayerNorm）
# 与 model.py 完全一致，无任何改动。
# =============================================================================

class SinusoidalPositionEncoder(torch.nn.Module):
    """ """

    def __init__(self, d_model=80, dropout_rate=0.1):
        super().__init__()

    def encode(self, positions=None, depth=None, dtype=torch.float32):
        batch_size = positions.size(0)
        positions = positions.type(dtype)
        device = positions.device
        log_timescale_increment = torch.log(torch.tensor([10000], dtype=dtype, device=device)) / (
            depth / 2 - 1
        )
        inv_timescales = torch.exp(
            torch.arange(depth / 2, device=device).type(dtype) * (-log_timescale_increment)
        )
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(
            inv_timescales, [1, 1, -1]
        )
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return encoding.type(dtype)

    def forward(self, x):
        batch_size, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps + 1, device=x.device)[None, :]
        position_encoding = self.encode(positions, input_dim, x.dtype).to(x.device)
        return x + position_encoding


class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


# =============================================================================
# [LoRA改动] 2/4 ─ MultiHeadedAttentionSANM.__init__
# 这是最核心的改动：真正让 lora_list 参数生效。
#
# 对比原始 model.py（约 L132-L133）：
#   self.linear_out   = nn.Linear(n_feat, n_feat)         # 固定，忽略 lora_list
#   self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)   # 固定，忽略 lora_list
#
# 修复后：当 lora_list 不为 None 时，替换为 LoRA 版本。
# =============================================================================

class MultiHeadedAttentionSANM(nn.Module):
    """Multi-Head Attention layer (SANM) with optional LoRA support."""

    def __init__(
        self,
        n_head,
        in_feat,
        n_feat,
        dropout_rate,
        kernel_size,
        sanm_shfit=0,
        lora_list=None,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
    ):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head

        # ── [LoRA改动] 原始写法（固定 nn.Linear，lora_list 被忽略）────────────
        # self.linear_out   = nn.Linear(n_feat, n_feat)
        # self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        #
        # ── 修复：根据 lora_list 决定是否替换为 LoRA Linear ──────────────────

        if lora_list is not None:
            # output projection（"o"）
            if "o" in lora_list:
                # lora.Linear = nn.Linear + 低秩旁路 (B @ A * scaling)
                # 预训练权重冻结，只有 lora_A / lora_B 参与梯度
                self.linear_out = lora.Linear(
                    n_feat, n_feat,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
            else:
                self.linear_out = nn.Linear(n_feat, n_feat)

            # Q/K/V 合并投影
            # lora_qkv_list 是长度为 3 的布尔列表：[enable_q, enable_k, enable_v]
            lora_qkv_list = ["q" in lora_list, "k" in lora_list, "v" in lora_list]
            if not any(lora_qkv_list):
                self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
            else:
                # lora.MergedLinear: 对同一大 Linear 的不同输出段分别加 LoRA
                # 例如 enable_lora=[True, False, True] → 只对 q 和 v 加 LoRA
                self.linear_q_k_v = lora.MergedLinear(
                    in_feat,
                    n_feat * 3,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    enable_lora=lora_qkv_list,
                )
        else:
            # lora_list=None：与原始 model.py 完全等价
            self.linear_out = nn.Linear(n_feat, n_feat)
            self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)

        # ── 以下与原始 model.py 一致 ──────────────────────────────────────────
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward_fsmn(self, inputs, mask, mask_shfit_chunk=None):
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            if mask_shfit_chunk is not None:
                mask = mask * mask_shfit_chunk
            inputs = inputs * mask
        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x

    def forward_qkv(self, x):
        b, t, d = x.size()
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        q_h = torch.reshape(q, (b, t, self.h, self.d_k)).transpose(1, 2)
        k_h = torch.reshape(k, (b, t, self.h, self.d_k)).transpose(1, 2)
        v_h = torch.reshape(v, (b, t, self.h, self.d_k)).transpose(1, 2)
        return q_h, k_h, v_h, v

    def forward_attention(self, value, scores, mask, mask_att_chunk_encoder=None):
        n_batch = value.size(0)
        if mask is not None:
            if mask_att_chunk_encoder is not None:
                mask = mask * mask_att_chunk_encoder
            mask = mask.unsqueeze(1).eq(0)
            min_value = -float("inf")
            scores = scores.masked_fill(mask, min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(self, x, mask, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask, mask_shfit_chunk)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask, mask_att_chunk_encoder)
        return att_outs + fsmn_memory

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        q_h, k_h, v_h, v = self.forward_qkv(x)
        if chunk_size is not None and look_back > 0 or look_back == -1:
            if cache is not None:
                k_h_stride = k_h[:, :, : -(chunk_size[2]), :]
                v_h_stride = v_h[:, :, : -(chunk_size[2]), :]
                k_h = torch.cat((cache["k"], k_h), dim=2)
                v_h = torch.cat((cache["v"], v_h), dim=2)
                cache["k"] = torch.cat((cache["k"], k_h_stride), dim=2)
                cache["v"] = torch.cat((cache["v"], v_h_stride), dim=2)
                if look_back != -1:
                    cache["k"] = cache["k"][:, :, -(look_back * chunk_size[1]) :, :]
                    cache["v"] = cache["v"][:, :, -(look_back * chunk_size[1]) :, :]
            else:
                cache_tmp = {
                    "k": k_h[:, :, : -(chunk_size[2]), :],
                    "v": v_h[:, :, : -(chunk_size[2]), :],
                }
                cache = cache_tmp
        fsmn_memory = self.forward_fsmn(v, None)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, None)
        return att_outs + fsmn_memory, cache


# =============================================================================
# LayerNorm / sequence_mask / EncoderLayerSANM：与 model.py 完全一致，无改动。
# =============================================================================

class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device=None):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask = mask.detach()
    return mask.type(dtype).to(device) if device is not None else mask.type(dtype)


class EncoderLayerSANM(nn.Module):
    def __init__(
        self,
        in_size,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        super(EncoderLayerSANM, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(in_size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.in_size = in_size
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate

    def forward(self, x, mask, cache=None, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        skip_layer = False
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.concat_after:
            x_concat = torch.cat(
                (
                    x,
                    self.self_attn(
                        x, mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    ),
                ),
                dim=-1,
            )
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
            else:
                x = stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x, mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
            else:
                x = stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x, mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.in_size == self.size:
            attn, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)
            x = residual + attn
        else:
            x, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm2(x)

        return x, cache


# =============================================================================
# [LoRA改动] 3/4 ─ SenseVoiceEncoderSmall（注册为 SenseVoiceEncoderSmallLoRA）
#
# 对比原始 model.py：
#   encoder_selfattn_layer_args0 = (n_head, in_feat, out_feat, attn_drop, ks, shift)
#   encoder_selfattn_layer_args  = (n_head, out_feat, out_feat, attn_drop, ks, shift)
#
# 修复后追加 4 个 lora_* 参数，使 MultiHeadedAttentionSANM 能正确构建 LoRA 层。
# =============================================================================

@tables.register("encoder_classes", "SenseVoiceEncoderSmallLoRA")
class SenseVoiceEncoderSmall(nn.Module):
    """
    SenseVoice Encoder with LoRA support.

    在 encoder_conf 中新增以下可选参数（全部有默认值，不填即等同原版）：
        lora_list    (str|None): 参与 LoRA 的投影层，如 "qkvo"，None=不使用LoRA
        lora_rank    (int):      LoRA 秩，默认 8
        lora_alpha   (int):      LoRA alpha，默认 16
        lora_dropout (float):   LoRA dropout，默认 0.1
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        tp_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=SinusoidalPositionEncoder,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        kernel_size: int = 11,
        sanm_shfit: int = 0,
        selfattention_layer_type: str = "sanm",
        # [LoRA改动] 显式声明 4 个 LoRA 参数（原版通过 **kwargs 接收但不使用）
        lora_list: Optional[str] = None,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self._output_size = output_size

        self.embed = SinusoidalPositionEncoder()
        self.normalize_before = normalize_before

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (output_size, linear_units, dropout_rate)

        encoder_selfattn_layer = MultiHeadedAttentionSANM

        # [LoRA改动] 在原有 6-tuple 基础上追加 4 个 LoRA 参数 ──────────────────
        # 原始：(attention_heads, input_size,  output_size, attention_dropout_rate, kernel_size, sanm_shfit)
        # 修复：在末尾追加 lora_list, lora_rank, lora_alpha, lora_dropout
        encoder_selfattn_layer_args0 = (
            attention_heads, input_size, output_size,
            attention_dropout_rate, kernel_size, sanm_shfit,
            lora_list, lora_rank, lora_alpha, lora_dropout,  # ← 新增
        )
        encoder_selfattn_layer_args = (
            attention_heads, output_size, output_size,
            attention_dropout_rate, kernel_size, sanm_shfit,
            lora_list, lora_rank, lora_alpha, lora_dropout,  # ← 新增
        )

        if lora_list is not None:
            logging.info(
                f"[SenseVoiceLoRA] LoRA enabled: list={lora_list}, "
                f"rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}"
            )
        else:
            logging.info("[SenseVoiceLoRA] LoRA disabled (full fine-tuning mode)")

        self.encoders0 = nn.ModuleList(
            [
                EncoderLayerSANM(
                    input_size, output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(1)
            ]
        )
        self.encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size, output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(num_blocks - 1)
            ]
        )
        self.tp_encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size, output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(tp_blocks)
            ]
        )

        self.after_norm = LayerNorm(output_size)
        self.tp_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(self, xs_pad: torch.Tensor, ilens: torch.Tensor):
        maxlen = xs_pad.shape[1]
        masks = sequence_mask(ilens, maxlen=maxlen, device=ilens.device)[:, None, :]

        xs_pad *= self.output_size() ** 0.5
        xs_pad = self.embed(xs_pad)

        for layer_idx, encoder_layer in enumerate(self.encoders0):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        for layer_idx, encoder_layer in enumerate(self.encoders):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1).int()

        for layer_idx, encoder_layer in enumerate(self.tp_encoders):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.tp_norm(xs_pad)
        return xs_pad, olens


# =============================================================================
# [LoRA改动] 4/4 ─ SenseVoiceSmall（注册为 SenseVoiceSmallLoRA）
#
# 新增两个训练开关（在顶层 yaml 中配置，不在 encoder_conf 内）：
#   lora_only (bool): True = 冻结非 LoRA 参数，只有 lora_A/lora_B 参与梯度
#   lora_bias (str):  "none" / "all" / "lora_only"
#
# 冻结后典型可训练参数量约为全参的 0.1% ~ 1%（取决于 rank 和 lora_list）
# =============================================================================

@tables.register("model_classes", "SenseVoiceSmallLoRA")
class SenseVoiceSmall(nn.Module):
    """CTC-attention hybrid Encoder-Decoder model with LoRA support."""

    def __init__(
        self,
        specaug: str = None,
        specaug_conf: dict = None,
        normalize: str = None,
        normalize_conf: dict = None,
        encoder: str = None,
        encoder_conf: dict = None,
        ctc_conf: dict = None,
        input_size: int = 80,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        length_normalized_loss: bool = False,
        **kwargs,
    ):
        super().__init__()

        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        if normalize is not None:
            normalize_class = tables.normalize_classes.get(normalize)
            normalize = normalize_class(**normalize_conf)

        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()

        if ctc_conf is None:
            ctc_conf = {}
        ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)

        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.error_calculator = None
        self.ctc = ctc
        self.length_normalized_loss = length_normalized_loss
        self.encoder_output_size = encoder_output_size

        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.textnorm_int_dict = {25016: 14, 25017: 15}
        self.embed = torch.nn.Embedding(
            7 + len(self.lid_dict) + len(self.textnorm_dict), input_size
        )
        self.emo_dict = {
            "unk": 25009, "happy": 25001, "sad": 25002,
            "angry": 25003, "neutral": 25004,
        }
        self.criterion_att = LabelSmoothingLoss(
            size=self.vocab_size,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
        )

        # [LoRA改动] 参数冻结逻辑 ──────────────────────────────────────────────
        # 原始 model.py：无此逻辑，所有参数默认全部可训练
        # 修复：当 lora_only=True 时，只让 lora_A / lora_B（及可选的 bias）参与梯度
        lora_only = kwargs.get("lora_only", False)
        if lora_only:
            lora_bias = kwargs.get("lora_bias", "none")
            mark_only_lora_as_trainable(self, bias=lora_bias)
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            logging.info(
                f"[SenseVoiceLoRA] lora_only=True, bias={lora_bias} | "
                f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
            )
        else:
            logging.info("[SenseVoiceLoRA] lora_only=False, all parameters trainable")

    @staticmethod
    def from_pretrained(model: str = None, **kwargs):
        from funasr import AutoModel
        model, kwargs = AutoModel.build_model(model=model, trust_remote_code=True, **kwargs)
        return model, kwargs

    # ── 以下 forward/encode/_calc_*/inference/post/export 与 model.py 完全一致 ──

    def forward(self, speech, speech_lengths, text, text_lengths, **kwargs):
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, text)

        loss_ctc, cer_ctc = None, None
        loss_rich, acc_rich = None, None
        stats = dict()

        loss_ctc, cer_ctc = self._calc_ctc_loss(
            encoder_out[:, 4:, :], encoder_out_lens - 4, text[:, 4:], text_lengths - 4
        )
        loss_rich, acc_rich = self._calc_rich_ce_loss(encoder_out[:, :4, :], text[:, :4])

        loss = loss_ctc + loss_rich
        stats["loss_ctc"] = torch.clone(loss_ctc.detach()) if loss_ctc is not None else None
        stats["loss_rich"] = torch.clone(loss_rich.detach()) if loss_rich is not None else None
        stats["loss"] = torch.clone(loss.detach()) if loss is not None else None
        stats["acc_rich"] = acc_rich

        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(self, speech, speech_lengths, text, **kwargs):
        if self.specaug is not None and self.training:
            speech, speech_lengths = self.specaug(speech, speech_lengths)
        if self.normalize is not None:
            speech, speech_lengths = self.normalize(speech, speech_lengths)

        lids = torch.LongTensor(
            [
                [
                    (
                        self.lid_int_dict[int(lid)]
                        if torch.rand(1) > 0.2 and int(lid) in self.lid_int_dict
                        else 0
                    )
                ]
                for lid in text[:, 0]
            ]
        ).to(speech.device)
        language_query = self.embed(lids)

        styles = torch.LongTensor(
            [[self.textnorm_int_dict[int(style)]] for style in text[:, 3]]
        ).to(speech.device)
        style_query = self.embed(styles)
        speech = torch.cat((style_query, speech), dim=1)
        speech_lengths += 1

        event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
            speech.size(0), 1, 1
        )
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3

        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
        return encoder_out, encoder_out_lens

    def _calc_ctc_loss(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens):
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rich_ce_loss(self, encoder_out, ys_pad):
        decoder_out = self.ctc.ctc_lo(encoder_out)
        loss_rich = self.criterion_att(decoder_out, ys_pad.contiguous())
        acc_rich = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_pad.contiguous(),
            ignore_label=self.ignore_id,
        )
        return loss_rich, acc_rich

    def get_timestamps_text(self, yseq: Tensor, tokenizer: SentencepiecesTokenizer = None,
                            frame_duration_sec=0.06, continue_frame_buff=2):
        if isinstance(yseq, np.ndarray):
            arr = yseq
        else:
            arr = yseq.detach().cpu().numpy() if hasattr(yseq, 'numpy') else np.array(yseq)
        if len(arr) <= 4:
            return [], ''
        arr = arr[4:]
        is_nonzero = arr != 0
        boundary_indices = np.where(is_nonzero)[0]
        if len(boundary_indices) == 0:
            return [], ''
        token_int = arr[boundary_indices]
        text = tokenizer.decode(token_int.tolist())
        token_list = tokenizer.text2tokens(text)
        mask = np.ones_like(token_int)
        mask[1:] &= ~((token_int[1:] == 124) & (token_int[:-1] == 124))
        blank_index_list = np.where(mask == 0)[0]
        if len(blank_index_list) > 0:
            logging.warn(f'has multi blank. text={text}, token_int={token_int}')
            for blank_index in blank_index_list:
                token_list.insert(blank_index, '▁')
        if len(token_list) != len(boundary_indices):
            logging.error(
                f'len(token_list) != len(boundary_indices), '
                f'len(token_list)={len(token_list)}, len(boundary_indices)={len(boundary_indices)}'
            )
            return [], ''
        pre_token = None
        start_idx = 0
        timestamps = []
        for i in range(len(boundary_indices)):
            curr_token = token_list[i]
            if (pre_token != curr_token) or (boundary_indices[i] - start_idx > continue_frame_buff):
                if pre_token:
                    start_frame = start_idx
                    if pre_token == curr_token:
                        end_frame = boundary_indices[i] + 1 if (i < len(boundary_indices) - 1) else boundary_indices[-1]
                    else:
                        end_frame = boundary_indices[i - 1] + 1 if (i < len(boundary_indices) - 1) else boundary_indices[-1]
                    timestamps.append([pre_token, start_frame * frame_duration_sec, end_frame * frame_duration_sec])
                pre_token = curr_token
                start_idx = boundary_indices[i]
        if pre_token:
            timestamps.append([pre_token, start_idx * frame_duration_sec, boundary_indices[-1] * frame_duration_sec])
        text = ''.join([sub_list[0] for sub_list in timestamps])
        text = text.replace('▁', ' ')
        return timestamps, text

    def inference(self, data_in, data_lengths=None, key=["wav_file_tmp_name"],
                  tokenizer=None, frontend=None, **kwargs):
        meta_data = {}
        if isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank":
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in, fs=frontend.fs, audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"), tokenizer=tokenizer,
            )
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = (
                speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
            )

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        language = kwargs.get("language", "auto")
        language_query = self.embed(
            torch.LongTensor([[self.lid_dict[language] if language in self.lid_dict else 0]]).to(
                speech.device
            )
        ).repeat(speech.size(0), 1, 1)

        use_itn = kwargs.get("use_itn", False)
        textnorm = kwargs.get("text_norm", None)
        output_timestamp = kwargs.get("output_timestamp", False)

        if textnorm is None:
            textnorm = "withitn" if use_itn else "woitn"
        textnorm_query = self.embed(
            torch.LongTensor([[self.textnorm_dict[textnorm]]]).to(speech.device)
        ).repeat(speech.size(0), 1, 1)
        speech = torch.cat((textnorm_query, speech), dim=1)
        speech_lengths += 1

        event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
            speech.size(0), 1, 1
        )
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3

        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        ctc_logits = self.ctc.log_softmax(encoder_out)
        if kwargs.get("ban_emo_unk", False):
            ctc_logits[:, :, self.emo_dict["unk"]] = -float("inf")

        results = []
        b, n, d = encoder_out.size()
        if isinstance(key[0], (list, tuple)):
            key = key[0]
        if len(key) < b:
            key = key * b
        for i in range(b):
            x = ctc_logits[i, : encoder_out_lens[i].item(), :]
            yseq = x.argmax(dim=-1)
            if output_timestamp:
                timestamp, text = self.get_timestamps_text(yseq, tokenizer)
            else:
                timestamp = None
                yseq = torch.unique_consecutive(yseq, dim=-1)
                mask = yseq != self.blank_id
                token_int = yseq[mask].tolist()
                text = tokenizer.decode(token_int)

            ibest_writer = None
            if kwargs.get("output_dir") is not None:
                if not hasattr(self, "writer"):
                    self.writer = DatadirWriter(kwargs.get("output_dir"))
                ibest_writer = self.writer[f"1best_recog"]
            if ibest_writer is not None:
                ibest_writer["text"][key[i]] = text

            if output_timestamp:
                text_timestamp_list = self.post(timestamp)
                text = ' '.join([sub_list[0] for sub_list in text_timestamp_list])
                timestamp_list = [sub_list[1:] for sub_list in text_timestamp_list]
                result_i = {"key": key[i], "text": text, "timestamp": timestamp_list}
            else:
                result_i = {"key": key[i], "text": text}
            results.append(result_i)
        return results, meta_data

    def post(self, timestamp):
        timestamp_new = []
        for i, t in enumerate(timestamp):
            word, start, end = t
            start = int(start * 1000)
            end = int(end * 1000)
            prev_word = timestamp_new[-1][0] if len(timestamp_new) > 0 else None
            if word == '▁':
                continue
            elif word.startswith("▁"):
                word = word[1:]
                timestamp_new.append([word, start, end])
            elif prev_word and prev_word.isalpha() and prev_word.isascii() and word.isalpha() and word.isascii():
                timestamp_new[-1][2] = end
                timestamp_new[-1][0] += word
            else:
                timestamp_new.append([word, start, end])
        return timestamp_new

    def export(self, **kwargs):
        from .export_meta import export_rebuild_model
        if "max_seq_len" not in kwargs:
            kwargs["max_seq_len"] = 512
        models = export_rebuild_model(model=self, **kwargs)
        return models
