#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)


import torch
import os.path
import librosa
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple

from funasr_onnx.utils.utils import (
    CharTokenizer,
    Hypothesis,
    ONNXRuntimeError,
    OrtInferSession,
    TokenIDConverter,
    get_logger,
    read_yaml,
)
from funasr_onnx.utils.sentencepiece_tokenizer import SentencepiecesTokenizer
from funasr_onnx.utils.frontend import WavFrontend
from funasr_onnx.utils.utils import pad_list

logging = get_logger()


class SenseVoiceSmall:
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
            self,
            model_dir: Union[str, Path] = None,
            batch_size: int = 1,
            device_id: Union[str, int] = "-1",
            plot_timestamp_to: str = "",
            quantize: bool = False,
            intra_op_num_threads: int = 4,
            cache_dir: str = None,
            **kwargs,
    ):

        if not Path(model_dir).exists():
            try:
                from modelscope.hub.snapshot_download import snapshot_download
            except:
                raise "You are exporting model from modelscope, please install modelscope and try it again. To install modelscope, you could:\n" "\npip3 install -U modelscope\n" "For the users in China, you could install with the command:\n" "\npip3 install -U modelscope -i https://mirror.sjtu.edu.cn/pypi/web/simple"
            try:
                model_dir = snapshot_download(model_dir, cache_dir=cache_dir)
            except:
                raise "model_dir must be model_name in modelscope or local path downloaded from modelscope, but is {}".format(
                    model_dir
                )

        model_file = os.path.join(model_dir, "model.onnx")
        if quantize:
            model_file = os.path.join(model_dir, "model_quant.onnx")
        if not os.path.exists(model_file):
            print(".onnx does not exist, begin to export onnx")
            try:
                from funasr import AutoModel
            except:
                raise "You are exporting onnx, please install funasr and try it again. To install funasr, you could:\n" "\npip3 install -U funasr\n" "For the users in China, you could install with the command:\n" "\npip3 install -U funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple"

            model = AutoModel(model=model_dir)
            model_dir = model.export(type="onnx", quantize=quantize, **kwargs)

        config_file = os.path.join(model_dir, "config.yaml")
        cmvn_file = os.path.join(model_dir, "am.mvn")
        config = read_yaml(config_file)

        self.tokenizer = SentencepiecesTokenizer(
            bpemodel=os.path.join(model_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model")
        )
        config["frontend_conf"]["cmvn_file"] = cmvn_file
        self.frontend = WavFrontend(**config["frontend_conf"])
        self.ort_infer = OrtInferSession(
            model_file, device_id, intra_op_num_threads=intra_op_num_threads
        )
        self.ort_infer_emb = OrtInferSession(
            os.path.join(model_dir, "sensevoice_model_hot_emb.onnx"), device_id,
            intra_op_num_threads=intra_op_num_threads
        )
        self.ort_infer_hot_module = OrtInferSession(
            os.path.join(model_dir, "sensevoice_model_hot_module.onnx"), device_id,
            intra_op_num_threads=intra_op_num_threads
        )
        self.ort_infer_nohot_module = OrtInferSession(
            os.path.join(model_dir, "sensevoice_model_nohot_module.onnx"), device_id,
            intra_op_num_threads=intra_op_num_threads
        )
        self.batch_size = batch_size
        self.blank_id = 0

    def proc_hotword(self, hotwords):
        hotword_str_list = hotwords.split("|")
        if len(hotword_str_list) > 0:
            hotword_list = [np.array(self.tokenizer.encode(i), dtype=np.int64) for i in hotword_str_list]
            hotword_list.insert(0, np.array([1], dtype=np.int64))
        else:
            hotword_list = [np.array([1], dtype=np.int64)]
        hotwords_length = np.array([len(i) for i in hotword_list], dtype=np.int32)
        max_length = max([len(arr) for arr in hotword_list])
        hotwords = pad_list(hotword_list, pad_value=-1, max_len=max_length)
        return hotwords.astype(np.int64), hotwords_length

    def __call__(self, wav_content: Union[str, np.ndarray, List[str]], hotwords_str: str, hotwords_score=1.0, **kwargs):
        hotwords_use = False
        if hotwords_str != '':
            print('当前热词：', hotwords_str.split('|'))
            hotwords_use = True
            hotwords_list, hotwords_length = self.proc_hotword(hotwords_str)
            context_emb = self.ort_infer_emb([hotwords_list, hotwords_length])[0]
            hotwords_score = np.array([hotwords_score], dtype=np.float32)

        waveform_list = self.load_data(wav_content, self.frontend.opts.frame_opts.samp_freq)
        waveform_nums = len(waveform_list)

        asr_res = []
        for beg_idx in range(0, waveform_nums, self.batch_size):
            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])
            encoder_out, encoder_out_lens = self.infer(
                feats,
                feats_len,
            )
            if hotwords_use == True:
                ctc_logits = self.ort_infer_hot_module([encoder_out, context_emb, hotwords_score])[0]
            else:
                ctc_logits = self.ort_infer_nohot_module([encoder_out])[0]

            for b in range(feats.shape[0]):
                # back to torch.Tensor
                if isinstance(ctc_logits, np.ndarray):
                    ctc_logits = torch.from_numpy(ctc_logits).float()
                # support batch_size=1 only currently
                x = ctc_logits[b, : encoder_out_lens[b].item(), :]
                yseq = x.argmax(dim=-1)
                yseq = torch.unique_consecutive(yseq, dim=-1)

                mask = yseq != self.blank_id
                token_int = yseq[mask].tolist()

                asr_res.append(self.tokenizer.decode(token_int))

        return asr_res

    def load_data(self, wav_content: Union[str, np.ndarray, List[str]], fs: int = None) -> List:
        def load_wav(path: str) -> np.ndarray:
            waveform, _ = librosa.load(path, sr=fs)
            return waveform

        if isinstance(wav_content, np.ndarray):
            return [wav_content]

        if isinstance(wav_content, str):
            return [load_wav(wav_content)]

        if isinstance(wav_content, list):
            return [load_wav(path) for path in wav_content]

        raise TypeError(f"The type of {wav_content} is not in [str, np.ndarray, list]")

    def extract_feat(self, waveform_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        feats, feats_len = [], []
        for waveform in waveform_list:
            speech, _ = self.frontend.fbank(waveform)
            feat, feat_len = self.frontend.lfr_cmvn(speech)
            feats.append(feat)
            feats_len.append(feat_len)

        feats = self.pad_feats(feats, np.max(feats_len))
        feats_len = np.array(feats_len).astype(np.int32)
        return feats, feats_len

    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, "constant", constant_values=0)

        feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats

    def infer(
            self,
            feats: np.ndarray,
            feats_len: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        outputs = self.ort_infer([feats, feats_len])
        return outputs