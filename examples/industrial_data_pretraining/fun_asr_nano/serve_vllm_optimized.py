#!/usr/bin/env python3
"""Fun-ASR-Nano vLLM 优化推理服务 (serve_vllm_optimized.py)

与 serve_vllm.py 的核心区别：
1. 独立进程部署 —— vLLM 完全与 asr_gateway 进程隔离，初始化/崩溃不影响网关
2. 所有 CPU/GPU 密集计算通过 ThreadPoolExecutor 执行，不阻塞 asyncio 事件循环
3. audio_buffer 使用 list + 延迟合并，避免 O(n²) 内存拷贝
4. WebSocket 只在 VAD 段产生变化时推送，减少无效传输
5. SPK/CAMPPlus import 提升到模块顶层
6. 新增 GET /health 健康检查接口

接口（与 serve_vllm.py 完全兼容）：
  POST /asr                      文件上传
  POST /v1/audio/transcriptions  OpenAI Whisper 兼容
  WS   /ws                       流式 WebSocket
  GET  /health                   健康检查

用法：
  CUDA_VISIBLE_DEVICES=0 python serve_vllm_optimized.py --port 8000
  CUDA_VISIBLE_DEVICES=0 python serve_vllm_optimized.py --port 8000 --workers 1
"""

import asyncio
import argparse
import concurrent.futures
import io
import logging
import os
import time
import warnings

import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    raise ImportError("pip install fastapi uvicorn python-multipart")

from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM
from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD
from funasr import AutoModel

# 提前 import SPK 工具，避免请求路径里动态 import 开销
try:
    from funasr.models.campplus.utils import sv_chunk, postprocess, distribute_spk
    from funasr.models.campplus.cluster_backend import ClusterBackend
    _SPK_UTILS_AVAILABLE = True
except ImportError:
    _SPK_UTILS_AVAILABLE = False
    logger.warning("CAMPPlus SPK utils not available; speaker diarization disabled")


# ============================================================
# Global state
# ============================================================
_engine: FunASRNanoVLLM = None
_vad_model = None
_spk_model = None
_args = None
# 专用线程池：所有 CPU/GPU 密集任务在此池中运行，不阻塞 asyncio 事件循环
_thread_pool: concurrent.futures.ThreadPoolExecutor = None


# ============================================================
# Utilities
# ============================================================

def truncate_repetition(text: str, min_repeat_len: int = 3, max_repeats: int = 3) -> str:
    """检测并截断 ASR 输出中的重复模式。"""
    if not text or len(text) < 20:
        return text
    n = len(text)
    for length in range(min_repeat_len, min(n // max_repeats, 30)):
        for start in range(n - length * max_repeats):
            chunk = text[start:start + length]
            if text[start:start + length * max_repeats] == chunk * max_repeats:
                return text[:start + length]
    return text


def prepare_audio_for_inference(audio_data: np.ndarray, sr: int, target_sr: int = 16000):
    """返回单声道 float32、target_sr 采样率的音频。"""
    audio_data = np.asarray(audio_data)
    if audio_data.ndim > 1:
        channel_axis = -1 if audio_data.shape[-1] <= audio_data.shape[0] else 0
        audio_data = audio_data.mean(axis=channel_axis)
    if sr != target_sr:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio_data.astype(np.float32), sr


# ============================================================
# Engine loading  (同步，只在启动时调用一次)
# ============================================================

def load_engine(args):
    """加载所有模型（阻塞，仅在进程启动时调用）。"""
    global _engine, _vad_model, _spk_model, _args, _thread_pool
    _args = args

    if _engine is not None:
        return  # 幂等

    logger.info(f"Loading vLLM engine: {args.model}")
    _engine = FunASRNanoVLLM.from_pretrained(
        model=args.model,
        hub=args.hub,
        device=args.device,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    logger.info(f"Loading VAD: {args.vad_model}")
    _vad_model = AutoModel(model=args.vad_model, device=args.device, disable_update=True)

    if args.spk_model:
        logger.info(f"Loading SPK: {args.spk_model}")
        _spk_model = AutoModel(model=args.spk_model, device=args.device, disable_update=True)
    else:
        logger.info("SPK disabled")

    _thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers,
        thread_name_prefix="vllm_worker",
    )
    logger.info("All models ready!")


# ============================================================
# Core processing  (同步，运行在 _thread_pool 里)
# ============================================================

def process_audio(
    audio_data: np.ndarray,
    sr: int = 16000,
    language: str = None,
    hotwords=None,
    use_vad: bool = True,
    use_spk: bool = False,
    use_timestamp: bool = True,
) -> dict:
    """核心处理：VAD 分段 → vLLM 批量 ASR → 时间戳 → SPK。

    纯同步函数，由调用方通过 run_in_executor 放入线程池执行。
    """
    audio_data, sr = prepare_audio_for_inference(audio_data, sr)

    # VAD 分段
    if use_vad and len(audio_data) > sr * 1:
        vad_res = _vad_model.generate(input=audio_data, fs=sr)
        segments = vad_res[0]["value"]
    else:
        segments = [[0, int(len(audio_data) * 1000 / sr)]]

    if not segments:
        return {"text": "", "segments": [], "duration": len(audio_data) / sr}

    seg_audios = []
    seg_times = []
    for seg in segments:
        s0 = int(seg[0] * sr / 1000)
        s1 = int(seg[1] * sr / 1000)
        seg_audio = audio_data[s0:s1]
        if len(seg_audio) > sr * 0.3:
            seg_audios.append(seg_audio)
            seg_times.append((seg[0], seg[1]))

    if not seg_audios:
        return {"text": "", "segments": [], "duration": len(audio_data) / sr}

    # vLLM 批量 ASR（所有段一次投入，充分利用 batch throughput）
    gen_kwargs = {"max_new_tokens": 500}
    if language:
        gen_kwargs["language"] = language
    if hotwords:
        gen_kwargs["hotwords"] = hotwords

    results = _engine.generate(inputs=seg_audios, **gen_kwargs)

    output_segments = []
    full_text_parts = []

    for r, (start_ms, end_ms) in zip(results, seg_times):
        r["text"] = truncate_repetition(r["text"])
        seg_info = {
            "text": r["text"],
            "start": start_ms / 1000,
            "end": end_ms / 1000,
        }
        if use_timestamp and "timestamps" in r:
            offset = start_ms / 1000
            seg_info["words"] = [
                {
                    "word": ts["token"],
                    "start": ts["start_time"] + offset,
                    "end": ts["end_time"] + offset,
                }
                for ts in r["timestamps"]
            ]
        output_segments.append(seg_info)
        full_text_parts.append(r["text"])

    # 说话人识别（可选）
    if use_spk and _spk_model is not None and _SPK_UTILS_AVAILABLE:
        try:
            vad_segs = [
                [s["start"], s["end"], audio_data[int(s["start"] * sr): int(s["end"] * sr)]]
                for s in output_segments
            ]
            chunks = sv_chunk(vad_segs)
            if chunks:
                speech_list = [ch[2] for ch in chunks]
                spk_res = _spk_model.generate(input=speech_list, cache={}, is_final=True)
                embs = torch.cat([r["spk_embedding"] for r in spk_res], dim=0)
                cluster = ClusterBackend(merge_thr=0.78).to(_args.device)
                labels = cluster(embs.cpu(), oracle_num=None)
                if not isinstance(labels, np.ndarray):
                    labels = np.array(labels)
                all_sorted = sorted(chunks, key=lambda x: x[0])
                sv_output = postprocess(all_sorted, None, labels, embs.cpu())
                sentences = [
                    {"text": s["text"], "start": int(s["start"] * 1000), "end": int(s["end"] * 1000)}
                    for s in output_segments
                ]
                distribute_spk(sentences, sv_output)
                for i, s in enumerate(sentences):
                    output_segments[i]["speaker"] = f"SPK{s.get('spk', 0)}"
        except Exception as e:
            logger.warning(f"SPK diarization failed: {e}")

    return {
        "text": " ".join(full_text_parts),
        "segments": output_segments,
        "duration": len(audio_data) / sr,
    }


def build_openai_verbose_json(result: dict, language: str = None) -> dict:
    segments = []
    for i, seg in enumerate(result["segments"]):
        item = {
            "id": i,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "words": seg.get("words", []),
        }
        if "speaker" in seg:
            item["speaker"] = seg["speaker"]
        segments.append(item)
    return {
        "task": "transcribe",
        "language": language or "zh",
        "duration": result["duration"],
        "text": result["text"],
        "segments": segments,
    }


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="Fun-ASR-Nano vLLM Optimized Server", version="2.0")


@app.on_event("startup")
async def startup():
    if _engine is None:
        raise RuntimeError("Engine not loaded. Call load_engine() before uvicorn.run().")


async def _run_in_pool(fn, *args, **kwargs):
    """把同步函数提交到专用线程池，返回协程，不阻塞事件循环。"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_thread_pool, lambda: fn(*args, **kwargs))


# --- HTTP REST: POST /asr ---
@app.post("/asr")
async def asr_endpoint(
    file: UploadFile = File(...),
    language: str = Form(default=None),
    hotwords: str = Form(default=""),
    spk: bool = Form(default=False),
    timestamp: bool = Form(default=True),
):
    """ASR 文件上传接口，返回 text + segments + timestamps + speaker。"""
    content = await file.read()
    audio_data, sr = await _run_in_pool(sf.read, io.BytesIO(content))
    hw_list = [w.strip() for w in hotwords.split(",") if w.strip()] if hotwords else None
    t0 = time.perf_counter()
    result = await _run_in_pool(
        process_audio, audio_data, sr, language, hw_list, True, spk, timestamp,
    )
    t1 = time.perf_counter()
    result["processing_time"] = round(t1 - t0, 3)
    result["rtf"] = round((t1 - t0) / result["duration"], 4) if result["duration"] > 0 else 0
    return JSONResponse(content=result)


# --- OpenAI API: POST /v1/audio/transcriptions ---
@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default="fun-asr-nano"),
    language: str = Form(default=None),
    response_format: str = Form(default="json"),
    timestamp_granularities: str = Form(default="word"),
    spk: bool = Form(default=False),
):
    """OpenAI Whisper 兼容转写接口（扩展支持 spk）。"""
    content = await file.read()
    audio_data, sr = await _run_in_pool(sf.read, io.BytesIO(content))
    use_ts = "word" in timestamp_granularities or "segment" in timestamp_granularities
    result = await _run_in_pool(
        process_audio, audio_data, sr, language, None, True, spk, use_ts,
    )
    if response_format == "text":
        return JSONResponse(content=result["text"])
    elif response_format == "verbose_json":
        return JSONResponse(content=build_openai_verbose_json(result, language=language))
    else:
        return JSONResponse(content={"text": result["text"]})


# --- WebSocket: ws://host:port/ws ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """流式 WebSocket ASR。

    改进点：
    - audio_buffer 用 list 累积，只在切片时合并，避免 O(n²) 拷贝
    - VAD 确认段通过 run_in_executor 异步推理，不卡接收循环
    - 只在 locked_sentences 有新增时才 send_json
    - STOP 时 SPK 聚类也异步化
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: {websocket.client}")

    vad = DynamicStreamingVAD(_vad_model)
    pcm_chunks: list = []            # 用 list 累积帧，延迟 concat
    locked_sentences: list = []
    language = None
    hotwords = None
    use_spk = False
    is_active = False
    last_sent_count = 0              # 上次 send_json 时的句子数量

    def _get_audio_buffer() -> np.ndarray:
        return np.concatenate(pcm_chunks) if pcm_chunks else np.array([], dtype=np.float32)

    async def _async_generate(seg_audio: np.ndarray) -> dict:
        gen_kw = {"max_new_tokens": 500}
        if language:
            gen_kw["language"] = language
        if hotwords:
            gen_kw["hotwords"] = hotwords
        res = await _run_in_pool(_engine.generate, [seg_audio], **gen_kw)
        return res[0]

    async def _run_spk(audio_buffer: np.ndarray):
        """说话人聚类，线程池异步执行。"""
        if not (_SPK_UTILS_AVAILABLE and _spk_model is not None):
            return

        def _worker():
            vad_segs = [
                [s["start"] / 1000, s["end"] / 1000,
                 audio_buffer[int(s["start"] * 16): int(s["end"] * 16)]]
                for s in locked_sentences
            ]
            chunks = sv_chunk(vad_segs)
            if not chunks:
                return
            speech_list = [ch[2] for ch in chunks]
            spk_res = _spk_model.generate(input=speech_list, cache={}, is_final=True)
            embs = torch.cat([r["spk_embedding"] for r in spk_res], dim=0)
            cluster = ClusterBackend(merge_thr=0.78).to(_args.device)
            labels = cluster(embs.cpu(), oracle_num=None)
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels)
            all_sorted = sorted(chunks, key=lambda x: x[0])
            sv_output = postprocess(all_sorted, None, labels, embs.cpu())
            spk_sents = [
                {"text": s["text"], "start": int(s["start"]), "end": int(s["end"])}
                for s in locked_sentences
            ]
            distribute_spk(spk_sents, sv_output)
            for i, ss in enumerate(spk_sents):
                locked_sentences[i]["spk"] = ss.get("spk", 0)

        try:
            await _run_in_pool(_worker)
        except Exception as e:
            logger.warning(f"SPK failed: {e}")

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                cmd = message["text"].strip()

                if cmd.upper() == "START":
                    vad.reset()
                    pcm_chunks.clear()
                    locked_sentences.clear()
                    last_sent_count = 0
                    is_active = True
                    await websocket.send_json({"event": "started"})

                elif cmd.upper().startswith("LANGUAGE:"):
                    language = cmd[9:].strip() or None
                    await websocket.send_json({"event": "language_set", "language": language})

                elif cmd.upper().startswith("HOTWORDS:"):
                    hotwords = [w.strip() for w in cmd[9:].split(",") if w.strip()]
                    await websocket.send_json({"event": "hotwords_set", "hotwords": hotwords})

                elif cmd.upper().startswith("SPK:"):
                    use_spk = cmd[4:].strip().lower() in ("true", "1", "on", "yes")
                    await websocket.send_json({"event": "spk_set", "spk": use_spk})

                elif cmd.upper() == "STOP":
                    if is_active and pcm_chunks:
                        audio_buffer = _get_audio_buffer()

                        # 处理 finalize 产生的剩余段
                        final_segs = vad.finalize()
                        for seg in final_segs:
                            seg_audio = audio_buffer[int(seg[0] * 16): int(seg[1] * 16)]
                            if len(seg_audio) > 8000:
                                res = await _async_generate(seg_audio)
                                if res["text"].strip():
                                    locked_sentences.append({
                                        "text": res["text"], "start": seg[0], "end": seg[1],
                                    })

                        # 处理仍在说话中的部分
                        if vad.is_speaking:
                            end_ms = int(len(audio_buffer) * 1000 / 16000)
                            start_ms = (
                                int(vad.current_speech_start)
                                if hasattr(vad, "current_speech_start") and vad.current_speech_start
                                else 0
                            )
                            seg_audio = audio_buffer[int(start_ms * 16):]
                            if len(seg_audio) > 8000:
                                res = await _async_generate(seg_audio)
                                if res["text"].strip():
                                    locked_sentences.append({
                                        "text": res["text"], "start": start_ms, "end": end_ms,
                                    })

                        # 说话人聚类（全量，仅在 STOP 时做一次）
                        if use_spk and locked_sentences:
                            await _run_spk(audio_buffer)

                        await websocket.send_json({
                            "sentences": locked_sentences,
                            "is_final": True,
                            "duration_ms": int(len(audio_buffer) * 1000 / 16000),
                        })
                        is_active = False
                    await websocket.send_json({"event": "stopped"})

            elif "bytes" in message and is_active:
                pcm = np.frombuffer(message["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                pcm_chunks.append(pcm)

                new_confirmed = vad.feed(torch.from_numpy(pcm).float())

                for seg in new_confirmed:
                    audio_buffer = _get_audio_buffer()
                    seg_audio = audio_buffer[int(seg[0] * 16): int(seg[1] * 16)]
                    if len(seg_audio) > 8000:
                        res = await _async_generate(seg_audio)
                        if res["text"].strip():
                            locked_sentences.append({
                                "text": res["text"], "start": seg[0], "end": seg[1],
                            })

                # 只在有新句子时推送
                if len(locked_sentences) > last_sent_count:
                    last_sent_count = len(locked_sentences)
                    total_samples = sum(len(c) for c in pcm_chunks)
                    await websocket.send_json({
                        "sentences": locked_sentences,
                        "is_final": False,
                        "duration_ms": int(total_samples * 1000 / 16000),
                    })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)


# ============================================================
# Health check
# ============================================================
@app.get("/health")
async def health():
    return {"status": "ok", "engine": "FunASRNanoVLLM", "version": "2.0"}


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fun-ASR-Nano vLLM Optimized Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model", type=str, default="FunAudioLLM/Fun-ASR-Nano-2512")
    parser.add_argument("--hub", type=str, default="ms")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--vad-model", type=str, default="fsmn-vad")
    parser.add_argument("--spk-model", type=str,
                        default="iic/speech_eres2netv2_sv_zh-cn_16k-common")
    parser.add_argument("--workers", type=int, default=1,
                        help="线程池大小（通常 1，与单 GPU 对应）")
    _args = parser.parse_args()

    # 模型在主线程同步加载完毕后再启动 uvicorn
    # 避免在 asyncio 内部触发 vLLM 初始化导致的死锁
    load_engine(_args)
    uvicorn.run(app, host=_args.host, port=_args.port)
