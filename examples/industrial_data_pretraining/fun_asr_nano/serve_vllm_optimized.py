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
import logging.handlers
import os
import time
import warnings

import numpy as np
import soundfile as sf
import torch
import librosa

warnings.filterwarnings('ignore')

# logger 占位符；在 main() 解析完参数后通过 setup_logging() 真正初始化
logger = logging.getLogger(__name__)


def setup_logging(
    log_dir: str = "/logs",
    log_days: int = 15,
    level: int = logging.INFO,
) -> None:
    """配置生产级日志：控制台 + 按天滚动文件，自动删除超过 log_days 天的旧日志。

    Args:
        log_dir:  日志目录（容器内路径或宿主机挂载路径）。
        log_days: 最多保留的天数（含当天），超出则在滚动时删除。
        level:    日志级别，默认 INFO。
    """
    os.makedirs(log_dir, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # 1. 控制台 handler（保持与原来行为一致）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    # 2. 按天滚动文件 handler
    log_file = os.path.join(log_dir, "serve_vllm.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",          # 每天凌晨滚动
        interval=1,
        backupCount=log_days,    # 保留最近 log_days 个备份文件
        encoding="utf-8",
        utc=False,
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    logger.info(
        f"Logging initialized → dir={log_dir}, keep_days={log_days}, "
        f"level={logging.getLevelName(level)}"
    )

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
# 推理专用线程池（workers=1 对应单 GPU，避免 vLLM 并发冲突）
_thread_pool: concurrent.futures.ThreadPoolExecutor = None
# IO 专用线程池：sf.read / librosa.resample 等纯 CPU 操作放这里，
# 与推理池分离，避免占用唯一的推理槽位。
_io_thread_pool: concurrent.futures.ThreadPoolExecutor = None
# ClusterBackend 单例：避免每次 SPK 请求都重建并迁移到 GPU
_cluster_backend = None

import threading
# SPK 模型全局锁：ClusterBackend / _spk_model.generate 内部可能有共享 cache。
# VAD 已确认线程安全，无需加锁。
_spk_lock = threading.Lock()


# ============================================================
# Utilities
# ============================================================

def truncate_repetition(text: str, min_repeat_len: int = 3, max_repeats: int = 3) -> str:
    """检测并截断 ASR 输出中的重复模式。

    注：chunk * max_repeats 不可避免地会产生临时字符串，但 ASR 输出一般不超过 500
    字符，周期 ≤ 27，实际开销微不足道。
    """
    if not text or len(text) < 20:
        return text
    n = len(text)
    for length in range(min_repeat_len, min(n // max_repeats, 30)):
        for start in range(n - length * max_repeats):
            chunk = text[start: start + length]
            if text[start: start + length * max_repeats] == chunk * max_repeats:
                return text[: start + length]
    return text


def prepare_audio_for_inference(audio_data: np.ndarray, sr: int, target_sr: int = 16000):
    """返回单声道 float32、target_sr 采样率的音频。"""
    audio_data = np.asarray(audio_data)
    if audio_data.ndim > 1:
        channel_axis = -1 if audio_data.shape[-1] <= audio_data.shape[0] else 0
        audio_data = audio_data.mean(axis=channel_axis)
    if sr != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio_data.astype(np.float32), sr


# ============================================================
# Engine loading  (同步，只在启动时调用一次)
# ============================================================

def load_engine(args):
    """加载所有模型（阻塞，仅在进程启动时调用）。"""
    global _engine, _vad_model, _spk_model, _args, _thread_pool, _io_thread_pool, _cluster_backend
    _args = args

    # 修复：以 _thread_pool 作为幂等判断（它是最后初始化的，
    # 只有全部模型都加载成功后才不为 None；避免半初始化状态被跳过）
    if _thread_pool is not None:
        return

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
    _vad_model = AutoModel(model=args.vad_model, device=args.device, disable_update=True, max_single_segment_time=30000,)

    if args.spk_model:
        logger.info(f"Loading SPK: {args.spk_model}")
        spk_kwargs = {"device": args.device, "disable_update": True}
        
        # 修复 FunASR Bug：如果 SPK_MODEL 传了本地绝对路径，由于缺少 config.yaml，
        # AutoModel 会把路径当成类名去 registry 查，导致 "xxx is not registered" 报错。
        # 这里我们在传入本地路径时，自动猜测其真实的 registered_name。
        if os.path.isdir(args.spk_model):
            spk_kwargs["model_path"] = args.spk_model
            name_lower = args.spk_model.lower()
            if "eres2netv2" in name_lower:
                spk_kwargs["model"] = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
            elif "cam++" in name_lower or "campplus" in name_lower:
                spk_kwargs["model"] = "iic/speech_campplus_sv_zh-cn_16k-common"
            else:
                spk_kwargs["model"] = args.spk_model
        else:
            spk_kwargs["model"] = args.spk_model

        _spk_model = AutoModel(**spk_kwargs)
        # 性能优化：提前构建 ClusterBackend 单例并迁移到目标设备，
        # 避免每次请求重建 + GPU 迁移的开销。
        _cluster_backend = ClusterBackend(merge_thr=0.78).to(args.device)
        logger.info("ClusterBackend singleton initialized")
    else:
        logger.info("SPK disabled")

    # IO 线程池：纯 CPU 操作（sf.read / librosa.resample）专用，
    # 与推理池分离，避免占用唯一的推理槽位。
    _io_thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=4,  # IO 操作可多线程，4 已经绰绰有余
        thread_name_prefix="io_worker",
    )
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
    use_timestamp: bool = False,
) -> dict:
    """核心处理：VAD 分段 → vLLM 批量 ASR → 时间戳 → SPK。

    纯同步函数，由调用方通过 run_in_executor 放入线程池执行。
    任何异常都会被 re-raise，由调用方（FastAPI handler）捕获并返回 500。
    """
    try:
        audio_data, sr = prepare_audio_for_inference(audio_data, sr)
        _t_start = time.perf_counter()

        # VAD 分段（VAD 模型已确认线程安全，无需加锁，并发请求可同时跑）
        if use_vad and len(audio_data) > sr * 1:
            vad_res = _vad_model.generate(input=audio_data, fs=sr)
            segments = vad_res[0]["value"]
        else:
            segments = [[0, int(len(audio_data) * 1000 / sr)]]
        _t_vad = time.perf_counter()
        logger.debug(
            f"[process_audio] VAD: dur={len(audio_data)/sr:.1f}s "
            f"segs={len(segments)} vad_time={_t_vad-_t_start:.3f}s"
        )

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
        # max_new_tokens：每个 VAD 段通常是 1-2 句话，150 token 绰绰有余。
        # 降低此值可以减少 vLLM 自回归循环次数，明显缩短生成耗时。
        gen_kwargs = {"max_new_tokens": int(os.environ.get("MAX_NEW_TOKENS", 150))}
        if language:
            gen_kwargs["language"] = language
        if hotwords:
            gen_kwargs["hotwords"] = hotwords

        results = _engine.generate(inputs=seg_audios, **gen_kwargs)
        _t_asr = time.perf_counter()
        logger.info(
            f"[process_audio] PERF dur={len(audio_data)/sr:.1f}s segs={len(seg_audios)} "
            f"vad={_t_vad-_t_start:.3f}s asr={_t_asr-_t_vad:.3f}s total={_t_asr-_t_start:.3f}s"
        )

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

        # 说话人识别（可选，失败不影响 ASR 结果）
        if use_spk and _spk_model is not None and _SPK_UTILS_AVAILABLE:
            try:
                vad_segs = [
                    [s["start"], s["end"], audio_data[int(s["start"] * sr): int(s["end"] * sr)]]
                    for s in output_segments
                ]
                chunks = sv_chunk(vad_segs)
                if chunks:
                    speech_list = [ch[2] for ch in chunks]
                    # 加锁：_spk_model.generate 内部可能有共享 cache
                    with _spk_lock:
                        spk_res = _spk_model.generate(input=speech_list, cache={}, is_final=True)
                    embs = torch.cat([r["spk_embedding"] for r in spk_res], dim=0)
                    # 注：ClusterBackend 存在并发状态问题（内部缓存可能互相覆盖），
                    # 每次新建实例是最安全的方式。它是纯 CPU 低开销操作，无需 GPU 迁移。
                    cluster = ClusterBackend(merge_thr=0.78)
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

    except Exception:
        logger.exception("process_audio failed (will return 500 to caller)")
        raise


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
    # 修复：使用 get_running_loop()，Python 3.10+ 中 get_event_loop()
    # 在非主线程/非 asyncio 上下文里会抛 DeprecationWarning 并最终报错。
    loop = asyncio.get_running_loop()
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
    logger.info(
        f"[/asr] recv filename={file.filename!r} "
        f"language={language!r} spk={spk} timestamp={timestamp}"
    )
    content = await file.read()
    file_size = len(content)
    # 性能优化：sf.read 是纯 CPU/IO 操作，走独立 IO 线程池，
    # 不与 vLLM 推理争抢唯一的 _thread_pool 槽位。
    loop = asyncio.get_running_loop()
    audio_data, sr = await loop.run_in_executor(
        _io_thread_pool, lambda: sf.read(io.BytesIO(content))
    )
    hw_list = [w.strip() for w in hotwords.split(",") if w.strip()] if hotwords else None
    t0 = time.perf_counter()
    result = await _run_in_pool(
        process_audio, audio_data, sr, language, hw_list, True, spk, timestamp,
    )
    t1 = time.perf_counter()
    elapsed = round(t1 - t0, 3)
    result["processing_time"] = elapsed
    result["rtf"] = round(elapsed / result["duration"], 4) if result["duration"] > 0 else 0
    logger.info(
        f"[/asr] done filename={file.filename!r} size={file_size}B "
        f"dur={result['duration']:.2f}s proc={elapsed}s rtf={result['rtf']} "
        f"segs={len(result['segments'])} text_len={len(result['text'])}"
    )
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
    logger.info(
        f"[/v1/audio/transcriptions] recv filename={file.filename!r} "
        f"model={model!r} language={language!r} fmt={response_format!r} spk={spk}"
    )
    content = await file.read()
    # 性能优化：sf.read 走 IO 线程池
    loop = asyncio.get_running_loop()
    audio_data, sr = await loop.run_in_executor(
        _io_thread_pool, lambda: sf.read(io.BytesIO(content))
    )
    use_ts = "word" in timestamp_granularities or "segment" in timestamp_granularities
    t0 = time.perf_counter()
    result = await _run_in_pool(
        process_audio, audio_data, sr, language, None, True, spk, use_ts,
    )
    elapsed = round(time.perf_counter() - t0, 3)
    logger.info(
        f"[/v1/audio/transcriptions] done filename={file.filename!r} "
        f"dur={result['duration']:.2f}s proc={elapsed}s "
        f"rtf={round(elapsed/result['duration'],4) if result['duration']>0 else 0}"
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
    client_id = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
    await websocket.accept()
    logger.info(f"[WS] connected client={client_id}")
    ws_t0 = time.perf_counter()

    vad = DynamicStreamingVAD(_vad_model)
    pcm_chunks: list = []            # 用 list 累积帧，延迟 concat
    _pcm_total_samples: list = [0]   # 用 list 包装 int，让嵌套函数可以修改
    locked_sentences: list = []
    language = None
    hotwords = None
    use_spk = False
    is_active = False
    last_sent_count = 0              # 上次 send_json 时的句子数量

    def _get_audio_buffer() -> np.ndarray:
        return np.concatenate(pcm_chunks) if pcm_chunks else np.array([], dtype=np.float32)

    async def _async_generate(seg_audio: np.ndarray) -> dict:
        gen_kw = {"max_new_tokens": int(os.environ.get("MAX_NEW_TOKENS", 150))}
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
            with _spk_lock:
                spk_res = _spk_model.generate(input=speech_list, cache={}, is_final=True)
            embs = torch.cat([r["spk_embedding"] for r in spk_res], dim=0)
            # 每次新建 ClusterBackend，避免全局单例并发覆盖
            cluster = ClusterBackend(merge_thr=0.78)
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
                    _pcm_total_samples[0] = 0
                    locked_sentences.clear()
                    last_sent_count = 0
                    is_active = True
                    ws_t0 = time.perf_counter()  # 重置计时（支持同一连接多次 START）
                    logger.info(f"[WS] START client={client_id} language={language!r} spk={use_spk}")
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

                        # 处理 finalize 产生的剩余段（改为 gather 并发，与实时路径保持一致）
                        final_segs = vad.finalize()
                        valid_final = [
                            (seg, audio_buffer[int(seg[0] * 16): int(seg[1] * 16)])
                            for seg in final_segs
                            if len(audio_buffer[int(seg[0] * 16): int(seg[1] * 16)]) > 8000
                        ]
                        if valid_final:
                            f_seg_list, f_audio_list = zip(*valid_final)
                            f_results = await asyncio.gather(
                                *[_async_generate(a) for a in f_audio_list],
                                return_exceptions=True,
                            )
                            for seg, res in zip(f_seg_list, f_results):
                                if isinstance(res, Exception):
                                    logger.warning(f"[WS] finalize _async_generate failed seg={seg}: {res}")
                                    continue
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

                        dur_ms = int(len(audio_buffer) * 1000 / 16000)
                        elapsed_s = round(time.perf_counter() - ws_t0, 3)
                        logger.info(
                            f"[WS] STOP client={client_id} "
                            f"dur={dur_ms/1000:.2f}s proc={elapsed_s}s "
                            f"rtf={round(elapsed_s/(dur_ms/1000),4) if dur_ms>0 else 0} "
                            f"sentences={len(locked_sentences)}"
                        )
                        await websocket.send_json({
                            "sentences": locked_sentences,
                            "is_final": True,
                            "duration_ms": dur_ms,
                        })
                        is_active = False
                    await websocket.send_json({"event": "stopped"})

            elif "bytes" in message and is_active:
                pcm = np.frombuffer(message["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                pcm_chunks.append(pcm)
                _pcm_total_samples[0] += len(pcm)  # O(1) 计数，无需 concat

                # vad.feed 移入线程池：它内部可能有 GPU 操作或较重 CPU 工作，
                # 放入线程池可避免阻塞当前连接的事件循环。
                # 每个 WS 连接的 vad 是独立实例，不需加全局锁。
                new_confirmed = await _run_in_pool(vad.feed, torch.from_numpy(pcm).float())

                if new_confirmed:
                    # 多 VAD 段并发推理，避免串行 await 导致后续 PCM 帧积压丢失。
                    # _get_audio_buffer() 只调用一次，所有段共用同一份合并结果。
                    audio_buffer = _get_audio_buffer()
                    valid_segs = [
                        (seg, audio_buffer[int(seg[0] * 16): int(seg[1] * 16)])
                        for seg in new_confirmed
                        if len(audio_buffer[int(seg[0] * 16): int(seg[1] * 16)]) > 8000
                    ]
                    if valid_segs:
                        seg_list, audio_list = zip(*valid_segs)
                        # asyncio.gather 并发提交所有段到线程池
                        results = await asyncio.gather(
                            *[_async_generate(a) for a in audio_list],
                            return_exceptions=True,
                        )
                        for seg, res in zip(seg_list, results):
                            if isinstance(res, Exception):
                                logger.warning(f"[WS] _async_generate failed for seg={seg}: {res}")
                                continue
                            if res["text"].strip():
                                locked_sentences.append({
                                    "text": res["text"], "start": seg[0], "end": seg[1],
                                })

                # 只在有新句子时推送
                if len(locked_sentences) > last_sent_count:
                    last_sent_count = len(locked_sentences)
                    # 性能优化：直接使用计数器，避免 sum(len(c) for c in pcm_chunks)
                    await websocket.send_json({
                        "sentences": locked_sentences,
                        "is_final": False,
                        "duration_ms": int(_pcm_total_samples[0] * 1000 / 16000),
                    })

    except WebSocketDisconnect:
        logger.info(f"[WS] disconnected client={client_id}")
    except Exception as e:
        logger.error(f"[WS] error client={client_id}: {e}", exc_info=True)


# ============================================================
# Health check
# ============================================================
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "FunASRNanoVLLM",
        "version": "2.0",
        "model": _args.model if _args else None,
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fun-ASR-Nano vLLM Optimized Server")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("SERVE_PORT", 8000)))
    parser.add_argument("--host", type=str,
                        default=os.environ.get("SERVE_HOST", "0.0.0.0"))
    parser.add_argument("--model", type=str,
                        default=os.environ.get("ASR_MODEL", "FunAudioLLM/Fun-ASR-Nano-2512"))
    parser.add_argument("--hub", type=str,
                        default=os.environ.get("MODEL_HUB", "ms"))
    parser.add_argument("--device", type=str,
                        default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--dtype", type=str,
                        default=os.environ.get("DTYPE", "bf16"))
    parser.add_argument("--max-model-len", type=int,
                        default=int(os.environ.get("MAX_MODEL_LEN", 4096)))
    parser.add_argument("--gpu-memory-utilization", type=float,
                        default=float(os.environ.get("GPU_MEM_UTIL", 0.5)))
    parser.add_argument("--vad-model", type=str,
                        default=os.environ.get("VAD_MODEL", "fsmn-vad"))
    parser.add_argument("--spk-model", type=str,
                        default=os.environ.get(
                            "SPK_MODEL",
                            "iic/speech_eres2netv2_sv_zh-cn_16k-common",
                        ))
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SERVE_WORKERS", 1)),
                        help="线程池大小（通常 1，与单 GPU 对应）")
    # ── 日志参数（同时支持环境变量，方便在 Docker 里覆盖） ──
    parser.add_argument(
        "--log-dir",
        type=str,
        default=os.environ.get("LOG_DIR", "/logs"),
        help="日志目录，默认 /logs（容器内，建议通过 -v 挂载到宿主机）",
    )
    parser.add_argument(
        "--log-days",
        type=int,
        default=int(os.environ.get("LOG_DAYS", 15)),
        help="日志按天滚动，保留最近 N 天，默认 15",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别，默认 INFO",
    )
    _args = parser.parse_args()

    # 初始化日志（在模型加载之前，确保所有加载日志都能落盘）
    setup_logging(
        log_dir=_args.log_dir,
        log_days=_args.log_days,
        level=getattr(logging, _args.log_level.upper()),
    )

    logger.info(
        f"Starting serve_vllm_optimized — "
        f"model={_args.model!r} host={_args.host} port={_args.port} "
        f"device={_args.device} dtype={_args.dtype} "
        f"gpu_mem={_args.gpu_memory_utilization} workers={_args.workers}"
    )

    # 模型在主线程同步加载完毕后再启动 uvicorn
    # 避免在 asyncio 内部触发 vLLM 初始化导致的死锁
    load_engine(_args)
    logger.info(f"Server ready → http://{_args.host}:{_args.port}")
    uvicorn.run(app, host=_args.host, port=_args.port)
