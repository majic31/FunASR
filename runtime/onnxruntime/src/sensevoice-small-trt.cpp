/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
 *
 * SenseVoiceSmall TensorRT 8.4 GPU implementation.
 *
 * Architecture notes (confirmed from model.py analysis):
 *  - SenseVoiceEncoderSmall: SANM layers (MultiHeadedAttentionSANM)
 *    Each SANM layer has:
 *      - FSMN block: Conv1d(n_feat, n_feat, kernel=11, groups=n_feat) + ConstantPad1d
 *        This is a depthwise conv with FIXED kernel — fully static, dynamic-T friendly.
 *      - Multi-head self-attention (QKV + softmax)
 *      - FFN (Linear → ReLU → Linear)
 *  - All ops are standard CUDA ops, TRT handles them natively.
 *  - Dynamic axes: {speech: [batch, feats_len], language/textnorm: [batch]}
 *  - No recurrence, no custom ops — clean ONNX export.
 *
 * ONNX model inputs (from export_meta.py):
 *   speech          [B, T, 560]     float32
 *   speech_lengths  [B]             int32
 *   language        [B]             int32
 *   textnorm        [B]             int32
 *
 * ONNX model outputs:
 *   ctc_logits      [B, T', vocab]  float32
 *   encoder_out_lens [B]            int32
 *
 * TRT Engine build (run once on target GPU, see export_trt.sh):
 *   trtexec --onnx=model.onnx --saveEngine=model.trt --fp16 \
 *     --minShapes=speech:1x10x560,speech_lengths:1,language:1,textnorm:1 \
 *     --optShapes=speech:4x200x560,speech_lengths:4,language:4,textnorm:4 \
 *     --maxShapes=speech:8x1500x560,speech_lengths:8,language:8,textnorm:8
 */

#include "precomp.h"
#include "sensevoice-small-trt.h"
#include <fstream>
#include <cstring>
#include <thread>   // std::this_thread::get_id()

using namespace std;
namespace funasr {

// ─────────────────────────────────────────────────────────────────
//  Static member definitions
// ─────────────────────────────────────────────────────────────────
TRTLogger SenseVoiceTRT::trt_logger_;
thread_local std::unique_ptr<TRTThreadCtx> SenseVoiceTRT::tls_ctx_;

// ─────────────────────────────────────────────────────────────────
//  TRTThreadCtx destructor — free all CUDA resources
// ─────────────────────────────────────────────────────────────────
TRTThreadCtx::~TRTThreadCtx() {
    // Free device buffers
    for (int i = 0; i < 6; ++i) {
        if (bindings[i]) { cudaFree(bindings[i]); bindings[i] = nullptr; }
    }
    // Free pinned host buffers
    if (host_logits) { cudaFreeHost(host_logits); host_logits = nullptr; }
    if (host_outlen) { cudaFreeHost(host_outlen); host_outlen = nullptr; }
    // Destroy context and stream
    if (ctx)    { ctx->destroy();           ctx    = nullptr; }
    if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
}

// ─────────────────────────────────────────────────────────────────
//  Constructor / Destructor
// ─────────────────────────────────────────────────────────────────
SenseVoiceTRT::SenseVoiceTRT() {}

SenseVoiceTRT::~SenseVoiceTRT() {
    // Thread-local ctx is destroyed by threads themselves.
    // We only clean up shared engine resources here.
    if (engine_)  { engine_->destroy();  engine_  = nullptr; }
    if (runtime_) { runtime_->destroy(); runtime_ = nullptr; }
    if (vocab)    { delete vocab;        vocab    = nullptr; }
}

// ─────────────────────────────────────────────────────────────────
//  LoadEngine: deserialize a serialized TRT engine from disk
// ─────────────────────────────────────────────────────────────────
bool SenseVoiceTRT::LoadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        LOG(ERROR) << "SenseVoiceTRT: engine file not found: " << engine_path;
        return false;
    }
    file.seekg(0, std::ios::end);
    size_t size = (size_t)file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    runtime_ = nvinfer1::createInferRuntime(trt_logger_);
    if (!runtime_) {
        LOG(ERROR) << "SenseVoiceTRT: failed to create TRT runtime";
        return false;
    }
    engine_ = runtime_->deserializeCudaEngine(buffer.data(), size);
    if (!engine_) {
        LOG(ERROR) << "SenseVoiceTRT: failed to deserialize engine from " << engine_path;
        return false;
    }
    LOG(INFO) << "SenseVoiceTRT: engine loaded from " << engine_path
              << "  (bindings=" << engine_->getNbBindings() << ")";
    return true;
}

// ─────────────────────────────────────────────────────────────────
//  InitAsr
// ─────────────────────────────────────────────────────────────────
void SenseVoiceTRT::InitAsr(const std::string& am_model,
                              const std::string& am_cmvn,
                              const std::string& am_config,
                              const std::string& token_file,
                              int thread_num) {
    LoadConfigFromYaml(am_config.c_str());

    fbank_opts_.frame_opts.dither      = 0;
    fbank_opts_.mel_opts.num_bins      = n_mels;
    fbank_opts_.frame_opts.samp_freq   = asr_sample_rate;
    fbank_opts_.frame_opts.window_type = window_type;
    fbank_opts_.frame_opts.frame_shift_ms  = frame_shift;
    fbank_opts_.frame_opts.frame_length_ms = frame_length;
    fbank_opts_.energy_floor           = 0;
    fbank_opts_.mel_opts.debug_mel     = false;

    if (vocab) { delete vocab; vocab = nullptr; }
    vocab = new Vocab(token_file.c_str());
    LoadCmvn(am_cmvn.c_str());

    // am_model points to a directory; look for model.trt inside it
    std::string engine_path = am_model;
    // If am_model is a directory path, append the filename
    if (am_model.back() == '/') {
        engine_path = am_model + "model.trt";
    } else {
        // Check if it ends with .trt already
        if (am_model.size() < 4 || am_model.substr(am_model.size()-4) != ".trt") {
            // Treat as directory
            engine_path = am_model + "/model.trt";
        }
    }

    if (!LoadEngine(engine_path)) {
        LOG(ERROR) << "SenseVoiceTRT: InitAsr failed — cannot load engine " << engine_path;
        exit(-1);
    }

    WarmUp();
}

// ─────────────────────────────────────────────────────────────────
//  GetOrCreateCtx — returns (or lazily creates) the per-thread context
// ─────────────────────────────────────────────────────────────────
TRTThreadCtx* SenseVoiceTRT::GetOrCreateCtx() {
    if (tls_ctx_) return tls_ctx_.get();

    auto tc = std::make_unique<TRTThreadCtx>();

    // Create an independent execution context for this thread
    tc->ctx = engine_->createExecutionContext();
    if (!tc->ctx) {
        LOG(ERROR) << "SenseVoiceTRT: failed to create IExecutionContext";
        return nullptr;
    }

    // Create a CUDA stream for this thread (async, no sync with other threads)
    if (cudaStreamCreate(&tc->stream) != cudaSuccess) {
        LOG(ERROR) << "SenseVoiceTRT: cudaStreamCreate failed";
        return nullptr;
    }

    // Resolve binding indices by name (engine guarantees stable names from ONNX)
    tc->idx_speech     = engine_->getBindingIndex("speech");
    tc->idx_speech_len = engine_->getBindingIndex("speech_lengths");
    tc->idx_language   = engine_->getBindingIndex("language");
    tc->idx_textnorm   = engine_->getBindingIndex("textnorm");
    tc->idx_ctc_logits = engine_->getBindingIndex("ctc_logits");
    tc->idx_out_lens   = engine_->getBindingIndex("encoder_out_lens");

    if (tc->idx_speech < 0 || tc->idx_ctc_logits < 0) {
        LOG(ERROR) << "SenseVoiceTRT: binding names not found in engine. "
                   << "Ensure engine was built from the correct ONNX model.";
        return nullptr;
    }

    tls_ctx_ = std::move(tc);
    LOG(INFO) << "SenseVoiceTRT: created per-thread execution context (thread_id="
              << std::this_thread::get_id() << ")";
    return tls_ctx_.get();
}

// ─────────────────────────────────────────────────────────────────
//  EnsureBuffers — (re)allocate GPU/host buffers only when needed
// ─────────────────────────────────────────────────────────────────
void SenseVoiceTRT::EnsureBuffers(TRTThreadCtx* tc,
                                   int64_t batch, int64_t T, int64_t feat_dim,
                                   int64_t vocab_size) {
    // speech: [B, T, feat_dim] float32
    size_t speech_bytes = (size_t)(batch * T * feat_dim) * sizeof(float);
    if (speech_bytes > tc->alloc_speech_bytes) {
        if (tc->bindings[tc->idx_speech]) cudaFree(tc->bindings[tc->idx_speech]);
        cudaMalloc(&tc->bindings[tc->idx_speech], speech_bytes);
        tc->alloc_speech_bytes = speech_bytes;
    }

    // speech_lengths, language, textnorm: [B] int32 — allocate once for max batch=8
    if (!tc->bindings[tc->idx_speech_len]) {
        cudaMalloc(&tc->bindings[tc->idx_speech_len], 8 * sizeof(int32_t));
    }

    // language: [B] int32
    if (!tc->bindings[tc->idx_language]) {
        cudaMalloc(&tc->bindings[tc->idx_language], 8 * sizeof(int32_t));
    }

    // textnorm: [B] int32
    if (!tc->bindings[tc->idx_textnorm]) {
        cudaMalloc(&tc->bindings[tc->idx_textnorm], 8 * sizeof(int32_t));
    }

    // ctc_logits: [B, T, vocab_size] float32
    //  T' (output length) == T (no subsampling in SenseVoiceSmall encoder)
    size_t logits_bytes = (size_t)(batch * T * vocab_size) * sizeof(float);
    if (logits_bytes > tc->alloc_logits_bytes) {
        if (tc->bindings[tc->idx_ctc_logits]) cudaFree(tc->bindings[tc->idx_ctc_logits]);
        cudaMalloc(&tc->bindings[tc->idx_ctc_logits], logits_bytes);
        tc->alloc_logits_bytes = logits_bytes;

        // Also grow pinned host buffer for async memcpy
        if (tc->host_logits) cudaFreeHost(tc->host_logits);
        cudaMallocHost((void**)&tc->host_logits, logits_bytes);
        tc->host_logits_bytes = logits_bytes;
    }

    // encoder_out_lens: [B] int32
    if (!tc->bindings[tc->idx_out_lens]) {
        cudaMalloc(&tc->bindings[tc->idx_out_lens], 8 * sizeof(int32_t));
        cudaMallocHost((void**)&tc->host_outlen, 8 * sizeof(int32_t));
    }
}

// ─────────────────────────────────────────────────────────────────
//  RunInference — set shapes, copy inputs, enqueue, copy outputs
// ─────────────────────────────────────────────────────────────────
std::string SenseVoiceTRT::RunInference(TRTThreadCtx* tc,
                                         const std::vector<float>& wav_feats,
                                         int64_t num_frames, int64_t feat_dim,
                                         int32_t svs_lid, int32_t svs_itnid) {
    const int32_t batch = 1;

    // Query vocab_size from engine's output binding
    // (we use max profile shape to size buffers conservatively)
    auto out_dims  = engine_->getBindingDimensions(tc->idx_ctc_logits);
    int64_t vocab_size = out_dims.d[2];  // last dim is vocab; may be -1 for dynamic but max is bounded
    if (vocab_size <= 0) vocab_size = 25055; // fallback: SenseVoiceSmall vocab

    EnsureBuffers(tc, batch, num_frames, feat_dim, vocab_size);

    // ── 1. Set dynamic input shapes ─────────────────────────────
    // Use nvinfer1::Dims directly (TRT 8.4 does not expose Dims3/Dims1 convenience types
    // as a public header typedef — use the base Dims struct instead)
    nvinfer1::Dims in_dims;
    in_dims.nbDims = 3;
    in_dims.d[0] = batch;
    in_dims.d[1] = (int)num_frames;
    in_dims.d[2] = (int)feat_dim;
    tc->ctx->setBindingDimensions(tc->idx_speech, in_dims);

    nvinfer1::Dims scalar_dims;
    scalar_dims.nbDims = 1;
    scalar_dims.d[0]   = batch;
    tc->ctx->setBindingDimensions(tc->idx_speech_len, scalar_dims);
    tc->ctx->setBindingDimensions(tc->idx_language,   scalar_dims);
    tc->ctx->setBindingDimensions(tc->idx_textnorm,   scalar_dims);

    if (!tc->ctx->allInputDimensionsSpecified()) {
        LOG(ERROR) << "SenseVoiceTRT: not all input dimensions specified";
        return "";
    }

    // ── 2. Async H→D copy of inputs ─────────────────────────────
    cudaMemcpyAsync(tc->bindings[tc->idx_speech],
                    wav_feats.data(),
                    wav_feats.size() * sizeof(float),
                    cudaMemcpyHostToDevice, tc->stream);

    int32_t h_len[1]      = { (int32_t)num_frames };
    int32_t h_lang[1]     = { svs_lid };
    int32_t h_textnorm[1] = { svs_itnid };

    cudaMemcpyAsync(tc->bindings[tc->idx_speech_len],
                    h_len, sizeof(int32_t), cudaMemcpyHostToDevice, tc->stream);
    cudaMemcpyAsync(tc->bindings[tc->idx_language],
                    h_lang, sizeof(int32_t), cudaMemcpyHostToDevice, tc->stream);
    cudaMemcpyAsync(tc->bindings[tc->idx_textnorm],
                    h_textnorm, sizeof(int32_t), cudaMemcpyHostToDevice, tc->stream);

    // ── 3. Inference (TRT 8.4 API: enqueueV2) ───────────────────
    void* all_bindings[6];
    all_bindings[tc->idx_speech]     = tc->bindings[tc->idx_speech];
    all_bindings[tc->idx_speech_len] = tc->bindings[tc->idx_speech_len];
    all_bindings[tc->idx_language]   = tc->bindings[tc->idx_language];
    all_bindings[tc->idx_textnorm]   = tc->bindings[tc->idx_textnorm];
    all_bindings[tc->idx_ctc_logits] = tc->bindings[tc->idx_ctc_logits];
    all_bindings[tc->idx_out_lens]   = tc->bindings[tc->idx_out_lens];

    bool ok = tc->ctx->enqueueV2(all_bindings, tc->stream, nullptr);
    if (!ok) {
        LOG(ERROR) << "SenseVoiceTRT: enqueueV2 failed";
        return "";
    }

    // ── 4. Get actual output shape after inference ──────────────
    auto out_shape = tc->ctx->getBindingDimensions(tc->idx_ctc_logits);
    int64_t T_out       = out_shape.d[1];
    int64_t vocab_out   = out_shape.d[2];

    // ── 5. Async D→H copy of outputs ────────────────────────────
    size_t logits_bytes = (size_t)(batch * T_out * vocab_out) * sizeof(float);
    cudaMemcpyAsync(tc->host_logits,
                    tc->bindings[tc->idx_ctc_logits],
                    logits_bytes,
                    cudaMemcpyDeviceToHost, tc->stream);
    cudaMemcpyAsync(tc->host_outlen,
                    tc->bindings[tc->idx_out_lens],
                    sizeof(int32_t),
                    cudaMemcpyDeviceToHost, tc->stream);

    // Wait for this stream (only blocks this thread, other threads run freely)
    cudaStreamSynchronize(tc->stream);

    // ── 6. CTC decoding ────────────────────────────────────────
    int32_t valid_len = tc->host_outlen[0];
    std::vector<int32_t> paraformer_length = { valid_len };
    std::vector<int64_t> outputShape = { batch, T_out, vocab_out };

    return CTCSearch(tc->host_logits, paraformer_length, outputShape);
}

// ─────────────────────────────────────────────────────────────────
//  WarmUp — run one dummy forward on the init thread to JIT-compile kernels
// ─────────────────────────────────────────────────────────────────
void SenseVoiceTRT::WarmUp() {
    int32_t in_feat_dim  = fbank_opts_.mel_opts.num_bins;
    int32_t feature_dim  = lfr_m * in_feat_dim; // 7 * 80 = 560
    int64_t num_frames   = 10;

    std::vector<float> dummy_feats((size_t)(num_frames * feature_dim), 0.1f);

    TRTThreadCtx* tc = GetOrCreateCtx();
    if (!tc) {
        LOG(ERROR) << "SenseVoiceTRT: WarmUp failed — cannot create context";
        return;
    }
    try {
        std::string result = RunInference(tc, dummy_feats, num_frames, feature_dim, 0, 15);
        LOG(INFO) << "SenseVoiceTRT: WarmUp completed successfully.";
    } catch (std::exception const& e) {
        LOG(ERROR) << "SenseVoiceTRT: WarmUp error: " << e.what();
    }
}

// ─────────────────────────────────────────────────────────────────
//  Forward — main inference path (called from decoder threads)
// ─────────────────────────────────────────────────────────────────
std::vector<std::string> SenseVoiceTRT::Forward(float** din, int* len,
                                                  bool input_finished,
                                                  std::string svs_lang,
                                                  bool svs_itn,
                                                  int batch_in) {
    std::vector<std::string> results;
    results.reserve(batch_in);

    int32_t in_feat_dim = fbank_opts_.mel_opts.num_bins;
    int32_t feature_dim = lfr_m * in_feat_dim;

    int64_t svs_lid   = 0;
    if (lid_map.count(svs_lang)) svs_lid = (int64_t)lid_map.at(svs_lang);
    int32_t svs_itnid = svs_itn ? 14 : 15;

    // Get (or create) the per-thread TRT context
    TRTThreadCtx* tc = GetOrCreateCtx();
    if (!tc) {
        for (int i = 0; i < batch_in; ++i) results.push_back("");
        return results;
    }

    // Process each sample sequentially within this thread's stream
    // (batch_in items are already dispatched to this thread by the caller)
    for (int index = 0; index < batch_in; ++index) {
        std::string result;
        try {
            // ── Feature extraction (CPU, same as SenseVoiceTorch) ─
            std::vector<std::vector<float>> asr_feats;
            FbankKaldi(asr_sample_rate, din[index], len[index], asr_feats);
            if (asr_feats.empty()) {
                results.push_back(result);
                continue;
            }
            LfrCmvn(asr_feats);

            int64_t num_frames = (int64_t)asr_feats.size();
            std::vector<float> wav_feats;
            wav_feats.reserve((size_t)(num_frames * feature_dim));
            for (const auto& f : asr_feats)
                wav_feats.insert(wav_feats.end(), f.begin(), f.end());

            // ── GPU inference via TRT ──────────────────────────────
            result = RunInference(tc, wav_feats, num_frames, feature_dim,
                                  (int32_t)svs_lid, svs_itnid);
        } catch (std::exception const& e) {
            LOG(ERROR) << "SenseVoiceTRT::Forward[" << index << "] error: " << e.what();
        }
        results.push_back(result);
    }
    return results;
}

// Hotword-interface stub
std::vector<std::string> SenseVoiceTRT::Forward(float** din, int* len,
                                                  bool input_finished,
                                                  const std::vector<std::vector<float>>& hw_emb,
                                                  void* wfst_decoder,
                                                  int batch_in) {
    return Forward(din, len, input_finished, "auto", false, batch_in);
}

// ─────────────────────────────────────────────────────────────────
//  CTCSearch — identical to SenseVoiceTorch version (CTC greedy decode)
// ─────────────────────────────────────────────────────────────────
std::string SenseVoiceTRT::CTCSearch(float* in,
                                      const std::vector<int32_t>& paraformer_length,
                                      const std::vector<int64_t>& outputShape,
                                      float frame_duration_sec) {
    const std::string unicodeChar = "\xe2\x96\x81"; // UTF-8 "▁"
    int32_t vocab_size = (int32_t)outputShape[2];

    std::vector<int64_t> tokens;
    std::vector<float>   timestamp_list;
    std::string text;
    int32_t prev_id = -1;
    bool is_start_token = false;

    for (int32_t t = 0; t != paraformer_length[0]; ++t) {
        auto y = (int32_t)std::distance(
            static_cast<const float*>(in),
            std::max_element(
                static_cast<const float*>(in),
                static_cast<const float*>(in) + vocab_size));
        in += vocab_size;

        if (((y == blank_id) || (y != blank_id && y != prev_id)) && is_start_token) {
            is_start_token = false;
            timestamp_list.push_back(t * frame_duration_sec);
        }
        if (y != blank_id && y != prev_id) {
            if (tokens.size() >= 4) {
                timestamp_list.push_back(t * frame_duration_sec);
                is_start_token = true;
            }
            tokens.push_back((int64_t)y);
        }
        prev_id = y;
    }
    if (is_start_token)
        timestamp_list.push_back((paraformer_length[0] - 1) * frame_duration_sec);

    std::string str_lang, str_emo, str_event, str_itn;
    if (tokens.size() >= 4) {
        str_lang  = vocab->Id2String((int)tokens[0]);
        str_emo   = vocab->Id2String((int)tokens[1]);
        str_event = vocab->Id2String((int)tokens[2]);
        str_itn   = vocab->Id2String((int)tokens[3]);
    }

    for (int32_t i = 4; i < (int32_t)tokens.size(); ++i) {
        std::string word = vocab->Id2String((int)tokens[i]);
        size_t found = word.find(unicodeChar);
        if (found != std::string::npos)
            text += " " + word.substr(3);
        else
            text += word;
    }
    if (str_itn == "<|withitn|>") {
        if (str_lang == "<|zh|>") text += "\xe3\x80\x82"; // 。
        else text += ".";
    }

    std::ostringstream oss;
    for (size_t i = 0; i < timestamp_list.size(); ++i) {
        oss << timestamp_list[i];
        if (i != timestamp_list.size() - 1) oss << ",";
    }
    std::string stamp_str = oss.str();
    std::string res;
    res.reserve(text.size() + stamp_str.size() + str_lang.size() + str_emo.size() + str_event.size() + 12);
    res.append(text);
    res.append(" | ");
    res.append(stamp_str);
    res.append(" | ");
    res.append(str_lang);
    res.append(",");
    res.append(str_emo);
    res.append(",");
    res.append(str_event);
    return res;
}

// ─────────────────────────────────────────────────────────────────
//  Feature extraction helpers — identical to SenseVoiceTorch
// ─────────────────────────────────────────────────────────────────
void SenseVoiceTRT::FbankKaldi(float sample_rate, const float* waves, int len,
                                 std::vector<std::vector<float>>& asr_feats) {
    knf::OnlineFbank fbank_(fbank_opts_);
    std::vector<float> buf(len);
    for (int32_t i = 0; i != len; ++i) buf[i] = waves[i] * 32768.0f;
    fbank_.AcceptWaveform(sample_rate, buf.data(), buf.size());
    int32_t frames = fbank_.NumFramesReady();
    for (int32_t i = 0; i != frames; ++i) {
        const float* frame = fbank_.GetFrame(i);
        asr_feats.emplace_back(frame, frame + fbank_opts_.mel_opts.num_bins);
    }
}

void SenseVoiceTRT::LfrCmvn(std::vector<std::vector<float>>& asr_feats) {
    std::deque<std::vector<float>> feats_dq(asr_feats.begin(), asr_feats.end());

    int pad_count = (lfr_m - 1) / 2;
    for (int i = 0; i < pad_count; i++) feats_dq.push_front(feats_dq.front());

    int T_lrf = (int)ceil(1.0 * (int)asr_feats.size() / lfr_n);
    int T     = (int)feats_dq.size();

    std::vector<std::vector<float>> out_feats;
    out_feats.reserve(T_lrf);
    std::vector<float> p;
    for (int i = 0; i < T_lrf; i++) {
        p.clear();
        int available = T - i * lfr_n;
        if (lfr_m <= available) {
            for (int j = 0; j < lfr_m; j++)
                p.insert(p.end(), feats_dq[i * lfr_n + j].begin(), feats_dq[i * lfr_n + j].end());
        } else {
            int num_padding = lfr_m - available;
            for (int j = 0; j < available; j++)
                p.insert(p.end(), feats_dq[i * lfr_n + j].begin(), feats_dq[i * lfr_n + j].end());
            for (int j = 0; j < num_padding; j++)
                p.insert(p.end(), feats_dq.back().begin(), feats_dq.back().end());
        }
        out_feats.emplace_back(std::move(p));
    }
    for (auto& feat : out_feats)
        for (int j = 0; j < (int)means_list_.size(); j++)
            feat[j] = (feat[j] + means_list_[j]) * vars_list_[j];

    asr_feats = std::move(out_feats);
}

void SenseVoiceTRT::LoadConfigFromYaml(const char* filename) {
    YAML::Node config;
    try { config = YAML::LoadFile(filename); }
    catch (exception const& e) {
        LOG(ERROR) << "SenseVoiceTRT: error loading YAML: " << filename;
        exit(-1);
    }
    try {
        YAML::Node fc = config["frontend_conf"];
        YAML::Node ec = config["encoder_conf"];
        window_type     = fc["window"].as<string>();
        n_mels          = fc["n_mels"].as<int>();
        frame_length    = fc["frame_length"].as<int>();
        frame_shift     = fc["frame_shift"].as<int>();
        lfr_m           = fc["lfr_m"].as<int>();
        lfr_n           = fc["lfr_n"].as<int>();
        asr_sample_rate = fc["fs"].as<int>();
        encoder_size    = ec["output_size"].as<int>();
        if (config["lang"].IsDefined())
            language = config["lang"].as<string>();
    } catch (exception const& e) {
        LOG(ERROR) << "SenseVoiceTRT: YAML parse error: " << e.what();
        exit(-1);
    }
}

void SenseVoiceTRT::LoadCmvn(const char* filename) {
    ifstream cmvn_stream(filename);
    if (!cmvn_stream.is_open()) {
        LOG(ERROR) << "SenseVoiceTRT: failed to open CMVN: " << filename; exit(-1);
    }
    means_list_.clear(); vars_list_.clear();
    string line;
    while (getline(cmvn_stream, line)) {
        istringstream iss(line);
        vector<string> items{istream_iterator<string>{iss}, istream_iterator<string>{}};
        if (items[0] == "<AddShift>") {
            getline(cmvn_stream, line); istringstream mss(line);
            vector<string> ml{istream_iterator<string>{mss}, istream_iterator<string>{}};
            if (ml[0] == "<LearnRateCoef>")
                for (int j = 3; j < (int)ml.size()-1; j++) means_list_.push_back(stof(ml[j]));
        } else if (items[0] == "<Rescale>") {
            getline(cmvn_stream, line); istringstream vss(line);
            vector<string> vl{istream_iterator<string>{vss}, istream_iterator<string>{}};
            if (vl[0] == "<LearnRateCoef>")
                for (int j = 3; j < (int)vl.size()-1; j++) vars_list_.push_back(stof(vl[j]) * scale);
        }
    }
}

std::vector<std::vector<float>> SenseVoiceTRT::CompileHotwordEmbedding(std::string& hotwords) {
    std::vector<std::vector<float>> emb;
    emb.push_back(std::vector<float>(encoder_size, 0.0f));
    return emb;
}

std::string SenseVoiceTRT::Rescoring() {
    LOG(ERROR) << "SenseVoiceTRT::Rescoring() not implemented.";
    return "";
}

} // namespace funasr
