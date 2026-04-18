/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
 *
 * SenseVoiceSmall TensorRT GPU implementation.
 * Replaces LibTorch (TorchScript) with TensorRT for dramatically better throughput on A100.
 *
 * Key design:
 *   - Single shared ICudaEngine* (read-only, thread-safe)
 *   - Per-thread IExecutionContext* + cudaStream_t via thread_local
 *   - No global locks: concurrent Forward() calls from N decoder threads all execute in parallel
 *   - FP16 precision with A100 Tensor Core acceleration
 *   - Dynamic batch + dynamic sequence length via TRT Optimization Profiles
 *
 * Model file: model.trt (generated offline by trtexec/onnx2trt, see tools/export_trt.sh)
 *
 * TRT version: 8.4.x   (uses IExecutionContext::enqueueV2)
 */
#pragma once

#include "precomp.h"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <deque>

namespace funasr {

// ─────────────────────────────────────────────────────────────────
//  TRT Logger (wraps glog)
// ─────────────────────────────────────────────────────────────────
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
            case Severity::kERROR:   LOG(ERROR) << "[TRT] " << msg; break;
            case Severity::kWARNING: LOG(WARNING) << "[TRT] " << msg; break;
            case Severity::kINFO:    LOG(INFO) << "[TRT] " << msg; break;
            default: break;
        }
    }
};

// ─────────────────────────────────────────────────────────────────
//  Per-thread context (one per decoder thread)
// ─────────────────────────────────────────────────────────────────
struct TRTThreadCtx {
    nvinfer1::IExecutionContext* ctx = nullptr;
    cudaStream_t                stream = nullptr;

    // GPU pinned / device buffers  (allocated once on first use)
    // Input bindings: [speech, speech_lengths, language, textnorm]
    // Output bindings: [ctc_logits, encoder_out_lens]
    void* bindings[6] = {nullptr};   // device pointers, indexed by binding index

    // Current max allocated sizes (bytes) to avoid re-alloc every call
    size_t alloc_speech_bytes    = 0;
    size_t alloc_logits_bytes    = 0;

    // Host output buffer (pinned) for async memcpy back from GPU
    float*   host_logits  = nullptr;
    int32_t* host_outlen  = nullptr;
    size_t   host_logits_bytes = 0;

    // Binding indices (filled when engine is initialized)
    int idx_speech       = -1;
    int idx_speech_len   = -1;
    int idx_language     = -1;
    int idx_textnorm     = -1;
    int idx_ctc_logits   = -1;
    int idx_out_lens     = -1;

    ~TRTThreadCtx();
};

// ─────────────────────────────────────────────────────────────────
//  SenseVoiceTRT
// ─────────────────────────────────────────────────────────────────
class SenseVoiceTRT : public Model {
    /**
     * SenseVoiceSmall GPU (TensorRT 8.4) implementation.
     *
     * Thread-safety model:
     *   ICudaEngine    ─ shared, read-only after Init ─ thread-safe
     *   IExecutionContext ─ one per decoder thread (thread_local) ─ no sharing
     *   cudaStream_t   ─ one per decoder thread                  ─ no sharing
     *
     * The upstream LibTorch implementation serializes all Forward() calls via
     * internal c10 global locks (SM utilization ~30-50% on A100).
     * Here each thread runs its own CUDA stream concurrently → A100 SM
     * utilization can reach 70-90%.
     */
private:
    // ── Shared engine state (thread-safe after Init) ─────────────
    static TRTLogger                trt_logger_;
    nvinfer1::IRuntime*             runtime_    = nullptr;
    nvinfer1::ICudaEngine*          engine_     = nullptr;

    // ── Per-thread context ────────────────────────────────────────
    //  Created lazily on first Forward() call, destroyed with thread.
    thread_local static std::unique_ptr<TRTThreadCtx> tls_ctx_;

    // ── Feature extraction state (same as SenseVoiceTorch) ───────
    Vocab*  vocab = nullptr;
    const float scale = 1.0f;

    void LoadConfigFromYaml(const char* filename);
    void LoadCmvn(const char* filename);
    void LfrCmvn(std::vector<std::vector<float>>& asr_feats);

    // ── TRT helpers ───────────────────────────────────────────────
    bool    LoadEngine(const std::string& engine_path);
    TRTThreadCtx* GetOrCreateCtx();           // returns thread-local ctx, creating if needed
    void    EnsureBuffers(TRTThreadCtx* tc,
                          int64_t batch, int64_t T, int64_t feat_dim, int64_t vocab_size);
    std::string RunInference(TRTThreadCtx* tc,
                             const std::vector<float>& wav_feats,
                             int64_t num_frames, int64_t feat_dim,
                             int32_t svs_lid, int32_t svs_itnid);

public:
    SenseVoiceTRT();
    ~SenseVoiceTRT();

    void InitAsr(const std::string& am_model, const std::string& am_cmvn,
                 const std::string& am_config, const std::string& token_file,
                 int thread_num);

    void WarmUp();

    void FbankKaldi(float sample_rate, const float* waves, int len,
                    std::vector<std::vector<float>>& asr_feats);

    // Primary inference entry point (identical signature to SenseVoiceTorch)
    std::vector<std::string> Forward(float** din, int* len,
                                     bool input_finished = true,
                                     std::string svs_lang = "auto",
                                     bool svs_itn = false,
                                     int batch_in = 1);

    // Hotword stub (SenseVoice TRT does not use hotwords)
    std::vector<std::string> Forward(float** din, int* len,
                                     bool input_finished,
                                     const std::vector<std::vector<float>>& hw_emb,
                                     void* wfst_decoder,
                                     int batch_in);

    std::string CTCSearch(float* in,
                          const std::vector<int32_t>& paraformer_length,
                          const std::vector<int64_t>& outputShape,
                          float frame_duration_sec = 0.06f);

    std::string Rescoring();
    std::string GetLang()  { return language; }
    int  GetAsrSampleRate(){ return asr_sample_rate; }
    void SetBatchSize(int b){ batch_size_ = b; }
    int  GetBatchSize()    { return batch_size_; }
    void StartUtterance()  {}
    void EndUtterance()    {}
    void Reset()           {}

    void InitLm(const std::string&, const std::string&, const std::string&) {}
    void InitHwCompiler(const std::string&, int) {}
    void InitSegDict(const std::string&) {}
    std::vector<std::vector<float>> CompileHotwordEmbedding(std::string& hotwords);

    Vocab* GetVocab()   { return vocab; }
    Vocab* GetLmVocab() { return nullptr; }

    // ── Feature extraction parameters (same as SenseVoiceTorch) ──
    knf::FbankOptions fbank_opts_;
    std::vector<float> means_list_;
    std::vector<float> vars_list_;
    int lfr_m = PARA_LFR_M;
    int lfr_n = PARA_LFR_N;

    std::string language = "zh-cn";
    std::string window_type = "hamming";
    int frame_length   = 25;
    int frame_shift    = 10;
    int n_mels         = 80;
    int encoder_size   = 512;
    int asr_sample_rate = MODEL_SAMPLE_RATE;
    int batch_size_    = 1;
    int blank_id       = 0;

    // Language ID — same mapping as CPU/Torch versions
    std::map<std::string, int> lid_map = {
        {"auto",     0},
        {"zh",       3},
        {"en",       4},
        {"yue",      7},
        {"ja",       11},
        {"ko",       12},
        {"nospeech", 13}
    };
};

} // namespace funasr
