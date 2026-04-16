/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
 *
 * SenseVoiceSmall TorchScript GPU implementation.
 * Based on ParaformerTorch, but Forward() passes 4 inputs:
 *   (speech, speech_lengths, language, textnorm)
 */
#pragma once
#define C10_USE_GLOG
#include <deque>
#include <torch/serialize.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include "precomp.h"

namespace funasr {

    class SenseVoiceTorch : public Model {
    /**
     * SenseVoiceSmall GPU (TorchScript) implementation.
     * Supports 4-input forward: speech, speech_lengths, language, textnorm.
     */
    private:
        Vocab* vocab = nullptr;
        const float scale = 1.0;

        void LoadConfigFromYaml(const char* filename);
        void LoadCmvn(const char *filename);
        void LfrCmvn(std::vector<std::vector<float>> &asr_feats);

        using TorchModule = torch::jit::script::Module;
        std::shared_ptr<TorchModule> model_ = nullptr;

    public:
        SenseVoiceTorch();
        ~SenseVoiceTorch();

        void InitAsr(const std::string &am_model, const std::string &am_cmvn,
                     const std::string &am_config, const std::string &token_file,
                     int thread_num);

        void WarmUp();

        void FbankKaldi(float sample_rate, const float* waves, int len,
                        std::vector<std::vector<float>> &asr_feats);

        // GPU version of Forward: takes svs_lang and svs_itn to build lang/textnorm tensors
        std::vector<std::string> Forward(float** din, int* len,
                                         bool input_finished = true,
                                         std::string svs_lang = "auto",
                                         bool svs_itn = false,
                                         int batch_in = 1);

        // Forward overload for hotword-based interface (not used by SVS, stub only)
        std::vector<std::string> Forward(float** din, int* len,
                                         bool input_finished,
                                         const std::vector<std::vector<float>> &hw_emb,
                                         void* wfst_decoder,
                                         int batch_in);

        std::string CTCSearch(float* in,
                              const std::vector<int32_t> &paraformer_length,
                              const std::vector<int64_t> &outputShape,
                              float frame_duration_sec = 0.06f);

        std::string Rescoring();
        std::string GetLang() { return language; }
        int GetAsrSampleRate() { return asr_sample_rate; }
        void SetBatchSize(int batch_size) { batch_size_ = batch_size; }
        int GetBatchSize() { return batch_size_; }
        void StartUtterance() {}
        void EndUtterance() {}
        // SenseVoice does not use LM/hotword in GPU mode; stub implementations
        void InitLm(const std::string &lm_file,
                    const std::string &lm_cfg_file,
                    const std::string &lex_file) {}
        void InitHwCompiler(const std::string &hw_model, int thread_num) {}
        void InitSegDict(const std::string &seg_dict_model) {}
        std::vector<std::vector<float>> CompileHotwordEmbedding(std::string &hotwords);
        void Reset() {}

        Vocab* GetVocab() { return vocab; }
        // These are not used by SenseVoice but required by Model interface
        Vocab* GetLmVocab() { return nullptr; }

        knf::FbankOptions fbank_opts_;
        std::vector<float> means_list_;
        std::vector<float> vars_list_;
        int lfr_m = PARA_LFR_M;
        int lfr_n = PARA_LFR_N;

        std::string language = "zh-cn";

        std::string window_type = "hamming";
        int frame_length = 25;
        int frame_shift = 10;
        int n_mels = 80;
        int encoder_size = 512;
        int asr_sample_rate = MODEL_SAMPLE_RATE;
        int batch_size_ = 1;
        int blank_id = 0;

        // Language ID map: consistent with SenseVoiceSmall CPU version
        std::map<std::string, int> lid_map = {
            {"auto", 0},
            {"zh",   3},
            {"en",   4},
            {"yue",  7},
            {"ja",   11},
            {"ko",   12},
            {"nospeech", 13}
        };
    };

} // namespace funasr
