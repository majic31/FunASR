/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
 *
 * SenseVoiceSmall TorchScript GPU implementation.
 * Passes 4 inputs to model forward:
 *   (speech [B,T,D], speech_lengths [B], language [B], textnorm [B])
 */

#include "precomp.h"
#include "sensevoice-small-torch.h"
#include <cstddef>

using namespace std;
namespace funasr {

SenseVoiceTorch::SenseVoiceTorch() {}

void SenseVoiceTorch::InitAsr(const std::string &am_model,
                               const std::string &am_cmvn,
                               const std::string &am_config,
                               const std::string &token_file,
                               int thread_num) {
    LoadConfigFromYaml(am_config.c_str());

    fbank_opts_.frame_opts.dither       = 0;
    fbank_opts_.mel_opts.num_bins       = n_mels;
    fbank_opts_.frame_opts.samp_freq    = asr_sample_rate;
    fbank_opts_.frame_opts.window_type  = window_type;
    fbank_opts_.frame_opts.frame_shift_ms   = frame_shift;
    fbank_opts_.frame_opts.frame_length_ms  = frame_length;
    fbank_opts_.energy_floor        = 0;
    fbank_opts_.mel_opts.debug_mel  = false;

    // Guard against double-init: free previous vocab if exists
    if (vocab) { delete vocab; vocab = nullptr; }
    vocab = new Vocab(token_file.c_str());
    LoadCmvn(am_cmvn.c_str());

    if (!torch::cuda::is_available()) {
        LOG(ERROR) << "CUDA is not available! Please check your GPU settings";
        exit(-1);
    }
    LOG(INFO) << "CUDA is available, SenseVoiceTorch running on GPU";

    torch::jit::setTensorExprFuserEnabled(false);

    try {
        torch::jit::script::Module model = torch::jit::load(am_model, at::kCUDA);
        model_ = std::make_shared<TorchModule>(std::move(model));
        LOG(INFO) << "Successfully load SenseVoiceTorch model from " << am_model;
        torch::NoGradGuard no_grad;
        model_->eval();
        torch::jit::setGraphExecutorOptimize(false);
        torch::jit::FusionStrategy static0 = {{torch::jit::FusionBehavior::STATIC, 0}};
        torch::jit::setFusionStrategy(static0);
        WarmUp();
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load SenseVoiceTorch model: " << am_model << " " << e.what();
        exit(-1);
    }
}

void SenseVoiceTorch::WarmUp() {
    int32_t in_feat_dim = fbank_opts_.mel_opts.num_bins;
    int32_t feature_dim = lfr_m * in_feat_dim;
    int64_t batch_in   = 1;
    int64_t max_frames  = 10;

    std::vector<float> all_feats((size_t)(batch_in * max_frames * feature_dim), 0.1f);
    torch::Tensor feats = torch::from_blob(
        all_feats.data(), {batch_in, max_frames, feature_dim}, torch::kFloat).contiguous();

    // int64 for all index tensors
    torch::Tensor feat_lens     = torch::full({batch_in}, max_frames,  torch::kInt64);
    torch::Tensor lang_tensor   = torch::full({batch_in}, 0L,          torch::kInt64);
    torch::Tensor textnorm_tens = torch::full({batch_in}, 15L,         torch::kInt64);

    feats       = feats.to(at::kCUDA);
    feat_lens   = feat_lens.to(at::kCUDA);
    lang_tensor = lang_tensor.to(at::kCUDA);
    textnorm_tens = textnorm_tens.to(at::kCUDA);

    std::vector<torch::jit::IValue> inputs = {feats, feat_lens, lang_tensor, textnorm_tens};
    try {
        torch::NoGradGuard no_grad;
        auto outputs = model_->forward(inputs).toTuple()->elements();
        LOG(INFO) << "SenseVoiceTorch WarmUp finished successfully.";
    } catch (std::exception const &e) {
        LOG(ERROR) << "SenseVoiceTorch WarmUp error: " << e.what();
    }
}

void SenseVoiceTorch::LoadConfigFromYaml(const char *filename) {
    YAML::Node config;
    try {
        config = YAML::LoadFile(filename);
    } catch (exception const &e) {
        LOG(ERROR) << "Error loading YAML config: " << filename;
        exit(-1);
    }
    try {
        YAML::Node frontend_conf = config["frontend_conf"];
        YAML::Node encoder_conf  = config["encoder_conf"];

        window_type    = frontend_conf["window"].as<string>();
        n_mels         = frontend_conf["n_mels"].as<int>();
        frame_length   = frontend_conf["frame_length"].as<int>();
        frame_shift    = frontend_conf["frame_shift"].as<int>();
        lfr_m          = frontend_conf["lfr_m"].as<int>();
        lfr_n          = frontend_conf["lfr_n"].as<int>();
        asr_sample_rate = frontend_conf["fs"].as<int>();
        encoder_size   = encoder_conf["output_size"].as<int>();

        if (config["lang"].IsDefined())
            language = config["lang"].as<string>();
    } catch (exception const &e) {
        LOG(ERROR) << "Error parsing SenseVoiceTorch YAML config: " << e.what();
        exit(-1);
    }
}

void SenseVoiceTorch::LoadCmvn(const char *filename) {
    ifstream cmvn_stream(filename);
    if (!cmvn_stream.is_open()) {
        LOG(ERROR) << "Failed to open CMVN file: " << filename;
        exit(-1);
    }
    // Clear before loading to prevent data corruption on re-init
    means_list_.clear();
    vars_list_.clear();
    string line;
    while (getline(cmvn_stream, line)) {
        istringstream iss(line);
        vector<string> items{istream_iterator<string>{iss}, istream_iterator<string>{}};
        if (items[0] == "<AddShift>") {
            getline(cmvn_stream, line);
            istringstream mss(line);
            vector<string> ml{istream_iterator<string>{mss}, istream_iterator<string>{}};
            if (ml[0] == "<LearnRateCoef>") {
                for (int j = 3; j < (int)ml.size() - 1; j++)
                    means_list_.push_back(stof(ml[j]));
                continue;
            }
        } else if (items[0] == "<Rescale>") {
            getline(cmvn_stream, line);
            istringstream vss(line);
            vector<string> vl{istream_iterator<string>{vss}, istream_iterator<string>{}};
            if (vl[0] == "<LearnRateCoef>") {
                for (int j = 3; j < (int)vl.size() - 1; j++)
                    vars_list_.push_back(stof(vl[j]) * scale);
                continue;
            }
        }
    }
}

void SenseVoiceTorch::FbankKaldi(float sample_rate, const float* waves, int len,
                                  std::vector<std::vector<float>> &asr_feats) {
    knf::OnlineFbank fbank_(fbank_opts_);
    std::vector<float> buf(len);
    for (int32_t i = 0; i != len; ++i)
        buf[i] = waves[i] * 32768.0f;
    fbank_.AcceptWaveform(sample_rate, buf.data(), buf.size());

    int32_t frames = fbank_.NumFramesReady();
    for (int32_t i = 0; i != frames; ++i) {
        const float *frame = fbank_.GetFrame(i);
        asr_feats.emplace_back(frame, frame + fbank_opts_.mel_opts.num_bins);
    }
}

void SenseVoiceTorch::LfrCmvn(std::vector<std::vector<float>> &asr_feats) {
    // Use deque to avoid O(n) insert-at-front for padding frames
    std::deque<std::vector<float>> feats_dq(asr_feats.begin(), asr_feats.end());

    int pad_count = (lfr_m - 1) / 2;
    for (int i = 0; i < pad_count; i++)
        feats_dq.push_front(feats_dq.front());

    int T     = (int)feats_dq.size();
    int T_lrf = (int)ceil(1.0 * (T - pad_count) / lfr_n);
    // Recalculate T_lrf based on original frame count (before padding)
    T_lrf = (int)ceil(1.0 * (int)asr_feats.size() / lfr_n);

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

    // Apply CMVN
    for (auto &feat : out_feats)
        for (int j = 0; j < (int)means_list_.size(); j++)
            feat[j] = (feat[j] + means_list_[j]) * vars_list_[j];

    asr_feats = std::move(out_feats);
}

// CTCSearch: takes vectors by const-ref to avoid copies
std::string SenseVoiceTorch::CTCSearch(float *in,
                                        const std::vector<int32_t> &paraformer_length,
                                        const std::vector<int64_t> &outputShape,
                                        float frame_duration_sec) {
    const std::string unicodeChar = "\xe2\x96\x81"; // UTF-8 for the special space token
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
        if (str_lang == "<|zh|>") text += "\xe3\x80\x82"; // UTF-8 for Chinese period
        else text += ".";
    }

    // Build result string with reserve+append to avoid temporary string allocations
    std::ostringstream oss;
    for (size_t i = 0; i < timestamp_list.size(); ++i) {
        oss << timestamp_list[i];
        if (i != timestamp_list.size() - 1) oss << ",";
    }
    std::string stamp_str = oss.str();
    std::string res;
    res.reserve(text.size() + stamp_str.size() +
                str_lang.size() + str_emo.size() + str_event.size() + 12);
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

// Forward: TRUE batch inference — pad all samples to max_frames, build [B,T,D] tensors,
// and call model_->forward() ONCE per batch to fully utilize GPU Tensor Cores.
// This replaces the previous per-sample serial loop (fake batch=1 each time).
std::vector<std::string> SenseVoiceTorch::Forward(float** din, int* len,
                                                   bool input_finished,
                                                   std::string svs_lang,
                                                   bool svs_itn,
                                                   int batch_in) {
    std::vector<std::string> results;
    results.reserve(batch_in);

    int32_t in_feat_dim = fbank_opts_.mel_opts.num_bins;
    int32_t feature_dim = lfr_m * in_feat_dim;

    // Resolve language/textnorm IDs once for all samples in the batch
    int64_t svs_lid   = 0;
    if (lid_map.count(svs_lang))
        svs_lid = (int64_t)lid_map.at(svs_lang);
    int64_t svs_itnid = svs_itn ? 14L : 15L;

    // ── Step 1: Feature extraction for every sample ──────────────────────────
    // feats_flat[i]: flattened float array of shape [num_frames_i * feature_dim]
    std::vector<std::vector<float>> feats_flat(batch_in);
    std::vector<int64_t>            feat_frame_lens(batch_in, 0);
    int64_t max_frames = 0;

    for (int i = 0; i < batch_in; ++i) {
        std::vector<std::vector<float>> asr_feats;
        FbankKaldi(asr_sample_rate, din[i], len[i], asr_feats);
        if (asr_feats.empty()) {
            // empty audio — leave feats_flat[i] empty, feat_frame_lens[i] = 0
            continue;
        }
        LfrCmvn(asr_feats);

        int64_t nf = (int64_t)asr_feats.size();
        feat_frame_lens[i] = nf;
        if (nf > max_frames) max_frames = nf;

        feats_flat[i].reserve((size_t)nf * feature_dim);
        for (const auto &frame : asr_feats)
            feats_flat[i].insert(feats_flat[i].end(), frame.begin(), frame.end());
    }

    // All samples are empty — return empty strings immediately
    if (max_frames == 0) {
        results.assign(batch_in, "");
        return results;
    }

    // ── Step 2: Pad to max_frames and build a single [B, T, D] tensor ────────
    // Allocate zero-initialised buffer (zero-padding is the standard approach)
    std::vector<float> all_feats((size_t)batch_in * max_frames * feature_dim, 0.0f);
    for (int i = 0; i < batch_in; ++i) {
        if (feats_flat[i].empty()) continue;
        // Copy only the valid frames; the rest remain zero-padded
        std::memcpy(
            all_feats.data() + (size_t)i * max_frames * feature_dim,
            feats_flat[i].data(),
            feats_flat[i].size() * sizeof(float));
    }

    // feats: [B, max_frames, feature_dim] — kFloat on CPU, then move to CUDA
    torch::Tensor feats = torch::from_blob(
        all_feats.data(),
        {(int64_t)batch_in, max_frames, (int64_t)feature_dim},
        torch::kFloat).clone().to(at::kCUDA);   // .clone() because from_blob borrows memory

    // feat_lens: [B] int64  (SenseVoice encoder expects int64)
    torch::Tensor feat_lens = torch::from_blob(
        feat_frame_lens.data(),
        {(int64_t)batch_in},
        torch::kInt64).clone().to(at::kCUDA);

    // lang / textnorm: [B] int64 — same value broadcast to every sample
    torch::Tensor lang_tensor   = torch::full({(int64_t)batch_in}, svs_lid,   torch::kInt64).to(at::kCUDA);
    torch::Tensor textnorm_tens = torch::full({(int64_t)batch_in}, svs_itnid, torch::kInt64).to(at::kCUDA);

    // ── Step 3: Single batched GPU forward ───────────────────────────────────
    torch::Tensor am_scores_cpu;   // [B, T_out, vocab_size]
    torch::Tensor token_lens_cpu;  // [B]
    try {
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs = {feats, feat_lens, lang_tensor, textnorm_tens};
        auto raw_out = model_->forward(inputs);
        auto elems   = raw_out.toTuple()->elements();

        // outputs[0]: logits  [B, T_out, vocab_size]
        // outputs[1]: valid output token lengths  [B]
        am_scores_cpu  = elems[0].toTensor().to(at::kCPU).contiguous();
        token_lens_cpu = elems[1].toTensor().to(at::kCPU).contiguous();

        LOG(INFO) << "SenseVoiceTorch batch forward: batch_in=" << batch_in
                  << " max_frames=" << max_frames
                  << " am_scores=" << am_scores_cpu.sizes();
    } catch (std::exception const &e) {
        LOG(ERROR) << "SenseVoiceTorch batch forward error: " << e.what();
        results.assign(batch_in, "");
        return results;
    }

    // ── Step 4: Per-sample CTCSearch on CPU ──────────────────────────────────
    std::vector<int64_t> outputShape = {
        am_scores_cpu.size(0),
        am_scores_cpu.size(1),
        am_scores_cpu.size(2)};

    for (int i = 0; i < batch_in; ++i) {
        std::string result;
        if (feat_frame_lens[i] == 0) {
            // empty audio, skip decoding
            results.push_back(result);
            continue;
        }
        try {
            // Slice [T_out, vocab_size] for this sample; data_ptr gives contiguous ptr
            std::vector<int32_t> valid_len = {token_lens_cpu[i].item<int>()};
            result = CTCSearch(am_scores_cpu[i].data_ptr<float>(), valid_len, outputShape);
        } catch (std::exception const &e) {
            LOG(ERROR) << "SenseVoiceTorch CTCSearch[" << i << "] error: " << e.what();
        }
        results.push_back(result);
    }

    return results;
}

// Hotword stub — SenseVoice GPU does not use hotwords
std::vector<std::string> SenseVoiceTorch::Forward(float** din, int* len,
                                                   bool input_finished,
                                                   const std::vector<std::vector<float>> &hw_emb,
                                                   void* wfst_decoder,
                                                   int batch_in) {
    return Forward(din, len, input_finished, "auto", false, batch_in);
}

std::vector<std::vector<float>> SenseVoiceTorch::CompileHotwordEmbedding(std::string &hotwords) {
    std::vector<std::vector<float>> emb;
    std::vector<float> dummy(encoder_size, 0.0f);
    emb.push_back(dummy);
    return emb;
}

SenseVoiceTorch::~SenseVoiceTorch() {
    if (vocab) { delete vocab; vocab = nullptr; }
}

std::string SenseVoiceTorch::Rescoring() {
    LOG(ERROR) << "SenseVoiceTorch::Rescoring() not implemented.";
    return "";
}

} // namespace funasr
