#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(
    model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    # spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
    lm_file = '/Users/majie/project/funasr_docker/funasr-runtime-resources/models/damo/speech_ngram_lm_zh-cn-ai-wesp-fst',
    lm_weight = 0.1,
    decoding_ctc_weight = 0.3,
    maxlenratio = 0.5,
    minlenratio = 0.3,
    ngram_weight = 0.1
)


# example1
res = model.generate(
    input="/Users/majie/project/stress_test/data/tmp/RC00a6d25110688_20241009000028.wav",
    hotword="达摩院 魔搭",
    lm_weight=0.5,
    lm_path="/Users/majie/project/funasr_docker/funasr-runtime-resources/models/damo/speech_ngram_lm_zh-cn-ai-wesp-fst"
    # return_raw_text=True,     # return raw text recognition results splited by space of equal length with timestamp
    # preset_spk_num=2,         # preset speaker num for speaker cluster model
    # sentence_timestamp=True,  # return sentence level information when spk_model is not given
    # lm_file = '/Users/majie/project/funasr_docker/funasr-runtime-resources/models/damo/speech_ngram_lm_zh-cn-ai-wesp-fst',
    #lm_weight = 0.1,
    #decoding_ctc_weight = 0.3
)
print(res)


"""
# tensor or numpy as input
# example2
import torchaudio
import os
wav_file = os.path.join(model.model_path, "example/asr_example.wav")
input_tensor, sample_rate = torchaudio.load(wav_file)
input_tensor = input_tensor.mean(0)
res = model.generate(input=[input_tensor], batch_size_s=300, is_final=True)


# example3
import soundfile

wav_file = os.path.join(model.model_path, "example/asr_example.wav")
speech, sample_rate = soundfile.read(wav_file)
res = model.generate(input=[speech], batch_size_s=300, is_final=True)
"""
