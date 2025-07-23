#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from pathlib import Path

from funasr_onnx.sensevoice_bin_hotword import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess


model_dir = "dengcunqin/SenseVoiceSmall_hotword"

model = SenseVoiceSmall(model_dir, batch_size=10, quantize=False)

# inference
wav_or_scp = ["/Users/majie/project/stress_test/data/tmp/RC00be39b310465_20241009000102.wav",
              "/Users/majie/Downloads/普通话转写/标准/R00bb4a0210206_20250114135859.wav"]

res = model(wav_or_scp, language="auto", use_itn=True, hotwords_str='',hotwords_score=1.0)
print([rich_transcription_postprocess(i) for i in res])
