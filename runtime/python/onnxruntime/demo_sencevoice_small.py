#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/QwenAudio/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from pathlib import Path
from funasr_onnx import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess


model_dir = "iic/SenseVoiceSmall"

model = SenseVoiceSmall(model_dir, batch_size=10, quantize=False)

# inference
wav_or_scp = ["{}/.cache/modelscope/hub/{}/example/en.mp3".format(Path.home(), model_dir)]
wav_or_scp = ["/Users/majie/data/empty/R003ea91910448_20250521131923.wav"]
res = model("/Users/majie/data/R002f1dcf10892_20250704152817.wav", language="auto", use_itn=True)
print([rich_transcription_postprocess(i) for i in res])
