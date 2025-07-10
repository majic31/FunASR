#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from pathlib import Path
from funasr_torch import SenseVoiceSmall
from funasr_torch.utils.postprocess_utils import rich_transcription_postprocess


model_dir = "iic/SenseVoiceSmall"

model = SenseVoiceSmall(model_dir, device="cpu")

wav_or_scp = ["/Users/majie/data/007c8a9d31371da3224b90c123daa35d_0017_00013_part.wav"]

res = model(wav_or_scp, language="zh", use_itn=False)
print([rich_transcription_postprocess(i) for i in res])
