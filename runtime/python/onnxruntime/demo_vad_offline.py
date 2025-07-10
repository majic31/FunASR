from funasr_onnx import Fsmn_vad
from pathlib import Path

model_dir = "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
wav_path = "{}/.cache/modelscope/hub/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/example/vad_example.wav".format(
    Path.home()
)
wav_path = "/Users/majie/project/stress_test/data/tmp/RC00a6d25110688_20241009000028.wav"

model = Fsmn_vad(model_dir)

result = model(wav_path)
print(result)
