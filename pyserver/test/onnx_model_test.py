from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
from funasr_onnx import SenseVoiceSmall
from funasr_onnx import Fsmn_vad
import time

start = time.time()
model_dir = "iic/SenseVoiceSmall"
asr_model = SenseVoiceSmall(model_dir, batch_size=10, quantize=False, intra_op_num_threads=4)
end = time.time()
print(f'load asr model. spend: {end - start:.2f} s')
start = end
model_dir = "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
vad_model = Fsmn_vad(model_dir)
end = time.time()
print(f'load vad model. spend: {end - start:.2f} s')

start = end
wav_or_scp = ['/Users/majie/project/stress_test/data/tmp/RC00a6d25110688_20241009000028.wav', ]
result = vad_model(wav_or_scp)
print(result)
end = time.time()
print(f'compute vad. spend = {end - start:.2f} s')

start = time.time()
for i in range(1):
    res = asr_model(wav_or_scp, language="auto", use_itn=True)
    print(res)
end = time.time()
print(f'compute asr. spend = {end - start:.2f} s')
