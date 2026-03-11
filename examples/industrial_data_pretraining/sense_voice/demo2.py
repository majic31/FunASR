from funasr import AutoModel
model = AutoModel(
    model='iic/SenseVoiceSmall',
    vad_model='fsmn-vad',
    punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    device='cpu',
)
import os
# Use a short test file that exists
test_file = f'/Users/majie/data/problem/R00a54bd310843_20250924170324.wav'
if os.path.exists(test_file):
    res = model.generate(input=test_file,
                        hotword='拐进去', 
                        output_timestamp=True,
                        sentence_timestamp=True,
                        batch_size_s=60)
    print(f'{res}')
else:
    print('Test file not found:', test_file)
    