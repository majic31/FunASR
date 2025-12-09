#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"


model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"sil_to_speech_time_thres":150, "speech_to_sil_time_thres":150, "lookahead_time_end_point":0, "speech_noise_thres":0.9, 'lookback_time_start_point':0, 'speech_2_noise_ratio': 0.8},
    device="cpu",
    punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    lm_model="",
    #spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
    #spk_mode='vad_segment'
)

# en
# res = model.generate(
#     input=f"{model.model_path}/example/en.mp3",
#     cache={},
#     language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
#     use_itn=True,
#     batch_size_s=60,
#     merge_vad=True,  #
#     merge_length_s=15,
# )
# text = rich_transcription_postprocess(res[0]["text"])
# print(text)

# en with timestamp
# # res = model.generate(
# #     input=f"{model.model_path}/example/en.mp3",
# #     cache={},
# #     language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
# #     use_itn=True,
# #     batch_size_s=60,
# #     merge_vad=True,  #
# #     merge_length_s=15,
# #     output_timestamp=True,
# # )
# # print(res)
# # text = rich_transcription_postprocess(res[0]["text"])
# # print(text)
#
# # zh
# res = model.generate(
#     input=f"{model.model_path}/example/zh.mp3",
#     cache={},
#     language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
#     use_itn=True,
#     batch_size_s=60,
#     merge_vad=True,  #
#     merge_length_s=15,
# )
# text = rich_transcription_postprocess(res[0]["text"])
# print(text)
token_int = [10860, 10153, 18339, 9957, 13190, 10153, 17692, 10153, 10767, 13185, 14122, 10405, 17972, 18339, 18486, 8, 124, 124, 9930, 10681, 10012, 10153, 18339, 18486, 13190, 10290, 10008, 14950]

# zh with timestamp
res = model.generate(
    # input=f"/Users/majie/data/R00286cdf10379_20250424083149_16k.wav",
    # input=f"/tmp/sample/粤语/R00d06da310515_20250513115423.wav",
    # input=f"/Users/majie/data/南海供电局/R003bd03810474_20250515084246.wav",
    # input = f"/Users/majie/project/stress_test/data/tmp/RC0084697210723_20241009000002.wav",
    # input = f"/Users/majie/data/测试数据.wav",
    #input = f"/Users/majie/data/R004e741e10473_20250524140803.wav",
    # input=f"/Users/majie/data/R003b364e10438_20250419152021.wav",
    # input = f"/Users/majie/project/stress_test/data/tmp/RC0084681410801_20241009000025.wav",
    # input = f"{model.model_path}/example/yue.mp3",
    # input = f"/Users/majie/data/007c8a9d31371da3224b90c123daa35d_0017_00013_part.wav",
    input = f'/Volumes/KINGSTON/data/20250917-4007157501.wav',
    cache={},
    # language="zh",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=False,
    batch_size_s=60,
    merge_vad=False,  #
    merge_length_s=15,
    output_timestamp=True,
    sentence_timestamp=True
)

# res = model.generate(
#     # input=f"/Users/majie/data/R00286cdf10379_20250424083149_16k.wav",
#     # input=f"/tmp/sample/粤语/R00d06da310515_20250513115423.wav",
#     input=f"/Users/majie/data/R003c63ef10361_20250427143123.wav",
#     cache={},
#     language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
#     use_itn=True,
#     batch_size_s=60,
#     merge_vad=True,  #
#     merge_length_s=15,
#     output_timestamp=False
# )
print(res)
#text = rich_transcription_postprocess(res[0]["text"])
#print(text)

# yue
# res = model.generate(
#     input=f"{model.model_path}/example/yue.mp3",
#     cache={},
#     language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
#     use_itn=True,
#     batch_size_s=60,
#     merge_vad=True,  #
#     merge_length_s=15,
# )
# text = rich_transcription_postprocess(res[0]["text"])
# print(text)
#
# # ja
# res = model.generate(
#     input=f"{model.model_path}/example/ja.mp3",
#     cache={},
#     language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
#     use_itn=True,
#     batch_size_s=60,
#     merge_vad=True,  #
#     merge_length_s=15,
# )
# text = rich_transcription_postprocess(res[0]["text"])
# print(text)
#
#
# # ko
# res = model.generate(
#     input=f"{model.model_path}/example/ko.mp3",
#     cache={},
#     language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
#     use_itn=True,
#     batch_size_s=60,
#     merge_vad=True,  #
#     merge_length_s=15,
# )
# text = rich_transcription_postprocess(res[0]["text"])
# print(text)
