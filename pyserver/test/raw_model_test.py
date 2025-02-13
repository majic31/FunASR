from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"

model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc",
                  spk_model="cam++", preset_spk_num=2
                  )

res = model.generate(input=f"/Users/majie/project/stress_test/data/tmp/RC00a6d25110688_20241009000028.wav",
            batch_size_s=300,
            hotword='魔搭')
print(res)

