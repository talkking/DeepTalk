import os

seedtts_path = "/data/lijiang/data/seedtts_testset/zh"
output_path = "/data/lijiang/data/seedtts_testset/codec_rec/vita_tts_result/zh/wav"

wav_res_text_path = os.path.join(seedtts_path, "meta.lst")
#with open(os.path.join(output_path, "wav_text.txt"), "w") as fout:
with open("/data/lijiang/data/seedtts_testset/codec_rec/vita_tts_result/zh_result/wav_text.txt", "w") as fout:
    for line in open(wav_res_text_path).readlines():
        line = line.strip()
        filename, prompt_text, prompt_path, target_text = line.split('|')
        wav_path = os.path.join(output_path, filename+'.wav')
        print(f"{wav_path}|{target_text}", file=fout)
