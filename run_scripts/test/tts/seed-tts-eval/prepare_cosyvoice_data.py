import os

seedtts_path = "/data/lijiang/data/seedtts_testset/zh"
output_path = "/data/lijiang/data/seedtts_testset/zh_sparktts_rec"

wav_res_text_path = os.path.join(seedtts_path, "meta.lst")
#with open(os.path.join(output_path, "wav_text.txt"), "w") as fout:
with open("seedtts_cosyvoice_rec/seedtts_cosyvoice_rec_zh/tsv", "w") as fout:
    print(seedtts_path + "/wavs", file=fout)
    for line in open(wav_res_text_path).readlines():
        line = line.strip()
        filename, prompt_text, prompt_path, target_text = line.split('|')
        print(filename+".wav", file=fout)
