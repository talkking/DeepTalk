import sys, os
from tqdm import tqdm
import multiprocessing
from jiwer import compute_measures
from zhon.hanzi import punctuation
import string
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration 
import soundfile as sf
import scipy
import zhconv
from funasr import AutoModel
import json

punctuation_all = punctuation + string.punctuation

wavname_res_text_path = sys.argv[1] # wavname and text gt
wav_rootdir = sys.argv[2]
res_path = sys.argv[3]
lang = sys.argv[4] # zh or en
device = "cuda"

def load_en_model():
    model_id = "/mnt/data/alanhshao/models/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    return processor, model

def load_zh_model():
    model = AutoModel(model="/mnt/data/alanhshao/models/paraformer-zh")
    return model

def process_one(hypo, truth):
    raw_truth = truth
    raw_hypo = hypo

    for x in punctuation_all:
        if x == '\'':
            continue
        truth = truth.replace(x, '')
        hypo = hypo.replace(x, '')

    truth = truth.replace('  ', ' ')
    hypo = hypo.replace('  ', ' ')

    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()
    else:
        raise NotImplementedError

    measures = compute_measures(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures["wer"]
    subs = measures["substitutions"] / len(ref_list)
    dele = measures["deletions"] / len(ref_list)
    inse = measures["insertions"] / len(ref_list)
    return (raw_truth, raw_hypo, wer, subs, dele, inse)


def run_asr(wav_rootdir, wav_res_text_path, res_path):
    if lang == "en":
        processor, model = load_en_model()
    elif lang == "zh":
        model = load_zh_model()

    params = []
    # for line in open(wav_res_text_path).readlines():
    #     line = line.strip()
    #     if len(line.split('|')) == 2:
    #         wav_res_path, text_ref = line.split('|')
    #     elif len(line.split('|')) == 3:
    #         wav_res_path, wav_ref_path, text_ref = line.split('|')
    #     elif len(line.split('|')) == 4: # for edit
    #         wav_res_path, _, text_ref, wav_ref_path = line.split('|')
    #     else:
    #         raise NotImplementedError

    #     if not os.path.exists(wav_res_path):
    #         continue
    for line in open(wav_res_text_path).readlines():
        #data = json.loads(line)
        
        wavname = line.split("\t")[0]
        wav_res_path = os.path.join(wav_rootdir, wavname)
        #import pdb; pdb.set_trace()
        try:
          text_hyp = line.split("\t")[1].replace('<｜begin▁of▁sentence｜>', '').replace('<｜end▁of▁sentence｜>', '').replace('【', '').replace('】', '').replace('\n', '')
          text_ref = text_hyp #data['']
          params.append((wav_res_path, text_ref))
        except:
            continue
    fout = open(res_path, "w")

    n_higher_than_50 = 0
    wers_below_50 = []
    for wav_res_path, text_ref in tqdm(params):
        
        if lang == "en":
            wav, sr = sf.read(wav_res_path)
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
            input_features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(device)
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        elif lang == "zh":
            res = model.generate(input=wav_res_path,
                    batch_size_s=300)
            transcription = res[0]["text"]
            transcription = zhconv.convert(transcription, 'zh-cn')
        try:
            raw_truth, raw_hypo, wer, subs, dele, inse = process_one(transcription, text_ref)
        except:
            raw_truth, raw_hypo, wer, subs, dele, inse = "", "", 1.0 , 0.0, 1.0, 0.0
        wavname = wav_res_path.split("/")[-1]
        fout.write(f"{wavname}\t{raw_hypo}\n")
        fout.flush()

run_asr(wav_rootdir, wavname_res_text_path, res_path)
