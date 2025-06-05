import soundfile as sf
import torchaudio
import argparse
import os
from torchaudio.transforms import Resample
from snac import SNAC
import torch
parser = argparse.ArgumentParser()
parser.add_argument('input_path', type=str)
parser.add_argument('output_path', type=str)
args = parser.parse_args()
model = SNAC.from_pretrained("/data/lijiang/models/snac_24khz").eval().cuda()

os.makedirs(args.output_path, exist_ok=True)

_, _, names = next(os.walk(args.input_path))
tokens = {}
snac_sr = 24_000
with torch.no_grad():
    for filename in names:
        wav_path = os.path.join(args.input_path, filename)
        output_path = os.path.join(args.output_path, filename)
        waveform, sample_rate = torchaudio.load(wav_path)

        if sample_rate != snac_sr:
            transform = Resample(16_000, snac_sr).to('cuda')
            wav_24khz = transform(waveform.unsqueeze(0)).to("cuda")
        else:
            wav_24khz = waveform.to("cuda")
        codes = model.encode(wav_24khz.unsqueeze(0))
        code_12hz, code_24hz, code_48hz = codes
        codes_cat = torch.cat([code_12hz, code_24hz.view(-1, 2).T, code_48hz.view(-1, 4).T], dim=0)
        codes_flat = codes_cat.T.reshape(-1)
        codes_flat_str = " ".join([f"{x}" for x in codes_flat.cpu().numpy()]) + "\n"
        tokens[filename.replace(".wav", "")] = codes_flat_str
        rec_wav = model.decode(codes).view(-1).cpu().numpy().reshape(-1)
        sf.write(output_path, rec_wav, snac_sr)
        print(f"finish: {output_path}")

import json
with open(os.path.join(args.output_path, "snac.json"), "w") as f:
    json.dump(tokens, f, indent=4)
