import os
import torch
import json
import transformers
import pathlib
import einops
import random
import numpy as np
import soundfile as sf
from typing import Optional, Union
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from src.model.language_model.puremoe import VITADeepseekV2ForCausalLM, VITADeepseekV2Config
from transformers import AutoModelForCausalLM
from tqdm import tqdm
from snac import SNAC
from time import time
from itertools import groupby
from src.utils import data_util
from src.utils.sampling import sample
from src.scripts.train_puremoe import ModelArguments
from src.utils import conversation as conversation_lib, move_to_cuda, move_to_cpu
from src.constants import IGNORE_INDEX, AUDIO_PH, PAD_TOKEN, EMOTION_SP
import json
import csv




@dataclass
class InferenceArguments:
    max_code_length: Optional[int] = field(default=None)
    snac_sr: Optional[int] = field(default=24000)
    snac_model: Optional[str] = field(default="hubertsiuzdak/snac_24khz")
    output_path: Optional[str] = field(default=None)
    save_audio: Optional[bool] = field(default=True)
    output_text_only: Optional[bool] = field(default=False)
    questions: Optional[str] = field(default=None)
    use_audio_input: Optional[bool] = field(default=True)

audio_num_codebook = 7
parser = transformers.HfArgumentParser((ModelArguments, data_util.DataArguments, InferenceArguments))
model_args, data_args, infer_args = parser.parse_args_into_dataclasses()
model_path = model_args.model_name_or_path
config = VITADeepseekV2Config.from_pretrained(model_path)
text_vocab_size_padded = config.text_vocab_size_padded #152000
audio_vocab_size_padded = config.audio_vocab_size_padded #4160
EOA = config.audio_additional_tokens["EOA"] #4096
PAD_A  = config.audio_additional_tokens["PAD_A"] #4097
BOA = config.audio_additional_tokens["BOA"] #4098
ANS_A  = config.audio_additional_tokens["ANS_A"] #4099
PAD_T  = config.text_additional_tokens["PAD_T"] #151937
EOT    = config.text_additional_tokens["EOT"] #151936
IM_END = 100001 #tokenizer.encode("<｜end▁of▁sentence｜>") #config.audio_additional_tokens["IM_END"] #151645
F10    = 4103                                                                                                                                         
M29    = 4104


### heting begin
FC_TOKEN = 27
NEUTRAL_TOKEN = 151648
JOY_TOKEN     = 151649
SADNESS_TOKEN = 151650
FEAR_TOKEN    = 151651
ANGER_TOKEN   = 151652
SUPRISE_TOKEN = 151653
DISGUST_TOKEN = 151654
SORRY_TOKEN   = 151655
TIRQ         = "<|tirq|>"         # 151656 text interrupt
AIRQ_DENIAL  = "<|airq_denial|>"  # 151657 audio interrupt: denial and discontent
AIRQ_INQUIRY = "<|airq_inquiry|>" # 151658 audio interrupt: further inquiry
AIRQ_CHANGE  = "<|airq_change|>"  # 151659 audio interrupt: change topic
ANEG_AFFIRM  = "<|airq_affirm|>"  # 151660 audio negative interrupt: affirmative acknowledgement
ANEG_NOISE   = "<|airq_noise|>"   # 151661 audio negative interrupt:background noise

SPECIAL_START = 151681 # "<|"
SPECIAL_END   = 151682 # "|>"
SPECIAL_START = 151662 # "<|"
SPECIAL_END   = 151663 # "|>"

STATE_TOKENS = [
    TIRQ,
    AIRQ_DENIAL,
    AIRQ_INQUIRY,
    AIRQ_CHANGE,
    ANEG_AFFIRM,
    ANEG_NOISE,
]

### heting end


audio_encoder_type="unknown"
 


def apply_repetition_penalty(logits: torch.Tensor, 
                            generated_tokens: Union[list, torch.tensor], 
                            penalty: float = 1.0) -> torch.Tensor:
    """
    应用重复惩罚到logits
    :param logits: [vocab_size] 当前步的logits
    :param generated_tokens: 已生成的所有token IDs列表
    :param penalty: 惩罚系数（>1抑制重复，<1鼓励重复）
    """
    if penalty == 1.0 or not generated_tokens:
        return logits
    
    logits = logits[0, -1]
    #import pdb; pdb.set_trace()
    # 只处理已出现的唯一token
    for token in set(generated_tokens):
        if logits[token] < 0:
            logits[token] *= penalty  # 降低负logits（高概率token）
        else:
            logits[token] /= penalty  # 降低正logits（低概率token）
    return logits[None, None]

def next_token(
    model, 
    audios=None,
    attention_mask=None,
    input_ids=None,
    audio_lengths=None,
    audio_attention_mask=None,
    past_key_values=None,
    **kwargs,
) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        audios=audios, 
        audio_lengths=audio_lengths, 
        audio_attention_mask=audio_attention_mask,
        past_key_values=past_key_values,
        use_cache=True
    )
    batch_size = input_ids.shape[0]
    assert batch_size == 1 or batch_size == 2, batch_size
    # if batch size is 2, use first item to predict audio codec and use second item to predict text
    logits_t = outputs.logits[-1:,:,:text_vocab_size_padded] # last item in batch
    # logits_t = outputs.logits[:1,:,:text_vocab_size_padded] # first item in batch
    
    # apply repetition penalty
    logits_t = apply_repetition_penalty(logits_t , kwargs["text_tokens"], penalty=1.2)
    
    next_t = sample(logits_t, top_k=1).to(input_ids[0]).repeat(batch_size).unsqueeze(-1) # B x 1

    next_a, next_ua = [], [] # layer shifted/unshifted audio tokens

    for i in range(audio_num_codebook):
        start = text_vocab_size_padded + i * audio_vocab_size_padded
        end = text_vocab_size_padded + (i+1) * audio_vocab_size_padded
        logits_a_i = outputs.logits[:1, :,start:end]
        ua_i = input_ids.new_zeros(batch_size,1).fill_(PAD_A)
        ua_i[:1, :] = sample(logits_a_i, top_k=10) # B x 1 # first item in batch
        a_i = codec_layer_shift(ua_i, i) # B x 1
        next_a.append(a_i)
        next_ua.append(ua_i)
    
    next_a = torch.cat(next_a, dim=-1) # B x 7
    next_ua = torch.cat(next_ua, dim=-1) # B x 7
    past_key_values = outputs.past_key_values
    return next_t, next_a, next_ua, past_key_values


def decode_audio(snac, audio_codes_padded):
    T, N = audio_codes_padded.shape # length of auido codes and number of codebooks
    audio_codes = torch.zeros((T-N-1, N)).to(audio_codes_padded) # 1 for EOA
    for i in range(N):
        audio_codes[:,i] = audio_codes_padded[i+1:-(N-i), i]
    # print(audio_codes)
    print("number of audios codes out of range", (audio_codes>=4096).sum(), audio_codes.numel())
    audio_codes[audio_codes>=4096] = 0
    (
        code_12hz, code_24hz, code_48hz
    ) = (
        audio_codes[:,0:1], 
        audio_codes[:,1:3],
        audio_codes[:,3:]
    )
    codes = [
        code_12hz.reshape(1, -1), 
        code_24hz.reshape(1, -1), 
        code_48hz.reshape(1, -1)
    ]
    audio = snac.decode(codes).view(-1)
    return audio


def load_wav(wavpath, sample_rate=16_000):
    wavpaths = [wavpath] if type(wavpath) is not list else wavpath
    
    wavs = []
    for i, wdata in enumerate(wavpaths):
        if type(wdata) is dict:
            wpath, start, end, audio_length = \
                wdata["wavpath"], wdata["start"], wdata["end"], wdata["audio_length"]
        else:
            wpath, start, end, audio_length = wdata, 0, None, None
        wav, sr = sf.read(wpath, start=start, stop=end)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert sr == sample_rate, f"Audio sampling rate {sr} != {sample_rate}"
        assert audio_length is None or len(wav) == audio_length, \
            f"Audio length {len(wav)} != {audio_length} of {wpath} with start {start} and end {end}"
        assert end is None or (end - start == audio_length), \
            f"Audio length {audio_length} != end {end} - start {start}"
        if i > 0:
            interval = random.uniform(0.35, 0.75)
            si_leng = int(interval * sample_rate)
            silence = np.zeros(si_leng)
            wavs.append(silence)
        wavs.append(wav)
    wav_cat = np.concatenate(wavs)
    wav_cat = torch.from_numpy(wav_cat).float().unsqueeze(0)
    return wav_cat, sr

def load_wav_feat(wavpaths, audio_processor, sample_rate=16_000, audio_feature_rate=50):
    wav, sr = load_wav(wavpaths)
    assert sr == sample_rate, f"{sr} != {sample_rate}"
    if audio_encoder_type == "whisper":
        wav = wav[0]
        audio_length = len(wav)
        audio = audio_processor(wav, sampling_rate=sr, return_tensors="pt").input_features
        audio_length = int(audio_length / sample_rate * audio_feature_rate) + 1
    elif audio_encoder_type == "whale":
        # wav = torch.from_numpy(wav).float().unsqueeze(0)
        audio, audio_length = audio_processor.process(waveform=wav, sample_rate=sr)
    return audio, audio_length

def codec_layer_shift(input_id, layer):
    return input_id + text_vocab_size_padded + layer * audio_vocab_size_padded

def prepare_inputs_whisper(
        source, use_audio_input, tokenizer, audio_processor, add_system_prompt, 
        system_prompt=None,
        past_input_dict=None, generated=None
    ):
    shifted_PAD_A = torch.LongTensor([codec_layer_shift(PAD_A, i) for i in range(audio_num_codebook)])
    shifted_BOA   = torch.LongTensor([codec_layer_shift(BOA, i)   for i in range(audio_num_codebook)])
    shifted_EOA   = torch.LongTensor([codec_layer_shift(EOA, i)   for i in range(audio_num_codebook)])
    shifted_ANS_A = torch.LongTensor([codec_layer_shift(ANS_A, i) for i in range(audio_num_codebook)])
    shifted_F10   = torch.LongTensor([codec_layer_shift(F10, i)   for i in range(audio_num_codebook)])
    shifted_M29   = torch.LongTensor([codec_layer_shift(M29, i)   for i in range(audio_num_codebook)])
    AUDIO_PH_idx  = tokenizer.convert_tokens_to_ids(AUDIO_PH)
    PAD_TOKEN_idx = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
    #conv = conversation_lib.conv_qwen2.copy()
    conv = conversation_lib.conv_deepseek.copy()
    conv.messages = []

    audios, audio_lengths = torch.zeros([0,80,3000]), torch.zeros([0]).long()
    # import pdb; pdb.set_trace()
    if past_input_dict is not None:
        audio_lengths = past_input_dict["audio_lengths"]
        num_audio = len(audio_lengths) // 2
        audio_lengths = audio_lengths[:num_audio]
        audios = past_input_dict["audios"][:num_audio]
    has_audio_input = "wavpath" in source

    if has_audio_input and use_audio_input:
        audio, audio_length = load_wav_feat(source["wavpath"], audio_processor)
        message = AUDIO_PH * (audio_length + 2)
        # audios.append(audio)
        audios = torch.cat([audios, audio])
        audio_lengths = torch.cat([
            audio_lengths, torch.LongTensor([audio_length])
        ])

    else:
        message = source["content"]
    role = source["role"]
    #if add_system_prompt:
    conv.append_message(role, message)
    prompt = conv.get_prompt()
    #else:
    #prompt = f"<|im_start|>user\n{message}<|im_end|>\n"
    #prompt += "<|im_start|>assistant\n"
    prompt += "Assistant: "
    speaker = source.get("speaker", "ANS_A")                                                                                                          
    speaker = "M29"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
    if past_input_dict is not None:
        input_ids = torch.cat([
            past_input_dict["input_ids"][0,:,-1], generated, input_ids]
        )

    input_codec = input_ids.new_zeros([len(input_ids), audio_num_codebook]).fill_(IGNORE_INDEX) # T x 7
    input_codec[:,:] = shifted_PAD_A[None,:]

    i_chunk, start, end = 0, 0, 0
    audio_attention_mask = input_ids == AUDIO_PH_idx
    for is_placeholder, chunk in groupby(audio_attention_mask.clone()):
        chunk_length = len(list(chunk))
        assert chunk_length > 2 # chunk has at least 1 BOA, 1 EOA, and 1 audio token
        end += chunk_length
        if is_placeholder:
            assert chunk_length == audio_lengths[i_chunk] + 2
            input_codec[start] = shifted_BOA
            input_codec[end-1] = shifted_EOA
            audio_attention_mask[[start,end-1]] = False
            i_chunk += 1
        start = end
    input_ids = torch.cat([input_codec, input_ids.unsqueeze(-1)], dim=-1) # T x 8
    batched_input_ids = input_ids.unsqueeze(0).repeat(2, 1, 1) # 2 x T x 8
    speaker_token = eval(f"shifted_{speaker}")
    #batched_input_ids[0, -1, :-1] = shifted_ANS_A # the last position of the first item in the batch is ANS_A
    batched_input_ids[0, -1, :-1] = speaker_token
    batched_audio_attention_mask = audio_attention_mask.unsqueeze(0).expand(2, -1) # 2 x T
    audio_lengths = audio_lengths.repeat(2) 
    attention_mask = batched_input_ids[...,-1].ne(PAD_TOKEN_idx)
    assert attention_mask.all()
    
    audios = torch.cat([audios, audios]) 
    input_dict = {
        "input_ids": batched_input_ids,
        "labels": None,
        "attention_mask": attention_mask, 
        "audios": audios,
        "audio_lengths": audio_lengths,
        "audio_attention_mask": batched_audio_attention_mask
    }
    return input_dict


def prepare_inputs_whale(
        source, use_audio_input, tokenizer, audio_processor, add_system_prompt, 
        system_prompt=None,
        past_input_dict=None, generated=None
    ):
    shifted_PAD_A = torch.LongTensor([codec_layer_shift(PAD_A, i) for i in range(audio_num_codebook)])
    shifted_BOA   = torch.LongTensor([codec_layer_shift(BOA, i)   for i in range(audio_num_codebook)])
    shifted_EOA   = torch.LongTensor([codec_layer_shift(EOA, i)   for i in range(audio_num_codebook)])
    shifted_ANS_A = torch.LongTensor([codec_layer_shift(ANS_A, i) for i in range(audio_num_codebook)])
    shifted_F10   = torch.LongTensor([codec_layer_shift(F10, i)   for i in range(audio_num_codebook)])
    shifted_M29   = torch.LongTensor([codec_layer_shift(M29, i)   for i in range(audio_num_codebook)])
    AUDIO_PH_idx  = tokenizer.convert_tokens_to_ids(AUDIO_PH)
    PAD_TOKEN_idx = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
    conv = conversation_lib.conv_qwen2.copy()
    conv.messages = []

    # audios, audio_lengths = torch.zeros([0,80,3000]), torch.zeros([0]).long()
    H = 80
    audios, audio_lengths, audio_feature_lengths = torch.zeros(0, H), torch.zeros([0]).long(), torch.zeros([0]).long()
    # import pdb; pdb.set_trace()
    if past_input_dict is not None:
        audio_lengths = past_input_dict["audio_lengths"]
        num_audio = len(audio_lengths) // 2
        audio_lengths = audio_lengths[:num_audio]
        audios = past_input_dict["audios"][:num_audio]
        audio_feature_lengths = past_input_dict["audio_feature_lengths"]
        audio_feature_lengths = audio_feature_lengths[:num_audio]
    has_audio_input = "wavpath" in source

    if has_audio_input and use_audio_input:
        audio, audio_length = load_wav_feat(source["wavpath"], audio_processor)
        audio_feature_length = len(audio)
        # import pdb; pdb.set_trace()
        message = AUDIO_PH * (audio_length + 2)
        # audios.append(audio)
        audio_lengths = torch.cat([
            audio_lengths, torch.LongTensor([audio_length])
        ])

        audio_feature_lengths = torch.cat([
            audio_feature_lengths, 
            torch.LongTensor([audio_feature_length])
        ])
        B = len(audio_feature_lengths) # 1 for new sample
        T = audio_feature_lengths.max()
        new_audios = torch.zeros(B, T, H)
        for i, (a, al) in enumerate(zip(audios, audio_feature_lengths[:B-1])):
            new_audios[i, :al] = a[:al]
        new_audios[-1, :audio_feature_length] = audio

        audios = new_audios
        state = AIRQ_INQUIRY
        # import pdb; pdb.set_trace()

    else:
        message = source["content"]
        state = TIRQ
    role = source["role"]
    speaker = source.get("speaker", "ANS_A")
    speaker = "M29"
    speaker = "F10"
    if add_system_prompt:  
        if system_prompt:
            conv.system = system_prompt
        conv.append_message(role, message)
        prompt = conv.get_prompt()
        print(prompt)
    else:
        prompt = f"<|im_start|>{role}\n{message}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
    if past_input_dict is not None:
        input_ids = torch.cat([
            past_input_dict["input_ids"][0,:,-1], generated, input_ids]
        )

    input_codec = input_ids.new_zeros([len(input_ids), audio_num_codebook]).fill_(IGNORE_INDEX) # T x 7
    input_codec[:,:] = shifted_PAD_A[None,:]

    i_chunk, start, end = 0, 0, 0
    audio_attention_mask = input_ids == AUDIO_PH_idx
    for is_placeholder, chunk in groupby(audio_attention_mask.clone()):
        chunk_length = len(list(chunk))
        assert chunk_length > 2 # chunk has at least 1 BOA, 1 EOA, and 1 audio token
        end += chunk_length
        if is_placeholder:
            assert chunk_length == audio_lengths[i_chunk] + 2
            input_codec[start] = shifted_BOA
            input_codec[end-1] = shifted_EOA
            audio_attention_mask[[start,end-1]] = False
            i_chunk += 1
        start = end
    input_ids = torch.cat([input_codec, input_ids.unsqueeze(-1)], dim=-1) # T x 8
    batched_input_ids = input_ids.unsqueeze(0).repeat(2, 1, 1) # 2 x T x 8
    speaker_token = eval(f"shifted_{speaker}")
    # import pdb; pdb.set_trace()
    # batched_input_ids[0, -1, :-1] = shifted_ANS_A # the last position of the first item in the batch is ANS_A
    batched_input_ids[0, -1, :-1] = speaker_token # the last position of the first item in the batch is ANS_A
    batched_audio_attention_mask = audio_attention_mask.unsqueeze(0).expand(2, -1) # 2 x T
    audio_lengths = audio_lengths.repeat(2) 
    audio_feature_lengths = audio_feature_lengths.repeat(2)
    # audio_feature_lengths = torch.LongTensor([len(a) for a in audios])
    attention_mask = batched_input_ids[...,-1].ne(PAD_TOKEN_idx)
    assert attention_mask.all()
    
    audios = torch.cat([audios, audios]) 
    state_start = tokenizer.convert_tokens_to_ids(TIRQ)
    state_end = state_start + len(STATE_TOKENS) 
    input_dict = {
        "input_ids": batched_input_ids,
        "labels": None,
        "attention_mask": attention_mask, 
        "audios": audios,
        "audio_lengths": audio_lengths,
        "audio_feature_lengths": audio_feature_lengths,
        "audio_attention_mask": batched_audio_attention_mask,
        "state_start": state_start,
        "state_end": state_end,
        "default_state": state,
        "max_input_length": 1e10,
        "infer": True
    }
    return input_dict

def prepare_inputs(
        source, use_audio_input, tokenizer, audio_processor, add_system_prompt, 
        system_prompt=None,
        past_input_dict=None, generated=None
):
    if audio_encoder_type == "whale":
        input_dict = prepare_inputs_whale(
            source, use_audio_input, tokenizer, audio_processor, add_system_prompt, system_prompt, past_input_dict, generated
        )
    elif audio_encoder_type == "whisper":
        input_dict = prepare_inputs_whisper(
            source, use_audio_input, tokenizer, audio_processor, add_system_prompt, system_prompt, past_input_dict, generated
        )
    return input_dict

def get_past_kv(past_kv, index):
    B = 2
    ix = (B + index) % B
    past_kv_i = tuple([tuple([x[ix:ix+1] for x in l]) for l in past_kv])
    return past_kv_i

def repeat_past_kv(past_kv, n):
    past_kv_n = tuple([tuple([x.repeat(n,1,1,1) for x in l]) for l in past_kv])
    return past_kv_n
    
def is_emotion(token):
    return NEUTRAL_TOKEN <= token[-1] <= SORRY_TOKEN

def batch_parallel_decode(model, tokenizer, input_dict, infer_args, device):
    audio_pads_shifted = torch.LongTensor([
        codec_layer_shift(PAD_A, i) for i in range(audio_num_codebook)
    ]).to(device)
    text_pad = torch.LongTensor([PAD_T]).to(device)
    text_ends = False
    audio_ends = False
    audio_num_layer_ends = -1
    audio_tokens, text_tokens = [], []
    input_dict["text_tokens"] = text_tokens
    for t in range(infer_args.max_code_length):
        if not infer_args.save_audio and text_ends:
            break
        if audio_num_layer_ends == audio_num_codebook:
            break
        input_dict["text_tokens"] = text_tokens
        next_t, next_a, next_ua, past_kv = next_token(
            model, **input_dict
        )
        # past_kv (num_layer x (2 x [B, 2, T, 128]) )

        if t < audio_num_codebook:
            num_pad = audio_num_codebook - t
            next_a[0,-num_pad:] = audio_pads_shifted[-num_pad:]
            next_ua[0,-num_pad:] = PAD_A
        if text_ends:
            next_t[0] = text_pad
        if audio_ends:
            next_a[0,:audio_num_layer_ends] = audio_pads_shifted[:audio_num_layer_ends]
            next_ua[0,:audio_num_layer_ends] = PAD_A
            audio_num_layer_ends += 1
        audio_tokens.append(next_ua[0])
        if len(text_tokens) > 0 and text_tokens[-1] == IM_END:
            next_t[:] = EOT
            # save past_key_values of second item and retain only first item in the batch
            next_t = next_t[:1]
            next_a = next_a[:1]
            # past_kv_t = get_past_kv(past_kv, 1)
            past_kv = get_past_kv(past_kv, 0)

        text_tokens.append(next_t[0])

        if next_t[0] == EOT:
            text_ends = True
        if next_ua[0,0] == EOA:
            audio_ends = True
            audio_num_layer_ends = 1
        next_input_ids = torch.cat([next_a, next_t], dim=-1)
        batch_size = next_input_ids.shape[0]
        if infer_args.output_text_only:
            next_input_ids = torch.cat([audio_pads_shifted.unsqueeze(0).repeat(batch_size, -1), next_t], dim=-1)
        next_input_ids = next_input_ids.view(batch_size,1,audio_num_codebook+1)
        input_dict = {
            "input_ids": next_input_ids,
            "past_key_values": past_kv
        }
        # if not text_ends:
        #     current_text = tokenizer.decode(torch.cat(text_tokens)) 
            #print(current_text)
    text = tokenizer.decode(torch.cat(text_tokens)) 
    print(text)
    text_tokens = torch.cat(text_tokens)
    audio_tokens = torch.stack(audio_tokens)
    return text_tokens, audio_tokens

    
def get_audio_encoder_type(audio_encoder):
    if "whisper" in audio_encoder.lower():
        audio_encoder_type = "whisper" 
    elif "audio-encoder-qwen2-7b-instruct" in audio_encoder.lower():
        audio_encoder_type = "whale"
    elif "audio-encoder-qwen2.5-7b" in audio_encoder.lower():
        audio_encoder_type = "whale"
    else:
        raise ValueError(f"Unknown encoder type {model_args.audio_encoder}")
    return audio_encoder_type

@torch.inference_mode()
def demo(conversations, use_audio_input=True):
    
    print("use_audio_input", use_audio_input)
    parser = transformers.HfArgumentParser((ModelArguments, data_util.DataArguments, InferenceArguments))
    model_args, data_args, infer_args = parser.parse_args_into_dataclasses()
    data_util.sync_data_args(model_args, data_args)
    global audio_encoder_type
    audio_encoder_type = get_audio_encoder_type(model_args.audio_encoder)
    if use_audio_input:
        dirname = os.path.dirname(infer_args.output_path)
        filename = os.path.basename(infer_args.output_path)
        infer_args.output_path =  os.path.join(dirname, "Audio")
        os.makedirs(infer_args.output_path, exist_ok=True)
        infer_args.output_path = os.path.join(infer_args.output_path, filename)
        print(infer_args.output_path)
    print(model_args)
    print(data_args)
    print(infer_args)
    print(audio_encoder_type)
    #import pdb; pdb.set_trace()

    device = torch.cuda.current_device()
    model = VITADeepseekV2ForCausalLM.from_pretrained(model_args.model_name_or_path, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2").eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    audio_processor = model.get_audio_encoder().audio_processor
    snac = SNAC.from_pretrained(infer_args.snac_model, cache_dir=model_args.cache_dir).eval().to(device)
    # import pdb; pdb.set_trace()

    # past_kv_t = None
    custom_format = (
        "{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} "
        "[ETA: {remaining}, Speed: {rate_fmt}]"
    )
    progress = tqdm(
        total=len(conversations),
        desc="Inference",
        bar_format=custom_format,
        unit="sample",
        colour="#00ff00"  # 绿色进度条
    )
    already_infer = {}
    if os.path.exists(f"{infer_args.output_path}"):
        with open(f"{infer_args.output_path}", "r", encoding='utf-8') as f:
            for line in f:
                name = line.split('\t')[0]
                already_infer[name] = True

    wavdir = os.path.dirname(infer_args.output_path)
    os.makedirs(f"{wavdir}/wav", exist_ok=True)
    with open(infer_args.output_path, "a", encoding="utf-8") as f:
        for i, conversation in enumerate(conversations):
            past_input_dict, generated = None, None
            if isinstance(conversation, dict) and "conversations" in conversation:
                idx = conversation["id"]
                conversation = conversation["conversations"]
            for turn, source in enumerate(conversation):
                if source["role"] != "user":
                    continue
                wavname = os.path.basename(source['wavpath'])
                if wavname in already_infer:
                    continue
                add_system_prompt = i == 0
                t0 = time()
                input_dict = prepare_inputs(source, use_audio_input, tokenizer, audio_processor, add_system_prompt, past_input_dict, generated)
                # if past_kv_t is not None:
                #    input_dict["past_key_values"] = repeat_past_kv(past_kv_t, 2)
                input_dict = move_to_cuda(input_dict, device)
                text_tokens, audio_tokens = batch_parallel_decode(model, tokenizer, input_dict, infer_args, device)

                if infer_args.save_audio:
                    wav = decode_audio(snac, audio_tokens).cpu().numpy().reshape(-1)
                    wavname = os.path.basename(source['wavpath'])
                    sf.write(f'{wavdir}/wav/{wavname}', wav, infer_args.snac_sr)
                    t1 = time()
                    gen_time = t1 - t0
                    wav_dur = len(wav) / infer_args.snac_sr
                    print(f"Used {gen_time:.4f}s to generate {wav_dur:.4f}s audio with RTF: {gen_time/wav_dur}")

                text = tokenizer.decode(text_tokens) if text_tokens is not None else "..."
                text = text.replace("\n", " ").replace("<｜end▁of▁sentence｜>", "").replace("<｜begin▁of▁sentence｜>", "")
                line = f"{idx}\t{text}\n"
                f.write(line)
                f.flush()
                #break

                past_input_dict = move_to_cpu(input_dict)
                generated = text_tokens[(text_tokens!=PAD_T)&(text_tokens!=EOT)].cpu()
                #break
            progress.update(1)
            
    
        
        
if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, data_util.DataArguments, InferenceArguments))
    model_args, data_args, infer_args = parser.parse_args_into_dataclasses()
    conversations = []
    with open(infer_args.questions) as f:
        for l in f:
            data = json.loads(l)
            #import pdb; pdb.set_trace()
            question_wav = data['wavpath']
            question_text = data['question'].strip()
            conv = {
                "conversations": [{"role": "user", "content": question_text, "wavpath": question_wav}], "id": pathlib.Path(question_wav).name
            }
            conversations.append(conv)
    demo(conversations, use_audio_input=infer_args.use_audio_input)
    
