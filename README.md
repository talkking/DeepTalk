# DeepTalk: Towards Seamless and Smart Speech Interaction with Adaptive Modality-Specific MoE


<font size=7><div align='center' > [[📖 DeepTalk Paper](https://openreview.net/pdf?id=dXwFgSRVix)] [[🤖 Model Weight](https://huggingface.co/shaohang/DeepTalk)]  </div></font>





## :fire: News



* **`2025.06.23`** 🌟 We are proud to launch DeepTalk, a native end-to-end large speech model with MoE architecture.


## 📄 Contents <!-- omit in toc -->


- [Highlights](#-highlights)
- [Experimental Results](#-experimental-results)
- [Training](#-training)
- [Evaluation](#-evaluation)


## ✨ Highlights

- **High IQ, high EQ**. Solves the catastrophic forgetting problem of native multimodality. Maximizes the language generalization capabilities of the original text LLM.
- **New Architecture**. The first model for end-to-end voice interaction based on MoE architecture.
- **Low Latency**. Based on the MoE architecture, since the activation parameter is only 2.4B which is much lower than other 7B dense models, and we adopt the modeling method of parallel generation of speech-text, the inference delay will be greatly reduced, and the delay of the most end-to-end speech interactions is within 500ms.
  



## :label: TODO 

- [x] Release training code and inference code.
- [x] Release checkpoints.
- [ ] Release the cleaned open-source data JSON and audio.



## 📈 Experimental Results
- **Evaluation on LLM benchmark**.
  
  <img width="543" alt="Clipboard_Screenshot_1750646258" src="https://github.com/user-attachments/assets/65cef43b-f991-4cb3-9b5b-3208f837f607" />

- **Comparison of Spoken Question Answering**.

  <img width="560" alt="Clipboard_Screenshot_1750646291" src="https://github.com/user-attachments/assets/54d6bbb5-3269-4c38-9182-eeba8150bd37" />



- **Evaluation on Text to Speech**.

  <img width="280" alt="Clipboard_Screenshot_1750646311" src="https://github.com/user-attachments/assets/4ba26b02-70ec-45d8-a268-0e320371fff3" />


- **Evaluation on Automatic Speech Recognition**.

  <img width="540" alt="Clipboard_Screenshot_1750646334" src="https://github.com/user-attachments/assets/87b22016-9c32-41a2-a6a5-c95d88ead5bd" />




## 📔 Requirements and Installation

### Get the Code
```
git clone https://github.com/talkking/DeepTalk.git
cd DeepTalk
pip install -r requirements.txt
```



### Data Format
#### **Speech QA Data Format**


```jsonc
{
  "conversations": [
    {
      "content": "水以什么类型的结构转动涡轮机？选项有：水力发电坝、水坑、污水泵、地下溪流、水槽。为什么人类会选择“水力发电坝”来回答这个问题？",
      "wavpath": "path/to/AudioQA-1M/q.wav",
      "codec": "path/to/AudioQA-1M/q.codec",
      "role": "user"
    },
    {
      "content": "人类会选择“水力发电坝”来回答这个问题，因为水力发电坝是专门设计来利用水的力量旋转涡轮机发电的。其他选项，比如水坑、污水泵、地下溪流或水槽，都不是为了发电设计的。水力发电坝的用途和问题中的水轮机机制最匹配，所以是正确答案。",
      "wavpath": "path/to/AudioQA-1M/a.wav",
      "codec": "path/to/AudioQA-1M/a.codec",
      "role": "assistant"
    }
  ]
}
```

#### **ASR Data Format**


```jsonc
{
  "messages": [
    {
      "content": "Convert the speech to text.\n<|audio|>",
      "wavpath": "path/to/wav/q.wav",
      "role": "user"
    },
    {
      "content": "没有跟大家说是在做什么",
      "role": "assistant"
    }
  ]
}
```

#### **TTS Data Format**


```jsonc
{
  "messages": [
    {
      "content": "Convert the text to speech.\n那我情愿无药可救。",
      "role": "user"
    },
    {
      "content": "<|audio|>",
      "wavpath": "path/to/wav/a.wav",
      "role": "assistant"
    }
  ]
}
```

## 🎲 Training
Three types of model training are supported, deeptalk, moextend, puremoe.


The following tutorial will take `DeepTalk` as an example.


### Stage-1 (Audio-Text Alignment)

```
bash run_scripts/train/alignment_s1.sh
```

The above script may need some adjustments.

- Set `MODEL_NAME_OR_PATH` and `AUDIO_ENCODER` to your base model folder.
- Set `WENET_DIR ` to your dataset folder.
- Modify other variables as needed for your environment.

### Stage-2 (Unimodal Expert Specialization Training)

#### Stage-2.1 (Audio Expert Specialization Training)

```
bash run_scripts/train/deeptalk/deeptalk_s2p1.sh
```
#### Stage-2.2 (Text Expert Specialization Training)

```
bash run_scripts/train/deeptalk/deeptalk_s2p2.sh
```


#### Stage-3 (Joint Training of Modality Experts)

```
bash run_scripts/train/deeptalk/deeptalk_s3.sh
```

#### Stage-rl (Audio Generation with Reinforcement Learning)

The above script may need some adjustments.

- Set `audio_dpo_data` to your audio dpo dataset folder.
- audio dpo format as follows:
```jsonc
[
  {
    "role": "user",
    "content": "Repeat the sentence inside the brackets without any explanation. \n【and a couched lion with shaggy head resting upo
his fore paws we watched her press beads of proper size and color into the eye sockets skilfully finish the base upon which each fig
e lay】"
  },
  {
    "role": "assistant",
    "content": "and a couched lion with shaggy head resting upon his fore paws we watched her press beads of proper size and color i
o the eye sockets skilfully finish the base upon which each figure lay",
    "win_wavpath": "/mnt/data/alanhshao/vita-e2e/datasets/dpo_data/win/2691-156755-0005.wav",
    "win_codec": "/mnt/data/alanhshao/vita-e2e/datasets/dpo_data/win/2691-156755-0005.snac",
    "win_reward": 0.17142857142857143,
    "lose_wavpath": "/mnt/data/alanhshao/vita-e2e/datasets/dpo_data/lose/2691-156755-0005.wav",
    "lose_codec": "/mnt/data/alanhshao/vita-e2e/datasets/dpo_data/lose/2691-156755-0005.snac",
    "lose_reward": 0.3142857142857143
  }
]
```


## 🔎 Evaluation

### Evaluation on LLM benchmark
#### Step 1 Refer to the following script to extract the weights of the LLM backbone
```
python run_scripts/test/llm/vllm_inference/scripts/save_deepseekV2_model.py
```
#### Step 2 Refer to the following script to deploy LLM backbone with vllm
```
run_scripts/test/llm/vllm_inference/network/deepseek_v2-vllm063.py
```
#### Step 3 Evaluation with OpenCompass

https://github.com/open-compass/opencompass.git

### Evaluation on SQA
```
bash run_scripts/test/sqa/eval_qa_ngpu.sh
```

### Evaluation on ASR
```
bash run_scripts/test/asr/infer_asr.sh
```

### Evaluation on TTS
```
bash run_scripts/test/tts/infer_tts_ngpu.sh
```



## :black_nib: Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.   


```bibtex
@article{shao2025deeptalk,
  title={DeepTalk: Towards Seamless and Smart Speech Interaction with Adaptive Modality-Specific MoE},
  author={Shao, Hang and Gao, Heting and Shen, Yunhang and Chen, Jiawei and Li, Lijiang and Long, Zuwei and Tong, Bo and Li, Ke and Sun, Xing},
  journal={arXiv preprint arXiv:2506.21864},
  year={2025}
}
```
