# <img src="assets/deeptalk.png" width="5%" height="5%">DeepTalk: Towards Seamless and Smart Speech Interaction with Adaptive Modality-Specific MoE



<font size=7><div align='center' > [[📖 DeepTalk Paper](https://openreview.net/pdf?id=dXwFgSRVix)] [[🤖 Model Weight](https://huggingface.co/shaohang/DeepTalk)]  </div></font>





## :fire: News



* **`2025.06.21`** 🌟 We are proud to launch DeepTalk, a native end-to-end large speech model with MoE architecture.


## 📄 Contents <!-- omit in toc -->


- [Highlights](#-highlights)
- [Models](#-models)
- [Experimental Results](#-experimental-results)
- [Training](#-training)
- [Inference](#-inference)
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
- **LLM benchmark**.
  
  <img width="543" alt="Clipboard_Screenshot_1750646258" src="https://github.com/user-attachments/assets/65cef43b-f991-4cb3-9b5b-3208f837f607" />

- **Comparison of Spoken Question Answering**.

  <img width="560" alt="Clipboard_Screenshot_1750646291" src="https://github.com/user-attachments/assets/54d6bbb5-3269-4c38-9182-eeba8150bd37" />



- **Comparison of Text to Speech**.

  <img width="280" alt="Clipboard_Screenshot_1750646311" src="https://github.com/user-attachments/assets/4ba26b02-70ec-45d8-a268-0e320371fff3" />


- **Comparison of Automatic Speech Recognition**.

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
      "content": "<|audio|>",
      "wavpath": "path/to/AudioQA-1M/q.wav",
      "role": "user"
    },
    {
      "content": "好的，这样排列更合理：这些生物废弃物如鸡蛋壳、蛤壳、贻贝壳比其他工业废渣更有价值。研究表明，它们在能源、材料、环境保护等领域有广泛应用。高效利用贝壳能提高资源利用效率，减少废弃物，减轻环境负担。特别是在这些领域中，鸡蛋壳因为含有丰富的钙元素，被用于制造医药品和肥料。\n<|audio|>",
      "wavpath": "path/to/AudioQA-1M/a.wav",
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
      "role": "user"
    },
    {
      "content": "没有跟大家说是在做什么",
      "role": "assistant"
    }
  ],
  "audios": [
    "datasets/wenet-e2e/wenetspeech/data/cuts_L_fixed.00000000/X00/X0000016296_135343932_S00019.wav"
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
      "role": "assistant"
    }
  ],
  "audios": [
    "datasets/Wenetspeech4TTS/WenetSpeech4TTS/Premium/WenetSpeech4TTS_Premium_9/wavs/X0000001735_50639692_S00035.wav"
  ]
}
```

## 🎲 Training


The following tutorial will take `VITA-Audio-Boost` as an example.

- To train `VITA-Audio-Balance` and other variants, you should modify the `text-audio-interval-ratio`.

  VITA-Audio-Boost:
  ```
  --text-audio-interval-ratio 1 10 4 10 \
  ```

  VITA-Audio-Balance:
  ```
  --text-audio-interval-ratio 1 4 3 8 4 10 \
  ```

- To train `VITA-Audio-Plus-*`, you should use the script like `scripts/deepspeed/sts_qwen25/finetune_sensevoice_glm4voice...`

### Stage-1 (Audio-Text Alignment)

```
bash scripts/deepspeed/sts_qwen25/finetune_glm4voice_stage1.sh 8192 `date +'%Y%m%d_%H%M%S'`
```

The above script may need some adjustments.

- Set `ROOT_PATH` to your code root folder.
- Set `LOCAL_ROOT_PATH` to a temporary code root folder.
- Modify other variables as needed for your environment.

### Stage-2 (Single MCTP Module Training)

```
bash scripts/deepspeed/sts_qwen25/finetune_glm4voice_mtp1_stage1.sh 8192 `date +'%Y%m%d_%H%M%S'`
```

The above script may need some adjustments.

- Set `ROOT_PATH` to your code root folder.
- Set `LOCAL_ROOT_PATH` to a temporary code root folder.
- Set `MODEL_NAME_OR_PATH` to the path of the model trained in Stage 1.
- Modify other variables as needed for your environment.

### Stage-3 (Multiple MCTP Modules Training)

```
bash scripts/deepspeed/sts_qwen25/finetune_glm4voice_mtp10_stage1.sh 8192 `date +'%Y%m%d_%H%M%S'`
```

The above script may need some adjustments.

- Set `ROOT_PATH` to your code root folder.
- Set `LOCAL_ROOT_PATH` to a temporary code root folder.
- Set `MODEL_NAME_OR_PATH` to the path of the model trained in Stage 2.
- Modify other variables as needed for your environment.

### Stage-4 (Supervised Fine-tuning)

```
bash scripts/deepspeed/sts_qwen25/finetune_glm4voice_mtp10_stage2.sh 2048 `date +'%Y%m%d_%H%M%S'`
```

The above script may need some adjustments.

- Set `ROOT_PATH` to your code root folder.
- Set `LOCAL_ROOT_PATH` to a temporary code root folder.
- Set `MODEL_NAME_OR_PATH` to the path of the model trained in Stage 3.
- Modify other variables as needed for your environment.



## 📐 Inference

Here we implement a simple script for inference.

It includes examples of speech-to-speech, ASR, and TTS tasks, as well as streaming and non-streaming inference speed testing.

```
python tools/inference_sts.py
```

- Set `model_name_or_path` to VITA-Audio weights.
- Set `audio_tokenizer_path` to the path of the audio encoder.
- Set `flow_path` to the path of the audio decoder.


## 🔎 Evaluation

Evaluate SQA, ASR, and TTS benchmarks
```
bash scripts/deepspeed/evaluate_sts.sh
```


## &#x1F4E3; Statement

**VITA-Audio is trained on large-scale open-source corpus, and its output has randomness. Any content generated by VITA-Audio does not represent the views of the model developers. We are not responsible for any problems arising from the use, misuse, and dissemination of VITA-Audio, including but not limited to public opinion risks and data security issues.**


## :black_nib: Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.   



```bibtex
@misc{,
      title={VITA-Audio: Fast Interleaved Cross-Modal Token Generation for Efficient Large Speech-Language Model}, 
      author={Zuwei Long and Yunhang Shen and Chaoyou Fu and Heting Gao and Lijiang Li and Peixian Chen and Mengdan Zhang and Hang Shao and Jian Li and Jinlong Peng and Haoyu Cao and Ke Li and Rongrong Ji and Xing Sun},
      year={2025},
      eprint={2505.03739},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.03739}, 
}
```
