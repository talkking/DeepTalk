import sys
import csv
import json
import tqdm

prompt = {}
prompt["en"] = '''Repeat the sentence inside the brackets without any explanation. \n【{}】'''
prompt["zh"] = '''复述中括号里面的句子，不需要做任何解释：\n【{}】'''

src_json_file = sys.argv[1]
res_json_file = sys.argv[2]
lang = sys.argv[3]

conversations = []
with open(src_json_file, "r") as f, open(res_json_file, "w") as f1:
    for data in f:
        line = json.loads(data)
        text = line['messages'][1]['content']
        wav = line['audios'][0].split("/")[-1].split(".")[0]
        new_prompt = prompt[lang].format(text)
        conv = [{"role": "user", "content": new_prompt, "uttid": wav}]
        conversations.append(conv)
    json.dump(conversations, f1, ensure_ascii=False, indent=2)
    
