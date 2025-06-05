import torch
from safetensors.torch import save_file
import sys
import os
import json

# 假设您已经定义并加载了模型
# model = YourModelClass()
sys.path.append("/mnt/data/alanhshao/vita-e2e")
from src.model import LUCYDeepseekV2ForCausalLM as VITADeepseekV2ForCausalLM, LUCYDeepseekV2Config as VITADeepseekV2Config
#model = VITAQwen2ForCausalLM.from_pretrained("/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s3_zh_parallel_tte-100124-134115/checkpoint-72000")
model = VITADeepseekV2ForCausalLM.from_pretrained(sys.argv[1])
config = VITADeepseekV2Config.from_pretrained(sys.argv[1])
lm_head_weight = model.lm_head.weight.data.clone()
print(lm_head_weight.shape)

new_vocab_size = config.text_vocab_size_padded
model.lm_head = torch.nn.Linear(2048, new_vocab_size, bias=False)
model.lm_head.weight.data = lm_head_weight[:new_vocab_size]
print("Before\n", model)

# 提取我们感兴趣的模块的权重
def extract_weights(model, modules_to_keep):
    state_dict = {}
    layers = 0
    for name, param in model.named_parameters():
        for module in modules_to_keep:
            if name.startswith("model." + module) or name.startswith("lm_head"):
                #if name.startswith("model.layers"):
                #   layers += 1
                #   if layers > layer_num*12:
                #      continue
                state_dict[name] = param
    return state_dict

# 指定要保留的模块
modules_to_keep = ['embed_tokens', 'layers', 'norm', 'lm_head']

metadata = {'format': 'pt'}

# 提取权重
weights_to_save = extract_weights(model, modules_to_keep)
print("After\n", weights_to_save)
def save_sharded_model(
    state_dict,
    output_dir,
    max_part_size=1024**3,  # 默认每块最大1GB
    prefix="model"
):
    os.makedirs(output_dir, exist_ok=True)

    # 第一步：临时分块（不保存文件，仅记录块数据）
    temp_parts = []
    current_size = 0
    current_part = {}

    for name, tensor in state_dict.items():
        tensor_size = tensor.nelement() * tensor.element_size()

        # 如果当前块已满，保存临时块并重置
        if current_size + tensor_size > max_part_size and current_part:
            temp_parts.append(current_part)
            current_part = {}
            current_size = 0

        current_part[name] = tensor
        current_size += tensor_size

    # 保存最后一个临时块
    if current_part:
        temp_parts.append(current_part)

    # 确定总块数和编号位数（固定5位，如 "00008"）
    total_parts = len(temp_parts)
    total_size = current_size
    num_digits = 5  # 固定5位编号

    # 第二步：生成正式分块文件并构建索引
    index = {"weight_map": {}, "metadata": {"total_size": total_size}}
    for part_idx, part_data in enumerate(temp_parts):
        # 生成符合格式的文件名（如 model-00003-of-00008.safetensors）
        part_filename = (
            f"{prefix}-{part_idx:0{num_digits}d}-of-"
            f"{total_parts:0{num_digits}d}.safetensors"
        )
        part_path = os.path.join(output_dir, part_filename)

        # 保存分块文件
        save_file(part_data, part_path, metadata=metadata)

        # 更新索引
        for key in part_data:
            index["weight_map"][key] = part_filename

    # 保存索引文件
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    return index

# 分块保存（每块最大500MB）
index = save_sharded_model(
    weights_to_save,
    output_dir="outputs/sharded_model",
    max_part_size=8000 * 1024**2,  # 500MB
    prefix="model"
)

#print(f"分块文件命名示例: model-00000-of-00003.safetensors")
#print(f"总块数: {index['metadata']['total_parts']}")

# 将提取的权重保存为 safetensors 文件
#save_file(weights_to_save, 'model.safetensors', metadata=metadata)
