import os
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Based on https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py
# Note that this does NOT guard against no-op merges. I would suggest testing the output.

if len(sys.argv) != 4:
    print("Usage: python export_hf_checkpoint.py <source> <lora> <dest>")
    exit(1)

source_path = sys.argv[1]
lora_path = sys.argv[2]
dest_path = sys.argv[3]

# Check if CUDA is available and get the GPU device name
device_available = torch.cuda.is_available()

# Initialize device_map as using CPU
device_map = {"": "cpu"}

# Check if GPU with sufficient memory is available
if device_available:
    cuda_device = torch.device('cuda')
    total_memory = torch.cuda.get_device_properties(cuda_device).total_memory / 1e9  # In GB
    
    if total_memory > 18: # 7B 模型需要 17GB+ 的内存才能merge
        device_map = {"": "cuda"}

print(f"Device Map: {device_map}")
print(f"Loading base model from {source_path}")

base_model = AutoModelForCausalLM.from_pretrained(
    source_path,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map=device_map,
    trust_remote_code=True,
)

print(f"Loading lora from {lora_path}")

lora_model = PeftModel.from_pretrained(
    base_model,
    lora_path,
    device_map=device_map,
    torch_dtype=torch.float16,
)

print(f"Merging model and lora")

# merge weights - new merging method from peft
lora_model = lora_model.merge_and_unload()
lora_model.train(False)

# lora_model_sd = lora_model.state_dict()
# deloreanized_sd = {
#     k.replace("base_model.model.", ""): v
#     for k, v in lora_model_sd.items()
#     if "lora" not in k
# }

print(f"Saving merged model to {dest_path}")

base_model.save_pretrained(
    dest_path, max_shard_size="10GB"
    # ,state_dict=deloreanized_sd
)

print(f"Saving tokenizer to {dest_path}")

tokenizer = AutoTokenizer.from_pretrained(source_path)
tokenizer.save_pretrained(dest_path)

# We don't use added_tokens but it will be saved, have to remove it to prevent llama.cpp failure
added_tokens_file = "added_tokens.json"

# 构建完整的added_tokens文件路径
added_tokens_path = os.path.join(dest_path, added_tokens_file)

# 如果文件存在，则重命名它，添加 .nouse 后缀
if os.path.exists(added_tokens_path):
    new_name = f"{added_tokens_path}.nouse"
    os.rename(added_tokens_path, new_name)
    print(f"Renamed {added_tokens_path} to {new_name}")
else:
    print(f"{added_tokens_path} does not exist, no need to rename.")
