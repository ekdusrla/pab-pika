#!/usr/bin/bash
#SBATCH -J qwen3b-ce
#SBATCH -A ugrad_ce
#SBATCH -p batch_ce_ugrad
#SBATCH --gres=gpu:high_perf:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/qwen3b-ce-%j.out

set -euo pipefail

cd /data/dada0228/plan-budget/P-and-B

export HF_HOME=/data/dada0228/huggingface_cache
ENV_PY=/data/dada0228/conda_envs/plan_budget/bin/python

echo "===== Job information ====="
date
hostname
echo "HF_HOME=$HF_HOME"
df -h "$HF_HOME"
nvidia-smi

echo "===== Python environment ====="
"$ENV_PY" -c "
import torch, transformers, accelerate
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('accelerate:', accelerate.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print('GPU:', torch.cuda.get_device_name(i))
    print('Total memory (GiB):',
          round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2))
"

echo "===== Load Qwen2.5-3B-Instruct ====="
"$ENV_PY" -u - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.to("cuda")
model.eval()

print("Model loaded successfully.", flush=True)
print("Allocated GPU memory (GiB):",
      round(torch.cuda.memory_allocated(0) / 1024**3, 2), flush=True)

messages = [
    {"role": "user", "content": "What is 17 multiplied by 24? Explain briefly."}
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)

answer = output[0][inputs["input_ids"].shape[1]:]
print("Test response:", tokenizer.decode(answer, skip_special_tokens=True), flush=True)
print("Peak GPU memory (GiB):",
      round(torch.cuda.max_memory_allocated(0) / 1024**3, 2), flush=True)
print("Diagnostic completed successfully.", flush=True)
PY

echo "===== Finished ====="
date
