#!/usr/bin/bash
#SBATCH -J qwen3b-3090
#SBATCH -A ugrad_ce
#SBATCH -p batch_ce_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/qwen3b-3090-%j.out

set -euo pipefail
cd /data/dada0228/plan-budget/P-and-B

export HF_HOME=/data/dada0228/huggingface_cache
export MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
export LOCAL_MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
export LOCAL_LLM=1
export LOCAL_MAX_NEW_TOKENS=512

ENV_PY=/data/dada0228/conda_envs/plan_budget/bin/python

echo "===== HOST ====="
hostname

echo "===== GPU ====="
nvidia-smi

echo "===== ENVIRONMENT ====="
"$ENV_PY" - <<'PY'
import torch, transformers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("GPU memory (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-3B-Instruct"
print("Loading tokenizer:", model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)
model = model.to("cuda")
model.eval()

prompt = "What is 17 multiplied by 24? Explain briefly."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("Generating...")
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

print("===== OUTPUT =====")
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
print("===== TEST PASSED =====")
PY
