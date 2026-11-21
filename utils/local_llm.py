import os
import threading
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = os.getenv("LOCAL_MODEL_ID") or os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "1024"))

_tokenizer = None
_model = None
_load_lock = threading.Lock()


def _load_model():
    global _tokenizer, _model

    if _model is not None:
        return _tokenizer, _model

    with _load_lock:
        if _model is None:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is unavailable. Check the PyTorch/CUDA installation."
                )

            print(f"[LocalLLM] Loading model: {MODEL_ID}")

            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            model.to("cuda")
            model.eval()

            _tokenizer = tokenizer
            _model = model

            print(f"[LocalLLM] Model loaded on {_model.device}")

    return _tokenizer, _model


def _generate(
    prompt,
    max_tokens=8192,
    temperature=0.1,
    top_p=0.9,
    is_chat=False,
):
    tokenizer, model = _load_model()

    if is_chat:
        prompt_text = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = prompt

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
    )
    inputs = {key: value.to("cuda") for key, value in inputs.items()}

    requested_tokens = max(1, int(max_tokens))
    max_new_tokens = min(requested_tokens, MAX_NEW_TOKENS)

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    if temperature is not None and temperature > 0:
        generation_args.update({
            "do_sample": True,
            "temperature": max(float(temperature), 0.01),
            "top_p": float(top_p),
        })
    else:
        generation_args["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **generation_args,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_length:]
    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    prompt_tokens = input_length
    completion_tokens = len(generated_ids)

    eos_token_id = tokenizer.eos_token_id
    reached_eos = (
        eos_token_id is not None
        and completion_tokens > 0
        and int(generated_ids[-1]) == eos_token_id
    )
    reached_token_cap = (
        completion_tokens >= max_new_tokens
        and not reached_eos
    )

    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        max_new_tokens=max_new_tokens,
        reached_eos=reached_eos,
        reached_token_cap=reached_token_cap,
        finish_reason=(
            "eos" if reached_eos
            else "length" if reached_token_cap
            else "other"
        ),
    )

    return text, usage


def complete(prompt, max_tokens=8192, temperature=0.1, top_p=0.9, **kwargs):
    text, usage = _generate(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        is_chat=False,
    )

    return SimpleNamespace(
        choices=[SimpleNamespace(text=text)],
        usage=usage,
    )


def chat(messages, max_tokens=8192, temperature=0.1, top_p=0.9, **kwargs):
    text, usage = _generate(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        is_chat=True,
    )

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text)
            )
        ],
        usage=usage,
    )
