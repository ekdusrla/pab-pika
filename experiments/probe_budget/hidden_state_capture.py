"""Core library: manual token-by-token decoding that captures hidden states
the instant each sub-question boundary is produced -- zero extra forward
passes.

THIS FILE DOES NOT LOAD A MODEL. It only defines functions that take an
already-loaded `model` and `tok` (tokenizer) as arguments. Something else
(generate_training_data.py for building the probe's training set, or your
real inference pipeline later) must load the model once and call these
functions. This split exists so the exact same decoding logic can be reused
by both "build training data" and "real inference" without duplicating it.

Why no training is needed for the marker:
  STEP_MARKER is literal text, not a special reserved token. The model is
  simply shown it in a few-shot example in the system prompt and imitates it,
  the same way it already imitates the literal word "Hint:". No fine-tuning,
  no vocabulary changes.

Why no extra forward pass is needed:
  model.generate() hides per-step hidden states in an awkward nested
  structure, so earlier versions did a second forward pass after generation
  to recover them cleanly. Here we instead call the model's forward()
  ourselves, one token at a time, with KV caching (past_key_values). Every
  step's hidden state is already computed in order to pick the next token --
  we simply keep a reference to it instead of discarding it, for every layer.
"""
import torch

STEP_MARKER = '<<<STEP>>>'   # literal text, not a tokenizer special token


@torch.no_grad()
def generate_with_step_hidden_states(
    model, tok, messages: list[dict], max_new_tokens: int = 512,
):
    """Greedy-decode `messages` and return (full_text, list_of_hidden_state_dicts).

    list_of_hidden_state_dicts[i] = {layer_idx: 1D np.array} captured at the
    moment the i-th STEP_MARKER was completed -- one entry per sub-question,
    in order, covering every layer (embedding layer + every transformer
    layer), so a probe-training step can later pick the best layer via
    cross-validation.
    """
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tok(prompt, return_tensors='pt').input_ids.to(model.device)

    past_key_values = None
    next_input = input_ids
    generated_ids: list[int] = []
    captured: list[dict[int, "torch.Tensor"]] = []

    marker_ids = tok(STEP_MARKER, add_special_tokens=False).input_ids
    lookback = len(marker_ids) + 4  # small buffer for tokenizer boundary effects

    for _ in range(max_new_tokens):
        out = model(
            input_ids=next_input,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = out.past_key_values

        # Hidden state at every layer for the position that is about to
        # produce the next token. Already computed for the logits above --
        # capturing it costs nothing extra.
        current_hidden_all_layers = {
            layer_idx: h[0, -1, :].float().cpu().numpy()
            for layer_idx, h in enumerate(out.hidden_states)
        }

        next_token_id = out.logits[0, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token_id.item())

        if next_token_id.item() == tok.eos_token_id:
            break

        tail_text = tok.decode(generated_ids[-lookback:], skip_special_tokens=True)
        if tail_text.rstrip().endswith(STEP_MARKER):
            captured.append(current_hidden_all_layers)

        next_input = next_token_id.unsqueeze(0)

    full_text = tok.decode(generated_ids, skip_special_tokens=True)
    return full_text, captured


# Original P-and-B paper (Appendix H) decomposition prompt, verbatim, with
# ONE addition on top: an instruction + few-shot demonstration of STEP_MARKER
# after each sub-question+hint. Everything else matches the paper.
DECOMP_SYSTEM = (
    "-Goal-\n"
    "You are an experienced expert in domain and exam question designer. Your role is to help "
    "students break down challenging math problems into a series of simpler, high-level "
    "sub-questions.\n"
    "We don't want too many detailed sub-questions, which are not beneficial for testing "
    "students' ability in an exam. Each sub-question should build on the previous one so that, "
    "once all have been answered, the complete solution is clear.\n"
    "Your output should be a list of sub-questions with brief hints explaining the purpose of "
    "each step, but you should not reveal your internal chain-of-thought either the final "
    "solution.\n\n"
    "Instructions for Decomposition:\n"
    "First, analyze the problem and identify the key ideas needed to solve it. Then, generate a "
    "series of 2 to 5 sub-questions that lead the student step by step to the complete solution. "
    "The difficulty level of the problem is presented out of 5, where 1 is easy, and 5 is hard. "
    "Please adjust the number of sub-questions based on the level. Ideally, we want fewer "
    "sub-questions for easy problems and more sub-questions for challenging problems.\n"
    "DO NOT perform reasoning, directly output those sub-questions based on your gut feelings; "
    "only output the list of sub-questions with brief hints for each.\n"
    "Your answer should be a list of numbered sub-questions. Each sub-question should have a "
    "brief accompanying hint that explains what the student will achieve by answering that "
    "part.\n\n"
    # --- addition on top of the paper's prompt: the step marker ---
    "In addition to the above, after EACH sub-question and its hint, you must output the exact "
    f"text '{STEP_MARKER}' alone on its own line before continuing to the next sub-question. "
    "This marker must appear after every sub-question without exception, exactly as shown in "
    "the example below.\n\n"
    "Example Decomposition:\n"
    "**Problem:** Find the remainder when (9 \u00d7 99 \u00d7 999 \u00d7 \u00b7 \u00b7 \u00b7 \u00d7 "
    "99\u00b7\u00b7\u00b79 [999 9's]) is divided by 1000.\n"
    "**Level:** 3 out of 5\n"
    "**Decomposed Sub-questions:**\n"
    "1. Compute the product modulo 8.\n"
    "Hint: Simplify each term using (10 \u2261 2 mod 8), noting that (10^k \u2261 0 mod 8) for "
    "k \u2265 3, leading to terms of (\u22121 mod 8).\n"
    f"{STEP_MARKER}\n"
    "2. Compute the product modulo 125.\n"
    "Hint: Recognize (10^3 \u2261 0 mod 125), so terms for (k \u2265 3) become (\u22121 mod 125). "
    "Calculate the product of the first two terms and combine with the remaining terms.\n"
    f"{STEP_MARKER}\n"
    "3. Solve the system of congruences using the Chinese Remainder Theorem.\n"
    "Hint: Combine the results from modulo 8 and modulo 125 to find a common solution modulo "
    "1000.\n"
    f"{STEP_MARKER}\n"
)


def build_decomp_messages_with_marker(problem: str, level: int) -> list[dict]:
    """Original P-and-B decomposition prompt (system message above) + the
    paper's own user-turn wording, unchanged except for the STEP_MARKER
    instruction/example that now lives inside DECOMP_SYSTEM."""
    return [
        {'role': 'system', 'content': DECOMP_SYSTEM},
        {'role': 'user', 'content': (
            f"A student has presented you with the following math problem:\n"
            f"Problem: {problem}\n"
            f"Level: {level} out of 5\n"
            f"**REMEMBER**, you are not allowed to think about it, please directly generate the "
            f"answer in the following:\n"
            f"Decomposed Sub-questions:"
        )},
    ]


def parse_steps_from_marked_text(text: str) -> list[str]:
    """Split the generated text on STEP_MARKER to recover the sub-question
    strings themselves (for saving alongside credits / for building the
    reasoner's prompt later)."""
    parts = [p.strip() for p in text.split(STEP_MARKER)]
    return [p for p in parts if p]
