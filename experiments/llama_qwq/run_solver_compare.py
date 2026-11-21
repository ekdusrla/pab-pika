"""Compare Vanilla and prompt-budgeted PAB using saved Planner results.

No project utility imports, dotenv loading, model loading, or Planner requests.
Uses only the adjacent conservative math evaluator. Run --help for the CLI.
"""

import argparse
import json
import math
import time
import hashlib
from pathlib import Path
from urllib.parse import urlparse

from math_answer_eval import evaluate_answer


LEVEL_WORDS = {1: 200, 2: 250, 3: 350, 4: 450, 5: 600}


def allocate_words(credits, total, decay):
    """Original credit × schedule allocation; leftovers favor largest weights."""
    import numpy as np

    n = len(credits)
    if not n or type(total) is not int or total <= 0:
        raise ValueError("Nonempty credits and positive integer budget required")
    if any(not isinstance(c, (int, float)) or isinstance(c, bool) or not math.isfinite(c) or c <= 0 for c in credits):
        raise ValueError("Credits must be finite and positive")
    schedules = {
        "constant": lambda i: 1.0,
        "linear": lambda i: n - i,
        "polynomial": lambda i: (n - i) ** 2,
        "exponential": lambda i: 0.9 ** i,
        "cosine": lambda i: (
            0.5 * (1 + np.cos(np.pi * i / (n - 1))) + 0.1
            if n > 1 else 1.0
        ),
    }
    schedule_weights = np.array([schedules[decay](i) for i in range(n)], dtype=float)
    weights = np.array(credits) * schedule_weights
    weights /= weights.sum()
    budgets = np.floor(weights * total).astype(int)
    # Retain NumPy arithmetic and tie ordering from the original implementation.
    for i in np.argsort(-weights)[:total - budgets.sum()]:
        budgets[i] += 1
    return budgets.tolist()


def messages_for(item, method, budgets, instruction, output_format):
    reference = item.get("reference") or "None."
    if method == "vanilla":
        system = (
            f"{instruction}\n\n"
            "Please reason step by step, and conclude your answer in the following format:\n\n"
            f"{output_format}\n"
        )
        user = f"Question: {item['query']}\nReference: {reference}\nOutput: <think>\n"
    else:
        system = (
            f"{instruction}\n\n"
            "The problem is given by an overall description, difficulty level out of 5, "
            "followed by a series of sub-questions as a hint.\n"
            "All the credit is given when you provide a correct final answer for the overall problem.\n"
            "Please solve the question efficiently and clearly to achieve as much credit as possible."
        )
        decomposed = "\n\n".join(
            f"{step} Please only think a little, and directly solve it using up to {budget} words."
            for step, budget in zip(item["steps"], budgets)
        )
        user = (
            "Let's start the exam. You are being given this math problem:\n"
            f"**Problem (100pt):** {item['query']}\n"
            f"**Reference:** {reference}\n**Level:** {item['level']} out of 5\n\n"
            "You may think following these sub-questions or feel free to use other methods "
            "that works the best towards getting the final answer:\n"
            f"{decomposed}\n\nPlease provide your final answer strictly following the format:\n"
            f"{output_format}\n\nOutput: <think>\n"
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]



from experiment_state import (
    SCHEMA_VERSION, original_definitions, read_manifest, digest, file_hash, versions,
    lock, atomic_save, load_json, load_checkpoint, stage, run_stage, common_arguments,
    check_arguments, termination_handler, StopRequested, ServerDead, attempt_cost,
)
from math_answer_eval import evaluator_info


def method_order(index, order):
    methods = ["vanilla", "weighted_pab"]
    return methods[::-1] if order == "pab-first" or (order == "alternate" and index % 2) else methods


def cost(records):
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "latency_seconds"):
        values = [r.get(key) for r in records]
        result[key] = round(sum(values), 6) if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values) else None
    result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"] if result["prompt_tokens"] is not None and result["completion_tokens"] is not None else None
    return result


def cost_summary(records, denominator):
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds"):
        values = [cost([r])[key] for r in records]
        known = [v for v in values if v is not None]
        result[key] = {"known_sum": round(sum(known), 6), "missing_records": len(values) - len(known),
                       "mean": round(sum(known) / denominator, 6) if len(values) == len(known) and denominator else None}
    return result


def planner_rows(source, manifest):
    if source.get("config", {}).get("manifest_fingerprint") != manifest["fingerprint"]:
        raise ValueError("Planner checkpoint must use the same frozen manifest")
    records = {r["id"]: r for r in source["stats"]}
    if len(records) != 500:
        raise ValueError("Planner checkpoint must preserve all 500 ids, including pending/failed")
    for original in manifest["records"]:
        row = records.get(original["id"])
        if not row or any(row.get(k) != original[k] for k in ("query", "answer", "level", "record_fingerprint")):
            raise ValueError("Planner/source record mapping mismatch")
    return records


def attach_plans(document, plans):
    import copy
    for row in document["comparisons"]:
        plan = plans[row["id"]]
        if plan["status"] == "success":
            signature = digest({k: plan[k] for k in ("steps", "credits", "decomposition", "assessment")})
            if row["methods"]["weighted_pab"]["status"] == "success" and row.get("plan_fingerprint") != signature:
                raise ValueError("Successful PAB result belongs to a different plan")
            row.update(steps=plan["steps"], credits=plan["credits"], plan_fingerprint=signature)
            row["words_per_step"] = allocate_words(plan["credits"], LEVEL_WORDS[row["level"]], document["config"]["decay"])
        row["planner"] = {k: copy.deepcopy(plan.get(k)) for k in ("decomposition", "assessment", "planner_latency_seconds", "stages", "status")}


def refresh_solver(document):
    for row in document["comparisons"]:
        for method, target in row["methods"].items():
            if target["status"] == "success":
                result = target["result"]
                finish = result["finish_reason"]
                row[method] = {"status": "success", "content": result["content"], "reasoning_content": result["reasoning_content"],
                               **result["usage"], "finish_reason": finish,
                               "completion_status": "complete" if finish == "stop" else "truncated" if finish == "length" else "other_finish",
                               "evaluation": target.get("evaluation", {"status": "undetermined", "reason": "evaluation_pending"})}
                text = (result.get("reasoning_content") or "") + " " + (result.get("content") or "")
                row[method]["budget_observation"] = {
                    "allocated_words": row["words_budget"] if method == "weighted_pab" else None,
                    "generated_whitespace_words": len(text.split()),
                    "generated_completion_tokens": result["usage"].get("completion_tokens"),
                    "per_step_usage_observable": False,
                    "note": "Words are approximate whitespace counts over all returned text. API tokens are a different unit; per-step compliance is unobservable."}
            else:
                row[method] = {"status": target["status"], "evaluation": {"status": "undetermined", "reason": "generation_incomplete"}}
        planner_usage = [row["planner"].get(k) or {} for k in ("decomposition", "assessment")]
        row["costs"] = {"planner": cost(planner_usage),
                        "vanilla_total": cost([row["vanilla"]]),
                        "pab_solver": cost([row["weighted_pab"]]),
                        "pab_total": cost(planner_usage + [row["weighted_pab"]])}
    document["summary"] = summarize(document["comparisons"])


def summarize(rows):
    result = {"expected_questions": 500, "methods": {}}
    for method in ("vanilla", "weighted_pab"):
        generated = [r for r in rows if r["methods"][method]["status"] == "success"]
        completed = [r for r in generated if r[method].get("finish_reason") == "stop"]
        correct = sum(r[method]["evaluation"]["status"] == "correct" for r in rows)
        completed_correct = sum(r[method]["evaluation"]["status"] == "correct" for r in completed)
        result["methods"][method] = {
            "api_success": len(generated), "normal_completion": len(completed), "correct": correct,
            "incorrect": sum(r[method]["evaluation"]["status"] == "incorrect" for r in rows),
            "undetermined": sum(r[method]["evaluation"]["status"] == "undetermined" for r in rows),
            "accuracy_all_500": correct / 500,
            "accuracy_api_success_subset": correct / len(generated) if generated else None,
            "accuracy_completed_subset": completed_correct / len(completed) if completed else None,
            "truncated": sum(r[method].get("finish_reason") == "length" for r in generated),
            "successful_solver_cost": cost_summary([r[method] for r in generated], len(generated)),
            "all_attempt_cost": attempt_cost([r["methods"][method] for r in rows])}
    paired = [r for r in rows if all(r["methods"][m]["status"] == "success" for m in ("vanilla", "weighted_pab"))]
    planner_stages = [s for r in rows for s in (r["planner"].get("stages") or {}).values()]
    result["planner_all_attempt_cost"] = attempt_cost(planner_stages)
    result["pab_all_attempt_cost_including_planner"] = attempt_cost(planner_stages + [r["methods"]["weighted_pab"] for r in rows])
    result["paired_subset"] = {"count": len(paired), "ids": [r["id"] for r in paired],
        "vanilla": cost_summary([r["vanilla"] for r in paired], len(paired)),
        "pab_solver": cost_summary([r["weighted_pab"] for r in paired], len(paired)),
        "pab_including_planner": cost_summary([r["costs"]["pab_total"] for r in paired], len(paired))}
    delta = {}
    for name in ("pab_solver", "pab_including_planner"):
        delta[name] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds"):
            left = result["paired_subset"]["vanilla"][key]
            right = result["paired_subset"][name][key]
            complete = not left["missing_records"] and not right["missing_records"] and bool(paired)
            difference = round(right["known_sum"] - left["known_sum"], 6) if complete else None
            delta[name][key] = {"difference": difference,
                               "mean_difference": round(difference / len(paired), 6) if difference is not None else None,
                               "percent_change": round((right["known_sum"] / left["known_sum"] - 1) * 100, 6)
                               if complete and left["known_sum"] else None}
    result["paired_subset"]["pab_minus_vanilla"] = delta
    result["cost_note"] = "Known attempt sums expose missing usage; not monetary cost. Historical Planner time + current Solver time excludes startup/model switching. Accuracy denominator remains 500 even when Planner fails."
    return result


def run_comparison(args, document, client, save, evaluator, tokenizer=None):
    for row in document["comparisons"][:args.limit]:
        row["call_order"] = method_order(row["source_index"], args.order)
        for method in row["call_order"]:
            target = row["methods"][method]
            if target["status"] != "success":
                if method == "weighted_pab" and row["planner"].get("status") != "success":
                    target["status"] = "blocked_planner"
                    save()
                    continue
                budgets = row.get("words_per_step", [])
                messages = messages_for(row, method, budgets, document["config"]["instruction"], document["config"]["output_format"])
                if tokenizer is not None:
                    count = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
                    if count + args.max_tokens > args.max_model_len:
                        target.update(status="error", validation_error={"type": "ContextLimit", "prompt_tokens": count})
                        save()
                        continue
                payload = {"model": args.model, "messages": messages, "max_tokens": args.max_tokens,
                           "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed}
                run_stage(target, payload, client, save, lambda response: {}, args.retries, args.retry_wait, args.server_pid)
            if target["status"] == "success" and target.get("evaluation_fingerprint") != evaluator["fingerprint"]:
                evaluate_method(row, method, evaluator, args.evaluator)
                save()
        print(row["id"], row["vanilla"]["status"], row["weighted_pab"]["status"], flush=True)
    return all(r["methods"][m]["status"] == "success" and r[m]["evaluation"]["status"] in ("correct", "incorrect")
               for r in document["comparisons"][:args.limit] for m in ("vanilla", "weighted_pab"))


def evaluate_method(row, method, evaluator, mode):
    target = row["methods"][method]
    response = target["result"]
    evaluated = evaluate_answer(response["content"], row["answer"], response["finish_reason"], "success", mode=mode)
    target["evaluation"] = evaluated
    target["evaluation_fingerprint"] = evaluator["fingerprint"]
    target.setdefault("evaluations", []).append({"evaluator": evaluator, "result": evaluated})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument("--planner-input", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--order", choices=["alternate", "vanilla-first", "pab-first"], default="alternate")
    parser.add_argument("--decay", choices=["constant", "linear", "polynomial", "exponential", "cosine"], default="constant")
    parser.add_argument("--evaluator", choices=["original", "conservative"], default="original")
    parser.add_argument("--reevaluate-only", action="store_true")
    parser.add_argument("--tokenizer-path")
    args = parser.parse_args()
    check_arguments(args)
    if args.max_tokens <= 0 or args.max_model_len <= args.max_tokens or not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2 or not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        parser.error("Invalid generation/context settings")
    if args.reevaluate_only and not args.resume:
        parser.error("Reevaluation requires --resume")
    manifest = read_manifest(args.manifest)
    source = load_json(args.planner_input)
    plans = planner_rows(source, manifest)
    _, instruction, output_format = original_definitions()
    config = {"kind": "solver", "manifest_fingerprint": manifest["fingerprint"],
              "planner_config_fingerprint": digest(source["config"]),
              "model": args.model, "revision": args.revision, "api_base": args.api_base,
              "instruction": instruction, "output_format": output_format, "level_words": LEVEL_WORDS,
              "decay": args.decay, "position_decay_applied": args.decay != "constant", "order": args.order,
              "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
              "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
              "timeout": args.timeout, "retries": args.retries, "retry_wait": args.retry_wait,
              "code_sha256": file_hash(__file__), "state_code_sha256": file_hash(Path(__file__).with_name("experiment_state.py")),
              "generation_packages": {k: v for k, v in versions().items() if k not in ("math-verify", "sympy")},
              "server_settings": {"dtype": "float16", "quantization": None, "tensor_parallel_size": 4,
                                  "max_num_seqs": 1, "gpu_memory_utilization": 0.85},
              "launch_script_sha256": file_hash(Path(__file__).with_name("solver_compare_y4.sbatch"))}
    evaluator = evaluator_info(args.evaluator)
    with lock(args.output):
        document = load_checkpoint(args.output, config, args.resume, recover=not args.check_only)
        if document is None:
            document = {"schema_version": SCHEMA_VERSION, "config": config, "solver_model": args.model,
                        "planner_model": source["config"]["model"], "comparisons": [
                            {**r, "words_budget": LEVEL_WORDS[r["level"]],
                             "methods": {"vanilla": stage(), "weighted_pab": stage()}}
                            for r in manifest["records"]]}
        attach_plans(document, plans)
        document["evaluator"] = evaluator
        if args.check_only:
            print("Solver configuration/checkpoint/evaluator validated")
            return 0
        def save():
            refresh_solver(document)
            atomic_save(args.output, document)
        save()
        with termination_handler():
            try:
                if args.reevaluate_only:
                    for row in document["comparisons"]:
                        for method, target in row["methods"].items():
                            if target["status"] == "success":
                                evaluate_method(row, method, evaluator, args.evaluator)
                                save()
                    return 0
                tokenizer = None
                if args.tokenizer_path:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
                from openai import OpenAI
                with OpenAI(base_url=args.api_base, api_key="DUMMY", timeout=args.timeout, max_retries=0) as client:
                    complete = run_comparison(args, document, client, save, evaluator, tokenizer)
            except StopRequested as exc:
                save()
                return 128 + exc.signum
            except ServerDead as exc:
                document["last_stop"] = str(exc)
                save()
                return 3
        return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())

