import argparse
import asyncio
import json
import os
import re
import random
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


MODEL = "casperhansen/llama-3.3-70b-instruct-awq"

# Only this audited pre-recovery version may cross the code-hash boundary.
# All other settings are still checked by load_checkpoint; original config stays intact.
RECOVERY_BASELINE_CODE = "b52ba0e2769facfa786f72a515a76c61bf97c6dd73589f0941d61817a5374ddb"
RECOVERY_INSTRUCTION = (
    "The preceding decomposition draft exceeds the allowed number of sub-questions. "
    "Rewrite the decomposition of the ORIGINAL problem into 2 to 5 numbered sub-questions, "
    "with an absolute maximum of 5. Preserve the original problem's mathematical meaning, "
    "all constraints, and the coverage needed to solve it. Merge closely related tasks into "
    "higher-level sub-questions instead of dropping essential tasks. Keep a brief Hint for "
    "each sub-question. Do not solve the problem or provide its final answer. Output only "
    "the revised numbered list. Do not output a sixth sub-question."
)


def parse_steps(text):
    matches = re.findall(
        r"(?m)^\s*\d+\.\s+(.+?)(?=^\s*\d+\.\s+|\Z)",
        text,
        flags=re.DOTALL,
    )
    steps = [s.strip() for s in matches if s.strip()]
    if not 2 <= len(steps) <= 5:
        raise ValueError(
            f"Expected 2-5 subquestions, got {len(steps)}: {text[:500]}"
        )
    return [f"{i}. {s}" for i, s in enumerate(steps, 1)]


def parse_credits(text, n):
    data = json.loads(text)
    credits = []
    for i in range(1, n + 1):
        item = data.get(str(i))
        if not isinstance(item, dict):
            raise ValueError(f"Missing credit for step {i}")
        credit = item.get("credit")
        if type(credit) is not int or credit <= 0:
            raise ValueError(f"Invalid credit for step {i}: {credit}")
        credits.append(credit)

    if sum(credits) != 100:
        raise ValueError(f"Credits sum to {sum(credits)}, not 100")
    return credits


def credit_schema(n):
    """Explicit schema avoids vLLM 0.6.6's json_object -> xgrammar fallback."""
    if not 2 <= n <= 5:
        raise ValueError("Expected 2-5 steps for credit assessment")
    assessment_properties = {
        "reason": {"type": "string"},
        "evaluated_level": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
    }
    properties = {
        "problem": {
            "type": "object",
            "properties": dict(assessment_properties),
            "required": ["reason", "evaluated_level"],
            "additionalProperties": False,
        }
    }
    for i in range(1, n + 1):
        properties[str(i)] = {
            "type": "object",
            "properties": {**assessment_properties, "credit": {"type": "integer"}},
            "required": ["reason", "evaluated_level", "credit"],
            "additionalProperties": False,
        }
    # Positivity and sum=100 remain enforced by parse_credits.
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }



from experiment_state import (
    SCHEMA_VERSION, original_definitions, read_manifest, digest, file_hash, versions,
    lock, atomic_save, load_checkpoint, stage, run_stage, common_arguments,
    check_arguments, termination_handler, StopRequested, ServerDead, attempt_cost,
)


def overlong_decomposition(target):
    """Require a finished, genuinely overlong response, not an API/parse error."""
    if target.get("status") not in ("error", "response_received"):
        return False
    result = target.get("result") or {}
    if result.get("finish_reason") != "stop":
        return False
    try:
        parse_steps(result.get("content") or "")
    except ValueError as exc:
        match = re.match(r"Expected 2-5 subquestions, got (\d+):", str(exc))
        return bool(match and int(match.group(1)) > 5)
    return False


def recovery_policy():
    return {"name": "overlong_decomposition_repair", "version": 1,
            "trigger": "finished decomposition contains more than 5 numbered steps",
            "normal_prompt_unchanged": True, "instruction": RECOVERY_INSTRUCTION,
            "seed_and_generation_options_unchanged": True,
            "validation": "2-5 steps; positive integer credits summing to 100"}


def prepare_recovery(document, manifest):
    """Freeze eligible ids and original-history hashes; never modify successful rows."""
    policy = recovery_policy()
    metadata = document.get("overlong_recovery")
    if metadata is None:
        targets = [r for r in document["stats"] if r["status"] != "success"
                   and overlong_decomposition(r["stages"]["decomposition"])
                   and r["stages"]["assessment"]["status"] != "success"]
        metadata = {
            "policy": policy, "policy_fingerprint": digest(policy),
            "recovery_code_sha256": file_hash(__file__),
            "baseline_code_sha256": document["config"]["code_sha256"],
            "manifest_fingerprint": manifest["fingerprint"],
            "target_ids": [r["id"] for r in targets],
            "protected_successes": {r["id"]: digest(r) for r in document["stats"] if r["status"] == "success"},
            "baseline_attempt_cost": attempt_cost([s for r in document["stats"] for s in r["stages"].values()]),
            "original_failures": {},
        }
        for row in targets:
            target = row["stages"]["decomposition"]
            metadata["original_failures"][row["id"]] = {
                "content": target["result"]["content"],
                "validation_error": target.get("validation_error"),
                "request": target.get("request"),
                "stages": {name: {"attempt_count": len(s["attempts"]), "attempts_fingerprint": digest(s["attempts"])}
                           for name, s in row["stages"].items()},
            }
        document["overlong_recovery"] = metadata
    if (metadata["policy_fingerprint"] != digest(policy)
            or metadata["recovery_code_sha256"] != file_hash(__file__)
            or metadata["manifest_fingerprint"] != manifest["fingerprint"]):
        raise ValueError("Recovery policy/code/manifest mismatch")
    validate_recovery_history(document)
    return metadata


def validate_recovery_history(document):
    metadata = document.get("overlong_recovery")
    if metadata is None:
        return
    by_id = {r["id"]: r for r in document["stats"]}
    for identifier, fingerprint in metadata["protected_successes"].items():
        if identifier not in by_id or digest(by_id[identifier]) != fingerprint:
            raise ValueError(f"Protected successful record changed: {identifier}")
    for identifier in metadata["target_ids"]:
        row = by_id[identifier]
        for name, baseline in metadata["original_failures"][identifier]["stages"].items():
            original = row["stages"][name]["attempts"][:baseline["attempt_count"]]
            if digest(original) != baseline["attempts_fingerprint"]:
                raise ValueError(f"Original attempt history changed: {identifier}/{name}")


def load_planner_checkpoint(args, config, manifest):
    if not getattr(args, "recover_overlong", False):
        return load_checkpoint(args.output, config, args.resume, recover=not args.check_only)
    if not args.resume or args.limit != 500:
        raise ValueError("Overlong recovery requires --resume and --limit 500")
    try:
        document = load_checkpoint(args.output, config, True, recover=False)
    except ValueError:
        baseline_config = {**config, "code_sha256": RECOVERY_BASELINE_CODE}
        document = load_checkpoint(args.output, baseline_config, True, recover=False)
    # Do not rewrite the old config to pretend all 500 plans used the repair prompt.
    prepare_recovery(document, manifest)
    return document


def planner_config(args, manifest, prompts):
    return {"kind": "planner", "manifest_fingerprint": manifest["fingerprint"],
            "model": args.model, "revision": args.revision, "api_base": args.api_base,
            "seed": args.seed, "temperature": 0.1, "top_p": 0.9,
            "max_tokens": [1024, 2048], "max_model_len": args.max_model_len,
            "prompt_fingerprint": digest(prompts), "benchmark_ids": manifest["benchmark_ids"],
            "code_sha256": file_hash(__file__), "state_code_sha256": file_hash(Path(__file__).with_name("experiment_state.py")),
            "packages": versions(), "retries": args.retries, "retry_wait": args.retry_wait,
            "timeout": args.timeout, "guided_backend": "outlines",
            "server_settings": {"dtype": "float16", "quantization": "awq", "tensor_parallel_size": 4,
                                "max_num_seqs": 1, "gpu_memory_utilization": 0.85},
            "launch_script_sha256": file_hash(Path(__file__).with_name("planner_smoke_y4.sbatch"))}


def refresh_planner(document):
    for row in document["stats"]:
        stages = row["stages"]
        for key in ("decomposition", "assessment"):
            target = stages[key]
            if target["status"] == "success":
                row[key] = target["result"]["usage"]
                row["steps" if key == "decomposition" else "credits"] = target["parsed"]
        if all(s["status"] == "success" for s in stages.values()):
            row["status"] = "success"
            row["planner_latency_seconds"] = round(sum(row[k]["latency_seconds"] for k in stages), 6)
        else:
            row["status"] = "error" if any(s["status"] in ("error", "interrupted") for s in stages.values()) else "pending"
    document["summary"] = {
        "expected_questions": 500, "success": sum(r["status"] == "success" for r in document["stats"]),
        "attempt_cost": attempt_cost([s for r in document["stats"] for s in r["stages"].values()])}
    if "overlong_recovery" in document:
        metadata = document["overlong_recovery"]
        selected = [r for r in document["stats"] if r["id"] in metadata["target_ids"]]
        stages = [s for r in selected for s in r["stages"].values()]
        def tagged_cost(purpose):
            return attempt_cost([{"attempts": [a for a in s["attempts"] if a.get("purpose") == purpose]} for s in stages])
        document["summary"]["overlong_recovery"] = {
            "target_ids": metadata["target_ids"], "success": sum(r["status"] == "success" for r in selected),
            "decomposition_attempt_cost": tagged_cost("overlong_decomposition_recovery"),
            "assessment_attempt_cost": tagged_cost("post_recovery_assessment"),
            "additional_attempt_cost": attempt_cost([{"attempts": [a for a in s["attempts"] if a.get("purpose") in (
                "overlong_decomposition_recovery", "post_recovery_assessment")]} for s in stages]),
        }


def run_planner(args, manifest, document, client, save, tokenizer=None):
    prompts, _, _ = original_definitions()
    by_id = {r["id"]: r for r in manifest["records"]}
    benchmarks = "\n".join(f"\nLevel {by_id[i]['level']} Example:\nQ: {by_id[i]['query']}\nA: {by_id[i]['solution']}\n"
                           for i in manifest["benchmark_ids"])
    recovery = prepare_recovery(document, manifest) if getattr(args, "recover_overlong", False) else None
    selected = [r for r in document["stats"] if r["id"] in recovery["target_ids"]] if recovery else document["stats"][:args.limit]
    for row in selected:
        for name in ("decomposition", "assessment"):
            target = row["stages"][name]
            if target["status"] == "success":
                continue
            if name == "assessment" and row["stages"]["decomposition"]["status"] != "success":
                break
            template = prompts["break_down" if name == "decomposition" else "evaluate_question_level_batch"]
            values = {"domain": "math", "problem": row["query"], "level": row["level"],
                      "benchmarks": benchmarks, "steps": row["stages"]["decomposition"].get("parsed", [])}
            messages = [{"role": role, "content": template[key].format(**values)}
                        for role, key in (("system", "system"), ("user", "user"))]
            if recovery and name == "decomposition":
                messages += [
                    {"role": "assistant", "content": recovery["original_failures"][row["id"]]["content"]},
                    {"role": "user", "content": RECOVERY_INSTRUCTION},
                ]
            cap = 1024 if name == "decomposition" else 2048
            if tokenizer is not None:
                count = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
                if count + cap > args.max_model_len:
                    target.update(status="error", validation_error={"type": "ContextLimit", "prompt_tokens": count, "max_tokens": cap})
                    save()
                    break
            payload = {"model": args.model, "messages": messages, "max_tokens": cap,
                       "temperature": 0.1, "top_p": 0.9, "seed": args.seed}
            if name == "assessment":
                payload["extra_body"] = {"guided_json": credit_schema(len(values["steps"])), "guided_decoding_backend": "outlines"}
            def validate(result):
                if result["finish_reason"] != "stop":
                    raise ValueError("Planner response truncated or abnormal finish")
                text = result["content"] or ""
                return parse_steps(text) if name == "decomposition" else parse_credits(text, len(values["steps"]))
            def save_call():
                if recovery:
                    baseline = recovery["original_failures"][row["id"]]["stages"][name]["attempt_count"]
                    purpose = "overlong_decomposition_recovery" if name == "decomposition" else "post_recovery_assessment"
                    for attempt in target["attempts"][baseline:]:
                        attempt["purpose"] = purpose
                        attempt["recovery_policy_fingerprint"] = recovery["policy_fingerprint"]
                    if target["status"] == "success":
                        # Original failure details remain in immutable recovery metadata.
                        target.pop("validation_error", None)
                save()
            run_stage(target, payload, client, save_call, validate, args.retries, args.retry_wait, args.server_pid)
        save()
        print(row["id"], row["status"], flush=True)
    return all(r["status"] == "success" for r in document["stats"][:args.limit])


def main():
    parser = argparse.ArgumentParser()
    common_arguments(parser)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--recover-overlong", action="store_true",
                        default=os.getenv("PLANNER_RECOVER_OVERLONG", "0") == "1",
                        help="Explicit resume-only repair of overlong decompositions; preserves successful rows")
    args = parser.parse_args()
    check_arguments(args)
    manifest = read_manifest(args.manifest)
    prompts, _, _ = original_definitions()
    config = planner_config(args, manifest, prompts)
    from openai import OpenAI
    with lock(args.output):
        document = load_planner_checkpoint(args, config, manifest)
        if args.check_only:
            print("Planner configuration/checkpoint validated", document.get("overlong_recovery", {}).get("target_ids", []) if document else [])
            return 0
        if document is None:
            document = {"schema_version": SCHEMA_VERSION, "config": config, "model": args.model,
                        "stats": [{**r, "status": "pending", "stages": {"decomposition": stage(), "assessment": stage()}}
                                  for r in manifest["records"]]}
        def save():
            validate_recovery_history(document)
            refresh_planner(document)
            validate_recovery_history(document)
            atomic_save(args.output, document)
        save()
        tokenizer = None
        if args.tokenizer_path:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
        with termination_handler():
            try:
                with OpenAI(base_url=args.api_base, api_key="DUMMY", timeout=args.timeout, max_retries=0) as client:
                    complete = run_planner(args, manifest, document, client, save, tokenizer)
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
