"""Read-only CPU analysis of frozen MATH-500 artifacts; write reports only.

Run with the vllm_server Python and -B. No OpenAI client or model is imported.
--reuse-evaluation uses only a hash-matched analysis cache, never changes checkpoints.
"""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics as stats

import numpy as np
from math_answer_eval import evaluate_answer, evaluator_info, extract_boxed, final_answer_text

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/llama_qwq"
METHODS = ("vanilla", "weighted_pab")
METRICS = ("prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds")
JOBS = ("153079", "153165", "153166", "153238")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def summary(values):
    return {"n": len(values), "sum": sum(values), "mean": stats.mean(values),
            "median": stats.median(values), "min": min(values), "max": max(values),
            "population_std": stats.pstdev(values)} if values else {}


def attempt_usage(target):
    attempts = target["attempts"]
    result = {"calls": len(attempts)}
    for key in ("prompt_tokens", "completion_tokens", "latency_seconds"):
        values = [a.get("usage", {}).get(key) for a in attempts]
        valid = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        result[key] = sum(valid)
        result[key + "_missing"] = len(values) - len(valid)
    result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    result["total_tokens_missing"] = sum(any(a.get("usage", {}).get(k) is None for k in ("prompt_tokens", "completion_tokens")) for a in attempts)
    return result


def combined_usage(targets):
    return attempt_usage({"attempts": [a for t in targets for a in t["attempts"]]})


def pct(new, old):
    return (new / old - 1) * 100 if old else None


def table(headers, rows):
    clean = lambda x: str(x).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(clean, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(clean, row)) + " |" for row in rows])


def number(x, decimals=3):
    return f"{x:,.{decimals}f}" if isinstance(x, float) else f"{x:,}" if isinstance(x, int) else str(x)


def corr(a, b, ranked=False):
    if len(a) < 3 or not np.std(a) or not np.std(b):
        return None
    if ranked:
        def ranks(values):
            ordered = sorted(set(values))
            rank = {}
            position = 1
            for value in ordered:
                count = values.count(value)
                rank[value] = position + (count - 1) / 2
                position += count
            return [rank[x] for x in values]
        a, b = ranks(a), ranks(b)
    return float(np.corrcoef(a, b)[0, 1])


def log_facts(job, role):
    prefix = "llama_planner" if role == "planner" else "qwq_compare"
    client_path = ROOT / "logs" / f"{prefix}_{job}.log"
    server_path = ROOT / "logs" / f"{'llama' if role == 'planner' else 'qwq'}_server_{job}.log"
    client = client_path.read_text().splitlines() if client_path.exists() else []
    server = server_path.read_text().splitlines() if server_path.exists() else []
    status_rows = [line.split() for line in client if re.fullmatch(r"\S+ (?:success|error|pending)(?: (?:success|error|pending|blocked_planner))?", line)]
    requests = {}
    for line in server:
        match = re.search(r"Received request (chatcmpl-[\w-]+):", line)
        if match:
            requests[match.group(1)] = line[:30]
    stamped = []
    for line in server:
        match = re.search(r"(?:INFO|WARNING|ERROR) (\d\d-\d\d \d\d:\d\d:\d\d)", line)
        if match:
            stamped.append((dt.datetime.strptime("2026-" + match.group(1), "%Y-%m-%d %H:%M:%S"), line))
    def time_for(text):
        matches = [time for time, line in stamped if text in line]
        return min(matches) if matches else None
    loading = time_for("Starting to load model")
    loaded = time_for("Loading model weights took")
    engine = time_for("init engine (profile")
    first_request = time_for("Received request")
    return {"job": job, "client_log": str(client_path.relative_to(ROOT)),
            "server_log": str(server_path.relative_to(ROOT)), "status_rows": status_rows,
            "request_ids": list(requests), "requests": len(requests),
            "http_chat_200": sum('POST /v1/chat/completions' in l and '200 OK' in l for l in server),
            "error_lines": [l[:1000] for l in server if ('ERROR' in l or 'CRITICAL' in l) and 'Received request' not in l],
            "warning_lines": list(dict.fromkeys(re.sub(r'\x1b\[[0-9;]*m', '', l) for l in server if ('WARNING' in l or 'Warning:' in l) and 'Received request' not in l)),
            "cancel_lines": [l for l in client if 'CANCELLED' in l or 'Traceback' in l],
            "load_start": str(loading) if loading else None, "load_end": str(loaded) if loaded else None,
            "parallel_weight_loading_seconds": (loaded - loading).total_seconds() if loaded and loading else None,
            "engine_ready": str(engine) if engine else None, "first_request": str(first_request) if first_request else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-evaluation", action="store_true")
    args = parser.parse_args()
    out = BASE / "reports"
    out.mkdir(exist_ok=True)
    paths = {"planner": BASE / "runs/math500_original_v1/planner.json",
             "solver": BASE / "runs/math500_original_v1/solver.json",
             "manifest": BASE / "math500_manifest.json", "evaluator_code": BASE / "math_answer_eval.py",
             "original_parser": ROOT / "utils/qwen_math_parser.py"}
    for job in JOBS:
        role = "planner" if job in JOBS[:2] else "solver"
        for prefix in (("llama_planner", "llama_server") if role == "planner" else ("qwq_compare", "qwq_server")):
            path = ROOT / "logs" / f"{prefix}_{job}.log"
            if path.exists():
                paths[f"log_{prefix}_{job}"] = path
    hashes = {name: sha(path) for name, path in paths.items()}
    planner_doc, solver_doc, manifest = (load(paths[k]) for k in ("planner", "solver", "manifest"))
    logs = {j: log_facts(j, "planner" if j in JOBS[:2] else "solver") for j in JOBS}
    checks, issues = [], []
    def check(label, condition):
        checks.append({"check": label, "passed": bool(condition)})
        if not condition:
            issues.append(label)
    check("manifest fingerprint", manifest["fingerprint"] == digest({k: v for k, v in manifest.items() if k != "fingerprint"}))
    check("manifest record fingerprints", all(r["record_fingerprint"] == digest({k: v for k, v in r.items() if k != "record_fingerprint"}) for r in manifest["records"]))
    datasets = {"manifest": manifest["records"], "planner": planner_doc["stats"], "solver": solver_doc["comparisons"]}
    reference_ids = [r["id"] for r in datasets["manifest"]]
    for name, data in datasets.items():
        check(name + " 500 unique ordered ids", len(data) == len(set(r["id"] for r in data)) == 500 and [r["id"] for r in data] == reference_ids)
        if name != "manifest":
            check(name + " original record mapping", all(all(row.get(k) == v for k, v in original.items()) for row, original in zip(data, datasets["manifest"])))
            check(name + " manifest config", (planner_doc if name == "planner" else solver_doc)["config"]["manifest_fingerprint"] == manifest["fingerprint"])
    check("solver planner config fingerprint", solver_doc["config"]["planner_config_fingerprint"] == digest(planner_doc["config"]))
    pmap = {r["id"]: r for r in datasets["planner"]}
    check("Planner 500 success", all(r["status"] == "success" and all(s["status"] == "success" for s in r["stages"].values()) for r in pmap.values()))
    check("Planner steps/credits constraints", all(2 <= len(r["steps"]) <= 5 and len(r["credits"]) == len(r["steps"]) and all(type(c) is int and c > 0 for c in r["credits"]) and sum(r["credits"]) == 100 for r in pmap.values()))
    recovery = planner_doc["overlong_recovery"]
    check("494 protected Planner records unchanged", len(recovery["protected_successes"]) == 494 and all(digest(pmap[k]) == v for k, v in recovery["protected_successes"].items()))
    check("6 original failure histories preserved", all(digest(pmap[identifier]["stages"][name]["attempts"][:old["attempt_count"]]) == old["attempts_fingerprint"] for identifier, original in recovery["original_failures"].items() for name, old in original["stages"].items()))
    for method in METHODS:
        check(method + " 500 API results", all(r["methods"][method]["status"] == "success" for r in datasets["solver"]))
    request_jobs = {request: job for job, facts in logs.items() for request in facts["request_ids"]}
    all_ids = []
    raw_errors = []
    for row in datasets["solver"]:
        plan = pmap[row["id"]]
        if any(row.get(k) != plan[k] for k in ("steps", "credits")) or row["planner"]["stages"] != plan["stages"]:
            raw_errors.append((row["id"], "plan"))
        expected_plan = digest({k: plan[k] for k in ("steps", "credits", "decomposition", "assessment")})
        if row.get("plan_fingerprint") != expected_plan:
            raw_errors.append((row["id"], "plan_fingerprint"))
        # Check allocation against configured, unchanged original constant-credit rule.
        credits = np.array(row["credits"], dtype=float)
        weights = credits / credits.sum()
        allocation = np.floor(weights * row["words_budget"]).astype(int)
        for i in np.argsort(-weights)[:row["words_budget"] - allocation.sum()]:
            allocation[i] += 1
        if allocation.tolist() != row["words_per_step"]:
            raw_errors.append((row["id"], "allocation"))
        for method in METHODS:
            target = row["methods"][method]
            result = target["result"]
            request = target["request"]
            for key in ("model", "max_tokens", "temperature", "top_p", "seed"):
                if request.get(key) != solver_doc["config"].get(key):
                    raw_errors.append((row["id"], method, "generation_setting", key))
            for key in ("prompt_tokens", "completion_tokens"):
                if result["usage"].get(key) != row[method].get(key):
                    raw_errors.append((row["id"], method, "usage_projection"))
            if any(result.get(k) != row[method].get(k) for k in ("content", "reasoning_content", "finish_reason")):
                raw_errors.append((row["id"], method, "response_projection"))
            if target["evaluation"] != row[method]["evaluation"]:
                raw_errors.append((row["id"], method, "evaluation_projection"))
            for attempt in target["attempts"]:
                if "raw_response" in attempt:
                    raw = attempt["raw_response"]
                    all_ids.append(raw["id"])
                    for key in ("prompt_tokens", "completion_tokens"):
                        if attempt["usage"].get(key) != (raw.get("usage") or {}).get(key):
                            raw_errors.append((row["id"], method, "attempt_usage"))
            successful_attempts = [a for a in target["attempts"] if a["status"] == "success"]
            if len(successful_attempts) != 1 or successful_attempts[0].get("raw_response") != result.get("raw_response"):
                raw_errors.append((row["id"], method, "result_attempt_identity"))
            raw_message = result["raw_response"]["choices"][0]["message"]
            if any(raw_message.get(k) != result.get(k) for k in ("content", "reasoning_content")):
                raw_errors.append((row["id"], method, "raw_message"))
            if any(a["request_fingerprint"] != digest(request) for a in target["attempts"]):
                raw_errors.append((row["id"], method, "request_fingerprint"))
    check("Solver plan/request/response/usage/evaluation consistency", not raw_errors)
    check("1000 unique successful response ids", len(all_ids) == len(set(all_ids)) == 1000)
    check("all Solver successful ids traceable to server logs", all(i in request_jobs for i in all_ids))
    check("constant allocation configuration", solver_doc["config"]["decay"] == "constant" and solver_doc["config"]["position_decay_applied"] is False)
    info = evaluator_info("original")
    check("same existing evaluator identity and dependencies", info == solver_doc["evaluator"])
    cache_key = digest({k: hashes[k] for k in ("planner", "solver", "manifest", "evaluator_code", "original_parser")})
    cache_path = out / "math500_analysis_metrics.json"
    old = load(cache_path) if args.reuse_evaluation and cache_path.exists() else {}
    if old.get("evaluation_cache_key") == cache_key and old.get("evaluator") == info:
        evaluated = old["reevaluated"]
        print("Using hash-matched CPU evaluation cache", flush=True)
    else:
        evaluated = {}
        for index, row in enumerate(datasets["solver"]):
            evaluated[row["id"]] = {}
            for method in METHODS:
                response = row["methods"][method]["result"]
                evaluated[row["id"]][method] = evaluate_answer(response["content"], row["answer"], response["finish_reason"], "success", mode="original")
            if (index + 1) % 25 == 0:
                print(f"CPU reevaluation {index + 1}/500", flush=True)
    mismatches = []
    records = []
    for row in datasets["solver"]:
        plan = pmap[row["id"]]
        plan_cost = combined_usage(list(plan["stages"].values()))
        assessment = json.loads(plan["stages"]["assessment"]["result"]["content"])
        credits = row["credits"]
        entropy = -sum(c / 100 * math.log(c / 100) for c in credits) / math.log(len(credits))
        record = {"id": row["id"], "source_index": row["source_index"], "level": row["level"],
                  "query": row["query"], "answer": row["answer"], "subquestions": len(row["steps"]),
                  "credits": credits, "credit_min": min(credits), "credit_max": max(credits),
                  "credit_normalized_entropy": entropy,
                  "planner_evaluated_level": assessment["problem"]["evaluated_level"],
                  "step_evaluated_levels": [assessment[str(i)]["evaluated_level"] for i in range(1, len(credits) + 1)],
                  "words_budget": row["words_budget"], "words_per_step": row["words_per_step"],
                  "recovered": row["id"] in recovery["target_ids"], "call_order": row["call_order"]}
        for key, value in plan_cost.items():
            record["planner_" + key] = value
        for method, short in (("vanilla", "v"), ("weighted_pab", "p")):
            target = row["methods"][method]
            response = target["result"]
            ev = evaluated[row["id"]][method]
            stored = target["evaluation"]
            if ev != stored:
                mismatches.append({"id": row["id"], "method": method, "stored": stored, "fresh": ev})
            record[short + "_evaluation"] = ev["status"]
            record[short + "_correct"] = ev["status"] == "correct"
            record[short + "_prediction"] = ev.get("prediction")
            record[short + "_eval_reason"] = ev.get("reason")
            record[short + "_finish"] = response["finish_reason"]
            record[short + "_has_final_boxed"] = extract_boxed(final_answer_text(response["content"])) is not None
            text = (response.get("reasoning_content") or "") + " " + (response.get("content") or "")
            record[short + "_words"] = len(text.split())
            record[short + "_reasoning_separate"] = response.get("reasoning_content") is not None
            raw_id = response["raw_response"]["id"]
            record[short + "_success_job"] = request_jobs.get(raw_id)
            usage = {**response["usage"]}
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            for key, value in usage.items():
                record[short + "_" + key] = value
            actual = attempt_usage(target)
            for key, value in actual.items():
                record[short + "_all_" + key] = value
        for key in METRICS:
            record["e2e_" + key] = record["planner_" + key] + record["p_all_" + key]
            record["e2e_" + key + "_missing"] = record.get("planner_" + key + "_missing", 0) + record.get("p_all_" + key + "_missing", 0)
            record["solver_delta_" + key] = record["p_" + key] - record["v_" + key]
            record["e2e_delta_" + key] = record["e2e_" + key] - record["v_all_" + key]
        record["p_words_minus_budget"] = record["p_words"] - record["words_budget"]
        record["p_words_budget_ratio"] = record["p_words"] / record["words_budget"]
        record["p_tokens_per_allocated_word"] = record["p_completion_tokens"] / record["words_budget"]
        record["accuracy_transition"] = ("both_correct" if record["v_correct"] and record["p_correct"] else
                                          "vanilla_only" if record["v_correct"] else "pab_only" if record["p_correct"] else "both_incorrect")
        records.append(record)
    check("CPU reevaluation exactly matches stored evaluations", not mismatches)
    check("Planner all-attempt summary matches recomputation", all(math.isclose(sum(r["planner_" + k] for r in records), planner_doc["summary"]["attempt_cost"][k]["known_sum"], abs_tol=1e-5) for k in METRICS))
    for method, short in (("vanilla", "v"), ("weighted_pab", "p")):
        check(method + " all-attempt summary matches", all(math.isclose(sum(r[short + "_all_" + k] for r in records), solver_doc["summary"]["methods"][method]["all_attempt_cost"][k]["known_sum"], abs_tol=1e-5) for k in METRICS))
    provenance = {}
    for method, short in (("vanilla", "v"), ("weighted_pab", "p")):
        provenance[method] = dict(collections.Counter(r[short + "_success_job"] for r in records))
    extra_attempts = [{"id": r["id"], "method": method, "attempts": r["methods"][method]["attempts"]}
                      for r in datasets["solver"] for method in METHODS if len(r["methods"][method]["attempts"]) != 1]
    transitions = dict(collections.Counter(r["accuracy_transition"] for r in records))
    distributions = {name: summary([r[name] for r in records]) for name in (
        [prefix + key for prefix in ("v_", "p_", "planner_", "e2e_") for key in METRICS] +
        ["v_words", "p_words", "p_words_budget_ratio", "p_tokens_per_allocated_word", "solver_delta_completion_tokens", "e2e_delta_total_tokens"])}
    b, c = transitions.get("vanilla_only", 0), transitions.get("pab_only", 0)
    n = b + c
    mcnemar = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)) if n else 1.0
    differences = np.array([int(r["p_correct"]) - int(r["v_correct"]) for r in records])
    rng = np.random.default_rng(42)
    bootstrap = [float(rng.choice(differences, size=500).mean() * 100) for _ in range(5000)]
    correlations = []
    for x in ("level", "subquestions", "credit_max", "credit_normalized_entropy", "words_budget", "planner_evaluated_level"):
        for y in ("solver_delta_completion_tokens", "p_completion_tokens", "p_words_budget_ratio"):
            a, b_values = [r[x] for r in records], [r[y] for r in records]
            correlations.append({"x": x, "y": y, "pearson": corr(a, b_values), "spearman": corr(a, b_values, True)})
    metrics = {"analysis_time": dt.datetime.now().astimezone().isoformat(), "input_hashes": hashes,
               "evaluation_cache_key": cache_key, "evaluator": info, "reevaluated": evaluated,
               "reevaluation_mismatches": mismatches, "integrity_checks": checks, "integrity_issues": issues,
               "raw_consistency_errors": raw_errors, "transitions": transitions, "distributions": distributions,
               "correlations": correlations, "response_job_provenance": provenance,
               "extra_solver_attempts": [{"id": x["id"], "method": x["method"], "attempts": [
                   {k: v for k, v in a.items() if k not in ("raw_response", "content", "reasoning_content")} for a in x["attempts"]]} for x in extra_attempts],
               "logs": logs, "mcnemar_exact_p": mcnemar,
               "accuracy_difference_bootstrap95_pp": np.quantile(bootstrap, [.025, .975]).tolist()}
    check("original input files unchanged after CPU analysis", all(sha(path) == hashes[name] for name, path in paths.items()))
    # The only writes are new analysis products; source hashes are checked again below.
    with (out / "math500_per_problem.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in record.items()})
    render_report(out, records, metrics, planner_doc, solver_doc, manifest)
    if not all(sha(path) == hashes[name] for name, path in paths.items()):
        raise RuntimeError("An original input changed during report generation")
    (out / "math500_analysis_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
    if issues:
        print("Integrity issues:", issues, flush=True)
    print("Reports written:", out, flush=True)


def render_report(out, rows, metrics, planner_doc, solver_doc, manifest):
    # Report construction below uses only recomputed per-problem quantities.
    sections = []
    add = sections.append
    N = len(rows)
    total = lambda field, subset=rows: sum(r[field] for r in subset)
    rate = lambda short, subset=rows: 100 * sum(r[short + "_correct"] for r in subset) / len(subset) if subset else 0
    change = lambda new, old: f"{pct(new, old):+.2f}%"
    add("# MATH-500 최종 분석: Llama 70B Planner를 사용하는 PAB 변형 실험\n")
    add(f"분석 시각: {metrics['analysis_time']}. 모든 계산은 기존 산출물을 읽는 CPU 작업이며 모델·GPU·새 LLM 호출은 수행하지 않았다.\n")
    add("## 1. 실험 환경 및 모델 구성\n")
    add(table(["항목", "설정"], [
        ["Planner", f"{planner_doc['config']['model']} / revision {planner_doc['config']['revision']}"],
        ["Solver", f"{solver_doc['config']['model']} / revision {solver_doc['config']['revision']}"],
        ["자원", "moana-y4, RTX 3090 × 4, TP=4, FP16; Planner AWQ, Solver 비양자화"],
        ["패키지", json.dumps(solver_doc['config']['generation_packages'])],
        ["생성", "max_tokens=8192, temperature=0.1, top_p=0.9, seed=42; 문제별 호출 순서 교대"],
        ["컨텍스트", "Planner 8192 / Solver 12288; max_num_seqs=1"],
        ["예산", "level 1~5 → 200/250/350/450/600 words; constant decay, 위치별 감쇠 없음"],
        ["구조", "정상 경로: Planner 분해 1회 + credit 평가 1회, Vanilla/PAB Solver 각각 1회"],
    ]))
    add("\n사용자가 제시한 원논문 Planner는 LLaMA-3.1-8B-Instruct이고, 본 실험은 Llama-3.3-70B-Instruct AWQ를 사용한다. 따라서 **원논문의 정확한 재현이 아닌 70B Planner PAB 변형 실험**이다. 논문 원문은 이번 분석에서 별도 검증하지 않았다. 8B 실행 결과가 없어 모델 크기 차이에 따른 실제 정확도·비용 차이는 단정할 수 없다.\n")
    add("## 2. 실험 완료 및 데이터 무결성 검증\n")
    labels_ko = {
        'manifest fingerprint':'Manifest 전체 fingerprint', 'manifest record fingerprints':'Manifest 문제별 fingerprint',
        'manifest 500 unique ordered ids':'Manifest 500개 ID·고유성·순서',
        'planner 500 unique ordered ids':'Planner 500개 ID·고유성·순서',
        'solver 500 unique ordered ids':'Solver 500개 ID·고유성·순서',
        'planner original record mapping':'Planner 원본 문제·정답·난이도 매핑',
        'solver original record mapping':'Solver 원본 문제·정답·난이도 매핑',
        'planner manifest config':'Planner Manifest 설정 일치', 'solver manifest config':'Solver Manifest 설정 일치',
        'solver planner config fingerprint':'Solver가 참조한 Planner 설정 fingerprint',
        'Planner 500 success':'Planner 500개 두 단계 성공', 'Planner steps/credits constraints':'Planner 단계 수·credit 검증',
        '494 protected Planner records unchanged':'기존 Planner 성공 494개 보존 해시',
        '6 original failure histories preserved':'Planner 최초 실패 6개 이력 보존',
        'vanilla 500 API results':'Vanilla API 결과 500개 보존', 'weighted_pab 500 API results':'PAB API 결과 500개 보존',
        'Solver plan/request/response/usage/evaluation consistency':'Solver 계획·요청·raw 응답·시도·usage·평가 정합성',
        '1000 unique successful response ids':'성공 응답 ID 1000개 고유성',
        'all Solver successful ids traceable to server logs':'성공 응답 ID와 서버 로그 매칭',
        'constant allocation configuration':'Constant 예산 배분 및 설정',
        'same existing evaluator identity and dependencies':'기존 평가기 코드·의존성 일치',
        'CPU reevaluation exactly matches stored evaluations':'CPU 재평가와 저장 판정 일치',
        'Planner all-attempt summary matches recomputation':'Planner 전체 시도 요약 재계산',
        'vanilla all-attempt summary matches':'Vanilla 전체 시도 요약 재계산',
        'weighted_pab all-attempt summary matches':'PAB 전체 시도 요약 재계산',
        'original input files unchanged after CPU analysis':'분석 전후 원본 파일 SHA-256 불변',
    }
    add(table(["검증", "결과"], [[labels_ko.get(c['check'],c['check']), "통과" if c['passed'] else "불일치"] for c in metrics['integrity_checks']]))
    add("\n완료는 API 응답의 checkpoint 보존을 뜻하며 정상 종료·정답을 뜻하지 않는다. Planner는 500개 모두 두 단계 성공, Solver는 방법별 500개 API 응답을 보존했다. ID·원본 순서·문제·정답·level·레코드 fingerprint 및 plans/credits/usage/raw 응답을 대조했다.\n")
    logrows = []
    for job in JOBS:
        facts = metrics['logs'][job]
        logrows.append([job, facts['requests'], facts['http_chat_200'], len(facts['status_rows']), len(facts['error_lines'])])
    add(table(["Job", "서버 접수 요청", "Chat HTTP 200", "클라이언트 문제 상태 행", "서버 ERROR/CRITICAL"], logrows))
    add("\n재개 로그의 문제 상태 행은 건너뛴 문제도 출력하므로 새 API 호출 수로 해석하지 않았다. raw 응답 ID와 서버 요청 ID를 연결한 성공 응답의 작업별 분포:\n")
    add(table(["방법", "Job별 현재 checkpoint 성공 응답 수"], [[m, json.dumps(metrics['response_job_provenance'][m])] for m in METHODS]))
    add("\nVanilla는 500회, PAB는 501회 시도했다. 성공 응답 ID는 1000개 모두 고유하고, 기존 성공을 다시 생성한 시도는 기록상 없다. 1개 PAB 중단 시도만 재개했다. Solver의 과거 checkpoint 스냅샷은 없어 모든 과거 바이트가 불변이었다는 절대 증명은 불가하지만, 보존된 응답 ID·시도 이력·로그는 정상 resume를 뒷받침한다. Planner 494개는 복구 전 저장된 해시로 불변을 확인했다.\n")
    add("추가 Solver 시도: `" + json.dumps(metrics['extra_solver_attempts'], ensure_ascii=False) + "`\n")
    add("## 3. Vanilla vs PAB Accuracy 비교\n")
    add(table(["방법", "정답", "오답", "판정 불가", "Accuracy(분모 500)", "stop", "length"], [
        [name, sum(r[s + '_correct'] for r in rows), sum(r[s + '_evaluation'] == 'incorrect' for r in rows), sum(r[s + '_evaluation'] == 'undetermined' for r in rows), f"{rate(s):.2f}%", sum(r[s + '_finish'] == 'stop' for r in rows), sum(r[s + '_finish'] == 'length' for r in rows)]
        for name, s in [('Vanilla', 'v'), ('PAB', 'p')]]))
    add(f"\nPAB−Vanilla = **{rate('p') - rate('v'):+.2f} percentage points**. 다음 비교는 같은 500개 문제의 쌍 비교다.\n")
    labels = {'both_correct': '둘 다 정답', 'vanilla_only': 'Vanilla만 정답', 'pab_only': 'PAB만 정답', 'both_incorrect': '둘 다 오답'}
    add(table(["정오 변화", "문제 수"], [[label, metrics['transitions'].get(key, 0)] for key, label in labels.items()]))
    add("\n기존 `math_answer_eval.py`의 original 모드를 수정 없이 1000응답에 CPU 재적용했다. 원본 Qwen `extract_answer`(최종 답 문구/boxed/답 문구/마지막 숫자 fallback) → `strip_string`(LaTeX·공백·단위 등 정규화) → `math_verify.parse/verify` 순서다. 단순 문자열 일치나 수동 채점으로 바꾸지 않았다. 파싱 실패·빈 추출·예외는 판정 불가이며 전체 정확도 분모에서 제외하지 않는다. finish_reason은 정오와 별개여서 잘린 응답도 이 기준으로 채점한다. 전체 content를 평가하므로 reasoning 중 숫자나 불완전 답에서 fallback이 작동할 수 있다.\n")
    add(f"재평가와 저장 평가의 불일치: **{len(metrics['reevaluation_mismatches'])}개**. evaluator fingerprint: `{metrics['evaluator']['fingerprint']}`.\n")
    add(table(["방법/종료", "문제 수", "정답", "해당 subset 정확도"], [[name + '/' + finish, len(sub), sum(r[s + '_correct'] for r in sub), f"{rate(s, sub):.2f}%"] for name, s in [('Vanilla','v'),('PAB','p')] for finish in ['stop','length'] for sub in [[r for r in rows if r[s + '_finish'] == finish]]]))
    add(f"\n탐색적 paired 검정: exact McNemar 양측 p={metrics['mcnemar_exact_p']:.4f}; 문제 단위 paired bootstrap 5000회(seed=42)의 정확도 차이 95% 구간={metrics['accuracy_difference_bootstrap95_pp'][0]:+.2f}~{metrics['accuracy_difference_bootstrap95_pp'][1]:+.2f} pp. 단일 seed·고정 데이터 결과이며 광범위한 일반화 또는 원인 증명이 아니다.\n")
    add("정확도 점추정은 PAB가 낮지만 이 paired 검정의 5% 기준에서 유의한 차이라고 단정할 수 없다. '동등함이 입증됐다'는 뜻도 아니다. 정상 종료 subset 정확도는 서로 다른 문제 집합을 조건으로 하므로 전체 500개 paired 정확도를 대체하지 않는다.\n")
    add("## 4. Vanilla vs PAB 토큰 사용량 비교\n")
    add("통계 단위는 문제 500개이며 표준편차는 모집단 표준편차(ddof=0)다. A/B는 최종 보존 성공 응답의 정확한 값이다. Planner/C는 실패·복구·중단 등 모든 시도를 포함한다. **†는 PAB 중단 호출 1개의 입력·출력 토큰이 미상인 기록값**이다. † 총합·평균은 실제 비용의 하한이며, 중앙값·최소/최대·표준편차는 알려진 값 벡터의 기술통계일 뿐 실제 비용 분포의 확정치가 아니다.\n")
    tokenrows = []
    for label, prefix, uncertain in [('A Vanilla Solver','v_',False),('B PAB Solver 성공응답','p_',False),('Planner 모든 시도','planner_',False),('C PAB Planner+Solver 모든 시도','e2e_',True)]:
        for key, klabel in [('prompt_tokens','입력'),('completion_tokens','생성'),('total_tokens','전체')]:
            values = metrics['distributions'][prefix + key]
            marker = '†' if uncertain else ''
            tokenrows.append([label,klabel]+[number(values[k],2)+marker for k in ['sum','mean','median','min','max','population_std']])
    add(table(['범위','토큰','총합','평균','중앙값','최소','최대','표준편차'],tokenrows))
    add("\nPAB Solver 모든 시도에서 알려진 토큰 합계는 성공응답 합계와 같지만, 중단 1회의 토큰은 미상이다. 이를 0토큰 호출로 단정하지 않았다.\n")
    add(table(['비교','입력 변화','생성 변화','전체 변화'],[[label]+[change(total(new+k),total('v_'+k)) for k in ['prompt_tokens','completion_tokens','total_tokens']] for label,new in [('PAB Solver 성공응답','p_'),('PAB 전체 알려진 시도 비용†','e2e_')]]))
    add("\n감소율=100×(Vanilla−PAB)/Vanilla, 변화율=100×(PAB/Vanilla−1). 총합의 비율을 사용하며 문제별 비율의 단순 평균은 아니다. †에서 Solver 절감률은 가능한 상한, 전체 비용 증가율은 하한이다. 서로 다른 Llama/QwQ tokenizer의 토큰 단순 합계는 FLOPs·GPU 시간·금전 비용과 동일하지 않다.\n")
    add("## 5. Vanilla vs PAB 추론 시간 비교\n")
    timerows=[]
    for label,field in [('Vanilla Solver','v_latency_seconds'),('PAB Solver 성공응답','p_latency_seconds'),('Planner 모든 시도','planner_latency_seconds'),('Planner+PAB 모든 시도','e2e_latency_seconds')]:
        values=metrics['distributions'][field]
        timerows.append([label]+[number(values[k],3) for k in ['sum','mean','median','min','max','population_std']])
    # Interrupted call has measured latency, even though token usage is unknown.
    timerows.insert(2,['PAB Solver 모든 시도']+[number(summary([r['p_all_latency_seconds'] for r in rows])[k],3) for k in ['sum','mean','median','min','max','population_std']])
    add(table(['범위(초)','합계','평균','중앙값','최소','최대','표준편차'],timerows))
    add(f"\n실패·중단을 포함한 PAB 전체 API latency 변화율은 **{change(total('e2e_latency_seconds'),total('v_all_latency_seconds'))}**다. API 시간은 클라이언트 요청부터 응답까지이며 서버 처리·대기·통신을 포함한다. Planner와 Solver는 별도 작업이므로 위 합은 합성 pipeline 비용이며 실제 한 번의 end-to-end wall-clock 측정이 아니다.\n")
    snapshot=out/'math500_slurm_snapshot.json'
    if snapshot.exists():
        metadata=load(snapshot)
        jobrows=[]
        for job in metadata['jobs']:
            submitted=dt.datetime.fromisoformat(job['submit']);start=dt.datetime.fromisoformat(job['start'])
            jobrows.append([job['id'],job['state'],job['exit_code'],job['start'],job['end'],job['elapsed'],(start-submitted).total_seconds()])
        add(table(['Job','State','ExitCode','시작','종료','Slurm 경과','대기(초)'],jobrows))
        add("\nSlurm 값은 읽기 전용 sacct 조회 스냅샷이다. 첫 Solver 작업 153166은 종료 신호로 실패했고 153238에서 재개했다. 이들 경과 시간을 단순 합해 추론 latency로 사용하지 않았다. 작업 사이 공백도 추론 시간에서 제외했다.\n")
    else:
        add('Slurm accounting 원본 스냅샷 없음: State/ExitCode·대기 시간은 확인 불가.\n')
    add(table(['Job','가중치 로딩 시작','로딩 종료','병렬 로딩 경과(초)','엔진 준비','첫 API 접수'],[[j,metrics['logs'][j]['load_start'],metrics['logs'][j]['load_end'],metrics['logs'][j]['parallel_weight_loading_seconds'],metrics['logs'][j]['engine_ready'],metrics['logs'][j]['first_request']] for j in JOBS]))
    add("\n로딩 경과는 로그의 병렬 worker 공통 구간이며 GPU별 시간을 합하지 않았다. 엔진 준비에는 profiling·KV cache·warmup 등이 포함된다. 서버 준비와 첫 요청 사이에는 polling·클라이언트 시작 등이 있어 순수 로딩으로 간주하지 않는다.\n")
    add("## 6. 난이도별 성능 분석\n")
    levelrows=[]
    for level in range(1,6):
        sub=[r for r in rows if r['level']==level]
        levelrows.append([level,len(sub),f"{rate('v',sub):.2f}%",f"{rate('p',sub):.2f}%",f"{rate('p',sub)-rate('v',sub):+.2f}",
            sum(r['accuracy_transition']=='vanilla_only' for r in sub),sum(r['accuracy_transition']=='pab_only' for r in sub),
            sum(r['accuracy_transition']=='both_correct' for r in sub),sum(r['accuracy_transition']=='both_incorrect' for r in sub),
            change(total('p_completion_tokens',sub),total('v_completion_tokens',sub)),change(total('p_latency_seconds',sub),total('v_latency_seconds',sub)),
            change(total('e2e_total_tokens',sub),total('v_all_total_tokens',sub)),change(total('e2e_latency_seconds',sub),total('v_all_latency_seconds',sub))])
    add(table(['Level','N','Vanilla Acc','PAB Acc','차이 pp','V만 정답','P만 정답','둘 다 정답','둘 다 오답','Solver 생성 변화','Solver 시간 변화','전체 토큰 변화†','전체 시간 변화'],levelrows))
    add("\n수준별 비용 변화는 그 수준의 총합 기준이다. 표시 난이도는 데이터셋 level이다. Planner가 다시 평가한 difficulty와 혼동하지 않았다. 난이도가 높을수록 PAB가 항상 유리하다고 단정하지 않으며, 각 수준의 표본 수·잘림·정확도 변화를 함께 해석해야 한다.\n")
    add("Level 5에서는 PAB가 1문제 더 맞혀 +0.75pp였지만, 생성 토큰 절감률은 16.93%로 다른 수준보다 작았다. Level 1~4의 정확도는 모두 낮아졌다. 따라서 고난도에서 효과가 일관되게 커진다고 주장할 근거는 없다.\n")
    add(table(['난이도','데이터셋 문제 수','Planner 재평가 문제 수','Planner 단계 평가 수'],[[level,sum(r['level']==level for r in rows),sum(r['planner_evaluated_level']==level for r in rows),sum(v==level for r in rows for v in r['step_evaluated_levels'])] for level in range(1,6)]))
    add("## 7. 문제별 결과 변화 분석\n")
    add(table(['개선/악화 기준','문제 수'],[[label,sum(condition(r) for r in rows)] for label,condition in [
        ('Solver 생성 토큰 감소',lambda r:r['p_completion_tokens']<r['v_completion_tokens']),
        ('Solver 총토큰 감소',lambda r:r['p_total_tokens']<r['v_total_tokens']),
        ('Solver 시간 감소',lambda r:r['p_latency_seconds']<r['v_latency_seconds']),
        ('Planner 포함 기록 총토큰 감소†',lambda r:r['e2e_total_tokens']<r['v_all_total_tokens']),
        ('Planner 포함 전체 시간 감소',lambda r:r['e2e_latency_seconds']<r['v_all_latency_seconds']),
        ('정확도 유지/개선 및 전체 시간 감소',lambda r:int(r['p_correct'])>=int(r['v_correct']) and r['e2e_latency_seconds']<r['v_all_latency_seconds']),
        ('PAB 전체 생성 words가 할당 합계 초과',lambda r:r['p_words']>r['words_budget'])]]))
    add(table(['단계 수','N','V Acc','P Acc','평균 V 생성','평균 P 생성','Solver 생성 변화'],[[n,len(sub),f"{rate('v',sub):.2f}%",f"{rate('p',sub):.2f}%",number(stats.mean(r['v_completion_tokens'] for r in sub),1),number(stats.mean(r['p_completion_tokens'] for r in sub),1),change(total('p_completion_tokens',sub),total('v_completion_tokens',sub))] for n in sorted(set(r['subquestions'] for r in rows)) for sub in [[r for r in rows if r['subquestions']==n]]]))
    add("\nCredits는 전부 양의 정수이며 합계 100이다. 단계 수·credits·정규화 entropy·Planner 평가 난이도·단계별 words를 CSV에 저장했다. 전체 생성 words는 reasoning+answer 공백 분리 수여서 수학 표기와 언어에 민감하다. 단일 Solver 호출이므로 실제 단계별 토큰/words 배분이나 엄격한 예산 준수는 확인 불가하다.\n")
    add(table(['지표','평균','중앙값','최소','최대'],[[label]+[number(summary([r[field] for r in rows])[k],3) for k in ['mean','median','min','max']] for label,field in [('할당 words 합계','words_budget'),('실제 PAB words','p_words'),('PAB words/할당 words','p_words_budget_ratio'),('PAB 생성 토큰/할당 word','p_tokens_per_allocated_word'),('최대 credit','credit_max'),('정규화 credit entropy','credit_normalized_entropy')]]))
    credit_stats=summary([c for r in rows for c in r['credits']])
    add(table(['전체 단계 credit 수','평균','중앙값','최소','최대','표준편차'],[[credit_stats['n']]+[number(credit_stats[k],3) for k in ['mean','median','min','max','population_std']]]))
    add('탐색적 상관관계(난이도 등 교란요인 미보정):\n')
    add(table(['X','Y','Pearson r','Spearman ρ'],[[c['x'],c['y'],number(c['pearson'],3),number(c['spearman'],3)] for c in metrics['correlations']]))
    add("\n이 상관관계는 인과관계가 아니다. 단계 수와 words budget은 난이도·문제 유형과 함께 변하므로 credit/계획 자체의 독립 효과로 해석할 수 없다.\n")
    for label,key,descending,subset in [
        ('PAB Solver 생성 증가 상위', 'solver_delta_completion_tokens', True, rows),
        ('Vanilla만 정답인 사례', 'solver_delta_completion_tokens', True, [r for r in rows if r['accuracy_transition']=='vanilla_only']),
        ('PAB만 정답인 사례', 'solver_delta_completion_tokens', False, [r for r in rows if r['accuracy_transition']=='pab_only'])]:
        selected=sorted(subset,key=lambda r:r[key],reverse=descending)[:5]
        add(label+' (정량 지표로 선택; 의미적 원인 판정은 하지 않음):\n')
        add(table(['ID','Level','단계','V→P 정답','V/P 생성','차이','V/P 종료','문제 앞부분'],[[r['id'],r['level'],r['subquestions'],f"{r['v_correct']}→{r['p_correct']}",f"{r['v_completion_tokens']}/{r['p_completion_tokens']}",r['solver_delta_completion_tokens'],f"{r['v_finish']}/{r['p_finish']}",r['query'][:110]] for r in selected]))
    add(table(['변화 그룹','N','평균 Level','평균 단계 수','평균 V 생성','평균 P 생성','PAB length 수'],[[label,len(sub)]+[number(stats.mean(r[k] for r in sub),2) if sub else '해당 없음' for k in ['level','subquestions','v_completion_tokens','p_completion_tokens']]+[sum(r['p_finish']=='length' for r in sub)] for key,label in labels.items() for sub in [[r for r in rows if r['accuracy_transition']==key]]]))
    add("## 8. Planner 오버헤드 분석\n")
    baseline=planner_doc['overlong_recovery']['baseline_attempt_cost']
    repair=planner_doc['summary']['overlong_recovery']['additional_attempt_cost']
    invalid=[a for row in planner_doc['stats'] for s in row['stages'].values() for a in s['attempts'] if a['status']!='success']
    invalid_cost=attempt_usage({'attempts':invalid})
    add(table(['범위','호출','입력','생성','전체 토큰','호출 시간(초)'],[
        ['최초 Planner 작업',baseline['attempts'],baseline['prompt_tokens']['known_sum'],baseline['completion_tokens']['known_sum'],baseline['total_tokens']['known_sum'],number(baseline['latency_seconds']['known_sum'],6)],
        ['6문제 복구 분해+평가',repair['attempts'],repair['prompt_tokens']['known_sum'],repair['completion_tokens']['known_sum'],repair['total_tokens']['known_sum'],number(repair['latency_seconds']['known_sum'],6)],
        ['그중 최초 무효 분해(중복 합산 금지)',len(invalid),invalid_cost['prompt_tokens'],invalid_cost['completion_tokens'],invalid_cost['total_tokens'],number(invalid_cost['latency_seconds'],6)],
        ['Planner 총 실제 시도',int(total('planner_calls')),int(total('planner_prompt_tokens')),int(total('planner_completion_tokens')),int(total('planner_total_tokens')),number(total('planner_latency_seconds'),6)],
    ]))
    add(f"\nSolver checkpoint의 `paired_subset.pab_including_planner`는 최종 성공 Planner 두 응답만 합산해 최초 무효 분해 {invalid_cost['total_tokens']:,}토큰·{invalid_cost['latency_seconds']:.6f}초를 제외한다. 본 보고서 C는 이 비용을 복원하고, PAB Solver 중단 호출 시간도 포함한 `all_attempt_cost` 기준이다. 성공 호출만 비교하는 요약과 혼동하면 안 된다.\n")
    add(f"실제 알려진 PAB 전체 토큰은 **최소 {int(total('e2e_total_tokens')):,}**, Vanilla {int(total('v_all_total_tokens')):,} 대비 **최소 {pct(total('e2e_total_tokens'),total('v_all_total_tokens')):.2f}% 증가**다. API latency 합은 {number(total('e2e_latency_seconds'),3)}초로 Vanilla 대비 {pct(total('e2e_latency_seconds'),total('v_all_latency_seconds')):.2f}% 변했다. 더 큰 Planner가 정확도 향상을 가져왔다거나 8B보다 경제적이라는 주장은 이 결과만으로 불가하다.\n")
    add("## 9. PAB의 효과 및 한계\n")
    add(f"- 정확도: Vanilla {rate('v'):.2f}%, PAB {rate('p'):.2f}%로 PAB가 {abs(rate('p')-rate('v')):.2f}pp 낮았다.\n"
        f"- Solver 성공응답 생성 토큰: {-pct(total('p_completion_tokens'),total('v_completion_tokens')):.2f}% 감소. 미상 중단 호출을 포함한 실제 절감률은 이보다 작을 수 있다.\n"
        f"- Solver 성공응답 latency: {-pct(total('p_latency_seconds'),total('v_latency_seconds')):.2f}% 감소.\n"
        f"- Planner 포함 알려진 전체 토큰: {pct(total('e2e_total_tokens'),total('v_all_total_tokens')):.2f}% 이상 증가. 전체 API 시간 합은 {-pct(total('e2e_latency_seconds'),total('v_all_latency_seconds')):.2f}% 감소.\n")
    add("\n따라서 정확도를 유지한 채 전체 비용을 줄인다는 결론은 성립하지 않는다. 이번 설정은 Solver 출력·API 시간을 줄이는 대신 정확도와 Planner 포함 토큰 비용을 지불한 trade-off다. 시간과 토큰의 서로 다른 결론은 모델별 prefill/decode 속도·입력 길이·생성량 등이 다르기 때문일 수 있으나 이 실험으로 원인을 분리하지 못한다.\n")
    add("주요 한계:\n\n- 단일 모델쌍·단일 seed·고정 MATH-500이다. 8B Planner 대조 실험이 없다.\n- constant이므로 위치 decay의 효과를 검증하지 않았다. 계획·credit·words 지시·추가 난이도 안내 효과를 분리하지 않았다.\n- benchmark 예시 5개는 평가 집합에서 뽑혀 정답 풀이가 Planner 문맥에 포함된다. 중복 ID: `" + ', '.join(manifest['benchmark_ids']) + "`. 원본 저장소 방식을 유지한 제한이며 독립 benchmark 조건과 다르다.\n- 잘린 출력도 원본 evaluator의 fallback으로 정답 처리될 수 있다. 정답의 의미적 타당성을 전부 수동 검증한 것은 아니다.\n- 중단 PAB 호출의 토큰은 미상이다. 세션/호출 순서 효과와 wall-clock 변동도 존재한다.\n- 두 tokenizer 토큰 합산은 모델별 FLOPs·에너지·요금이 아니다.\n")
    add("서버 로그에는 FP16 변환·custom allreduce 비활성화·NCCL 종료 정리·공유 메모리 경고가 있다. OOM이나 HTTP 오류의 원인으로 단정할 기록은 없으며, 첫 Solver 작업의 취소/종료 신호와 1개 중단 호출은 별도로 기록됐다.\n")
    add("## 10. 최종 결론 및 후속 실험 제안\n")
    add("기존 산출물 기준 Planner/Vanilla/PAB는 모두 500문제 결과를 보존했고, resume로 작업을 완결했다. 이 70B Planner 변형은 Solver 효율에 이점이 있지만 정확도와 전체 토큰 관점에서 Vanilla를 지배하지 않는다. 시간 우선/토큰 우선/정확도 우선 중 운영 목적에 따라 평가해야 한다.\n\n후속 실험 제안(이번에는 실행하지 않음):\n\n1. 동일한 평가·생성 설정에서 8B Planner와 70B Planner의 직접 대조.\n2. 여러 seed와 paired confidence interval, 오류·잘림 사례의 수동 감사.\n3. plan-only/균등 credit/credit-only/기존 decay별 ablation.\n4. 독립 benchmark 예시를 사용해 평가 집합 중복 제거.\n5. 요청별 GPU 실행 시간·에너지·원가 측정과 서버 측 중단 usage 기록.\n")
    add("### 재현 기준 및 원본 보호\n")
    add("재현 명령: `/data/dada0228/conda_envs/vllm_server/bin/python -B experiments/llama_qwq/analyze_math500_final.py` (권장 환경 `PYTHONDONTWRITEBYTECODE=1`). 보고서만 다시 생성할 때 `--reuse-evaluation`은 아래 원본·평가기 해시가 일치하는 CPU 재평가 캐시만 사용한다.\n")
    add(table(['입력','SHA-256'],[[str((BASE/'runs/math500_original_v1'/('planner.json' if k=='planner' else 'solver.json')).relative_to(ROOT)) if k in ('planner','solver') else k,v] for k,v in metrics['input_hashes'].items()]))
    add("\n문제별 CSV는 원본 ID와 source_index를 함께 보존한다. `p_all_*`/`planner_*`/`e2e_*`는 모든 시도의 알려진 합계이며 `*_missing`은 미상 호출 수다. `v_*`/`p_*`는 최종 보존 응답이고, `solver_delta_*`는 PAB−Vanilla다. 평가 결과는 별도 분석 캐시에 저장하며 checkpoint·평가 코드·로그를 고치지 않는다.\n")
    (out / "math500_final_analysis.md").write_text("\n\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    main()
