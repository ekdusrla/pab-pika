"""Frozen MATH manifest, atomic checkpoints and bounded API attempts.

Only reads named source files; never imports project utility/dataset packages.
"""
import argparse
import ast
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import textwrap
import time
from importlib.metadata import version, PackageNotFoundError

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 2
METRICS = ("prompt_tokens", "completion_tokens", "latency_seconds")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions():
    result = {}
    for name in ("openai", "vllm", "torch", "numpy", "transformers", "math-verify", "sympy"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def original_definitions():
    """Extract only literal prompt assignments; avoid import-time side effects."""
    path = ROOT / "dataset/break_down_question.py"
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "PROMPTS" or
        isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == "PROMPTS"
        for t in n.targets)]
    namespace = {"textwrap": textwrap}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    math_path = ROOT / "dataset/math_dataset.py"
    cls = next(n for n in ast.parse(math_path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "MathDatasetLoader")
    for node in cls.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("_instruction_prompt", "_output_format") for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(math_path), "exec"), namespace)
    return namespace["PROMPTS"], namespace["_instruction_prompt"], namespace["_output_format"]


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(path) + ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another writer holds {path}")
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def atomic_save(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)
    # Defer termination only over the short durable commit section.
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
    try:
        with Path(str(path) + ".tmp").open("w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(path) + ".tmp", path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def make_manifest(dataset):
    source = load_json(dataset)["stats"]
    if len(source) != 500:
        raise ValueError(f"Expected exactly 500 records, got {len(source)}")
    rows, ids = [], set()
    for index, item in enumerate(source):
        original_id = item.get("id")
        if type(original_id) not in (str, int) or str(original_id) == "":
            raise ValueError(f"Missing/invalid source id at {index}")
        identifier = str(original_id)
        if identifier in ids:
            raise ValueError(f"Duplicate id {identifier}")
        ids.add(identifier)
        for key in ("query", "answer", "solution"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Invalid {key} for {identifier}")
        if type(item.get("level")) is not int or item["level"] not in range(1, 6):
            raise ValueError(f"Invalid level for {identifier}")
        row = {key: item[key] for key in ("query", "answer", "solution", "level")}
        row.update(id=identifier, original_id=original_id, source_index=index)
        row["record_fingerprint"] = digest(row)
        rows.append(row)
    rng = random.Random(42)
    benchmarks = [rng.choice([r for r in rows if r["level"] == level])["id"]
                  for level in range(1, 6) if any(r["level"] == level for r in rows)]
    body = {"schema_version": SCHEMA_VERSION, "dataset": "MATH-500 local source",
            "source_sha256": file_hash(dataset), "count": 500, "records": rows,
            "benchmark_ids": benchmarks, "benchmark_seed": 42,
            "benchmark_overlap_policy": "Examples sampled from evaluation dataset as in repository; overlap explicitly recorded."}
    return {**body, "fingerprint": digest(body)}


def read_manifest(path):
    document = load_json(path)
    body = {k: v for k, v in document.items() if k != "fingerprint"}
    if document.get("fingerprint") != digest(body) or document.get("count") != 500 or len(document.get("records", [])) != 500:
        raise ValueError("Manifest fingerprint/count mismatch")
    ids = []
    for index, row in enumerate(document["records"]):
        if row["source_index"] != index or row["record_fingerprint"] != digest({k: v for k, v in row.items() if k != "record_fingerprint"}):
            raise ValueError("Manifest record changed")
        ids.append(row["id"])
    if len(set(ids)) != 500:
        raise ValueError("Manifest duplicate ids")
    return document


def load_checkpoint(path, config, resume, recover=True):
    path = Path(path)
    if path.exists() and not resume:
        raise FileExistsError(f"Existing checkpoint protected: {path}; use --resume")
    if not resume:
        if Path(str(path) + ".tmp").exists():
            raise FileExistsError("Uncommitted checkpoint exists; use --resume")
        return None
    candidates = (path, Path(str(path) + ".tmp"))
    document = None
    for candidate in candidates:
        try:
            document = load_json(candidate)
            break
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    if document is None:
        raise ValueError("No recoverable checkpoint")
    if document.get("schema_version") != SCHEMA_VERSION or document.get("config") != json.loads(json.dumps(config)):
        raise ValueError("Resume configuration mismatch; existing checkpoint unchanged")
    if candidate != path and recover:
        atomic_save(path, document)
    return document


class StopRequested(BaseException):
    def __init__(self, signum):
        self.signum = signum


class ServerDead(RuntimeError):
    pass


@contextlib.contextmanager
def termination_handler():
    def stop(signum, _):
        raise StopRequested(signum)
    old = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for signum, handler in old.items():
            signal.signal(signum, handler)


def stage():
    return {"status": "pending", "attempts": []}


def check_server(pid):
    if pid:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            raise ServerDead("Server process exited")


def response_record(response, elapsed):
    raw = response.model_dump(mode="json")
    choice = raw["choices"][0]
    message = choice["message"]
    usage = raw.get("usage") or {}
    return {"raw_response": raw, "content": message.get("content"),
            "reasoning_content": message.get("reasoning_content"),
            "finish_reason": choice.get("finish_reason"),
            "usage": {"prompt_tokens": usage.get("prompt_tokens"),
                      "completion_tokens": usage.get("completion_tokens"),
                      "latency_seconds": round(elapsed, 6)}}


def run_stage(target, payload, client, save, validate, retries=2, retry_wait=2, server_pid=None):
    """Persist response before validation; completed calls are never replayed."""
    if target["status"] == "success":
        return True
    if target["status"] == "response_received":
        try:
            target["parsed"] = validate(target["result"])
            target["status"] = "success"
            if target["attempts"]:
                target["attempts"][-1]["status"] = "success"
            save()
            return True
        except Exception as exc:
            target["status"] = "error"
            target["validation_error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
            save()
            # A received response with invalid semantics is not an API transient.
            return False
    for attempt in target["attempts"]:
        if attempt["status"] == "in_flight":
            attempt["status"] = "interrupted"
            attempt["cost_unknown"] = True
    for retry in range(retries + 1):
        check_server(server_pid)
        target["request"] = payload
        attempt = {"number": len(target["attempts"]) + 1, "status": "in_flight", "request_fingerprint": digest(payload),
                   "usage": {"prompt_tokens": None, "completion_tokens": None, "latency_seconds": None}}
        target["attempts"].append(attempt)
        target["status"] = "running"
        save()
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(**payload)
            result = response_record(response, time.perf_counter() - start)
        except StopRequested:
            attempt.update(status="interrupted", cost_unknown=True)
            attempt["usage"]["latency_seconds"] = round(time.perf_counter() - start, 6)
            target["status"] = "interrupted"
            save()
            raise
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            body = getattr(exc, "body", None)
            # Keep API response body, never request headers or credential-bearing repr.
            error = {"type": type(exc).__name__, "status_code": status_code,
                     "body": body if isinstance(body, (dict, list, str, type(None))) else str(type(body))}
            attempt.update(status="error", error=error, cost_unknown=True)
            attempt["usage"]["latency_seconds"] = round(time.perf_counter() - start, 6)
            if isinstance(body, dict) and isinstance(body.get("usage"), dict):
                for key in ("prompt_tokens", "completion_tokens"):
                    attempt["usage"][key] = body["usage"].get(key)
            target["status"] = "error"
            save()
            lower = json.dumps(body).lower()
            if "out of memory" in lower or "engine is dead" in lower or "engine has died" in lower:
                raise ServerDead("Server OOM/engine failure; checkpoint preserved")
            check_server(server_pid)
            connection_error = type(exc).__name__ in ("APIConnectionError", "APITimeoutError")
            transient = connection_error or status_code in (408, 429, 500, 502, 503, 504)
            if transient and retry < retries:
                time.sleep(min(retry_wait * (2 ** retry), 30))
                continue
            if connection_error:
                raise ServerDead("Connection retries exhausted; checkpoint preserved")
            return False
        attempt.update(status="response_received", **result)
        target.update(status="response_received", result=result)
        save()
        try:
            target["parsed"] = validate(result)
            target["status"] = "success"
            attempt["status"] = "success"
        except Exception as exc:
            target["status"] = "error"
            attempt["status"] = "invalid_response"
            target["validation_error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
        save()
        return target["status"] == "success"


def attempt_cost(stages):
    attempts = [a for s in stages for a in s.get("attempts", [])]
    result = {"attempts": len(attempts)}
    for key in METRICS:
        values = [a.get("usage", {}).get(key) for a in attempts]
        known = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
        result[key] = {"known_sum": round(sum(known), 6), "missing_attempts": len(values) - len(known)}
    total = [a.get("usage", {}).get("prompt_tokens") + a.get("usage", {}).get("completion_tokens")
             for a in attempts if isinstance(a.get("usage", {}).get("prompt_tokens"), int)
             and isinstance(a.get("usage", {}).get("completion_tokens"), int)]
    result["total_tokens"] = {"known_sum": sum(total), "missing_attempts": len(attempts) - len(total)}
    return result


def common_arguments(parser):
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Validate manifest/config/checkpoint without API calls or result writes")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-wait", type=float, default=2)
    parser.add_argument("--server-pid", type=int)


def check_arguments(args):
    from urllib.parse import urlparse
    url = urlparse(args.api_base)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("Invalid API URL")
    if not 1 <= args.limit <= 500 or not 0 <= args.retries <= 5 or not 0 <= args.retry_wait <= 30 or not 0 < args.timeout <= 3600:
        raise ValueError("Invalid bounded limit/retry/timeout settings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "dataset/MATH-500/decomposed_questions.json"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    proposed = make_manifest(args.dataset)
    with lock(args.output):
        if args.output.exists():
            existing = read_manifest(args.output)
            if existing != proposed:
                raise ValueError("Existing frozen manifest differs; choose a NEW path")
        else:
            atomic_save(args.output, proposed)
    print("Manifest:", args.output, proposed["count"], proposed["fingerprint"])
