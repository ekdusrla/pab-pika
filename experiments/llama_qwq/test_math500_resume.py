"""CPU-only regression suite. All test checkpoints live in temporary directories."""
import ast
import copy
import json
from pathlib import Path
import random
import signal
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import experiment_state as state
import run_planner as planner
import run_solver_compare as solver
from math_answer_eval import evaluator_info, evaluate_answer

HERE = Path(__file__).resolve().parent


class HTTPError(Exception):
    def __init__(self, status=400):
        self.status_code = status
        self.body = {"error": "synthetic failure"}


class FakeResponse:
    def __init__(self, text, finish="stop", prompt=10, completion=5):
        self.raw = {"choices": [{"message": {"content": text, "reasoning_content": None}, "finish_reason": finish}],
                    "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}

    def model_dump(self, **_):
        return copy.deepcopy(self.raw)


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.chat = NS(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        value = next(self.responses)
        if isinstance(value, BaseException):
            raise value
        return value

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def args(**kwargs):
    values = dict(limit=1, model="test", revision="frozen", api_base="http://127.0.0.1:8000/v1",
                  seed=42, timeout=600, retries=0, retry_wait=0, server_pid=None, max_model_len=12288,
                  max_tokens=8192, temperature=0.1, top_p=0.9, order="alternate", evaluator="original")
    return NS(**{**values, **kwargs})


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = state.read_manifest(HERE / "math500_manifest.json")
        cls.evaluator = evaluator_info()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pab_resume_test_")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "checkpoint.json"

    def planner_document(self):
        return {"schema_version": state.SCHEMA_VERSION,
                "config": {"manifest_fingerprint": self.manifest["fingerprint"], "model": "test"},
                "stats": [{**r, "status": "pending", "stages": {"decomposition": state.stage(), "assessment": state.stage()}}
                          for r in self.manifest["records"]]}

    def solver_document(self, source):
        document = {"schema_version": state.SCHEMA_VERSION, "config": {"decay": "constant"},
                    "comparisons": [{**r, "words_budget": solver.LEVEL_WORDS[r["level"]],
                                     "methods": {"vanilla": state.stage(), "weighted_pab": state.stage()}}
                                    for r in self.manifest["records"]]}
        _, instruction, output_format = state.original_definitions()
        document["config"].update(instruction=instruction, output_format=output_format)
        solver.attach_plans(document, solver.planner_rows(source, self.manifest))
        return document

    def successful_plan(self, document):
        row = document["stats"][0]
        for key, parsed in (("decomposition", ["1. One", "2. Two"]), ("assessment", [40, 60])):
            target = row["stages"][key]
            response = state.response_record(FakeResponse("raw"), 1)
            target.update(status="success", parsed=parsed, result=response)
            target["attempts"] = [{"status": "success", **response}]
        planner.refresh_planner(document)

    def test_manifest(self):
        proposed = state.make_manifest(state.ROOT / "dataset/MATH-500/decomposed_questions.json")
        self.assertEqual(proposed, self.manifest)
        self.assertEqual(len({r["id"] for r in proposed["records"]}), 500)
        for index, row in enumerate(proposed["records"]):
            self.assertEqual(row["source_index"], index)

    def test_original_budget_parity(self):
        source = ast.parse((state.ROOT / "inference/_5_planned_local_weighted_model.py").read_text())
        mapping = {"constant": "constant_schedule", "linear": "linear_decay", "polynomial": "polynomial_decay",
                   "cosine": "cosine_anneal", "exponential": "exponential_decay"}
        names = {"allocate_tokens", *mapping.values()}
        namespace = {"np": np}
        exec(compile(ast.Module(body=[n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), "original", "exec"), namespace)
        rng = random.Random(42)
        for mode, function in mapping.items():
            for count in range(2, 6):
                for _ in range(30):
                    cuts = sorted(rng.sample(range(1, 100), count - 1))
                    credits = [b - a for a, b in zip([0] + cuts, cuts + [100])]
                    budget = rng.choice(list(solver.LEVEL_WORDS.values()))
                    actual = solver.allocate_words(credits, budget, mode)
                    self.assertEqual(actual, namespace["allocate_tokens"](credits, budget, namespace[function]).tolist())
                    self.assertEqual(sum(actual), budget)

    def test_planner_partial_resume(self):
        document = self.planner_document()
        def save():
            planner.refresh_planner(document)
            state.atomic_save(self.path, document)
        first = FakeClient([FakeResponse("1. One\n2. Two"), HTTPError()])
        self.assertFalse(planner.run_planner(args(), self.manifest, document, first, save))
        loaded = state.load_json(self.path)
        self.assertEqual(loaded["stats"][0]["stages"]["decomposition"]["status"], "success")
        self.assertEqual(loaded["stats"][0]["stages"]["assessment"]["status"], "error")
        second = FakeClient([FakeResponse('{"1":{"credit":40},"2":{"credit":60}}')])
        self.assertTrue(planner.run_planner(args(), self.manifest, document, second, save))
        self.assertEqual(len(second.calls), 1)
        self.assertIn("guided_json", second.calls[0]["extra_body"])
        empty = FakeClient([])
        self.assertTrue(planner.run_planner(args(), self.manifest, document, empty, save))
        self.assertEqual(empty.calls, [])
        self.assertEqual(len(document["stats"][0]["stages"]["assessment"]["attempts"]), 2)

    def test_solver_partial_resume_and_reevaluate(self):
        source = self.planner_document()
        self.successful_plan(source)
        document = self.solver_document(source)
        def save():
            solver.refresh_solver(document)
            state.atomic_save(self.path, document)
        first = FakeClient([FakeResponse(r"\boxed{2000}"), HTTPError()])
        self.assertFalse(solver.run_comparison(args(), document, first, save, self.evaluator))
        self.assertEqual(state.load_json(self.path)["comparisons"][0]["methods"]["vanilla"]["status"], "success")
        second = FakeClient([FakeResponse(r"\boxed{2000}")])
        self.assertTrue(solver.run_comparison(args(), document, second, save, self.evaluator))
        self.assertEqual(len(second.calls), 1)
        empty = FakeClient([])
        self.assertTrue(solver.run_comparison(args(), document, empty, save, self.evaluator))
        self.assertEqual(empty.calls, [])
        row = document["comparisons"][0]
        original = copy.deepcopy(row["methods"]["vanilla"]["result"])
        changed = {**self.evaluator, "fingerprint": "new-evaluator"}
        self.assertTrue(solver.run_comparison(args(), document, empty, save, changed))
        self.assertEqual(row["methods"]["vanilla"]["result"], original)
        self.assertEqual(row["methods"]["vanilla"]["evaluation_fingerprint"], "new-evaluator")
        self.assertEqual(document["summary"]["methods"]["vanilla"]["accuracy_all_500"], 1 / 500)
        self.assertEqual(document["summary"]["methods"]["vanilla"]["accuracy_completed_subset"], 1)
        self.assertEqual(row["costs"]["pab_total"]["total_tokens"], 45)
        self.assertEqual(document["summary"]["methods"]["weighted_pab"]["all_attempt_cost"]["prompt_tokens"]["missing_attempts"], 1)

    def test_planner_failure_does_not_filter_vanilla(self):
        source = self.planner_document()
        document = self.solver_document(source)
        client = FakeClient([FakeResponse(r"\boxed{2000}")])
        self.assertFalse(solver.run_comparison(args(), document, client, lambda: solver.refresh_solver(document), self.evaluator))
        self.assertEqual(len(document["comparisons"]), 500)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(document["comparisons"][0]["methods"]["weighted_pab"]["status"], "blocked_planner")

    def test_atomic_write_recovery_and_config(self):
        document = {"schema_version": state.SCHEMA_VERSION, "config": {"seed": 42, "levels": {1: 200}}}
        state.atomic_save(self.path, document)
        with patch.object(state.os, "replace", side_effect=OSError("interrupted write")):
            with self.assertRaises(OSError):
                state.atomic_save(self.path, {**document, "marker": "uncommitted"})
        self.assertNotIn("marker", state.load_json(self.path))
        self.assertNotIn("marker", state.load_checkpoint(self.path, document["config"], True))
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            state.load_checkpoint(self.path, {"seed": 99}, True)
        self.assertEqual(self.path.read_bytes(), before)
        self.path.unlink()
        recovered = state.load_checkpoint(self.path, document["config"], True)
        self.assertEqual(recovered["marker"], "uncommitted")
        with self.assertRaises(FileExistsError):
            state.load_checkpoint(self.path, document["config"], False)

    def test_lock_and_retry_and_signal(self):
        with state.lock(self.path):
            with self.assertRaises(RuntimeError):
                with state.lock(self.path):
                    pass
        target = state.stage()
        client = FakeClient([HTTPError(503), FakeResponse("ok")])
        with patch.object(state.time, "sleep") as sleep:
            self.assertTrue(state.run_stage(target, {}, client, lambda: state.atomic_save(self.path, target), lambda r: r["content"], retries=1))
            self.assertEqual(sleep.call_count, 1)
        self.assertEqual(len(target["attempts"]), 2)
        target = state.stage()
        with self.assertRaises(state.StopRequested):
            state.run_stage(target, {}, FakeClient([state.StopRequested(signal.SIGTERM)]), lambda: state.atomic_save(self.path, target), lambda r: {})
        self.assertEqual(state.load_json(self.path)["status"], "interrupted")

    def test_received_response_recovery_no_call(self):
        response = state.response_record(FakeResponse("ok"), 1)
        target = {"status": "response_received", "attempts": [{"status": "response_received", **response}], "result": response}
        client = FakeClient([])
        self.assertTrue(state.run_stage(target, {}, client, lambda: None, lambda r: r["content"]))
        self.assertEqual(client.calls, [])

    def test_real_cli_resume_mismatch_and_evaluation_only(self):
        manifest_path = str(HERE / "math500_manifest.json")
        planner_path = Path(self.temp.name) / "planner.json"
        solver_path = Path(self.temp.name) / "solver.json"
        base = ["runner", "--manifest", manifest_path, "--limit", "1", "--model", "test", "--revision", "frozen", "--retries", "0"]
        def execute(module, argv, client):
            with patch.object(sys, "argv", argv), patch("openai.OpenAI", return_value=client), patch("builtins.print"):
                return module.main()
        first = FakeClient([FakeResponse("1. One\n2. Two"), HTTPError()])
        self.assertEqual(execute(planner, base + ["--output", str(planner_path)], first), 2)
        second = FakeClient([FakeResponse('{"1":{"credit":40},"2":{"credit":60}}')])
        self.assertEqual(execute(planner, base + ["--output", str(planner_path), "--resume"], second), 0)
        self.assertEqual(len(second.calls), 1)
        old = planner_path.read_bytes()
        with self.assertRaises(ValueError):
            execute(planner, base + ["--output", str(planner_path), "--resume", "--seed", "43"], FakeClient([]))
        self.assertEqual(planner_path.read_bytes(), old)
        argv = base + ["--output", str(solver_path), "--planner-input", str(planner_path)]
        first = FakeClient([FakeResponse(r"\boxed{2000}"), HTTPError()])
        self.assertEqual(execute(solver, argv, first), 2)
        second = FakeClient([FakeResponse(r"\boxed{2000}")])
        self.assertEqual(execute(solver, argv + ["--resume"], second), 0)
        self.assertEqual(len(second.calls), 1)
        with patch.object(sys, "argv", argv + ["--resume", "--reevaluate-only", "--evaluator", "conservative"]), patch("openai.OpenAI", side_effect=AssertionError("No API client during reevaluation")):
            self.assertEqual(solver.main(), 0)
        loaded = state.load_json(solver_path)
        row = loaded["comparisons"][0]
        self.assertEqual(row["methods"]["vanilla"]["evaluation"]["status"], "correct")
        self.assertEqual(len(row["methods"]["vanilla"]["attempts"]), 1)

    def test_server_death_and_oom(self):
        target = state.stage()
        def save():
            state.atomic_save(self.path, target)
        with patch.object(state.os, "kill", side_effect=ProcessLookupError):
            with self.assertRaises(state.ServerDead):
                state.run_stage(target, {}, FakeClient([]), save, lambda r: {}, server_pid=123)
        failure = HTTPError(500)
        failure.body = {"error": "CUDA out of memory"}
        with self.assertRaises(state.ServerDead):
            state.run_stage(target, {}, FakeClient([failure]), save, lambda r: {})
        self.assertEqual(state.load_json(self.path)["attempts"][-1]["status"], "error")

    def test_original_evaluation_and_finish_independence(self):
        for predicted, reference in [(r"\boxed{\frac{1}{2}}", "0.5"),
                                     (r"\boxed{(3,\frac{\pi}{2})}", r"\left(3,\frac{\pi}{2}\right)"),
                                     (r"\boxed{(x^3-1)(x^2-x+1)}", "x^5-x^4+x^3-x^2+x-1")]:
            self.assertEqual(evaluate_answer(predicted, reference)["status"], "correct")
        self.assertEqual(evaluate_answer(r"\boxed{3}", "3", "length")["status"], "correct")
        self.assertEqual(evaluate_answer(r"\boxed{4}", "3")["status"], "incorrect")
        self.assertEqual(evaluate_answer("", "3")["status"], "undetermined")

    def test_all_500_aggregation_with_1000_fake_solver_calls(self):
        source = self.planner_document()
        for row in source["stats"]:
            for key, parsed in (("decomposition", ["1. One", "2. Two"]), ("assessment", [40, 60])):
                response = state.response_record(FakeResponse("raw"), 1)
                row["stages"][key].update(status="success", parsed=parsed, result=response,
                                          attempts=[{"status": "success", **response}])
        planner.refresh_planner(source)
        document = self.solver_document(source)
        client = FakeClient([FakeResponse("raw") for _ in range(1000)])
        for row in document["comparisons"]:
            for method in ("vanilla", "weighted_pab"):
                target = row["methods"][method]
                with patch.object(state.time, "perf_counter", side_effect=[0, 1]):
                    self.assertTrue(state.run_stage(target, {"model": "test"}, client, lambda: None, lambda r: {}))
                target["evaluation"] = {"status": "correct"}
        solver.refresh_solver(document)
        summary = document["summary"]
        self.assertEqual(len(client.calls), 1000)
        self.assertEqual(summary["methods"]["vanilla"]["accuracy_all_500"], 1)
        self.assertEqual(summary["methods"]["weighted_pab"]["normal_completion"], 500)
        self.assertEqual(summary["paired_subset"]["count"], 500)
        self.assertEqual(summary["paired_subset"]["vanilla"]["total_tokens"]["known_sum"], 7500)
        self.assertEqual(summary["paired_subset"]["pab_including_planner"]["total_tokens"]["known_sum"], 22500)
        self.assertEqual(summary["pab_all_attempt_cost_including_planner"]["total_tokens"]["known_sum"], 22500)
        self.assertEqual(summary["paired_subset"]["pab_minus_vanilla"]["pab_including_planner"]["total_tokens"]["percent_change"], 200)

    def recovery_fixture(self):
        source_path = HERE / "runs/math500_original_v1/planner.json"
        source = state.load_json(source_path)
        self.assertEqual(sum(r["status"] == "success" for r in source["stats"]), 494)
        configuration = {**source["config"], "code_sha256": state.file_hash(planner.__file__)}
        return source, configuration

    def test_actual_six_recovery_and_494_immutable(self):
        source, _ = self.recovery_fixture()
        document = copy.deepcopy(source)
        metadata = planner.prepare_recovery(document, self.manifest)
        self.assertEqual(metadata["target_ids"], ["12", "189", "279", "288", "381", "423"])
        self.assertEqual(len(metadata["protected_successes"]), 494)
        five = "\n".join(f"{i}. Task {i}\nHint: preserve meaning" for i in range(1, 6))
        credits = json.dumps({str(i): {"credit": 20} for i in range(1, 6)})
        fake = FakeClient([response for _ in range(6) for response in (FakeResponse(five), FakeResponse(credits))])
        def save():
            planner.validate_recovery_history(document)
            planner.refresh_planner(document)
            planner.validate_recovery_history(document)
        with patch("builtins.print"):
            self.assertTrue(planner.run_planner(args(limit=500, recover_overlong=True), self.manifest, document, fake, save))
        self.assertEqual(len(fake.calls), 12)
        for request in fake.calls[::2]:
            self.assertEqual(len(request["messages"]), 4)
            self.assertEqual(request["messages"][-1]["content"], planner.RECOVERY_INSTRUCTION)
            self.assertIn("6.", request["messages"][-2]["content"])
            self.assertIn("absolute maximum of 5", request["messages"][-1]["content"])
        for request in fake.calls[1::2]:
            self.assertIn("guided_json", request["extra_body"])
            self.assertEqual(len(request["messages"]), 2)
        for original, repaired in zip(source["stats"], document["stats"]):
            if original["status"] == "success":
                self.assertEqual(original, repaired)
            else:
                self.assertEqual(original["stages"]["decomposition"]["attempts"], repaired["stages"]["decomposition"]["attempts"][:1])
                self.assertEqual(len(repaired["steps"]), 5)
                self.assertEqual(sum(repaired["credits"]), 100)
        self.assertEqual(document["config"], source["config"])
        self.assertEqual(document["summary"]["success"], 500)
        self.assertEqual(document["summary"]["attempt_cost"]["attempts"], 1006)
        recovery_cost = document["summary"]["overlong_recovery"]
        self.assertEqual(recovery_cost["decomposition_attempt_cost"]["attempts"], 6)
        self.assertEqual(recovery_cost["assessment_attempt_cost"]["attempts"], 6)
        self.assertEqual(recovery_cost["additional_attempt_cost"]["total_tokens"]["known_sum"], 180)
        empty = FakeClient([])
        with patch("builtins.print"):
            self.assertTrue(planner.run_planner(args(limit=500, recover_overlong=True), self.manifest, document, empty, save))
        self.assertEqual(empty.calls, [])

    def test_recovery_interruption_and_resume(self):
        source, _ = self.recovery_fixture()
        document = copy.deepcopy(source)
        five = "\n".join(f"{i}. Task {i}" for i in range(1, 6))
        credits = json.dumps({str(i): {"credit": 20} for i in range(1, 6)})
        def save():
            planner.validate_recovery_history(document)
            planner.refresh_planner(document)
        with self.assertRaises(state.StopRequested), patch("builtins.print"):
            planner.run_planner(args(limit=500, recover_overlong=True), self.manifest, document,
                                FakeClient([FakeResponse(five), state.StopRequested(signal.SIGTERM)]), save)
        state.atomic_save(self.path, document)
        document = state.load_json(self.path)
        target = next(r for r in document["stats"] if r["id"] == "12")
        self.assertEqual(target["stages"]["decomposition"]["status"], "success")
        self.assertEqual(target["stages"]["assessment"]["status"], "interrupted")
        responses = [FakeResponse(credits)] + [r for _ in range(5) for r in (FakeResponse(five), FakeResponse(credits))]
        resumed = FakeClient(responses)
        with patch("builtins.print"):
            self.assertTrue(planner.run_planner(args(limit=500, recover_overlong=True), self.manifest, document, resumed, save))
        self.assertEqual(len(resumed.calls), 11)
        self.assertEqual(document["summary"]["attempt_cost"]["attempts"], 1007)
        self.assertEqual(document["summary"]["overlong_recovery"]["additional_attempt_cost"]["prompt_tokens"]["missing_attempts"], 1)

    def test_explicit_recovery_config_boundary_and_check_only(self):
        source, config = self.recovery_fixture()
        state.atomic_save(self.path, source)
        before = self.path.read_bytes()
        recovery_args = args(output=self.path, resume=True, check_only=True, limit=500, recover_overlong=True)
        loaded = planner.load_planner_checkpoint(recovery_args, config, self.manifest)
        self.assertEqual(len(loaded["overlong_recovery"]["target_ids"]), 6)
        self.assertEqual(before, self.path.read_bytes())
        with self.assertRaises(ValueError):
            planner.load_planner_checkpoint(recovery_args, {**config, "seed": 43}, self.manifest)
        with self.assertRaises(ValueError):
            planner.load_planner_checkpoint(recovery_args, {**config, "model": "different"}, self.manifest)
        with self.assertRaises(ValueError):
            planner.load_planner_checkpoint(args(output=self.path, resume=True, check_only=True, recover_overlong=False), config, self.manifest)

    def test_recovery_trigger_and_normal_prompts(self):
        for n in range(2, 7):
            text = "\n".join(f"{i}. Task {i}" for i in range(1, n + 1))
            target = {"status": "error", "result": {"content": text, "finish_reason": "stop"}}
            self.assertEqual(planner.overlong_decomposition(target), n > 5)
            if n <= 5:
                self.assertEqual(len(planner.parse_steps(text)), n)
        target["result"]["finish_reason"] = "length"
        self.assertFalse(planner.overlong_decomposition(target))
        target["result"] = {"content": "1. One", "finish_reason": "stop"}
        self.assertFalse(planner.overlong_decomposition(target))
        document = self.planner_document()
        client = FakeClient([FakeResponse("1. One\n2. Two"), FakeResponse('{"1":{"credit":40},"2":{"credit":60}}')])
        with patch("builtins.print"):
            planner.run_planner(args(), self.manifest, document, client, lambda: planner.refresh_planner(document))
        self.assertEqual(len(client.calls[0]["messages"]), 2)
        self.assertNotIn(planner.RECOVERY_INSTRUCTION, json.dumps(client.calls[0]))


if __name__ == "__main__":
    unittest.main()
