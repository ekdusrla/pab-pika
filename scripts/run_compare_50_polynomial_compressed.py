import asyncio
import json
import os
import sys
import time
from pathlib import Path
from functools import partial

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

if os.getenv("LOCAL_LLM", "0") != "1":
    raise RuntimeError(
        "LOCAL_LLM=1이 아닙니다. 외부 API 호출 방지를 위해 중단합니다."
    )

from dataset.math_dataset import MathDatasetLoader
from inference._0_vanilla_model import Vanilla_Model
from inference._5_planned_local_weighted_compressed_model import PlannedLocalWeighted_Compressed_Model
from utils.logger import QAProgressLogger


NUM_SAMPLES = 50
RESULT_DIR = PROJECT_ROOT / "results"

OUTPUTS = {
    "vanilla": {
        "progress": RESULT_DIR / "compare50_poly_compressed_vanilla_progress.json",
        "result": RESULT_DIR / "compare50_poly_compressed_vanilla_result.json",
    },
    "planned_local_weighted": {
        "progress": RESULT_DIR / "compare50_poly_compressed_planned_progress.json",
        "result": RESULT_DIR / "compare50_poly_compressed_planned_result.json",
    },
}
SUMMARY_PATH = RESULT_DIR / "compare50_poly_compressed_summary.json"


def save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_sample(index, data):
    return {
        "id": str(index),
        "query": data["query"],
        "steps": data.get("steps", []),
        "credits": data.get("credits", []),
        "reference": "",
        "solution": data["solution"],
        "answer": data["answer"],
        "level": data["level"],
    }


async def run_method(name, model, loader, logger, samples):
    print(f"\n{'=' * 70}")
    print(f"실험 시작: {name}")
    print(f"샘플 수: {len(samples)}")
    print(f"{'=' * 70}")

    durations = {}

    for i, sample in enumerate(samples, start=1):
        print(
            f"\n[{name}] {i}/{len(samples)} "
            f"ID={sample['id']} Level={sample['level']}"
        )
        print("Question:", sample["query"])

        start = time.perf_counter()
        await model.process_question(**sample, logger=logger)
        durations[sample["id"]] = round(time.perf_counter() - start, 3)

        stats = logger.progress_data.get("stats", [])
        current = next(
            (item for item in reversed(stats)
             if str(item.get("id")) == sample["id"]),
            None,
        )

        if current is None:
            raise RuntimeError(
                f"{name}: ID={sample['id']}의 평가 결과를 찾을 수 없습니다. "
                "진행 파일을 확인하세요."
            )

        print("Score:", current.get("score"))
        print("Prompt tokens:", current.get("prompt_tokens"))
        print("Completion tokens:", current.get("completion_tokens"))
        print("Elapsed seconds:", durations[sample["id"]])

    stats = logger.progress_data.get("stats", [])
    if len(stats) != len(samples):
        raise RuntimeError(
            f"{name}: 예상 결과 {len(samples)}개, 실제 결과 {len(stats)}개"
        )

    # 원본 통계를 유지하면서 샘플별 실행 시간만 추가합니다.
    results = []
    for item in stats:
        row = dict(item)
        row["elapsed_seconds"] = durations.get(str(item.get("id")))
        results.append(row)

    result_path = OUTPUTS[name]["result"]
    save_json(result_path, results)

    print(f"\n{name} 완료")
    print("결과:", result_path)
    print("진행 기록:", OUTPUTS[name]["progress"])

    return results


def summarize(name, results):
    scores = [r.get("score") for r in results]
    valid_scores = [s for s in scores if isinstance(s, (bool, int, float))]

    prompt = [
        r["prompt_tokens"] for r in results
        if isinstance(r.get("prompt_tokens"), (int, float))
    ]
    completion = [
        r["completion_tokens"] for r in results
        if isinstance(r.get("completion_tokens"), (int, float))
    ]
    durations = [
        r["elapsed_seconds"] for r in results
        if isinstance(r.get("elapsed_seconds"), (int, float))
    ]

    return {
        "method": name,
        "num_samples": len(results),
        "num_scored": len(valid_scores),
        "num_correct": sum(bool(s) for s in valid_scores),
        "accuracy": (
            sum(bool(s) for s in valid_scores) / len(valid_scores)
            if valid_scores else None
        ),
        "prompt_tokens_total": sum(prompt),
        "prompt_tokens_mean": sum(prompt) / len(prompt) if prompt else None,
        "completion_tokens_total": sum(completion),
        "completion_tokens_mean": (
            sum(completion) / len(completion) if completion else None
        ),
        "total_tokens_sum": sum(prompt) + sum(completion),
        "elapsed_seconds_total": round(sum(durations), 3),
        "elapsed_seconds_mean": (
            round(sum(durations) / len(durations), 3) if durations else None
        ),
    }


async def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    all_paths = [SUMMARY_PATH]
    for paths in OUTPUTS.values():
        all_paths.extend(paths.values())

    existing = [str(p) for p in all_paths if p.exists()]
    if existing:
        raise FileExistsError(
            "기존 파일을 덮어쓰지 않기 위해 중단합니다:\n"
            + "\n".join(existing)
        )

    loader = MathDatasetLoader(
        config={
            "decomposed": True,
            "num_workers": 1,
            "queue_size": 1,
        },
        logger=None,
        processor=None,
    )

    if len(loader.dataset) < NUM_SAMPLES:
        raise ValueError(f"샘플이 {NUM_SAMPLES}개보다 적습니다.")

    samples = [
        make_sample(i, loader.dataset[i])
        for i in range(NUM_SAMPLES)
    ]

    for sample in samples:
        if sample["level"] not in (1, 2, 3, 4, 5):
            raise ValueError(
                f"잘못된 level: ID={sample['id']}, level={sample['level']}"
            )
        if not sample["steps"]:
            raise ValueError(f"ID={sample['id']}의 steps가 비어 있습니다.")
        if len(sample["steps"]) != len(sample["credits"]):
            raise ValueError(
                f"ID={sample['id']}: steps와 credits 길이가 다릅니다."
            )

    print("Model:", os.getenv("MODEL_NAME"))
    print("Local mode:", os.getenv("LOCAL_LLM"))
    print("Dataset: MATH-500")
    print("Number of samples:", NUM_SAMPLES)
    print("Sample levels:", [s["level"] for s in samples])
    print("Execution: sequential")
    print("Decay: polynomial")
    print("Existing smoke-test files will not be modified.")

    # Vanilla
    v_logger = QAProgressLogger(str(OUTPUTS["vanilla"]["progress"]))
    vanilla = Vanilla_Model(logger=v_logger)
    vanilla.instruction = loader._instruction_prompt
    vanilla.output_format = loader._output_format
    vanilla.evaluator = loader.eval

    vanilla_results = await run_method(
        "vanilla", vanilla, loader, v_logger, samples
    )

    # Planned Local Weighted
    p_logger = QAProgressLogger(
        str(OUTPUTS["planned_local_weighted"]["progress"])
    )
    planned = PlannedLocalWeighted_Compressed_Model(
        instruction=loader._instruction_prompt,
        decay="polynomial",
        use_label_level=True,
        evaluator=loader.eval,
        logger=p_logger,
    )
    planned.output_format = loader._output_format

    planned_results = await run_method(
        "planned_local_weighted", planned, loader, p_logger, samples
    )

    summary = {
        "model": os.getenv("MODEL_NAME"),
        "dataset": "MATH-500",
        "num_samples": NUM_SAMPLES,
        "sample_ids": [s["id"] for s in samples],
        "sample_levels": [s["level"] for s in samples],
        "local_mode": True,
        "planned_decay": "polynomial",
        "vanilla": summarize("vanilla", vanilla_results),
        "planned_local_weighted": summarize(
            "planned_local_weighted", planned_results
        ),
    }

    save_json(SUMMARY_PATH, summary)

    print(f"\n{'=' * 70}")
    print("최종 요약")
    print(f"{'=' * 70}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n요약 파일:", SUMMARY_PATH)


if __name__ == "__main__":
    asyncio.run(main())
