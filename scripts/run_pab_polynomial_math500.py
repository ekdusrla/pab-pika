import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

ENV_PATH = PROJECT_ROOT / ".env_llama70b_local"
if not ENV_PATH.is_file():
    raise FileNotFoundError(f"환경 파일이 없습니다: {ENV_PATH}")

load_dotenv(ENV_PATH, override=True)

if os.getenv("LOCAL_LLM", "0") != "0":
    raise RuntimeError("OpenAI 호환 API 모드(LOCAL_LLM=0)가 필요합니다.")

from dataset.math_dataset import MathDatasetLoader
from utils.logger import QAProgressLogger
import inference
from utils.utils import generate_response

MODEL_NAME = os.environ["MODEL_NAME"]
GENERATION_CAP = 512
EXPECTED_COUNT = 500
NUM_WORKERS = 2
QUEUE_SIZE = 4

RESULTS_DIR = PROJECT_ROOT / "results"
PROGRESS_PATH = RESULTS_DIR / "pab_llama70b_awq_polynomial_math500_progress.json"
RESULT_PATH = RESULTS_DIR / "pab_llama70b_awq_polynomial_math500_results.json"


async def main():
    if RESULT_PATH.exists():
        raise FileExistsError(
            f"기존 결과 파일 보호를 위해 실행을 중단합니다: {RESULT_PATH}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    inference.CHAT_LLM.add(MODEL_NAME)

    async def capped_generate_response(prompt, *args, **kwargs):
        requested = kwargs.pop("max_tokens", GENERATION_CAP)
        kwargs["max_tokens"] = min(requested, GENERATION_CAP)
        return await generate_response(prompt, *args, **kwargs)

    from inference import _5_planned_local_weighted_model as model_module
    from inference._5_planned_local_weighted_model import PlannedLocalWeighted_Model

    model_module.generate_response = capped_generate_response

    logger = QAProgressLogger(str(PROGRESS_PATH))
    loader = MathDatasetLoader(
        config={
            "decomposed": True,
            "num_workers": NUM_WORKERS,
            "queue_size": QUEUE_SIZE,
        },
        logger=logger,
        processor=None,
    )

    if len(loader.dataset) != EXPECTED_COUNT:
        raise RuntimeError(
            f"데이터 수가 예상과 다릅니다: {len(loader.dataset)} != {EXPECTED_COUNT}"
        )

    model = PlannedLocalWeighted_Model(
        instruction=loader._instruction_prompt,
        decay="polynomial",
        use_label_level=True,
        evaluator=loader.eval,
        logger=logger,
    )
    model.output_format = loader._output_format
    model.loader = loader

    errors = []

    async def safe_process(**item):
        try:
            await model.process_question(**item, logger=logger)
        except Exception as exc:
            errors.append({"id": item["id"], "error": repr(exc)})
            print(f"[ERROR] sample_id={item['id']}: {exc!r}")

    loader.processor = safe_process

    print("=" * 65)
    print("Experiment: PaB polynomial on MATH-500")
    print("Model:", MODEL_NAME)
    print("Samples:", EXPECTED_COUNT)
    print("Workers:", NUM_WORKERS)
    print("Queue size:", QUEUE_SIZE)
    print("Generation cap:", GENERATION_CAP)
    print("Progress:", PROGRESS_PATH)
    print("Result:", RESULT_PATH)
    print("Previously processed IDs:", len(logger.processed))
    print("=" * 65)

    await loader.run()

    stats = logger.progress_data.get("stats", [])
    completed_ids = {str(row.get("id")) for row in stats}
    expected_ids = {str(i) for i in range(EXPECTED_COUNT)}
    missing_ids = expected_ids - completed_ids
    unexpected_ids = completed_ids - expected_ids

    print("\n===== MATH-500 run summary =====")
    print("Recorded records:", len(stats))
    print("Unique recorded IDs:", len(completed_ids))
    print("Missing IDs:", len(missing_ids))
    print("Unexpected IDs:", len(unexpected_ids))
    print("Processing errors this run:", len(errors))
    print("Progress file:", PROGRESS_PATH)

    if errors or missing_ids or unexpected_ids or len(completed_ids) != EXPECTED_COUNT:
        print("실험이 완전하지 않습니다. 진행 파일을 보존합니다.")
        if errors:
            print("Failed IDs:", [e["id"] for e in errors])
        if missing_ids:
            print("Missing ID preview:", sorted(missing_ids, key=int)[:30])
        raise RuntimeError(
            "오류 또는 누락이 있어 최종 결과 파일을 생성하지 않았습니다. "
            "진행 파일을 확인한 뒤 재실행하세요."
        )

    ordered = sorted(stats, key=lambda row: int(row["id"]))
    with RESULT_PATH.open("x", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)

    correct = sum(row.get("score") is True for row in ordered)
    print("Correct:", correct)
    print("Accuracy:", f"{correct / EXPECTED_COUNT:.4%}")
    print("Total prompt tokens:", sum(row.get("prompt_tokens") or 0 for row in ordered))
    print("Total completion tokens:", sum(row.get("completion_tokens") or 0 for row in ordered))
    print("Total tokens:", sum(row.get("total_tokens") or 0 for row in ordered))
    print("Result:", RESULT_PATH)


if __name__ == "__main__":
    asyncio.run(main())
