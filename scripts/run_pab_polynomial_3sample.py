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
from utils.data import BaseDatasetLoader
import inference
from utils.utils import generate_response

MODEL_NAME = os.environ["MODEL_NAME"]
GENERATION_CAP = 512
SAMPLE_COUNT = 3

RESULTS_DIR = PROJECT_ROOT / "results"
PROGRESS_PATH = RESULTS_DIR / "pab_llama70b_awq_polynomial_3sample_progress.json"
RESULT_PATH = RESULTS_DIR / "pab_llama70b_awq_polynomial_3sample_results.json"


async def main():
    # 결과 파일은 덮어쓰지 않습니다. 진행 파일은 재실행 시 이어서 사용합니다.
    if RESULT_PATH.exists():
        raise FileExistsError(f"기존 결과 파일 보호를 위해 중단합니다: {RESULT_PATH}")

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
        config={"decomposed": True, "num_workers": 2, "queue_size": 4},
        logger=logger,
        processor=None,
    )

    # 원본 데이터셋은 변경하지 않고, 이 인스턴스의 실행 범위만 3개로 제한합니다.
    loader.dataset = loader.dataset[:SAMPLE_COUNT]
    loader.data_generator = enumerate(loader.dataset)

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

    # 실패한 샘플의 예외를 포착해 다른 샘플 처리가 중단되거나
    # producer/consumer가 교착되는 것을 방지합니다.
    await loader.run()

    stats = logger.progress_data.get("stats", [])
    completed_ids = {str(row.get("id")) for row in stats}
    expected_ids = {str(i) for i in range(SAMPLE_COUNT)}
    missing_ids = expected_ids - completed_ids

    print("\n===== 3-sample PaB polynomial pilot =====")
    print("Model:", MODEL_NAME)
    print("Method: planned_local_weighted")
    print("Schedule: polynomial")
    print("Generation cap:", GENERATION_CAP)
    print("Completed sample records:", len(stats))
    print("Missing IDs:", sorted(missing_ids))
    print("Errors:", len(errors))
    print("Progress:", PROGRESS_PATH)

    if errors or missing_ids:
        print("일부 샘플이 완료되지 않았습니다. 진행 파일을 보존합니다.")
        raise RuntimeError(
            "파일 덮어쓰기 없이 실패 샘플을 확인하고 재실행할 수 있도록 "
            "최종 결과 파일은 생성하지 않았습니다."
        )

    ordered = sorted(stats, key=lambda row: int(row["id"]))
    with RESULT_PATH.open("x", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)

    for row in ordered:
        print(
            f"ID={row.get('id')} | score={row.get('score')} | "
            f"prompt_tokens={row.get('prompt_tokens')} | "
            f"completion_tokens={row.get('completion_tokens')} | "
            f"total_tokens={row.get('total_tokens')}"
        )

    print("Result:", RESULT_PATH)


if __name__ == "__main__":
    asyncio.run(main())
