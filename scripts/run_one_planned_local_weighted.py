import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

if os.getenv("LOCAL_LLM", "0") != "1":
    raise RuntimeError(
        "LOCAL_LLM=1이 아닙니다. 외부 API 호출 방지를 위해 실행을 중단합니다."
    )

from dataset.math_dataset import MathDatasetLoader
from inference._5_planned_local_weighted_model import PlannedLocalWeighted_Model
from utils.logger import QAProgressLogger


def main():
    progress_path = PROJECT_ROOT / "results/single_planned_local_weighted_progress.json"
    result_path = PROJECT_ROOT / "results/single_planned_local_weighted_result.json"

    # 기존 결과를 덮어쓰지 않습니다.
    if progress_path.exists():
        raise FileExistsError(f"이미 존재합니다: {progress_path}")
    if result_path.exists():
        raise FileExistsError(f"이미 존재합니다: {result_path}")

    progress_path.parent.mkdir(parents=True, exist_ok=True)

    logger = QAProgressLogger(str(progress_path))

    loader = MathDatasetLoader(
        config={
            "decomposed": True,
            "num_workers": 1,
            "queue_size": 1,
        },
        logger=logger,
        processor=None,
    )

    model = PlannedLocalWeighted_Model(
        instruction=loader._instruction_prompt,
        decay="constant",
        use_label_level=True,
        evaluator=loader.eval,
        logger=logger,
    )
    model.output_format = loader._output_format

    data = loader.dataset[0]

    sample = {
        "id": "0",
        "query": data["query"],
        "steps": data.get("steps", []),
        "credits": data.get("credits", []),
        "reference": "",
        "solution": data["solution"],
        "answer": data["answer"],
        "level": data["level"],
    }

    if not sample["steps"]:
        raise ValueError("첫 번째 샘플의 steps가 비어 있습니다.")
    if len(sample["steps"]) != len(sample["credits"]):
        raise ValueError(
            f"steps와 credits 길이가 다릅니다: "
            f"{len(sample['steps'])} vs {len(sample['credits'])}"
        )
    if sample["level"] not in (1, 2, 3, 4, 5):
        raise ValueError(f"유효하지 않은 level: {sample['level']}")

    print("=" * 60)
    print("Model:", os.getenv("MODEL_NAME"))
    print("Mode: local inference")
    print("Method: planned_local_weighted")
    print("Decay: constant")
    print("Dataset: MATH-500")
    print("Number of samples: 1")
    print("Sample ID:", sample["id"])
    print("Level:", sample["level"])
    print("Number of steps:", len(sample["steps"]))
    print("Credits:", sample["credits"])
    print("Question:", sample["query"])
    print("=" * 60)

    asyncio.run(model.process_question(**sample, logger=logger))

    stats = logger.progress_data.get("stats", [])
    if len(stats) != 1:
        raise RuntimeError(
            f"예상한 결과는 1개지만 {len(stats)}개가 기록됐습니다. "
            f"진행 파일을 확인하세요: {progress_path}"
        )

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(stats[0], f, ensure_ascii=False, indent=2)

    result = stats[0]
    print("\n===== 단일 샘플 실행 결과 =====")
    print("Prediction:", result.get("prediction"))
    print("Score:", result.get("score"))
    print("Prompt tokens:", result.get("prompt_tokens"))
    print("Completion tokens:", result.get("completion_tokens"))
    print("Result file:", result_path)
    print("Progress file:", progress_path)


if __name__ == "__main__":
    main()
