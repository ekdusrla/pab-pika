import asyncio
import json
import os
import sys
from pathlib import Path
from functools import partial

# Add the project root to Python's module search path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

if os.getenv("LOCAL_LLM", "0") != "1":
    raise RuntimeError(
        "LOCAL_LLM=1이 아닙니다. 외부 API 호출 방지를 위해 실행을 중단합니다."
    )

from inference._0_vanilla_model import Vanilla_Model
from dataset.math_dataset import MathDatasetLoader
from utils.logger import QAProgressLogger


def main():
    progress_path = Path("results/single_vanilla_smoke_progress.json")

    # 기존 진행 파일을 덮어쓰지 않습니다.
    if progress_path.exists():
        raise FileExistsError(
            f"{progress_path} 파일이 이미 있습니다. "
            "파일을 확인한 뒤 이름을 변경해 주세요."
        )

    logger = QAProgressLogger(str(progress_path))

    model = Vanilla_Model(logger=logger)

    loader = MathDatasetLoader(
        config={
            "decomposed": True,
            "num_workers": 1,
            "queue_size": 1,
        },
        logger=logger,
        processor=partial(model.process_question, logger=logger),
    )

    model.instruction = loader._instruction_prompt
    model.output_format = loader._output_format
    model.evaluator = loader.eval

    # 전체 데이터셋이 아닌 첫 번째 샘플만 선택합니다.
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

    print("=" * 60)
    print("Model:", os.getenv("MODEL_NAME"))
    print("Device: CUDA (local model adapter)")
    print("Dataset: MATH-500")
    print("Number of samples: 1")
    print("Sample ID:", sample["id"])
    print("Question:", sample["query"])
    print("=" * 60)

    asyncio.run(model.process_question(**sample, logger=logger))

    stats = logger.progress_data.get("stats", [])
    if len(stats) != 1:
        raise RuntimeError(
            f"예상한 결과 수는 1개지만 {len(stats)}개가 기록됐습니다."
        )

    result_path = Path("results/single_vanilla_smoke_result.json")
    if result_path.exists():
        raise FileExistsError(
            f"{result_path} 파일이 이미 있습니다. 덮어쓰지 않고 중단합니다."
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
