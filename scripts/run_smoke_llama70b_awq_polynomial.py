import argparse
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
    raise RuntimeError("이 테스트는 OpenAI 호환 API 모드(LOCAL_LLM=0)만 지원합니다.")

from dataset.math_dataset import MathDatasetLoader
from utils.logger import QAProgressLogger
import inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["planned_local_weighted"],
        required=True,
    )
    args = parser.parse_args()

    model_name = os.environ["MODEL_NAME"]
    api_base = os.environ["API_BASE"]

    # 모델 이름을 기존 chat-completion 경로로 라우팅합니다.
    # 공유 코드 파일은 수정하지 않습니다.
    inference.CHAT_LLM.add(model_name)

    from utils.utils import generate_response

    # 서버 컨텍스트 한도(2048)에 맞춘 실험용 생성 상한.
    # 원본 구현의 기본값 8192를 그대로 사용하지 않습니다.
    generation_cap = 512

    async def capped_generate_response(prompt, *args, **kwargs):
        requested = kwargs.pop("max_tokens", generation_cap)
        kwargs["max_tokens"] = min(requested, generation_cap)
        return await generate_response(prompt, *args, **kwargs)

    if args.method == "vanilla":
        from inference import _0_vanilla_model as model_module
        from inference._0_vanilla_model import Vanilla_Model

        model_module.generate_response = capped_generate_response
        model_name_for_file = "vanilla"
    else:
        from inference import _5_planned_local_weighted_model as model_module
        from inference._5_planned_local_weighted_model import (
            PlannedLocalWeighted_Model,
        )

        model_module.generate_response = capped_generate_response
        model_name_for_file = "planned_local_weighted"

    results_dir = PROJECT_ROOT / "results"
    progress_path = results_dir / f"llama70b_awq_smoke_polynomial_{model_name_for_file}_progress.json"
    result_path = results_dir / f"llama70b_awq_smoke_polynomial_{model_name_for_file}_result.json"

    # 기존 파일 보호: 하나라도 있으면 실행하지 않습니다.
    for path in (progress_path, result_path):
        if path.exists():
            raise FileExistsError(
                f"기존 파일을 보호하기 위해 실행을 중단합니다: {path}"
            )

    results_dir.mkdir(parents=True, exist_ok=True)
    logger = QAProgressLogger(str(progress_path))

    loader = MathDatasetLoader(
        config={"decomposed": True, "num_workers": 1, "queue_size": 1},
        logger=logger,
        processor=None,
    )

    if args.method == "vanilla":
        model = Vanilla_Model(logger=logger)
        model.instruction = loader._instruction_prompt
        model.output_format = loader._output_format
        model.evaluator = loader.eval
    else:
        model = PlannedLocalWeighted_Model(
            instruction=loader._instruction_prompt,
            decay="polynomial",
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

    if args.method == "planned_local_weighted":
        if not sample["steps"] or len(sample["steps"]) != len(sample["credits"]):
            raise ValueError("첫 번째 문제의 steps/credits 구성이 유효하지 않습니다.")

    print("=" * 60)
    print("Model:", model_name)
    print("API base:", api_base)
    print("Method:", args.method)
    print("Dataset: MATH-500")
    print("Samples: 1")
    print("Generation cap:", generation_cap)
    print("Sample ID:", sample["id"])
    print("Question:", sample["query"])
    print("=" * 60)

    asyncio.run(model.process_question(**sample, logger=logger))

    stats = logger.progress_data.get("stats", [])
    if len(stats) != 1:
        raise RuntimeError(
            f"예상 결과는 1개지만 {len(stats)}개가 기록됐습니다. "
            f"진행 파일을 확인하세요: {progress_path}"
        )

    # 실행 도중 같은 이름의 파일이 생겼다면 덮어쓰지 않습니다.
    if result_path.exists():
        raise FileExistsError(f"결과 파일이 이미 있습니다: {result_path}")

    with result_path.open("x", encoding="utf-8") as f:
        json.dump(stats[0], f, ensure_ascii=False, indent=2)

    result = stats[0]
    print("\n===== Smoke test result =====")
    print("Score:", result.get("score"))
    print("Prediction:", result.get("prediction"))
    print("Prompt tokens:", result.get("prompt_tokens"))
    print("Completion tokens:", result.get("completion_tokens"))
    print("Total tokens:", result.get("total_tokens"))
    print("Result:", result_path)
    print("Progress:", progress_path)


if __name__ == "__main__":
    main()
