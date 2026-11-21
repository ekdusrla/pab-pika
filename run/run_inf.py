import argparse
import functools
import time

from inference import *
from utils.logger import *
from utils.utils import *

if __name__ == "__main__":
    from dataset.math_dataset import MathDatasetLoader
    from dataset.instruction_dataset import InstructionDatasetLoader
    from dataset.travelplanner_dataset import TravelPlannerDatasetLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Evaluation dataset")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_MAP.keys(), help="Model to run inference with")
    parser.add_argument("--num-workers", type=int, default=500, help="Number of workers generating the answers")
    parser.add_argument("--queue-size", type=int, default=500, help="Queue size of data loading")
    parser.add_argument("--decay", type=str, choices=["constant", "linear", "polynomial", "exponential", "cosine"], help="Weighted model config")
    parser.add_argument("--postfix", type=str, help="Postfix added to the result file name")
    parser.add_argument("--keep", action="store_true", help="Keep the progress file")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Planned/decomposed json ({'stats': [...]}) to read instead of the default decomposed_questions.json")
    parser.add_argument("--raw-dataset", action="store_true",
                        help="Read the raw HF dataset instead of a decomposed json (vanilla needs no plan; ids = row index)")
    args = parser.parse_args()

    config = {
        "num_workers": args.num_workers,
        "queue_size": args.queue_size,
        "decomposed": not args.raw_dataset,
    }
    if args.data_path:
        config["data_path"] = args.data_path
    run_start = time.time()

    participant_model = MODEL_MAP[args.model](decay=args.decay)

    postfix = f"_{args.postfix}" if args.postfix else ""
    progress_path = f"results/{participant_model.name}_{args.dataset}_progress{postfix}.json"
    result_path = f"results/{participant_model.name}_{args.dataset}_results{postfix}.json"
    logger = QAProgressLogger(progress_path=progress_path)
    participant_model.logger = logger
    print(logger.processed_questions)

    loader = None
    if args.dataset.lower() == 'math':
        loader = MathDatasetLoader(
            config=config, 
            logger=logger,
            processor=functools.partial(participant_model.process_question, logger=logger)
        )
    elif args.dataset.lower() == 'instruction':
        loader = InstructionDatasetLoader(
            config=config, 
            logger=logger,
            processor=functools.partial(participant_model.process_question, logger=logger)
        )
    elif args.dataset.lower() == 'travelplanner':
        loader = TravelPlannerDatasetLoader(
            config=config, 
            logger=logger,
            processor=functools.partial(participant_model.process_question, logger=logger)
        )
    else:
        raise NotImplementedError("Dataset is not supported ❌")

    participant_model.loader = loader
    participant_model.instruction = loader._instruction_prompt
    participant_model.output_format = loader._output_format
    participant_model.evaluator = loader.eval
    
    loop = always_get_an_event_loop()
    loop.run_until_complete(
        loader.run()
    )

    run_wall_seconds = time.time() - run_start
    results = [
        {"id": int(stat["id"]), "query": stat["query"], "answer": stat["answer"],
        "prediction": stat["prediction"], "prompt_tokens": stat["prompt_tokens"],
        "completion_tokens": stat["completion_tokens"], "total_tokens": stat["total_tokens"],
        "reasoning_seconds": stat.get("reasoning_seconds"),
        "failed": stat["completion_tokens"] is None,
        # numpy scalar -> python (json); TravelPlanner keeps its pre-post-eval placeholder (-1)
        "score": stat["score"].item() if hasattr(stat["score"], "item") else stat["score"],
        "explanation": stat["explanation"]}
        for stat in logger.progress_data["stats"]
    ]
    results = sorted(results, key=lambda x: x["id"])
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Failed requests (llm_retry exhausted -> usage None) are scored as wrong and excluded
    # from token sums; their count is reported explicitly instead of crashing.
    # TravelPlanner: "score" here is a placeholder; the real score comes from run.run_eval.
    ok = [r for r in results if not r["failed"]]
    score = sum(r["score"] for r in results) / len(results) * 100.0
    tokens_sum = sum(r["completion_tokens"] for r in ok)
    prompt_sum = sum(r["prompt_tokens"] for r in ok)
    avg_tokens = tokens_sum / max(1, len(ok))
    secs = [r["reasoning_seconds"] for r in ok if r["reasoning_seconds"] is not None]
    stats = {
        "len": len(results),
        "n_failed_requests": len(results) - len(ok),
        "score": score,
        "token_sum": tokens_sum,
        "prompt_token_sum": prompt_sum,
        "avg_tokens": avg_tokens,
        "avg_prompt_tokens": prompt_sum / max(1, len(ok)),
        "score_over_avg_tokens": score / avg_tokens if avg_tokens else None,
        "mean_reasoning_seconds": sum(secs) / len(secs) if secs else None,
        "run_wall_seconds_this_invocation": round(run_wall_seconds, 1),
        "num_workers": args.num_workers,
        "data_path": args.data_path,
        "llm": MODEL_NAME
    }
    results.insert(0, stats)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    if not args.keep:
        os.remove(progress_path)

    logger.info(stats)
    logger.info(f"Done inference in {args.dataset} dataset on {args.model}_model ✅")