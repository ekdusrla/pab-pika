import argparse
import asyncio
import json

from inference import *
from utils.logger import *
from utils.utils import *

# probe_budget changes:
#  - f-strings with nested double quotes were a SyntaxError on Python 3.10
#  - concurrency limited by a semaphore (vLLM --max-num-seqs)
#  - failed reasoning requests (completion_tokens None) are kept, scored by the parser like any other
#    output (empty prediction -> PARSE_FAIL -> 0) and excluded from token sums
#  - output written to *_posteval.json so the run_inf result file is not overwritten


async def evaluate_all(results, concurrency):
    sem = asyncio.Semaphore(concurrency)

    async def one(r):
        async with sem:
            return await loader.post_eval(r['query'], r['prediction'] or "", r['answer'])
    return await asyncio.gather(*[one(r) for r in results])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Evaluation dataset")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_MAP.keys(), help="Model to run inference with")
    parser.add_argument("--postfix", type=str, help="Postfix added to the result file name")
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()

    pf = f"_{args.postfix}" if args.postfix else ""
    progress_path = f"results/{args.model}_{args.dataset}_progress{pf}.json"
    output_path = f"results/{args.model}_{args.dataset}_results{pf}_posteval.json"

    with open(progress_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    results = results['stats']
    results = sorted(results, key=lambda x: int(x["id"]))

    if args.dataset.lower() == 'travelplanner':  # This dataset needs post-evaluation
        from dataset.travelplanner_dataset import TravelPlannerDatasetLoader
        loader = TravelPlannerDatasetLoader(config={}, processor=None)
        items = asyncio.run(evaluate_all(results, args.concurrency))
        for idx in range(len(results)):
            results[idx]["score"] = items[idx][0]
            results[idx]["explanation"] = items[idx][1]

    ok = [s for s in results if s['completion_tokens'] is not None]
    tokens_sum = sum(s['completion_tokens'] for s in ok)
    score = sum(s['score'] for s in results) / len(results) * 100.0
    avg_tokens = tokens_sum / max(1, len(ok))
    stats = {
        "len": len(results),
        "n_failed_requests": len(results) - len(ok),
        "score": score,
        "token_sum": tokens_sum,
        "avg_tokens": avg_tokens,
        "score_over_avg_tokens": score / avg_tokens if avg_tokens else None,
        "llm": MODEL_NAME
    }
    results.insert(0, stats)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    logger.info(stats)
    logger.info(f"Done evaluation in {args.dataset} dataset on {args.model}_model ✅")
