"""CPU smoke for v3 (run from repo root on a compute node):
 1) eval_detail on TravelPlanner validation gold plans (annotated_plan) -> should mostly pass
 2) post_eval JSON extraction on a fake QwQ response (with <think>)
 3) parse_joint on synthetic LLaMA outputs (well-formed / same-line credit / missing credit)
"""
import asyncio
import json
import sys
from collections import Counter

sys.path.insert(0, '.'); sys.path.insert(0, 'experiments/probe_budget')
from datasets import load_dataset

from dataset.TravelPlanner.eval import eval_detail

ds = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
print('fields:', ds.column_names)
res = []
for it in ds:
    plan = [{'days': d + 1, 'current_city': (f"from {it['org']} to {it['dest']}" if d == 0 else it['dest']),
             'transportation': '-', 'breakfast': '-', 'attraction': '-', 'lunch': '-', 'dinner': '-',
             'accommodation': '-'} for d in range(it['days'])]
    try:
        res.append(eval_detail(dict(it), {'plan': plan}))
    except Exception as e:
        res.append({'error': repr(e)})
err = [r for r in res if 'error' in r]
ok = [r for r in res if 'error' not in r]
print(f'dummy plans: n={len(res)} errors={len(err)} {err[:2]}')
for k in ('pab_score', 'final_pass', 'commonsense_pass', 'hard_pass'):
    print(f'  {k}: {sum(r[k] for r in ok)}/{len(ok)}')
print('  fail msgs:', Counter(r['msg'][:60] for r in ok if not r['final_pass']).most_common(5))
gold = [None, *[{'days': d + 1, 'current_city': ds[0]['dest'], 'transportation': '-', 'breakfast': '-',
                 'attraction': '-', 'lunch': '-', 'dinner': '-', 'accommodation': '-'} for d in range(ds[0]['days'])]]
# 2) post_eval extraction
import dataset.travelplanner_dataset as tpd
q = ds[0]['query']


async def fake(prompt, **kw):
    return 'Let me think... {"bogus": 1}\n</think>\n```json\n' + json.dumps({'plan': gold[1:]}) + '\n```'
tpd.generate_response = fake
loader = tpd.TravelPlannerDatasetLoader(config={}, processor=None)
print('post_eval(fake gold):', asyncio.run(loader.post_eval(q, 'x</think>plan', '')))


async def fake_bad(prompt, **kw):
    return 'no json here'
tpd.generate_response = fake_bad
print('post_eval(bad):', asyncio.run(loader.post_eval(q, 'x', '')))

# 3) joint parser
from plan_v3 import parse_joint
good = ("Decomposed Sub-questions:\n\n1. Find a.\nHint: use b.\nLevel: 2, Credit: 30\n\n"
        "2. Find c.\nHint: use d.\nLevel: 4, Credit: 70\n")
same_line = "1. Find a.\nHint: use b. Level: 2, Credit: 30\n2. Find c.\nHint: use d. Level: 4, Credit: 70"
missing = "1. Find a.\nHint: use b.\nLevel: 2, Credit: 30\n2. Find c.\nHint: use d.\n"
for name, t in (('good', good), ('same_line', same_line), ('missing', missing)):
    try:
        print(name, parse_joint(t))
    except ValueError as e:
        print(name, 'ValueError', e)
