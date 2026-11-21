"""Read-only prerequisite check for the dedicated 8B Solver experiment."""
import argparse
import json
from pathlib import Path

from experiment_state import read_manifest, SCHEMA_VERSION
from run_planner import parse_steps, parse_credits
from run_solver_compare import planner_rows

MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
REVISION = '0e9e39f249a16976918f6564b8830bc894c89659'


def validate(path, manifest_path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Planner checkpoint missing: {path}. Run the 8B Planner first; wait for 500/500 completion before submitting Solver. No server started.')
    manifest = read_manifest(manifest_path)
    source = json.loads(path.read_text())
    if source.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported Planner checkpoint schema')
    config = source.get('config', {})
    if config.get('model') != MODEL or config.get('revision') != REVISION:
        raise ValueError('Expected the pinned Llama-3.1-8B Planner model/revision')
    rows = source.get('stats', [])
    if len(rows) != 500 or len({r['id'] for r in rows}) != 500:
        raise ValueError('Planner checkpoint must contain exactly 500 unique IDs')
    plans = planner_rows(source, manifest)
    incomplete = [r['id'] for r in rows if r.get('status') != 'success' or any(
        r.get('stages', {}).get(k, {}).get('status') != 'success' for k in ('decomposition', 'assessment'))]
    if incomplete:
        raise ValueError(f'Planner incomplete: {500-len(incomplete)}/500; unfinished IDs: {incomplete}. Resume/recover Planner first; do not submit Solver yet.')
    for row in plans.values():
        stages = row['stages']
        steps = row['steps']; credits = row['credits']
        if not 2 <= len(steps) <= 5 or len(steps) != len(credits) or any(type(c) is not int or c <= 0 for c in credits) or sum(credits) != 100:
            raise ValueError(f"Invalid steps/credits for ID {row['id']}")
        if steps != stages['decomposition']['parsed'] or credits != stages['assessment']['parsed']:
            raise ValueError(f"Stage/projection mismatch for ID {row['id']}")
        if parse_steps(stages['decomposition']['result']['content']) != steps or parse_credits(stages['assessment']['result']['content'], len(steps)) != credits:
            raise ValueError(f"Raw response/projection mismatch for ID {row['id']}")
    return manifest['fingerprint']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planner', required=True)
    parser.add_argument('--manifest', default='experiments/llama_qwq/math500_manifest.json')
    args = parser.parse_args()
    try:
        fingerprint = validate(args.planner, args.manifest)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f'8B Planner prerequisite failed: {error}\n')
    print(f'8B Planner verified: 500/500; steps 2-5; positive credits sum 100; manifest {fingerprint}')
