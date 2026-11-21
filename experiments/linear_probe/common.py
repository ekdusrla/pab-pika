"""CPU-only contracts, provenance and fail-closed dataset preparation."""
import hashlib
import importlib.metadata
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / 'experiments/llama_qwq'
sys.path.insert(0, str(LEGACY))
from experiment_state import digest, file_hash, original_definitions, read_manifest
from run_planner import parse_steps, parse_credits, credit_schema
from run_solver_compare import allocate_words, LEVEL_WORDS

MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
REVISION = '0e9e39f249a16976918f6564b8830bc894c89659'
FEATURE = 'last-fully-contained-token-v1;post-layer;original-numbered-step;no-marker'
SCHEMA = 1


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def output_path(path):
    """All mutations stay in the NEW experiment's runs tree, including symlinks."""
    path = Path(path).resolve()
    root = Path(__file__).parent.resolve() / 'runs'
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Outputs must be under experiments/linear_probe/runs/')
    return path


def save_new(path, obj):
    path = output_path(path)
    payload = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(payload)


def problem_hash(text):
    # Conservative exact normalized overlap check, not semantic equivalence detection.
    text = unicodedata.normalize('NFKC', text).casefold()
    return hashlib.sha256(re.sub(r'\s+', '', text).encode()).hexdigest()


def seal(body):
    return {**body, 'fingerprint': digest(body)}


def verify(doc):
    if doc.get('fingerprint') != digest({k: v for k, v in doc.items() if k != 'fingerprint'}):
        raise ValueError('Fingerprint mismatch')
    return doc


def runtime():
    result = {'python': sys.version, 'engine': 'transformers-manual-kv-cache'}
    for name in ('torch', 'transformers', 'numpy', 'scikit-learn', 'joblib', 'vllm', 'openai'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    result['source_sha256'] = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    result['baseline_source_sha256'] = {str(p.relative_to(ROOT)): file_hash(p) for p in (
        ROOT / 'dataset/break_down_question.py', LEGACY / 'run_planner.py',
        LEGACY / 'run_solver_compare.py', LEGACY / 'experiment_state.py')}
    return result


def prompt_fingerprint():
    return digest(original_definitions()[0]['break_down'])


def decomp_messages(row):
    template = original_definitions()[0]['break_down']
    values = dict(domain='math', problem=row['query'], level=row['level'])
    return [{'role': role, 'content': template[role].format(**values)} for role in ('system', 'user')]


def assessment_messages(row, steps, benchmarks):
    template = original_definitions()[0]['evaluate_question_level_batch']
    examples = '\n'.join(f"\nLevel {r['level']} Example:\nQ: {r['query']}\nA: {r['solution']}\n" for r in benchmarks)
    values = dict(domain='math', problem=row['query'], steps=steps, benchmarks=examples)
    return [{'role': role, 'content': template[role].format(**values)} for role in ('system', 'user')]


def credit_contract(credits, n):
    if len(credits) != n or not 2 <= n <= 5 or any(type(x) is not int or x <= 0 for x in credits) or sum(credits) != 100:
        raise ValueError('Expected 2-5 positive integer credits summing to 100')
    return credits


def project_credits(raw):
    """Fixed rule: clamp at zero, reserve 1 per step, largest remainders; stable ties."""
    if not 2 <= len(raw) <= 5 or any(not math.isfinite(float(x)) for x in raw):
        raise ValueError('Nonfinite predictions or invalid step count')
    weights = [max(float(x), 0.0) for x in raw]
    scale = max(weights)
    weights = [w / scale for w in weights] if scale else [1.0] * len(raw)
    total = sum(weights)
    normalized = [100 * w / total for w in weights]
    quotas = [(100 - len(raw)) * w / total for w in weights]
    credits = [1 + math.floor(q) for q in quotas]
    order = sorted(range(len(raw)), key=lambda i: (-(quotas[i] - math.floor(quotas[i])), i))
    for i in order[:100 - sum(credits)]:
        credits[i] += 1
    return {'raw': [float(x) for x in raw], 'normalized': normalized,
            'credits': credit_contract(credits, len(raw)),
            'rule': 'clamp-zero-reserve-one-largest-remainder-v1',
            'clamped_count': sum(float(x) < 0 for x in raw), 'uniform_fallback': scale == 0}


def prepare(source, eval_manifest, validation_fraction=0.2, seed=42):
    """Source: external candidate records plus separate benchmark records and provenance."""
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be in (0,1)')
    if not source.get('provenance') or not isinstance(source.get('records'), list):
        raise ValueError('External source needs provenance and records')
    forbidden = {problem_hash(r['query']) for r in eval_manifest['records']}
    seen, ids = set(), set()
    def convert(items, require_solution=False):
        rows = []
        for item in items:
            query = item.get('query', item.get('problem'))
            identifier = str(item.get('id', ''))
            if not isinstance(query, str) or not query.strip() or not identifier or identifier in ids:
                raise ValueError('Missing/duplicate source ID or question')
            if type(item.get('level')) is not int or item['level'] not in range(1, 6):
                raise ValueError('Invalid level')
            h = problem_hash(query)
            if h in forbidden or h in seen:
                raise ValueError('Evaluation overlap or duplicate normalized question')
            if require_solution and not item.get('solution'):
                raise ValueError('Benchmark needs solution')
            seen.add(h); ids.add(identifier)
            rows.append({'id': identifier, 'query': query, 'level': item['level'],
                         'solution': item.get('solution', ''), 'problem_hash': h})
        return rows
    benchmarks = convert(source.get('benchmarks', []), True)
    if {r['level'] for r in benchmarks} != set(range(1, 6)):
        raise ValueError('External benchmarks must cover levels 1-5')
    rows = convert(source['records'])
    if len(rows) < 2:
        raise ValueError('Need at least two independent training/validation problems')
    # Content-based deterministic split; no sub-question can cross the split.
    rows.sort(key=lambda r: digest([seed, r['problem_hash']]))
    n = max(1, min(len(rows) - 1, round(len(rows) * validation_fraction)))
    evaluation = [{**r, 'problem_hash': problem_hash(r['query'])} for r in eval_manifest['records']]
    return seal({'schema_version': SCHEMA, 'kind': 'data_manifest',
                 'evaluation_manifest_fingerprint': eval_manifest['fingerprint'],
                 'source_provenance': source['provenance'], 'source_fingerprint': digest(source),
                 'split_seed': seed, 'validation_fraction': validation_fraction,
                 'overlap_policy': 'NFKC+casefold+remove-whitespace; no semantic dedup guarantee',
                 'benchmarks': benchmarks,
                 'splits': {'train': rows[n:], 'validation': rows[:n], 'evaluation': evaluation}})


def validate_data(doc):
    verify(doc)
    seen = set()
    for rows in [doc['benchmarks'], *doc['splits'].values()]:
        for row in rows:
            h = problem_hash(row['query'])
            if h != row['problem_hash'] or h in seen:
                raise ValueError('Split overlap or changed question')
            seen.add(h)
    return doc
