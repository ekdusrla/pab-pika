"""Score frozen responses in memory and save an immutable regression ledger."""
from collections import Counter

from .audit import LEGACY, OUT, read, verify
from .evaluator import evaluate_answer, evaluator_info, rescue
from ..common import save_new, seal
from math_answer_eval import evaluate_answer as legacy, original_parser


def compare(rows):
    counts = Counter()
    changes, stored_mismatches = [], []
    for ident, content, answer, finish, status, stored in rows:
        old = legacy(content, answer, finish, status)
        revised = rescue(old, content, answer, finish, status)
        counts[f'{old["status"]}->{revised["status"]}'] += 1
        if stored is not None and old['status'] != stored:
            stored_mismatches.append(dict(id=ident, stored=stored, recomputed=old['status']))
        if old != revised:
            changes.append(dict(id=ident, before=old, after=revised))
    return dict(counts=dict(counts), changes=changes, legacy_stored_mismatches=stored_mismatches)


def regression():
    candidate = verify(read(OUT/'candidate_manifest.json'))
    control = {r['id']: r['answer'] for r in candidate['rows']}
    boxed = lambda answer: 'Answer: $\\boxed{' + answer + '}$.'
    reference = compare((i, boxed(a), a, 'stop', 'success', None) for i,a in control.items())
    parser = original_parser()
    diagnostics = []
    for i in candidate['reference_self_check_failures']:
        a = control[i]
        diagnostics.append(dict(id=i, reference=a,
                           legacy_normalized=parser.strip_string(parser.extract_answer(boxed(a), 'math'))))
    probes = (
        ('102', 'ab+2a+5b+10', True), ('102', '5(b+2)', False),
        ('97', r'\mathrm{East}', True), ('97', r'\text{west}', False),
        ('96', r'\{1-\sqrt{19},1+\sqrt{19}\}', True), ('96', r'1+\sqrt{19}', False),
        ('284', r'3\pm\sqrt{8}', True), ('284', r'3\pm\sqrt{2}', False),
        ('422', r'\{-2,1+\sqrt{5},1-\sqrt{5}\}', True), ('422', r'\{1\pm\sqrt{5}\}', False),
        ('257', r'\frac{1728}{2}\text{in}^{2}', True), ('257', r'864\text{cm}^{2}', False),
        ('467', r'\frac{30}{2}\mathrm{cm}^{2}', True), ('467', r'15\mathrm{cm}^{3}', False),
    )
    cases = [dict(id=i, prediction=p, expected=ok, result=evaluate_answer(boxed(p), control[i]))
             for i,p,ok in probes]
    cases += [dict(id='97', prediction='truncated', expected=False,
                   result=evaluate_answer(boxed(control['97']), control['97'], 'length')),
              dict(id='97', prediction='thinking only', expected=False,
                   result=evaluate_answer('<think>'+boxed(control['97']), control['97']))]
    raw = verify(read(OUT.parent/'solver_y1_v1/responses_solver_221051_main.json'))
    def pilot_rows():
        for r in raw['records']:
            s = r['stage']
            yield (r['id']+'|'+r['arm'], s['result']['content'], r['answer'],
                   s['result']['finish_reason'], s['status'], s['parsed']['answer_evaluation']['status'])
    pilot = compare(pilot_rows())
    history = {}
    for name in ('math500_llama31_8b_v1','math500_original_v1'):
        old = read(LEGACY/'runs'/name/'solver.json')
        def old_rows():
            for r in old['comparisons']:
                i = str(r['id'])
                assert r['answer'] == control[i]
                for arm in ('vanilla','weighted_pab'):
                    item = r[arm]
                    yield (i+'|'+arm, item['content'], control[i], item['finish_reason'],
                           item['status'], item['evaluation']['status'])
        history[name] = compare(old_rows())
    return dict(evaluator=evaluator_info(), candidate_fingerprint=candidate['fingerprint'],
                reference=reference, diagnostics=diagnostics, typed_cases=cases,
                pilot=pilot, historical_math500=history)


if __name__ == '__main__':
    result = regression()
    save_new(OUT/'final_scoring_regression.json', seal(result))
    print({k:v['counts'] for k,v in [('reference',result['reference']), ('pilot',result['pilot']),
                                    *result['historical_math500'].items()]})
    print('typed failures', [x for x in result['typed_cases'] if (x['result']['status']=='correct')!=x['expected']])
    print('stored mismatches', {k:len(v['legacy_stored_mismatches']) for k,v in
          [('pilot',result['pilot']),*result['historical_math500'].items()]})
