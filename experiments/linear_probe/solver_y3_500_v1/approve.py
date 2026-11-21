"""Publish the explicitly authorized MATH-500 re-evaluation after CPU qualification."""
from pathlib import Path
from tempfile import TemporaryDirectory

from .audit import OUT, probe_metadata
from .evaluator import evaluator_info
from .runner import PROTOCOL, code_hashes, experiment
from ..common import file_hash, read, save_new, seal, verify
from ..resources import validate_allocation
from experiment_state import atomic_save

CANDIDATE_SHA = '9c99be5b4c93e63ac1bcad2b3d503281b6bc2df9a99a211e9378e17c7282218c'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def approve():
    path = OUT/'candidate_manifest.json'
    require(file_hash(path) == CANDIDATE_SHA, 'Candidate bytes changed')
    candidate = verify(read(path))
    conf = read(PROTOCOL)
    require(conf['accuracy_margin_pp'] == 3 and conf['solver_seeds'] == [42], 'Margin or seed changed')
    require((conf['node'], conf['gpu_count'], conf['tensor_parallel'], conf['dtype'],
             conf['cpus'], conf['ram_gib'], conf['max_wall']) ==
            ('moana-y3', 4, 4, 'float16', 16, 64, '6-00:00:00'), 'GPU configuration changed')
    scheduler = read(OUT/'final_y3_scheduler.json')
    require(conf['partition'] == scheduler['partition'] and conf['max_wall'] == scheduler['qos_max_wall']
            and conf['account'] in scheduler['partition_accounts'] and conf['qos'] == scheduler['qos'],
            'Allocation disagrees with scheduler snapshot')
    validate_allocation(scheduler, conf['gpu_count'], conf['cpus'], conf['ram_gib']*1024)
    require(scheduler['gpus']-scheduler['gpu_allocated'] >= conf['gpu_count'], 'Snapshot shows too few available GPUs')
    regression = verify(read(OUT/'final_scoring_regression.json'))
    require(regression['evaluator'] == evaluator_info() and
            regression['candidate_fingerprint'] == candidate['fingerprint'], 'Regression inputs changed')
    ref = regression['reference']
    cases = regression['typed_cases']
    require(ref['counts'] == {'correct->correct':493, 'incorrect->correct':6, 'undetermined->correct':1}
            and {r['id'] for r in ref['changes']} == set(candidate['reference_self_check_failures']),
            'MATH-500 reference regression failed')
    require(len(cases) == 16 and all((x['result']['status']=='correct') == x['expected'] for x in cases),
            'Typed positive/negative regression failed')
    require(regression['pilot']['counts'] == {'correct->correct':25} and
            not regression['pilot']['legacy_stored_mismatches'], 'Frozen Pilot regression failed')
    history = regression['historical_math500']
    for name, result in history.items():
        require(sum(result['counts'].values()) == 1000 and not result['legacy_stored_mismatches']
                and all(change['after']['reason'].startswith('typed_boxed_rescue:')
                        for change in result['changes']), 'Historical response regression failed: '+name)
    power = read(OUT/'final_power.json')
    require(power['n'] == 500 and power['margin_pp'] == 3 and len(power['scenarios']) == 20,
            'Design power audit missing')
    require('Ran 26 tests' in (OUT/'final_cpu_validation.log').read_text()
            and (OUT/'final_cpu_validation.log').read_text().rstrip().endswith('OK'), 'CPU preflight failed')
    meta = probe_metadata()
    require(meta['best_layer_idx'] == 18 and meta['alpha'] == 100 and not meta['evaluation_labels_accessed']
            and not meta['validation_used_for_fit'], 'Unexpected Probe selection metadata')
    require(all(not x['matches'] for x in candidate['screens'].values())
            and candidate['previously_evaluated_count'] == 500, 'Benchmark provenance inconsistent')
    report_files = ('final_scoring_regression.json','final_power.json','final_y3_scheduler.json',
                    'final_cpu_validation.log')
    evidence = {str(OUT/name):file_hash(OUT/name) for name in report_files}
    approval = seal({**{k:v for k,v in candidate.items() if k not in ('fingerprint','blockers')},
                     'kind':'solver_y3_500_approved_v1','approved':True,
                     'evaluation_scope':'previously_used_math500_benchmark_reevaluation',
                     'confirmatory_claim_allowed':False, 'candidate_sha256':CANDIDATE_SHA,
                     'candidate_fingerprint':candidate['fingerprint'],
                     'protocol_sha256':file_hash(PROTOCOL),'code_sha256':code_hashes(),
                     'legacy_evaluator':candidate['evaluator'],'evaluator':regression['evaluator'],
                     'approved_margin_pp':3,'gpu_qualification':'deferred_to_allocation',
                     'authorization':'user approved unchanged MATH-500 benchmark re-evaluation',
                     'qualification_sha256':evidence,
                     'input_sha256':{**candidate['input_sha256'],str(path):CANDIDATE_SHA,**evidence}})
    with TemporaryDirectory() as tmp:
        preview = Path(tmp)/'approval.json'
        atomic_save(preview, approval)
        checked, _ = experiment(preview)
        require(checked['rows'] == candidate['rows'], 'Approval changed candidate records')
    target = OUT/'approved_manifest.json'
    save_new(target, approval)
    print('Verified approval SHA256:', file_hash(target))


if __name__ == '__main__':
    approve()
