"""Isolated qualification for the approved y3 execution configuration."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from importlib.metadata import version

from .cuda_diagnostics import device_access
from .gres_identity import load_map, prove_ordinals
from ..common import FEATURE, LEVEL_WORDS, allocate_words, credit_contract, digest, file_hash, read, save_new, seal, verify
from ..learning import predict
from ..resources import parse_nodes, validate_allocation
from ..solver_y1_v1.gpu_mapping import check_visible, pci_id, requested_ids, resolve_mapping
from ..solver_y1_v1.identity import command, discover
from ..solver_y3_500_v1 import hardware
from ..solver_y3_500_v1.audit import OUT, PILOT, PROBE, probe_metadata
from ..solver_y3_500_v1.evaluator import evaluator_info, validate_solver
from ..solver_y3_500_v1.runner import capture_problem, experiment, journal, solver_payload
from experiment_state import run_stage, termination_handler

ROOT = OUT / 'smoke'
PROBLEM_ID = '336'
PROFILE = hardware.PROFILE
EVIDENCE = ('preflight.json', 'capture.json', 'probe.json', 'server.json', 'request.json', 'memory.json')


def compare_sets(assigned, physical, driver, torch):
    cli_set, allocated = set(physical), set(assigned)
    sets = {}
    for name, result in (('driver',driver),('torch',torch)):
        if not result['passed']:
            sets[name] = None
            continue
        ids = [d['uuid'] for d in result['devices']]
        seen = set(ids)
        sets[name] = dict(missing_from_runtime=sorted(allocated-seen), unexpected_in_runtime=sorted(seen-allocated),
                          absent_from_nvidia_smi=sorted(seen-cli_set), nvidia_smi_only=sorted(cli_set-seen))
    return sets


def validate_driver(record):
    driver = record['driver_probe']
    torch = record['torch_probe']
    physical = record['physical_identity']['reconciled']
    check_visible(record['allocation_mapping'], [d['uuid'] for d in driver['devices']])
    if any(pci_id(d['pci']) != pci_id(physical[d['uuid']]['pci']) for d in driver['devices']):
        raise ValueError('Driver PCI mapping differs from NVML/proc')
    if [d['uuid'] for d in driver['devices']] != [d['uuid'] for d in torch['devices']]:
        raise ValueError('CUDA Driver and PyTorch local ordinal maps differ')


def preflight(path, conf):
    env = {k: os.environ.get(k) for k in ('SLURMD_NODENAME', 'SLURM_JOB_NUM_NODES', 'SLURM_JOB_GPUS',
                                          'SLURM_STEP_GPUS', 'CUDA_DEVICE_ORDER', 'CUDA_VISIBLE_DEVICES', 'SLURM_CPUS_PER_TASK', 'SLURM_MEM_PER_NODE')}
    record = dict(passed=False, environment=env, query_commands=[])
    try:
        if env['SLURMD_NODENAME'] != conf['node'] or env['SLURM_JOB_NUM_NODES'] != '1':
            raise ValueError('Require a single allocation on the selected node')
        cpus, memory = int(env['SLURM_CPUS_PER_TASK']), int(env['SLURM_MEM_PER_NODE'])
        if (cpus, memory) != (conf['cpus'], conf['ram_gib']*1024):
            raise ValueError('CPU or memory allocation differs from approval')
        raw = subprocess.check_output(['scontrol','show','node',conf['node'],'-o'],text=True)
        node, = parse_nodes(raw)
        record['node'] = node
        record['per_gpu_limits'] = validate_allocation(node,4,cpus,memory)
        expected = verify(read(PILOT))['legacy_solver_config']['generation_packages']
        record['packages'] = {p:version(p) for p in expected}
        if record['packages'] != expected:
            raise ValueError('GPU generation environment differs from Pilot')
        inventory = command(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free',
                             '--format=csv,noheader'],record['query_commands'])['stdout']
        record['inventory'] = inventory
        ids = requested_ids(env['SLURM_JOB_GPUS'])
        annotation = load_map(conf['node'],node['gpus']) if ids[0].isdecimal() else None
        if ids[0].isdecimal() and conf['node']=='moana-y5' and annotation is None:
            raise ValueError('y5 numeric GRES IDs need a root-controlled ordinal map; run one-GPU diagnostics first')
        if annotation is not None:
            # The original collector may skip its canonical numeric check only once
            # Slurm IDs are resolved via the root-controlled attestation.
            translated = [annotation['ordinals'][key]['uuid'] for key in ids]
            record['environment'] = {**env,'SLURM_JOB_GPUS':','.join(translated)}
        try:
            discover(record,inventory,node['gpus'],conf['node'])
        finally:
            record['environment'] = env
        physical = record['physical_identity']['reconciled']
        if annotation is None:
            mapping = resolve_mapping(env['SLURM_JOB_GPUS'],inventory,record['identity_inventory'])
        else:
            assigned = prove_ordinals(ids,annotation,physical)
            mapping = dict(assigned_ids=ids,assigned_uuids=sorted(assigned),
                           method='administrator-attested-ordinal-map')
            record['ordinal_map'] = annotation
        record['allocation_mapping'] = mapping
        assigned = mapping['assigned_uuids']
        if env['SLURM_STEP_GPUS']:
            step_ids = requested_ids(env['SLURM_STEP_GPUS'])
            if annotation is not None and step_ids[0].isdecimal():
                step = dict(assigned_ids=step_ids,assigned_uuids=sorted(prove_ordinals(step_ids,annotation,physical)))
            else:
                step = resolve_mapping(env['SLURM_STEP_GPUS'],inventory,record['identity_inventory'])
            record['step_mapping'] = step
            if step['assigned_uuids'] != assigned:
                raise ValueError('The SLURM step and job track different GPU UUIDs')
        record['device_access'] = device_access(['/dev/nvidiactl', '/dev/nvidia-uvm',
            *[f"/dev/nvidia{physical[u]['minor']}" for u in assigned]])
        for role in ('driver','torch'):
            try:
                child = subprocess.run([sys.executable,'-B','-m',
                                       __package__+'.cuda_diagnostics',role],
                                       capture_output=True,text=True,timeout=90)
                evidence = json.loads(child.stdout) if child.returncode == 0 else dict(passed=False)
                evidence.update(returncode=child.returncode, stderr=child.stderr)
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                evidence = dict(passed=False,error=str(exc))
            record[role+'_probe'] = evidence
        record['gpu_set_comparison'] = compare_sets(assigned, physical, record['driver_probe'], record['torch_probe'])
        if not record['driver_probe']['passed'] or not record['torch_probe']['passed']:
            raise ValueError('CUDA initialization/enumeration failed; see driver_probe, torch_probe and device_access')
        validate_driver(record)
        record['mapping'] = check_visible(mapping,[d['uuid'] for d in record['torch_probe']['devices']])
        record['devices'] = record['torch_probe']['devices']
        for d in record['devices']:
            d['pci'] = physical[d['uuid']]['pci']
            if (d['name'] != PROFILE['gpu_name'] or d['free'] < PROFILE['min_free_gib']*2**30
                    or not PROFILE['min_total_gib']*2**30 <= d['total'] <= PROFILE['max_total_gib']*2**30):
                raise ValueError('Expected RTX 3090 and free VRAM headroom')
        record['passed'] = True
    except Exception as exc:
        record['error'] = str(exc)
        raise
    finally:
        save_new(path,seal(record))


def capture(path, manifest, conf, row):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ..safety import MemoryGuard, require_device

    require_device(torch)
    snapshot = manifest['model_identity']['planner']['snapshot']
    tokenizer = AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    frozen = verify(read(OUT.parent/'pilot_v1/evaluation_recovered.json'))
    if digest(tokenizer.chat_template) != frozen['sessions'][0]['chat_template_sha256']:
        raise ValueError('Planner tokenizer differs from Frozen Pilot')
    guard = MemoryGuard(torch.cuda,limit_gib=21.)
    with termination_handler():
        model = AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,
                    torch_dtype=torch.float16,attn_implementation='eager').to('cuda').eval()
        torch.cuda.synchronize()
        guard.check('smoke_model_load')
        layer = probe_metadata()['best_layer_idx']
        j = journal(path,manifest,conf,row,'smoke_capture')
        capture_problem(j,row,model,tokenizer,guard,layer)
        state = j.record['stage']
        if state['status'] != 'success' or not state['result']['same_generated_ids']:
            raise ValueError('Planner control/capture failed or produced different token IDs')
        save_new(path/'capture.json',seal(dict(stage=state,layer=layer,feature_definition=FEATURE)))


def probe(path, manifest, row):
    import joblib

    source = verify(read(path/'capture.json'))
    cap = source['stage']['result']
    meta = probe_metadata()
    expected = {k:meta['runtime'][k] for k in ('scikit-learn','numpy','joblib')}
    if {k:version(k) for k in expected} != expected:
        raise ValueError('Probe CPU runtime differs from frozen artifact')
    pipeline = joblib.load(PROBE/'probe.joblib')
    if source['layer'] != meta['best_layer_idx'] or pipeline.n_features_in_ != meta['feature_dimension']:
        raise ValueError('Frozen feature layer or dimension mismatch')
    features = next(a['selected_features'] for a in source['stage']['attempts']
                    if a['role']=='capture' and a['status']=='success')
    prediction = predict(pipeline,{'capture':{'features':features}},source['layer'])
    credits = credit_contract(prediction['credits'],len(cap['steps']))
    total = LEVEL_WORDS[row['level']]
    budgets = allocate_words(credits,total,'constant')
    if sum(budgets) != total:
        raise ValueError('Budget total mismatch')
    save_new(path/'probe.json',seal(dict(manifest=manifest['fingerprint'], feature_definition=FEATURE,
             steps=cap['steps'],steps_fingerprint=digest(cap['steps']),layer=source['layer'],
             probe_fingerprint=meta['fingerprint'],prediction=prediction,budgets=budgets,total_words=total)))


def request(path, manifest, conf, row, port, pid):
    from openai import OpenAI

    if manifest['evaluator'] != evaluator_info():
        raise ValueError('Solver evaluator differs from approval')
    plan = verify(read(path/'probe.json'))
    payload = solver_payload(row,plan['steps'],'ridge',plan['budgets'],conf['solver_seeds'][0])
    j = journal(path,manifest,conf,row,'smoke_ridge')
    model = manifest['model_identity']['solver']['model']

    def qualify(result):
        parsed = validate_solver(result,row)
        if result['finish_reason'] != 'stop' or parsed['answer_evaluation']['status'] != 'correct':
            raise ValueError('Smoke requires a complete correct answer')
        return parsed

    with OpenAI(base_url=f'http://127.0.0.1:{port}/v1',api_key='DUMMY',timeout=360,max_retries=0) as client, termination_handler():
        if model not in {m.id for m in client.models.list().data}:
            raise ValueError('Wrong Solver model')
        if not run_stage(j.record['stage'],payload,client,j.save,qualify,retries=0,server_pid=pid):
            raise ValueError(f'Solver smoke failed; see {j.path}')
    save_new(path/'request.json',seal(dict(problem=row['id'],arm='ridge',stage=j.record['stage'])))


def smoke_hashes():
    folder = Path(__file__).parent
    return {name:file_hash(folder/name) for name in ('smoke.py','smoke.sbatch','cuda_diagnostics.py','gres_identity.py','main.sbatch','main_launch.sh')}


def qualify_evidence(path, manifest, conf, row):
    records = {name:verify(read(path/name)) for name in EVIDENCE}
    pre, mem = records['preflight.json'],records['memory.json']
    uuids = pre['mapping']['assigned_uuids']
    if pre['environment']['SLURMD_NODENAME'] != conf['node']:
        raise ValueError('Smoke evidence was captured on a different node')
    if (not pre['passed'] or not pre['mapping']['passed'] or len(uuids)!=4 or not mem['passed']
            or mem['samples']<=0 or set(mem['peaks_mib'])!=set(uuids)
            or any(not 0 < x <= PROFILE['used_limit_gib']*1024 for x in mem['peaks_mib'].values())):
        raise ValueError('GPU mapping or VRAM qualification failed')
    cap, probe_plan = records['capture.json'],records['probe.json']
    req, server = records['request.json'],records['server.json']
    args = server['argv']
    cap_result = cap['stage']['result']
    credits = credit_contract(probe_plan['prediction']['credits'],len(cap_result['steps']))
    if probe_plan['budgets'] != allocate_words(credits,LEVEL_WORDS[row['level']],'constant'):
        raise ValueError('Recorded Ridge budgets cannot be reproduced')
    if (cap['stage']['status']!='success' or not cap_result['same_generated_ids']
            or cap['feature_definition']!=FEATURE or probe_plan['feature_definition']!=FEATURE
            or cap['layer'] != manifest['probe_settings']['layer'] or cap['layer'] != probe_plan['layer']
            or probe_plan['steps'] != cap_result['steps']
            or probe_plan['steps_fingerprint'] != digest(cap_result['steps'])
            or probe_plan['probe_fingerprint'] != manifest['probe_fingerprint']
            or sum(probe_plan['budgets']) != LEVEL_WORDS[row['level']]):
        raise ValueError('Planner or Ridge qualification failed')
    if (server['manifest'] != manifest['fingerprint']
            or args[args.index('--tensor-parallel-size')+1] != '4'
            or args[args.index('--dtype')+1] != 'float16'
            or args[args.index('--model')+1] != manifest['model_identity']['solver']['snapshot']):
        raise ValueError('Solver startup record differs from approval')
    stage = req['stage']
    if (probe_plan['manifest']!=manifest['fingerprint'] or req['problem']!=row['id'] or req['arm']!='ridge'
            or stage['status']!='success' or len(stage['attempts'])!=1
            or stage['request']!=solver_payload(row,probe_plan['steps'],'ridge',probe_plan['budgets'],conf['solver_seeds'][0])
            or stage['result']['finish_reason']!='stop'
            or stage['parsed']['answer_evaluation']['status']!='correct'):
        raise ValueError('Solver request qualification failed')
    return {name:file_hash(path/name) for name in EVIDENCE}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action',choices=('preflight','capture','probe','serve','monitor','request','complete','check','check-main','main-preflight'))
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--job')
    p.add_argument('--node',choices=('moana-y3','moana-y5'),default=os.environ.get('PAB_SMOKE_NODE'))
    p.add_argument('--port',type=int)
    p.add_argument('--pid',type=int)
    a = p.parse_args()
    doc,conf = experiment(a.manifest)
    if a.node:
        if a.action in ('check-main','main-preflight') and a.node != conf['node']:
            p.error('A Smoke on another node cannot qualify the approved main job')
        conf = {**conf,'node':a.node}

    job = a.job if a.action in ('check','check-main') else os.environ.get('SLURM_JOB_ID')
    if not job or not job.isdecimal():
        p.error('A SLURM numeric job ID is required')
    path = ROOT / job
    row, = (r for r in doc['rows'] if r['id']==PROBLEM_ID)
    if a.action in ('check','check-main'):
        complete = verify(read(path/'complete.json'))
        expected = seal(dict(passed=True,job=job,manifest=doc['fingerprint'],node=conf['node'],problem=row['id'],
                             evidence=qualify_evidence(path,doc,conf,row),smoke_sha256=smoke_hashes()))
        if complete != expected:
            raise ValueError('Smoke completion evidence differs')
        print('Y3 SMOKE PASS',job)
        return
    if os.environ.get('SLURMD_NODENAME') != conf['node']:
        p.error('GPU stages require the approved node allocation')
    if a.action=='main-preflight':
        preflight(OUT/f'preflight_{job}.json',conf)
        return
    if a.action=='preflight':
        path.mkdir(exist_ok=False)
        preflight(path/'preflight.json',conf)
        return
    if not verify(read(path/'preflight.json'))['passed']:
        raise ValueError('GPU preflight did not pass')
    if a.action=='capture':
        capture(path,doc,conf,row)
    elif a.action=='probe':
        probe(path,doc,row)
    elif a.action=='serve':
        hardware.serve(doc,'solver',a.port,path/'server.json')
    elif a.action=='monitor':
        hardware.monitor(path/'preflight.json',a.pid,path/'memory.json')
    elif a.action=='request':
        request(path,doc,conf,row,a.port,a.pid)
    else:
        files = qualify_evidence(path,doc,conf,row)
        save_new(path/'complete.json',seal(dict(passed=True,job=job,manifest=doc['fingerprint'],
                 node=conf['node'],problem=row['id'],evidence=files,smoke_sha256=smoke_hashes())))
        print('Y3 SMOKE PASS',job)


if __name__=='__main__':
    main()
