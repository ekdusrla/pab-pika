"""Allocation-only identity checks and single-server launch."""
import argparse
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys
import time

from .audit import OUT, PILOT
from .runner import experiment
from ..common import read, save_new, seal, verify
from ..resources import parse_nodes, validate_allocation
from ..solver_y1_v1.gpu_mapping import resolve_mapping, check_visible
from ..solver_y1_v1.identity import command, discover
from ..solver_y1_v1.plan import PROFILES
from experiment_state import atomic_save

PROFILE = PROFILES['solver']


def preflight(path, configuration):
    node = configuration['node']
    if os.environ.get('SLURMD_NODENAME') != node or int(os.environ.get('SLURM_JOB_NUM_NODES', '0')) != 1:
        raise ValueError('Require a single y3 allocation')
    cpus, memory = int(os.environ['SLURM_CPUS_PER_TASK']), int(os.environ['SLURM_MEM_PER_NODE'])
    if (cpus, memory) != (configuration['cpus'], configuration['ram_gib'] * 1024):
        raise ValueError('SLURM CPU/memory must match protocol')
    raw = subprocess.check_output(['scontrol', 'show', 'node', node, '-o'], text=True)
    host, = parse_nodes(raw)
    packages = verify(read(PILOT))['legacy_solver_config']['generation_packages']
    actual = {p: version(p) for p in packages}
    if actual != packages:
        raise ValueError(f'GPU runtime changed: {actual}')
    record = {'passed': False, 'node': host, 'packages': actual,
              'per_gpu_limits': validate_allocation(host, 4, cpus, memory), 'query_commands': []}
    try:
        inventory = command(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free',
                             '--format=csv,noheader'], record['query_commands'])['stdout']
        record['inventory'] = inventory
        discover(record, inventory, host['gpus'], node)
        mapping = resolve_mapping(os.environ.get('SLURM_JOB_GPUS', ''), inventory, record['identity_inventory'])
        import torch
        if torch.cuda.device_count() != 4:
            raise ValueError('Require four visible CUDA devices')
        record['cuda_version'] = torch.version.cuda
        record['devices'] = []
        for i in range(4):
            prop = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            record['devices'].append(dict(index=i, uuid=str(prop.uuid), name=prop.name, free=free, total=total))
            if (prop.name != PROFILE['gpu_name'] or free < PROFILE['min_free_gib']*2**30
                    or not PROFILE['min_total_gib']*2**30 <= total <= PROFILE['max_total_gib']*2**30):
                raise ValueError('Assigned RTX 3090 or free memory contract failed')
            torch.zeros(1, device=f'cuda:{i}')
            torch.cuda.synchronize(i)
        record['mapping'] = check_visible(mapping, [d['uuid'] for d in record['devices']])
        record['passed'] = True
    except Exception as exc:
        record['error'] = str(exc)
        raise
    finally:
        save_new(path, seal(record))


def serve(manifest, role, port, path):
    from vllm.engine.arg_utils import EngineArgs
    m = manifest['model_identity']['planner' if role == 'teacher' else 'solver']
    if EngineArgs(model=m['model']).enable_prefix_caching is not False:
        raise ValueError('Unexpected prefix cache default')
    cmd = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
           '--model', m['snapshot'], '--tokenizer', m['snapshot'], '--served-model-name', m['model'],
           '--tensor-parallel-size', '4', '--distributed-executor-backend', 'mp', '--dtype', 'float16',
           '--max-model-len', '8192' if role == 'teacher' else '12288', '--max-num-seqs', '1',
           '--gpu-memory-utilization', '.85', '--seed', '42', '--host', '127.0.0.1', '--port', str(port)]
    if role == 'teacher':
        cmd += ['--guided-decoding-backend', 'outlines']
    save_new(path, seal(dict(argv=cmd, manifest=manifest['fingerprint'], started_unix=time.time())))
    os.execv(sys.executable, cmd)


def monitor(preflight_path, pid, path):
    import signal
    devices = verify(read(preflight_path))['mapping']['assigned_uuids']
    state = dict(samples=0, peaks_mib=dict.fromkeys(devices, 0), passed=False)
    start = time.monotonic()
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while running:
            for uuid in devices:
                used = int(command(['nvidia-smi', '-i', uuid, '--query-gpu=memory.used',
                                    '--format=csv,noheader,nounits'])['stdout'].strip())
                state['peaks_mib'][uuid] = max(used, state['peaks_mib'][uuid])
                if used > PROFILE['used_limit_gib'] * 1024:
                    raise ValueError(f'{uuid} exceeded VRAM limit')
            state['samples'] += 1
            time.sleep(.5)
        state['passed'] = state['samples'] > 0
    except Exception as exc:
        state['error'] = str(exc)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        raise
    finally:
        state['wall_seconds'] = time.monotonic() - start
        atomic_save(path, seal(state))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('preflight', 'serve', 'monitor'))
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--role', choices=('teacher', 'solver'))
    p.add_argument('--port', type=int)
    p.add_argument('--pid', type=int)
    a = p.parse_args()
    doc, conf = experiment(a.manifest)
    job = os.environ['SLURM_JOB_ID']
    check = OUT / f'preflight_{job}.json'
    if a.action == 'preflight':
        preflight(check, conf)
    else:
        if not verify(read(check))['passed']:
            raise ValueError('Allocation did not pass preflight')
        if a.action == 'serve':
            serve(doc, a.role, a.port, OUT / f'server_{job}_{a.role}.json')
        else:
            monitor(check, a.pid, OUT / f'memory_{job}_{a.role}.json')
