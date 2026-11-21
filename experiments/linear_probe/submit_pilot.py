"""PRINT ONLY: resource-scaled sbatch commands. Never contacts SLURM."""
import argparse
import shlex
import re
import subprocess
from pathlib import Path
from .resources import parse_nodes, common_limits
from .common import output_path


def command(stage, run, cpus_per_gpu, ram_gb_per_gpu, target=None, previous=None, dependency=None):
    output_path(run)
    if cpus_per_gpu < 1 or ram_gb_per_gpu < 1:
        raise ValueError('Explicit positive site-policy CPU/RAM per GPU values required')
    if stage not in ('length','capture','teacher','solver'):
        raise ValueError('Unknown stage')
    gpus = 4 if stage == 'solver' else 1
    # Values are operator-supplied approved policy, not inferred from old sbatch files.
    exports = ['ALL',f'STAGE={stage}',f'RUN={run}',f'CPUS_PER_GPU={cpus_per_gpu}',f'RAM_GB_PER_GPU={ram_gb_per_gpu}']
    if any(',' in str(value) or '\n' in str(value) for value in (run,previous or '')):
        raise ValueError('SLURM export values cannot contain commas/newlines')
    if stage == 'length':
        if target not in (1024,2048,4096):
            raise ValueError('length target required')
        if target != 1024 and not previous:
            raise ValueError('Previous length report required')
        exports.append(f'TARGET={target}')
        if previous:
            exports.append(f'PREVIOUS={previous}')
    result = ['sbatch','--parsable','--nodes=1','--ntasks=1','--time=02:00:00',
        f'--gres=gpu:normal:{gpus}',f'--cpus-per-task={cpus_per_gpu*gpus}',
        f'--mem={ram_gb_per_gpu*gpus}G',f'--export={",".join(exports)}']
    if dependency:
        if not re.fullmatch(r'\d+(?::\d+)*',dependency):
            raise ValueError('Dependency must be numeric job IDs separated by colons')
        result.append(f'--dependency=afterok:{dependency}')
    result.append('experiments/linear_probe/pilot_gpu.sbatch')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('length','capture','teacher','solver'))
    p.add_argument('--run',required=True)
    p.add_argument('--nodes-file',type=Path,required=True,help='Fresh scontrol show nodes -o snapshot')
    p.add_argument('--cpus-per-gpu',type=int,help='Optional value no larger than all eligible node ratios')
    p.add_argument('--ram-gb-per-gpu',type=int,help='Optional value no larger than headroom-adjusted node ratios')
    p.add_argument('--constraint',help='One actual node feature, if cluster advertises an RTX 3090 feature')
    p.add_argument('--exclude',help='SLURM hostlist excluded from BOTH resource calculation and scheduling')
    p.add_argument('--split',choices=('train','validation','evaluation'),help='Optional independent capture/teacher shard')
    p.add_argument('--target',type=int);p.add_argument('--previous');p.add_argument('--dependency')
    a=p.parse_args(argv)
    nodes=parse_nodes(a.nodes_file.read_text())
    if a.exclude:
        # Slurm expands bracket expressions; never run a shell or infer GPU models from node names.
        excluded=set(subprocess.check_output(['scontrol','show','hostnames',a.exclude],text=True).splitlines())
        nodes=[node for node in nodes if node['name'] not in excluded]
    caps=common_limits(nodes,4 if a.stage=='solver' else 1,constraint=a.constraint)
    cpus=a.cpus_per_gpu if a.cpus_per_gpu is not None else caps['cpus_per_gpu']
    ram=a.ram_gb_per_gpu if a.ram_gb_per_gpu is not None else caps['ram_gb_per_gpu']
    if cpus>caps['cpus_per_gpu'] or ram>caps['ram_gb_per_gpu']:
        raise ValueError('Requested per-GPU values exceed the eligible nodes policy; refresh snapshot')
    cmd=command(a.stage,a.run,cpus,ram,a.target,a.previous,a.dependency)
    if a.split:
        if a.stage not in ('capture','teacher'):raise ValueError('--split is only for capture/teacher')
        position=next(i for i,value in enumerate(cmd) if value.startswith('--export='))
        cmd[position]+=f',SPLITS={a.split}'
    if a.constraint:cmd.insert(-1,f'--constraint={a.constraint}')
    if a.exclude:cmd.insert(-1,f'--exclude={a.exclude}')
    print(shlex.join(cmd))


if __name__=='__main__':
    main()
