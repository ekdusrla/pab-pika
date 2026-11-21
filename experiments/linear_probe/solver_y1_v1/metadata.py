"""Allocated y1 CPU-only CLI-help/config diagnostics, no CUDA or NVML calls."""
import os
from .plan import OUT,POLICY,require_compute
from .identity import command,gres_evidence
from .gpu_mapping import supported_fields
from ..common import seal,save_new
from ..resources import parse_nodes


def run():
    require_compute()
    if os.environ['SLURMD_NODENAME']!=POLICY['node']:raise ValueError('Submit CPU metadata task with --nodelist=moana-y1')
    doc={'passed':False,'kind':'cpu_only_cli_help_gres_metadata','no_cuda_or_nvml_calls':True,
         'node':os.environ['SLURMD_NODENAME'],'query_commands':[]}
    try:
        raw=command(['scontrol','show','node',POLICY['node'],'-o'],doc['query_commands'])['stdout']
        node=parse_nodes(raw)[0];doc['node_config']=node
        help_record=command(['nvidia-smi','--help-query-gpu'],doc['query_commands'],required=False)
        fields=supported_fields(help_record['stdout']) if help_record['returncode']==0 else set()
        doc['field_support']={f:(f in fields if help_record['returncode']==0 else None)
                              for f in ('index','uuid','minor_number','pci.bus_id')}
        doc['gres_evidence']=gres_evidence(node['name'],node['gpus'])
        doc['passed']=help_record['returncode']==0 and any(r['passed'] for r in doc['gres_evidence'])
        doc['scope']='Help/host configuration only; not actual GPU identity, VRAM, model loading or Smoke qualification'
    except Exception as exc:doc['error']={'type':type(exc).__name__,'message':str(exc)}
    output=OUT/f'metadata_{os.environ["SLURM_JOB_ID"]}.json';save_new(output,seal(doc))
    if not doc['passed']:raise RuntimeError('BLOCKED: help/GRES diagnostics incomplete; inspect '+str(output))
    print('Y1 CPU METADATA PASS (not GPU Smoke PASS): '+str(output),flush=True)


if __name__=='__main__':run()
