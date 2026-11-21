"""Compute-allocation-only discovery with CPU-mockable command capture."""
import csv
import hashlib
import os
from pathlib import Path
import subprocess

from .gpu_mapping import supported_fields,pci_id,canonical_gres_minors,requested_ids


class CommandFailure(RuntimeError):
    def __init__(self,record):
        self.record=record
        super().__init__(f"Command failed rc={record.get('returncode')}: {record['argv']}; stderr={record.get('stderr','')}")


def command(argv,records=None,required=True,runner=subprocess.run):
    record={'argv':argv,'stdout':'','stderr':'','returncode':None}
    try:
        result=runner(argv,text=True,capture_output=True,timeout=15)
        record.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    except (subprocess.TimeoutExpired,OSError) as exc:
        def string(x):return x.decode(errors='replace') if isinstance(x,bytes) else (x or '')
        record.update(error_type=type(exc).__name__,message=str(exc),
                      stdout=string(getattr(exc,'stdout',None)),stderr=string(getattr(exc,'stderr',None)))
    if records is not None:records.append(record)
    if required and record['returncode']!=0:raise CommandFailure(record)
    return record


def proc_record(text,bus):
    fields=dict(line.split(':',1) for line in text.splitlines() if ':' in line)
    fields={k.strip():v.strip() for k,v in fields.items()}
    u=fields.get('GPU UUID','');minor=fields.get('Device Minor','')
    if not u.startswith('GPU-') or not minor.isdigit():raise ValueError('Missing /proc UUID or Device Minor')
    pci_id(bus)
    return {'uuid':u,'minor':minor,'pci':bus,'source':'proc_driver'}


def reconcile(uuids,nvml,proc,cli_pci):
    """At least one physical source per UUID; reject conflicts across all sources."""
    result={}
    for u in uuids:
        rows=[r for source in (nvml,proc) if (r:=source.get(u)) is not None]
        if not rows:raise ValueError('No NVML or /proc physical identity for '+u)
        first=rows[0]
        if first['uuid']!=u or not str(first['minor']).isdigit():raise ValueError('Invalid physical identity')
        for r in rows:
            if r['uuid']!=u or r['minor']!=first['minor'] or pci_id(r['pci'])!=pci_id(first['pci']):
                raise ValueError('NVML/proc physical identity conflict for '+u)
        if u in cli_pci and pci_id(cli_pci[u])!=pci_id(first['pci']):raise ValueError('CLI/NVML/proc PCI identity conflict')
        result[u]=first
    if len({r['minor'] for r in result.values()})!=len(uuids) or len({pci_id(r['pci']) for r in result.values()})!=len(uuids):
        raise ValueError('Duplicate physical minor/PCI identities')
    return result


def gres_evidence(node,count):
    candidates=[Path('/etc/slurm/gres.conf'),Path('/etc/slurm-llnl/gres.conf')]
    if os.environ.get('SLURM_CONF'):candidates.append(Path(os.environ['SLURM_CONF']).parent/'gres.conf')
    evidence=[]
    for p in dict.fromkeys(candidates):
        row={'path':str(p)};evidence.append(row)
        try:
            if p.stat().st_size>262144:raise ValueError('GRES config exceeds bounded read limit')
            text=p.read_text();row.update(text=text,sha256=hashlib.sha256(text.encode()).hexdigest())
            row.update(passed=True,minors=canonical_gres_minors(text,node,count))
        except Exception as exc:row.update(passed=False,error_type=type(exc).__name__,error=str(exc))
    return evidence


def discover(doc,inventory,node_count,node):
    records=doc.setdefault('query_commands',[])
    doc['identity_stage']='help_query_gpu'
    help_result=command(['nvidia-smi','--help-query-gpu'],records,required=False)
    fields=supported_fields(help_result['stdout']) if help_result['returncode']==0 else set()
    doc['supported_query_fields']=sorted(fields)
    doc['query_field_support']={f:(f in fields if help_result['returncode']==0 else None)
                                for f in ('index','uuid','minor_number','pci.bus_id')}
    rows=[[x.strip() for x in row] for row in csv.reader(inventory.splitlines()) if row]
    uuids=[row[1] for row in rows]
    doc['physical_identity']={'nvml':{},'proc':{},'cli_pci':{},'source_errors':[],'gres_evidence':[]}
    state=doc['physical_identity'];cli=state['cli_pci']
    doc['identity_stage']='individual_supported_fields'
    # Only fields explicitly advertised by this driver's help are probed individually.
    for field in ('index','uuid','pci.bus_id'):
        if field not in fields:continue
        if field=='pci.bus_id':
            for u in uuids:
                r=command(['nvidia-smi','-i',u,'--query-gpu=pci.bus_id','--format=csv,noheader,nounits'],records,required=False)
                if r['returncode']==0:cli[u]=r['stdout'].strip()
        else:
            r=command(['nvidia-smi','--query-gpu='+field,'--format=csv,noheader,nounits'],records,required=False)
            if field=='uuid' and r['returncode']==0 and set(r['stdout'].split())!=set(uuids):
                raise ValueError('Independent CLI UUID query disagrees with inventory')
    # minor_number is deliberately not used as a CLI query field, even if listed.
    doc['identity_stage']='nvml'
    try:
        import pynvml as nv
        nv.nvmlInit()
        try:
            for u in uuids:
                h=nv.nvmlDeviceGetHandleByUUID(u);actual=nv.nvmlDeviceGetUUID(h)
                if isinstance(actual,bytes):actual=actual.decode()
                pci=nv.nvmlDeviceGetPciInfo(h).busId
                if isinstance(pci,bytes):pci=pci.decode()
                if actual!=u:raise ValueError('NVML UUID lookup mismatch')
                state['nvml'][u]={'uuid':actual,'minor':str(nv.nvmlDeviceGetMinorNumber(h)),'pci':pci,'source':'nvml'}
        finally:nv.nvmlShutdown()
    except ValueError:raise
    except Exception as exc:state['source_errors'].append({'source':'nvml','type':type(exc).__name__,'message':str(exc)})
    doc['identity_stage']='proc_driver'
    for p in sorted(Path('/proc/driver/nvidia/gpus').glob('*/information')):
        try:
            raw=p.read_text();r=proc_record(raw,p.parent.name)
            if r['uuid'] not in uuids:continue
            if r['uuid'] in state['proc']:raise ValueError('Duplicate /proc UUID')
            state['proc'][r['uuid']]={**r,'path':str(p),'raw':raw}
        except Exception as exc:state['source_errors'].append({'source':str(p),'type':type(exc).__name__,'message':str(exc)})
    doc['identity_stage']='reconcile_physical_sources'
    physical=reconcile(uuids,state['nvml'],state['proc'],cli)
    state['reconciled']=physical
    assigned=requested_ids(doc['environment'].get('SLURM_JOB_GPUS') or '')
    if not all(i.startswith('GPU-') for i in assigned):
        doc['identity_stage']='prove_gres_files'
        # Numeric GRES ordinals are NOT assumed to equal Linux minors without File evidence.
        state['gres_evidence']=gres_evidence(node,node_count)
        proofs=[r for r in state['gres_evidence'] if r['passed']]
        if not proofs:raise ValueError('BLOCKED: no node-local canonical GRES File evidence for global IDs; administrator mapping needed')
        state['gres_mapping_basis']=proofs[0]
    doc['identity_inventory']='\n'.join(', '.join((row[0],row[1],physical[row[1]]['minor'],physical[row[1]]['pci'])) for row in rows)
    doc['identity_stage']='complete'
    return doc['identity_inventory']
