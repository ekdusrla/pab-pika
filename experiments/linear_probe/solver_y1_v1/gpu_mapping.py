"""CPU-only parsing. Slurm GRES IDs and job-local nvidia-smi indices differ."""
import csv
import re
import shlex
import fnmatch


def requested_ids(raw):
    ids=[]
    for token in raw.split(','):
        token=token.strip()
        if token.startswith('GPU-'):ids.append(token);continue
        if not re.fullmatch(r'\d+(?:-\d+)?',token):raise ValueError('Unverifiable SLURM_JOB_GPUS syntax')
        ends=[int(x) for x in token.split('-')]
        if ends[-1]<ends[0] or ends[-1]-ends[0]>256:raise ValueError('Invalid GPU ID range')
        ids.extend(str(i) for i in range(ends[0],ends[-1]+1))
    if len(ids)!=4 or len(set(ids))!=4:raise ValueError('Require four distinct assigned GPU IDs')
    if any(s.startswith('GPU-') for s in ids) and not all(s.startswith('GPU-') for s in ids):
        raise ValueError('Mixed numeric and UUID assignment syntax')
    return ids


def resolve_mapping(raw,inventory,identity=None):
    rows=[[x.strip() for x in row] for row in csv.reader(inventory.splitlines()) if row]
    if not rows or any(len(row)!=5 for row in rows):raise ValueError('Malformed GPU inventory')
    by_index={row[0]:row[1] for row in rows};uuids={row[1] for row in rows}
    if len(by_index)!=len(rows) or len(uuids)!=len(rows) or any(not u.startswith('GPU-') for u in uuids):
        raise ValueError('Duplicate or missing inventory identity')
    ids=requested_ids(raw)
    if all(s.startswith('GPU-') for s in ids):
        assigned=set(ids);method='slurm_uuid'
    elif identity is not None:
        physical=[[x.strip() for x in row] for row in csv.reader(identity.splitlines()) if row]
        if not physical or any(len(row)!=4 for row in physical):raise ValueError('Malformed GPU minor/PCI identity inventory')
        if {row[1] for row in physical}!=uuids or len(physical)!=len(rows):raise ValueError('Identity/inventory UUID mismatch')
        by_minor={row[2]:row[1] for row in physical}
        if len(by_minor)!=len(physical) or any(not row[2].isdigit() for row in physical):raise ValueError('Invalid or duplicate device minor numbers')
        if any(row[0] not in by_index or by_index[row[0]]!=row[1] for row in physical):raise ValueError('Identity/index UUID mismatch')
        if len({row[3] for row in physical})!=len(physical) or any(row[3] in ('','N/A','[N/A]') for row in physical):
            raise ValueError('Missing or duplicate PCI identity')
        if not set(ids)<=set(by_minor):raise ValueError('SLURM GPU IDs absent from device minors; authoritative GRES mapping needed')
        assigned={by_minor[i] for i in ids};method='slurm_numeric_id_to_device_minor_uuid'
    else:
        # Legacy/full-node direct-index path for existing parser tests only.
        # Runtime preflight ALWAYS supplies the independent identity query.
        if not set(ids)<=set(by_index):raise ValueError('Assigned GPU index absent from inventory')
        assigned={by_index[i] for i in ids};method='direct_index_without_physical_identity'
    if not assigned<=uuids or len(assigned)!=4:raise ValueError('Assigned UUIDs absent or duplicated in inventory')
    return {'assigned_ids':ids,'assigned_uuids':sorted(assigned),'method':method,
            'nvidia_smi_indices_assumed_global':False if identity is not None else None}


def check_visible(mapping,visible):
    values=[u if u.startswith('GPU-') else 'GPU-'+u for u in visible]
    if len(values)!=4 or len(set(values))!=4:raise ValueError('Require four distinct CUDA-visible UUIDs')
    if set(values)!=set(mapping['assigned_uuids']):raise ValueError('CUDA-visible UUIDs differ from SLURM assigned GPUs')
    return {**mapping,'visible_uuids':sorted(values),'passed':True}


def supported_fields(help_text):
    # The driver help normally quotes field names. Also accept standalone names.
    return set(re.findall(r'"([a-zA-Z0-9_.]+)"',help_text)) | set(re.findall(r'(?m)^\s*([a-z][a-z0-9_.]+)\s*$',help_text))


def pci_id(value):
    text=value.decode() if isinstance(value,bytes) else str(value)
    m=re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])',text.strip())
    if not m:raise ValueError('Missing or malformed PCI bus ID')
    return tuple(int(x,16) for x in m.groups())


def canonical_gres_minors(text,node,count):
    """Conservative proof of GRES ordinal == /dev/nvidia minor, never a guess."""
    minors=[]
    for line in text.splitlines():
        fields=dict(t.split('=',1) for t in shlex.split(line,comments=True) if '=' in t)
        if not fields:continue
        if 'Include' in fields:raise ValueError('GRES Include requires administrator-expanded mapping')
        pattern=fields.get('NodeName',node)
        if not fnmatch.fnmatchcase(node,pattern):continue
        if fields.get('Name')!='gpu':continue
        if fields.get('Type')!='normal':raise ValueError('Require unambiguous normal GPU GRES configuration')
        file=fields.get('File','')
        m=re.fullmatch(r'/dev/nvidia(\d+)',file)
        span=re.fullmatch(r'/dev/nvidia\[(\d+)-(\d+)\]',file)
        if m:values=[int(m.group(1))]
        elif span:
            lo,hi=map(int,span.groups())
            if hi<lo or hi-lo>256:raise ValueError('Invalid GRES device range')
            values=list(range(lo,hi+1))
        else:raise ValueError('GRES File mapping unavailable/unsupported; administrator mapping needed')
        if 'Count' in fields and fields['Count']!=str(len(values)):raise ValueError('GRES Count/File mismatch')
        minors.extend(values)
    if minors!=list(range(count)):
        raise ValueError('Node-local GRES File list is not canonical/full; cannot prove global ID to minor mapping')
    return minors
