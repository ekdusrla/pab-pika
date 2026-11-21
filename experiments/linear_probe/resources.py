"""Node CPU/GPU and RealMemory/GPU policy, with explicit memory headroom."""
import math
import re
import shlex


def parse_nodes(text):
    nodes=[]
    for line in text.splitlines():
        if not line.strip():continue
        fields=dict(token.split('=',1) for token in shlex.split(line) if '=' in token)
        gres=fields.get('Gres','')
        counts=[int(m.group(1)) for m in re.finditer(r'(?:^|,)gpu:(?:[^,:()]+:)?(\d+)',gres)]
        if not counts:continue
        nodes.append({'name':fields['NodeName'],'cpus':int(fields['CPUTot']),
            'memory_mib':int(fields['RealMemory']),'gpus':sum(counts),
            'normal_gpus':sum(int(m.group(1)) for m in re.finditer(r'(?:^|,)gpu:normal:(\d+)',gres)),
            'partitions':fields.get('Partitions','').split(','),
            'features':fields.get('AvailableFeatures','').split(','),'raw':line})
    return nodes


def limits(node, headroom=.95):
    if not 0 < headroom < 1:raise ValueError('Memory headroom fraction must be in (0,1)')
    return {'cpus_per_gpu':node['cpus']//node['gpus'],
            'ram_gb_per_gpu':math.floor(node['memory_mib']/node['gpus']*headroom/1024)}


def common_limits(nodes, gpus, partition='batch_ce_ugrad', constraint=None, headroom=.95):
    eligible=[n for n in nodes if partition in n['partitions'] and n['normal_gpus']>=gpus
              and (constraint is None or constraint in n['features'])]
    if not eligible:raise ValueError('No eligible gpu:normal nodes in fresh scheduler snapshot')
    caps=[limits(n,headroom) for n in eligible]
    return {'cpus_per_gpu':min(c['cpus_per_gpu'] for c in caps),
            'ram_gb_per_gpu':min(c['ram_gb_per_gpu'] for c in caps),
            'eligible_nodes':[n['name'] for n in eligible],'memory_fraction':headroom}


def validate_allocation(node, gpus, cpus, memory_mib, headroom=.95):
    cap=limits(node,headroom)
    if gpus<1 or gpus>node['normal_gpus'] or cpus>gpus*cap['cpus_per_gpu'] or memory_mib>gpus*cap['ram_gb_per_gpu']*1024:
        raise ValueError('Allocation exceeds this node CPU/GPU or headroom-adjusted MEM/GPU policy')
    return cap
