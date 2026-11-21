"""Read an administrator-attested ordinal map without inferring File order."""
import json
from pathlib import Path
import re
import stat

from ..common import file_hash
from ..solver_y1_v1.gpu_mapping import pci_id


MAP_PATH = Path('/etc/slurm/pab-gpu-ordinal-map.json')
GRES_PATH = Path('/etc/slurm/gres.conf')


def load_map(node, count):
    if not MAP_PATH.exists():
        return None
    # Only a node administrator may install a mapping. Reject symlinks and writable parents.
    for p in (MAP_PATH, *MAP_PATH.parents):
        mode = p.lstat()
        if mode.st_uid != 0 or mode.st_mode & 0o022 or stat.S_ISLNK(mode.st_mode):
            raise ValueError('GRES ordinal map must be root-controlled and not writable by other users')
    if not MAP_PATH.is_file() or MAP_PATH.stat().st_size > 65536:
        raise ValueError('Invalid administrator GRES map file')
    doc = json.loads(MAP_PATH.read_text())
    if doc['node'] != node or doc['gres_sha256'] != file_hash(GRES_PATH) or not doc.get('source'):
        raise ValueError('Stale or unsupported administrator GRES map')
    devices = doc['ordinals']
    if set(devices) != {str(i) for i in range(count)} or len({v['uuid'] for v in devices.values()}) != count:
        raise ValueError('Administrator map must describe every active Slurm GPU ordinal uniquely')
    minors = set()
    buses = set()
    for v in devices.values():
        if not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}',v['uuid']) or not re.fullmatch(r'0|[1-9][0-9]*',v['minor']):
            raise ValueError('Invalid ordinal map UUID or minor')
        minors.add(v['minor'])
        buses.add(pci_id(v['pci']))
    if len(minors)!=count or len(buses)!=count:
        raise ValueError('Duplicate minor or PCI identity in administrator map')
    return doc


def prove_ordinals(ids, doc, physical):
    devices = [doc['ordinals'][key] for key in ids]
    for device in devices:
        actual = physical.get(device['uuid'])
        if (actual is None or actual['minor'] != device['minor']
                or pci_id(actual['pci']) != pci_id(device['pci'])):
            raise ValueError('Attested ordinal disagrees with independent UUID, minor or PCI evidence')
    return [d['uuid'] for d in devices]
