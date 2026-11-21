"""CUDA Driver API evidence independent of PyTorch and NVML."""
import ctypes as C
import os
import stat
import uuid


def device_access(paths):
    records = {}
    for path in paths:
        record = {}
        try:
            st = os.stat(path)
            record.update(character_device=stat.S_ISCHR(st.st_mode), major=os.major(st.st_rdev),
                          minor=os.minor(st.st_rdev))
            fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
            os.close(fd)
            record['opened'] = True
        except OSError as exc:
            record.update(opened=False, errno=exc.errno, error=str(exc))
        records[path] = record
    return records


def driver_probe():
    report = dict(passed=False, devices=[], calls=[], pid=os.getpid())
    try:
        lib = C.CDLL('libcuda.so.1')
        report['library'] = lib._name
        with open('/proc/self/maps') as stream:
            report['loaded_paths'] = sorted({line.split()[-1] for line in stream
                                             if '/libcuda.so' in line})

        def call(name, types, *args):
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = types, C.c_int
            status = fn(*args)
            entry = dict(api=name, code=status)
            report['calls'].append(entry)
            if status:
                label = C.c_char_p()
                lib.cuGetErrorName.argtypes = [C.c_int, C.POINTER(C.c_char_p)]
                lib.cuGetErrorName.restype = C.c_int
                if lib.cuGetErrorName(status, C.byref(label)) == 0 and label.value:
                    entry['error_name'] = label.value.decode()
                raise RuntimeError(f"{name}: {entry.get('error_name', status)}")

        intp = C.POINTER(C.c_int)
        call('cuInit', [C.c_uint], 0)
        driver, count = C.c_int(), C.c_int()
        call('cuDriverGetVersion', [intp], C.byref(driver))
        call('cuDeviceGetCount', [intp], C.byref(count))
        report.update(driver_version=driver.value, count=count.value)
        for index in range(count.value):
            dev, raw, bus = C.c_int(), (C.c_ubyte*16)(), C.create_string_buffer(32)
            call('cuDeviceGet', [intp,C.c_int], C.byref(dev), index)
            call('cuDeviceGetUuid', [C.c_void_p,C.c_int], C.byref(raw), dev.value)
            call('cuDeviceGetPCIBusId', [C.c_void_p,C.c_int,C.c_int],
                 bus, len(bus), dev.value)
            report['devices'].append(dict(index=index, uuid='GPU-'+str(uuid.UUID(bytes=bytes(raw))),
                                          pci=bus.value.decode()))
        report['passed'] = count.value > 0
    except (OSError, AttributeError, RuntimeError) as exc:
        report['error'] = str(exc)
    return report


def torch_probe(*, allocate=True):
    report = dict(passed=False, phase='import', devices=[], pid=os.getpid())
    try:
        import torch
        report.update(torch=torch.__version__, cuda_build=torch.version.cuda)
        report['phase'] = 'preinit_device_count'
        report['preinit_device_count'] = torch.cuda.device_count()
        report['phase'] = 'explicit_init'
        torch.cuda.init()
        report['phase'] = 'enumeration'
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            device = dict(index=index, uuid='GPU-'+str(getattr(prop,'uuid','')).removeprefix('GPU-'),
                          name=prop.name, free=free, total=total)
            report['devices'].append(device)
        report['phase'] = 'allocation'
        if allocate:
            for device in report['devices']:
                i = device['index']
                value = torch.zeros(1,device=f'cuda:{i}')
                torch.cuda.synchronize(i)
                del value
                device['allocation_passed'] = True
        report['passed'] = bool(report['devices'])
        report['phase'] = 'complete'
    except Exception as exc:
        report['error'] = str(exc)
    return report


def system_evidence(inventory, node, assigned):
    from pathlib import Path
    import subprocess

    from .gres_identity import load_map, prove_ordinals
    from ..resources import parse_nodes
    from ..solver_y1_v1.identity import gres_evidence, proc_record

    result = {'files': {}, 'physical': [], 'assigned_device': None}
    if node == 'moana-y5':
        try:
            query = subprocess.run(['slurmd','-G'],capture_output=True,text=True,timeout=10)
            result['effective_gres_diagnostic'] = dict(returncode=query.returncode,
                                                        stdout=query.stdout,stderr=query.stderr)
        except (OSError, subprocess.SubprocessError) as exc:
            result['effective_gres_diagnostic'] = {'error':str(exc)}
    for name in ('/proc/self/cgroup', '/proc/self/mountinfo', '/proc/driver/nvidia/version',
                 '/sys/module/nvidia/version', '/sys/module/nvidia_uvm/initstate',
                 '/proc/driver/nvidia/params'):
        try:
            value = Path(name).read_text()
            if name.endswith('mountinfo'):
                value = '\n'.join(line for line in value.splitlines() if 'cgroup' in line)
            result['files'][name] = value
        except OSError as exc:
            result['files'][name] = {'error': str(exc)}
    result['devices_list'] = {}
    cgroup = result['files']['/proc/self/cgroup']
    for line in cgroup.splitlines() if isinstance(cgroup, str) else ():
        _, controllers, group = line.split(':', 2)
        if 'devices' in controllers.split(','):
            path = Path('/sys/fs/cgroup/devices') / group.lstrip('/') / 'devices.list'
            try:
                result['devices_list'][str(path)] = path.read_text()
            except OSError as exc:
                result['devices_list'][str(path)] = {'error': str(exc)}
    uuids = {row.split(',')[0].strip() for row in inventory.splitlines()}
    for path in sorted(Path('/proc/driver/nvidia/gpus').glob('*/information')):
        try:
            physical = proc_record(path.read_text(),path.parent.name)
            if physical['uuid'] in uuids:
                result['physical'].append(physical)
        except (ValueError, OSError):
            continue
    for config in (Path('/etc/slurm/gres.conf'),Path('/etc/slurm-llnl/gres.conf')):
        try:
            if config.stat().st_size <= 262144:
                result.setdefault('gres_config',{})[str(config)] = config.read_text()
        except OSError as exc:
            result.setdefault('gres_config',{})[str(config)] = {'error':str(exc)}
    candidates = [r for r in result['physical'] if r['uuid']==assigned]
    if assigned.isdecimal() and node:
        try:
            raw = subprocess.check_output(['scontrol','show','node',node,'-o'],text=True,timeout=10)
            host, = parse_nodes(raw)
            proof = gres_evidence(node,host['gpus'])
            result['gres_proof'] = [{k:v for k,v in row.items() if k!='text'} for row in proof]
            annotation = load_map(node,host['gpus'])
            if annotation is not None:
                physical = {r['uuid']:r for r in result['physical']}
                uuid, = prove_ordinals([assigned],annotation,physical)
                candidates = [physical[uuid]]
                result['allocation_basis'] = 'administrator-attested-ordinal-map'
                result['ordinal_map'] = annotation
            elif node != 'moana-y5' and any(r['passed'] for r in proof):
                candidates = [r for r in result['physical'] if r['minor']==assigned]
                result['allocation_basis'] = 'canonical-GRES'
            else:
                result['allocation_error'] = 'Noncanonical GRES; need administrator-attested ordinal map or Slurm UUID assignment'
        except (OSError, KeyError, ValueError, subprocess.SubprocessError) as exc:
            result['gres_error'] = str(exc)
    if len(candidates)==1:
        result['assigned_device'] = candidates[0]
    return result


if __name__ == '__main__':
    import argparse
    import json
    import subprocess
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('role', choices=('driver','torch','one-gpu'))
    parser.add_argument('--native')
    args = parser.parse_args()
    if args.role in ('driver', 'torch'):
        print(json.dumps(driver_probe() if args.role=='driver' else torch_probe()))
    else:
        inventory = subprocess.run(['nvidia-smi','--query-gpu=uuid,pci.bus_id',
                                    '--format=csv,noheader'],capture_output=True,text=True,timeout=15)
        result = dict(environment={k:os.environ.get(k) for k in
                       ('SLURMD_NODENAME','SLURM_JOB_GPUS','SLURM_STEP_GPUS','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER')},
                      inventory=inventory.stdout, nvidia_smi_rc=inventory.returncode,
                      nvidia_smi_stderr=inventory.stderr,
                      device_access=device_access(['/dev/nvidiactl','/dev/nvidia-uvm']))
        result['system'] = system_evidence(inventory.stdout,os.environ.get('SLURMD_NODENAME'),
                                           os.environ.get('SLURM_JOB_GPUS',''))
        result['device_access'].update(device_access(['/dev/nvidia-uvm-tools']))
        assigned = result['system']['assigned_device']
        if assigned is not None:
            result['device_access'].update(device_access(['/dev/nvidia'+assigned['minor']]))
        if args.native:
            native = subprocess.run([args.native],capture_output=True,text=True,timeout=45)
            result['native'] = dict(returncode=native.returncode,stdout=native.stdout,stderr=native.stderr)
        for role in ('driver','torch'):
            child = subprocess.run([sys.executable,'-B','-m',__spec__.name,role],
                                   capture_output=True,text=True,timeout=90)
            result[role] = json.loads(child.stdout) if child.returncode==0 else dict(
                passed=False, returncode=child.returncode, error=child.stderr)
        print(json.dumps(result,indent=2))
        driver = result['driver']
        torch = result['torch']
        passed = (assigned is not None and ('native' not in result or result['native']['returncode']==0)
                  and inventory.returncode == 0 and driver['passed'] and torch['passed']
                  and len(driver['devices']) == len(torch['devices']) == 1
                  and driver['devices'][0]['uuid'] == torch['devices'][0]['uuid'] == assigned['uuid']
                  and driver['devices'][0]['uuid'] in {r.split(',')[0].strip() for r in inventory.stdout.splitlines()})
        sys.exit(0 if passed else 1)
