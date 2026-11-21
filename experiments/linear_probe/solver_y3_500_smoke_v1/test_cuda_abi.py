"""Verify both call interfaces with a CPU-only shared library fixture."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class CudaABITests(unittest.TestCase):
    def test_native_and_ctypes_init_agree(self):
        folder = Path(__file__).parent
        with TemporaryDirectory() as temp:
            path = Path(temp)
            stub = path/'stub.c'
            stub.write_text('''
#include <stdlib.h>
int cuInit(unsigned int flags) {
    return flags ? 1 : atoi(getenv("PAB_TEST_CUDA_RESULT"));
}
int cuGetErrorName(int code, const char **name) {
    *name = code == 3 ? "CUDA_ERROR_NOT_INITIALIZED" : "OTHER";
    return 0;
}
int cuDriverGetVersion(int *v) { *v=12010; return 0; }
int cuDeviceGetCount(int *n) { *n=0; return 0; }
''')
            subprocess.run(['cc','-shared','-fPIC',str(stub),'-o',str(path/'libcuda.so.1')],check=True)
            native = path/'native'
            subprocess.run(['cc','-Wall','-Wextra','-Werror',str(folder/'cuinit_native.c'),'-ldl','-o',str(native)],check=True)
            for status in (0,3):
                env = dict(os.environ,LD_LIBRARY_PATH=str(path),PAB_TEST_CUDA_RESULT=str(status))
                first = subprocess.run([str(native)],env=env,capture_output=True,text=True)
                second = json.loads(subprocess.check_output(
                    [sys.executable,'-B','-m',__package__+'.cuda_diagnostics','driver'],env=env,text=True))
                self.assertEqual(first.stdout.strip(),f'cuInit(0)={status}')
                self.assertEqual(first.returncode,int(status!=0))
                self.assertEqual(second['calls'][0]['code'],status)
                if status:
                    self.assertEqual(second['calls'][0]['error_name'],'CUDA_ERROR_NOT_INITIALIZED')
                    self.assertEqual(len(second['calls']),1)
                else:
                    self.assertEqual(second['driver_version'],12010)
