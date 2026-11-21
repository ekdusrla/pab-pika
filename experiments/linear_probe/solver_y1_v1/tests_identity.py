"""CPU-only fixtures; never invoke nvidia-smi, NVML, CUDA or SLURM."""
import subprocess
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from .gpu_mapping import supported_fields,pci_id,canonical_gres_minors
from .identity import command,CommandFailure,proc_record,reconcile,discover


class IdentityTests(unittest.TestCase):
    def test_supported_help_fields(self):
        fields=supported_fields('"uuid" or "gpu_uuid"\n"pci.bus_id"\n index\n')
        self.assertIn('uuid',fields);self.assertIn('pci.bus_id',fields)
        self.assertNotIn('minor_number',fields)

    def test_stderr_stdout_and_status_preserved(self):
        records=[]
        def runner(*a,**k):return SimpleNamespace(returncode=2,stdout='synthetic field error',stderr='synthetic stderr')
        with self.assertRaises(CommandFailure) as caught:
            command(['nvidia-smi','fake-query'],records,runner=runner)
        self.assertEqual(records[0],caught.exception.record)
        self.assertEqual(records[0]['returncode'],2)
        self.assertEqual(records[0]['stdout'],'synthetic field error');self.assertEqual(records[0]['stderr'],'synthetic stderr')

    def test_optional_field_failure_preserved(self):
        records=[]
        r=command(['fake'],records,required=False,runner=lambda *a,**k:SimpleNamespace(returncode=2,stdout='',stderr='unsupported'))
        self.assertEqual(r['returncode'],2);self.assertEqual(records[0]['stderr'],'unsupported')

    def test_timeout_diagnostics_preserved(self):
        def runner(*a,**k):raise subprocess.TimeoutExpired(a[0],15,output=b'partial',stderr=b'timeout')
        records=[]
        with self.assertRaises(CommandFailure):command(['fake'],records,runner=runner)
        self.assertEqual(records[0]['stdout'],'partial');self.assertEqual(records[0]['stderr'],'timeout')

    def test_pci_normalization(self):
        self.assertEqual(pci_id('00000000:01:00.0'),pci_id('0000:01:00.0'))
        with self.assertRaises(ValueError):pci_id('N/A')

    def test_proc_physical_identity(self):
        r=proc_record('GPU UUID: GPU-a\nDevice Minor: 3\n','0000:01:00.0')
        self.assertEqual((r['uuid'],r['minor']),('GPU-a','3'))
        with self.assertRaises(ValueError):proc_record('GPU UUID: GPU-a\n','0000:01:00.0')

    def test_independent_sources_must_agree(self):
        r={'uuid':'GPU-a','minor':'3','pci':'0000:01:00.0'}
        self.assertEqual(reconcile(['GPU-a'],{'GPU-a':r},{},{'GPU-a':'00000000:01:00.0'})['GPU-a']['minor'],'3')
        with self.assertRaisesRegex(ValueError,'conflict'):
            reconcile(['GPU-a'],{'GPU-a':r},{'GPU-a':{**r,'minor':'4'}},{})
        with self.assertRaisesRegex(ValueError,'No NVML'):
            reconcile(['GPU-a'],{},{},{})
        with self.assertRaisesRegex(ValueError,'conflict'):
            reconcile(['GPU-a'],{'GPU-a':r},{},{'GPU-a':'0000:02:00.0'})

    def test_canonical_gres_file_evidence(self):
        text='NodeName=moana-y1 Name=gpu Type=normal File=/dev/nvidia[0-7]'
        self.assertEqual(canonical_gres_minors(text,'moana-y1',8),list(range(8)))
        single='\n'.join(f'Name=gpu Type=normal File=/dev/nvidia{i}' for i in range(8))
        self.assertEqual(canonical_gres_minors(single,'moana-y1',8),list(range(8)))

    def test_ambiguous_or_incomplete_gres_rejected(self):
        for text in ('AutoDetect=nvml','Name=gpu Type=normal File=/dev/nvidia[0-2]',
                     'NodeName=moana-y2 Name=gpu Type=normal File=/dev/nvidia[0-7]',
                     'Name=gpu Type=normal File=/dev/nvidia[1-8]',
                     'Name=gpu Type=normal File=/dev/nvidia[0-7] Count=4'):
            with self.subTest(text=text),self.assertRaises(ValueError):canonical_gres_minors(text,'moana-y1',8)

    def discovery_fixture(self,gres):
        uuids=['GPU-'+s for s in 'abcd'];minors=dict(zip(uuids,('0','1','3','4')))
        buses={u:f'00000000:{i+1:02X}:00.0' for i,u in enumerate(uuids)}
        inventory='\n'.join(f'{i}, {u}, RTX 3090, 24576, 24000' for i,u in enumerate(uuids))
        nv=SimpleNamespace(nvmlInit=lambda:None,nvmlShutdown=lambda:None,
            nvmlDeviceGetHandleByUUID=lambda u:u,nvmlDeviceGetUUID=lambda u:u,
            nvmlDeviceGetMinorNumber=lambda u:int(minors[u]),
            nvmlDeviceGetPciInfo=lambda u:SimpleNamespace(busId=buses[u]))
        def fake_command(argv,records=None,required=True):
            if '--help-query-gpu' in argv:stdout='"uuid"\n"pci.bus_id"\n'
            elif '--query-gpu=uuid' in argv:stdout='\n'.join(uuids)
            elif '--query-gpu=pci.bus_id' in argv:stdout=buses[argv[argv.index('-i')+1]]
            else:raise AssertionError('Unsupported field or unexpected command: '+str(argv))
            r={'argv':argv,'returncode':0,'stdout':stdout,'stderr':''}
            if records is not None:records.append(r)
            return r
        doc={'environment':{'SLURM_JOB_GPUS':'0,1,3,4'}}
        with patch('experiments.linear_probe.solver_y1_v1.identity.command',side_effect=fake_command), \
             patch.dict('sys.modules',{'pynvml':nv}),patch.object(Path,'glob',return_value=[]), \
             patch.object(Path,'stat',return_value=SimpleNamespace(st_size=len(gres))),patch.object(Path,'read_text',return_value=gres):
            result=discover(doc,inventory,8,'moana-y1')
        return doc,result

    def test_discovery_without_minor_cli_field(self):
        doc,identity=self.discovery_fixture('Name=gpu Type=normal File=/dev/nvidia[0-7]')
        self.assertFalse(doc['query_field_support']['minor_number'])
        self.assertEqual(doc['identity_stage'],'complete')
        self.assertIn('GPU-c, 3,',identity)
        self.assertTrue(all('minor_number' not in ' '.join(r['argv']) for r in doc['query_commands']))

    def test_discovery_blocks_without_gres_proof(self):
        with self.assertRaisesRegex(ValueError,'no node-local canonical GRES File evidence'):
            self.discovery_fixture('Name=gpu Type=normal File=/dev/nvidia[0-2]')


if __name__=='__main__':unittest.main()
