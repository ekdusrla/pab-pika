"""CPU contracts for Smoke evidence and immutable main approval."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import smoke
from ..common import FEATURE, LEVEL_WORDS, allocate_words, digest, seal


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc, cls.conf = smoke.experiment(smoke.OUT/'approved_manifest.json')
        cls.row, = (r for r in cls.doc['rows'] if r['id']==smoke.PROBLEM_ID)

    def test_preflight_passes_slurm_environment_to_original_discovery(self):
        env = dict(SLURMD_NODENAME='moana-y3', SLURM_JOB_NUM_NODES='1', SLURM_JOB_GPUS='0-3',
                   SLURM_CPUS_PER_TASK='16',SLURM_MEM_PER_NODE='65536',CUDA_VISIBLE_DEVICES='0,1,2,3')
        class Recorded(Exception):
            pass
        def discover(record,*_):
            self.assertEqual(record['environment']['SLURM_JOB_GPUS'],'0-3')
            raise Recorded
        with patch.dict(smoke.os.environ,env,clear=True), patch.object(smoke.subprocess,'check_output'), \
             patch.object(smoke,'parse_nodes',return_value=[dict(gpus=8)]), \
             patch.object(smoke,'validate_allocation',return_value={}), patch.object(smoke,'command',return_value={'stdout':''}), \
             patch.object(smoke,'discover',side_effect=discover), patch.object(smoke,'save_new') as saved:
            with self.assertRaises(Recorded):
                smoke.preflight(Path('/unused'),self.conf)
        self.assertFalse(saved.call_args.args[1]['passed'])

    def test_completion_rejects_missing_and_failed_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            with self.assertRaises(FileNotFoundError):
                smoke.qualify_evidence(target,self.doc,self.conf,self.row)
            steps = ['1. Expand the difference.', '2. Evaluate.']
            total = LEVEL_WORDS[self.row['level']]
            budgets = allocate_words([50,50],total,'constant')
            layer = self.doc['probe_settings']['layer']
            uuids = [f'GPU-{i}' for i in range(4)]
            req = smoke.solver_payload(self.row,steps,'ridge',budgets,self.conf['solver_seeds'][0])
            manifest = self.doc['fingerprint']
            docs = dict(
                preflight=dict(passed=True,environment={'SLURMD_NODENAME':self.conf['node']},mapping=dict(passed=True,assigned_uuids=uuids)),
                memory=dict(passed=True,samples=1,peaks_mib=dict.fromkeys(uuids,21000)),
                capture=dict(layer=layer,feature_definition=FEATURE,
                             stage=dict(status='success',result=dict(same_generated_ids=True,steps=steps))),
                probe=dict(manifest=manifest,feature_definition=FEATURE,layer=layer,steps=steps,
                           steps_fingerprint=digest(steps),probe_fingerprint=self.doc['probe_fingerprint'],
                           prediction=dict(credits=[50,50]),budgets=budgets),
                server=dict(manifest=manifest,argv=['--tensor-parallel-size','4','--dtype','float16',
                                                  '--model',self.doc['model_identity']['solver']['snapshot']]),
                request=dict(problem=self.row['id'],arm='ridge',stage=dict(
                    status='success',attempts=[{}],request=req,result=dict(finish_reason='stop'),
                    parsed=dict(answer_evaluation=dict(status='correct')))))
            def write(name):
                (target/(name+'.json')).write_text(json.dumps(seal(docs[name])))
            for name in docs:
                write(name)
            self.assertEqual(set(smoke.qualify_evidence(target,self.doc,self.conf,self.row)),set(smoke.EVIDENCE))
            docs['request']['stage']['result']['finish_reason']='length'
            write('request')
            with self.assertRaisesRegex(ValueError,'request qualification'):
                smoke.qualify_evidence(target,self.doc,self.conf,self.row)
            docs['memory']['peaks_mib'][uuids[0]]=24000
            write('memory')
            with self.assertRaisesRegex(ValueError,'VRAM qualification'):
                smoke.qualify_evidence(target,self.doc,self.conf,self.row)

    def test_historical_gpu_allocation_matches_smi(self):
        from ..common import read, verify
        old = verify(read(smoke.ROOT/'221852/preflight.json'))
        mapping = smoke.resolve_mapping(old['environment']['SLURM_JOB_GPUS'], old['inventory'],
                                        old['identity_inventory'])
        physical = old['physical_identity']['reconciled']
        self.assertEqual(set(mapping['assigned_uuids']), set(physical))
        self.assertEqual(sorted(int(v['minor']) for v in physical.values()), [0,1,2,4])
        self.assertEqual(old['devices'], [])

    def test_runtime_gpu_set_and_ordinal_mismatch(self):
        ids = [f'GPU-{i}' for i in range(4)]
        devices = [dict(uuid=u,pci=f'0000:{i:02x}:00.0') for i,u in enumerate(ids)]
        physical = {d['uuid']:d for d in devices}
        driver = dict(passed=True,devices=devices)
        extra = dict(passed=True,devices=devices[:3]+[dict(uuid='GPU-extra')])
        got = smoke.compare_sets(ids,physical,driver,extra)
        self.assertEqual(got['torch']['missing_from_runtime'], ['GPU-3'])
        self.assertEqual(got['torch']['unexpected_in_runtime'], ['GPU-extra'])
        torch = dict(passed=True,devices=devices[::-1])
        record = {'physical_identity':{'reconciled':physical}, 'allocation_mapping':{'assigned_uuids':ids},
                  'driver_probe':driver,'torch_probe':torch}
        with self.assertRaisesRegex(ValueError,'ordinal maps differ'):
            smoke.validate_driver(record)
        record['torch_probe'] = driver
        smoke.validate_driver(record)

    def test_compatibility_launcher_changes_only_gate_and_preflight(self):
        here = Path(smoke.__file__).parent
        approved = here.parent/'solver_y3_500_v1'
        original = (approved/'launch.sh').read_text()
        new = (here/'main_launch.sh').read_text()
        restored = new.replace('SMOKE_JOB=${2:?Successful Smoke job ID required}\nshift 2\n','shift\n')
        restored = restored.replace('QUALIFIER=experiments.linear_probe.solver_y3_500_smoke_v1.smoke\n'
                                    '"$PY" -B -m "$QUALIFIER" check-main --manifest "$MANIFEST" --job "$SMOKE_JOB"\n','')
        restored = restored.replace('"$QUALIFIER" main-preflight','"$MODULE.hardware" preflight')
        self.assertEqual(restored,original)
        self.assertEqual((here/'main.sbatch').read_text().replace('solver_y3_500_smoke_v1/main_launch.sh',
                         'solver_y3_500_v1/launch.sh'),(approved/'gpu.sbatch').read_text())

    def test_attested_gres_ordinal_is_not_linux_minor(self):
        from .gres_identity import prove_ordinals
        from ..common import ROOT, read
        recorded = read(ROOT/'cuda_diag_221866.log')
        physical = {d['uuid']:d for d in recorded['system']['physical']}
        real = recorded['driver']['devices'][0]['uuid']
        self.assertEqual(recorded['environment']['SLURM_JOB_GPUS'],'1')
        self.assertEqual(physical[real]['minor'],'2')
        # This is a CPU fixture, never a trusted map for the node.
        self.assertEqual(prove_ordinals(['1'],{'ordinals':{'1':physical[real]}},physical),[real])
        wrong = {**physical[real],'minor':'1'}
        with self.assertRaisesRegex(ValueError,'disagrees'):
            prove_ordinals(['1'],{'ordinals':{'1':wrong}},physical)

    def test_y5_evidence_cannot_qualify_y3(self):
        with patch.object(smoke.sys,'argv',['smoke','check-main','--node','moana-y5','--job','221866',
                                          '--manifest',str(smoke.OUT/'approved_manifest.json')]):
            with self.assertRaises(SystemExit) as failure:
                smoke.main()
        self.assertEqual(failure.exception.code,2)
