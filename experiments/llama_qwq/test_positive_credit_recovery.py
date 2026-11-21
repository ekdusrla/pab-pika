"""Read-only production snapshot; fake API and atomic checkpoint tests in /tmp."""
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace as NS
from unittest.mock import patch
import run_planner as p
import planner_8b_recovery as rec
from experiment_state import read_manifest, original_definitions, atomic_save
from check_8b_planner import validate as check_complete
from test_math500_resume import FakeResponse, FakeClient

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'runs/math500_llama31_8b_v1/planner.json'


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=SOURCE.read_bytes()
        cls.original=json.loads(cls.raw)
        cls.manifest=read_manifest(HERE/'math500_manifest.json')
        with patch.dict(os.environ,PLANNER_RECOVER_OVERLONG='1',PLANNER_RECOVER_CREDITS='1'):
            rec.install(p)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='positive_credit_')
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'planner.json'
        self.doc=copy.deepcopy(self.original)
        self.args=NS(**{k:self.doc['config'][k] for k in ['model','revision','api_base','seed','max_model_len','retries','retry_wait','timeout']},limit=500,resume=True,check_only=False,output=self.path,server_pid=None,recover_overlong=True)
        rec.prepare(self.doc,self.manifest,(True,True))

    def tearDown(self):
        self.assertEqual(self.raw,SOURCE.read_bytes())

    def save(self):
        p.validate_recovery_history(self.doc)
        p.refresh_planner(self.doc)
        atomic_save(self.path,self.doc)

    def config(self):
        c=p.planner_config(self.args,self.manifest,original_definitions()[0])
        c['server_settings'].update(quantization=None,tensor_parallel_size=1)
        c['launch_script_sha256']=p.file_hash(HERE/'planner_llama31_8b_y4.sbatch')
        return c

    def run_response(self,credits):
        client=FakeClient([FakeResponse(json.dumps({str(i+1):{'credit':n} for i,n in enumerate(credits)}))])
        with redirect_stdout(io.StringIO()):result=p.run_planner(self.args,self.manifest,self.doc,client,self.save)
        self.assertEqual(len(client.calls),1)
        self.assertEqual(client.calls[0]['messages'][-1]['content'],rec.POSITIVE_CREDIT_INSTRUCTION)
        self.assertIn('guided_json',client.calls[0]['extra_body'])
        for row,old in zip(self.doc['stats'],self.original['stats']):
            if old['status']=='success':self.assertEqual(row,old)
            self.assertEqual(row['stages']['decomposition'],old['stages']['decomposition'])
            for name in row['stages']:
                count=len(old['stages'][name]['attempts'])
                self.assertEqual(row['stages'][name]['attempts'][:count],old['stages'][name]['attempts'])
        self.assertEqual(self.doc['eight_b_recovery'],self.original['eight_b_recovery'])
        self.assertEqual(self.doc['summary']['attempt_cost']['attempts'],self.original['summary']['attempt_cost']['attempts']+1)
        self.assertEqual(self.doc['summary']['attempt_cost']['prompt_tokens']['known_sum'],self.original['summary']['attempt_cost']['prompt_tokens']['known_sum']+10)
        self.assertEqual(self.doc['summary']['attempt_cost']['completion_tokens']['known_sum'],self.original['summary']['attempt_cost']['completion_tokens']['known_sum']+5)
        stages=[s for r in self.doc['stats'] for s in r['stages'].values()]
        self.assertEqual(self.doc['summary']['attempt_cost'],p.attempt_cost(stages))
        return result

    def test_positive_and_resume(self):
        self.assertTrue(self.run_response([20,1,29,50]))
        check_complete(self.path,HERE/'math500_manifest.json')
        self.doc=p.load_planner_checkpoint(self.args,self.config(),self.manifest)
        client=FakeClient([])
        with redirect_stdout(io.StringIO()):self.assertTrue(p.run_planner(self.args,self.manifest,self.doc,client,self.save))
        self.assertEqual(client.calls,[])

    def test_zero_remains_failure_and_can_resume(self):
        self.assertFalse(self.run_response([20,0,30,50]))
        self.assertEqual(self.doc['summary']['success'],499)
        self.doc=p.load_planner_checkpoint(self.args,self.config(),self.manifest)
        rec.verify(self.doc)

    def test_wrong_sum_remains_failure(self):
        self.assertFalse(self.run_response([20,1,30,50]))
        self.assertEqual(self.doc['summary']['success'],499)

    def test_repeated_zero_only(self):
        row=next(r for r in self.original['stats'] if r['id']=='402')
        target=copy.deepcopy(row['stages']['assessment'])
        self.assertTrue(rec.repeated_zero(target,4))
        target['attempts']=target['attempts'][:1]
        self.assertFalse(rec.repeated_zero(target,4))
        target['attempts']*=2
        for a in target['attempts']:a['content']='{"1":{"credit":20},"2":{"credit":1},"3":{"credit":29},"4":{"credit":51}}'
        self.assertFalse(rec.repeated_zero(target,4))

    def test_499_protection_and_adapter_boundary(self):
        self.assertEqual(len(self.doc['positive_credit_recovery']['protected_successes']),499)
        row=next(r for r in self.doc['stats'] if r['status']=='success')
        row['steps'][0]='changed'
        with self.assertRaises(ValueError):rec.verify(self.doc)
        doc=copy.deepcopy(self.original);doc['eight_b_recovery']['adapter_sha256']='untrusted'
        with self.assertRaises(ValueError):rec.prepare(doc,self.manifest,(True,True))


if __name__=='__main__':unittest.main()
