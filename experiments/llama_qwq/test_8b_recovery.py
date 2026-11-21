"""Real checkpoint read-only; fake recovery calls and atomic saves exclusively in /tmp."""
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
import planner_8b_recovery as recovery
from experiment_state import read_manifest, original_definitions, atomic_save, digest
from check_8b_planner import validate as check_complete
from test_math500_resume import FakeResponse, FakeClient

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'runs/math500_llama31_8b_v1/planner.json'


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_bytes=SOURCE.read_bytes()
        cls.original=json.loads(cls.source_bytes)
        cls.manifest=read_manifest(HERE/'math500_manifest.json')
        with patch.dict(os.environ,PLANNER_RECOVER_OVERLONG='1',PLANNER_RECOVER_CREDITS='1'):
            recovery.install(p)

    def setUp(self):
        self.doc=copy.deepcopy(self.original)
        self.temp=tempfile.TemporaryDirectory(prefix='8b_recovery_')
        self.addCleanup(self.temp.cleanup)
        self.output=Path(self.temp.name)/'planner.json'
        self.args=NS(**{k:self.doc['config'][k] for k in ('model','revision','api_base','seed','max_model_len','retries','retry_wait','timeout')},limit=500,resume=True,check_only=False,output=self.output,server_pid=None,recover_overlong=True)
        self.meta=recovery.prepare(self.doc,self.manifest,(True,True))

    def tearDown(self):
        self.assertEqual(SOURCE.read_bytes(),self.source_bytes)

    def save(self):
        p.validate_recovery_history(self.doc)
        p.refresh_planner(self.doc)
        atomic_save(self.output,self.doc)

    def replies(self, bad=False):
        replies=[]
        for r in self.doc['stats']:
            if r['id'] not in self.meta['targets']:continue
            if self.meta['targets'][r['id']]=='decomposition':
                replies.append(FakeResponse('1. First\n2. Second'))
                n=2
            else:n=len(r['steps'])
            credits=[100//n]*n
            credits[0]+=100-sum(credits)
            if bad:credits[0]+=1
            replies.append(FakeResponse(json.dumps({str(i+1):{'credit':c} for i,c in enumerate(credits)})))
        return replies

    def test_selection(self):
        from collections import Counter
        self.assertEqual(Counter(self.meta['targets'].values()),{'decomposition':24,'assessment':4})
        self.assertEqual(len(self.meta['protected_successes']),472)
        self.assertEqual({i for i,k in self.meta['targets'].items() if k=='assessment'},{'129','317','402','365'})

    def test_full_resume_cost_and_final_checker(self):
        client=FakeClient(self.replies())
        with redirect_stdout(io.StringIO()):self.assertTrue(p.run_planner(self.args,self.manifest,self.doc,client,self.save))
        self.assertEqual(len(client.calls),52)
        self.assertEqual(sum(recovery.CREDIT_INSTRUCTION in str(c) for c in client.calls),4)
        self.assertEqual(sum(p.RECOVERY_INSTRUCTION in str(c) for c in client.calls),24)
        self.assertEqual(self.doc['summary']['success'],500)
        costs=self.doc['summary']['eight_b_recovery']['additional_attempt_cost']
        self.assertEqual(costs['attempts'],52)
        self.assertEqual(costs['prompt_tokens']['known_sum'],520)
        self.assertEqual(costs['completion_tokens']['known_sum'],260)
        self.assertEqual(self.doc['summary']['attempt_cost']['attempts'],1028)
        check_complete(self.output,HERE/'math500_manifest.json')
        config=p.planner_config(self.args,self.manifest,original_definitions()[0])
        config['server_settings'].update(quantization=None,tensor_parallel_size=1)
        config['launch_script_sha256']=p.file_hash(HERE/'planner_llama31_8b_y4.sbatch')
        loaded=p.load_planner_checkpoint(self.args,config,self.manifest)
        self.doc=loaded
        empty=FakeClient([])
        with redirect_stdout(io.StringIO()):self.assertTrue(p.run_planner(self.args,self.manifest,self.doc,empty,self.save))
        self.assertEqual(empty.calls,[])
        recovery.verify(self.doc)

    def test_invalid_credit_keeps_failure(self):
        with redirect_stdout(io.StringIO()):self.assertFalse(p.run_planner(self.args,self.manifest,self.doc,FakeClient(self.replies(True)),self.save))
        self.assertEqual(self.doc['summary']['success'],472)
        self.assertEqual(sum(r['stages']['assessment']['status']=='error' for r in self.doc['stats']),28)
        recovery.verify(self.doc)

    def test_interrupted_credit_resumes_without_decomposition(self):
        first=FakeClient([FakeResponse('1. First\n2. Second'),p.StopRequested(15)])
        with redirect_stdout(io.StringIO()), self.assertRaises(p.StopRequested):
            p.run_planner(self.args,self.manifest,self.doc,first,self.save)
        config=p.planner_config(self.args,self.manifest,original_definitions()[0])
        config['server_settings'].update(quantization=None,tensor_parallel_size=1)
        config['launch_script_sha256']=p.file_hash(HERE/'planner_llama31_8b_y4.sbatch')
        self.doc=p.load_planner_checkpoint(self.args,config,self.manifest)
        second=FakeClient([FakeResponse('{"1":{"credit":50},"2":{"credit":50}}'),p.StopRequested(15)])
        with redirect_stdout(io.StringIO()), self.assertRaises(p.StopRequested):
            p.run_planner(self.args,self.manifest,self.doc,second,self.save)
        self.assertIn('guided_json',second.calls[0]['extra_body'])
        self.assertEqual(len(first.calls),2)
        self.assertEqual(len(second.calls),2)
        self.assertEqual(self.doc['summary']['success'],473)
        self.assertEqual(self.doc['summary']['eight_b_recovery']['additional_attempt_cost']['attempts'],4)
        recovery.verify(self.doc)

    def test_failed_attempt_known_cost_included(self):
        row=next(r for r in self.doc['stats'] if r['id']=='129')
        row['stages']['assessment']['attempts'].append({'status':'error',
            'recovery_adapter':self.meta['adapter_sha256'],'purpose':'credit_recovery',
            'usage':{'prompt_tokens':7,'completion_tokens':3,'latency_seconds':1.25}})
        self.save()
        additional=self.doc['summary']['eight_b_recovery']['additional_attempt_cost']
        self.assertEqual(additional['attempts'],1)
        self.assertEqual(additional['total_tokens']['known_sum'],10)
        self.assertEqual(additional['latency_seconds']['known_sum'],1.25)
        self.assertAlmostEqual(self.doc['summary']['attempt_cost']['latency_seconds']['known_sum'],
            self.original['summary']['attempt_cost']['latency_seconds']['known_sum']+1.25,places=6)
        self.assertEqual(self.doc['summary']['success'],472)

    def test_settings_mismatch_refused(self):
        self.save()
        config={**self.original['config'],'seed':999}
        with self.assertRaises(ValueError):p.load_planner_checkpoint(self.args,config,self.manifest)

    def test_decomposition_still_invalid_skips_credit(self):
        replies=[]
        for r in self.doc['stats']:
            if r['id'] not in self.meta['targets']:continue
            if self.meta['targets'][r['id']]=='decomposition':replies.append(FakeResponse('\n'.join(f'{i}. Step' for i in range(1,7))))
            else:replies.append(FakeResponse('{"1":{"credit":0}}'))
        client=FakeClient(replies)
        with redirect_stdout(io.StringIO()):self.assertFalse(p.run_planner(self.args,self.manifest,self.doc,client,self.save))
        self.assertEqual(len(client.calls),28)
        self.assertEqual(self.doc['summary']['success'],472)


if __name__=='__main__':unittest.main()
