"""Compute-node CPU tests only; fake CUDA counters and synthetic features."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .safety import (MemoryGuard,MemoryLimitExceeded,guarded_generate,validate_features,
                     apply_attempt,acknowledge_resume,require_device)
from .common import read,digest,REVISION


class FakeCUDA:
    def __init__(self):
        self.allocated=15*2**30;self.reserved=16*2**30
        self.reset_peak_memory_stats()
    def reset_peak_memory_stats(self):
        self.peak_allocated=self.allocated;self.peak_reserved=self.reserved
    def memory_allocated(self):return self.allocated
    def memory_reserved(self):return self.reserved
    def max_memory_allocated(self):return self.peak_allocated
    def max_memory_reserved(self):return self.peak_reserved
    def grow(self,gib):
        self.reserved=int(gib*2**30);self.peak_reserved=max(self.reserved,self.peak_reserved)


def cap():
    return {'steps':['1. A','2. B'],'features':[{'0':[1.,2.,3.]},{'0':[4.,5.,6.]}],
        'feature_positions':[0,1],'generated_ids':[3,9],'finish_reason':'stop',
        'usage':{'prompt_tokens':10,'completion_tokens':2,'latency_seconds':.1},
        'metrics':{'generation_seconds':.1,'generation_plus_feature_seconds':.12}}


def outcome(status='success'):
    return {'status':status,'result':cap() if status=='success' else None,
        'memory':{'allocated_bytes':15*2**30,'reserved_bytes':16*2**30,
            'peak_allocated_bytes':15*2**30,'peak_reserved_bytes':(22 if status=='memory_limit_exceeded' else 16)*2**30},
        'diagnostics':{'generated_ids':[3,9]},'error':None if status=='success' else {'message':'guard'},
        'completion_tokens':2,'length_coverage':{'worst_case_verified':False},
        'stop_required':status in ('memory_limit_exceeded','oom','interrupted')}


class GuardTests(unittest.TestCase):
    def test_exact_threshold_and_interval_abort(self):
        cuda=FakeCUDA();guard=MemoryGuard(cuda);guard.begin()
        cuda.grow(21);guard.check('equal_threshold')
        cuda.grow(21.01)
        with self.assertRaises(MemoryLimitExceeded):guard.check('model_forward')
        with self.assertRaises(ValueError):MemoryGuard(cuda,21.1)

    def test_generation_memory_failure_preserves_trace(self):
        cuda=FakeCUDA();guard=MemoryGuard(cuda)
        def fake(*args,**kw):
            kw['diagnostics'].update(prompt_tokens=939,generated_ids=[3],forward_calls=1)
            cuda.grow(22);kw['memory_check']('model_forward')
        model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[9]))
        with patch('experiments.linear_probe.capture.generate',side_effect=fake):
            result=guarded_generate(model,None,[],guard,[0],hidden_size=3)
        self.assertEqual(result['status'],'memory_limit_exceeded')
        self.assertEqual(result['completion_tokens'],1)
        self.assertTrue(result['stop_required']);self.assertFalse(result['safe'])
        self.assertEqual(result['memory']['peak_reserved_bytes'],22*2**30)

    def test_real_decode_hook_stops_before_next_token(self):
        from .test_probe import CaptureTests
        model,tok,_=CaptureTests().fake();cuda=FakeCUDA();guard=MemoryGuard(cuda)
        original=guard.check
        def check(phase):
            if phase=='model_forward':cuda.grow(22)
            original(phase)
        with patch.object(guard,'check',side_effect=check):
            result=guarded_generate(model,tok,[],guard,[0],hidden_size=3)
        self.assertEqual(result['status'],'memory_limit_exceeded')
        self.assertEqual(model.index,1)
        self.assertEqual(result['diagnostics']['forward_calls'],1)
        self.assertEqual(result['completion_tokens'],0)

    def test_early_eos_is_not_worst_case_proof(self):
        from .test_probe import CaptureTests
        model,tok,_=CaptureTests().fake()
        result=guarded_generate(model,tok,[],MemoryGuard(FakeCUDA()),[0,1],hidden_size=3)
        self.assertEqual(result['status'],'success')
        self.assertEqual(result['finish_reason'],'stop')
        self.assertLess(result['completion_tokens'],1024)
        self.assertFalse(result['length_coverage']['worst_case_verified'])
        self.assertFalse(result['length_coverage']['near_cap_90_percent'])

    def test_truncation_is_distinct(self):
        from .test_probe import CaptureTests
        model,tok,_=CaptureTests().fake()
        result=guarded_generate(model,tok,[],MemoryGuard(FakeCUDA()),[0],max_new_tokens=5,hidden_size=3)
        self.assertEqual(result['status'],'generation_truncated')
        self.assertEqual(result['completion_tokens'],5)
        self.assertEqual(result['finish_reason'],'length')

    def test_parse_oom_and_feature_failures(self):
        model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[9]))
        for exc,expected in [(ValueError('Expected 2-5 subquestions'),'parse_failure'),
                             (RuntimeError('CUDA out of memory'),'oom')]:
            with patch('experiments.linear_probe.capture.generate',side_effect=exc):
                result=guarded_generate(model,None,[],MemoryGuard(FakeCUDA()),[0],hidden_size=3)
            self.assertEqual(result['status'],expected)
        invalid=cap();invalid['features'][0]['0'][1]=float('nan')
        with self.assertRaises(ValueError):validate_features(invalid,[0],3)

    def test_non_3090_rejected_before_loading(self):
        cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:1,
                             get_device_name=lambda i:'Tesla V100')
        with self.assertRaisesRegex(RuntimeError,'before model loading'):
            require_device(SimpleNamespace(cuda=cuda))

    def test_memory_measurement_failure_is_not_a_pass(self):
        cuda=FakeCUDA();guard=MemoryGuard(cuda)
        with patch.object(cuda,'memory_reserved',side_effect=RuntimeError('driver error')):
            with self.assertRaisesRegex(RuntimeError,'measurement failed'):guard.begin()
            self.assertIsNone(guard.snapshot()['peak_reserved_bytes'])


class CollectionTests(unittest.TestCase):
    def test_completed_rows_survive_stop_and_explicit_resume(self):
        from .cli import collect_rows
        rows=[{'id':str(i),'query':str(i),'level':1,'problem_hash':str(i)} for i in (1,2,3)]
        data={'splits':{'train':rows}}
        doc={'config':{},'records':[],'sessions':[{}],'last_successful_problem_id':None,'safety_stop':None}
        with tempfile.TemporaryDirectory() as tmp, \
            patch('experiments.linear_probe.common.output_path',side_effect=lambda p:Path(p)), \
            patch('experiments.linear_probe.cli.output_path',side_effect=lambda p:Path(p)):
            args=SimpleNamespace(output=Path(tmp)/'capture.json',split='train',seed=42,capture_control=False)
            with patch('experiments.linear_probe.safety.guarded_generate',side_effect=[outcome(),outcome('memory_limit_exceeded')]) as generate:
                with self.assertRaises(RuntimeError):collect_rows(args,data,doc,None,None,MemoryGuard(FakeCUDA()),[0])
                self.assertEqual(generate.call_count,2)
            stored=read(args.output)
            self.assertEqual(stored['last_successful_problem_id'],'1')
            self.assertEqual(stored['safety_stop']['problem_id'],'2')
            self.assertEqual(len(stored['records']),2)  # Third problem never started.
            successful_hash=digest(stored['records'][0])
            with self.assertRaises(ValueError):acknowledge_resume(stored,True,False)
            acknowledge_resume(stored,True,True)
            with patch('experiments.linear_probe.safety.guarded_generate',side_effect=[outcome(),outcome()]) as generate:
                collect_rows(args,data,stored,None,None,MemoryGuard(FakeCUDA()),[0])
                self.assertEqual(generate.call_count,2)
            self.assertEqual(digest(stored['records'][0]),successful_hash)
            self.assertEqual(len(stored['records'][1]['attempts']),2)
            self.assertEqual(stored['last_successful_problem_id'],'3')
            self.assertEqual(len(stored['safety_stop_history']),1)

    def test_acknowledgement_does_not_relax_configuration(self):
        from .cli import configuration
        args=SimpleNamespace(resume=True,resume_after_safety_stop=True,memory_limit_gib=21.,seed=42)
        self.assertEqual(configuration(args),{'memory_limit_gib':21.,'seed':42})
        with self.assertRaises(ValueError):acknowledge_resume({},False,True)


class SmokeTests(unittest.TestCase):
    def test_longest_real_input_selection_excludes_benchmarks(self):
        from .pilot_smoke import longest_input
        class Tok:
            def apply_chat_template(self,messages,**kw):return list(range(len(messages[1]['content'])))
        def row(i,n):return {'id':i,'query':'x'*n,'level':1}
        data={'benchmarks':[row('not-collected',10000)],
              'splits':{'train':[row('a',5)],'validation':[row('b',100)],'evaluation':[row('c',10)]}}
        self.assertEqual(longest_input(data,Tok())['row']['id'],'b')

    def test_preflight_failure_json_and_no_overwrite(self):
        from .pilot_smoke import run
        with tempfile.TemporaryDirectory() as tmp, \
            patch('experiments.linear_probe.common.output_path',side_effect=lambda p:Path(p)), \
            patch('experiments.linear_probe.pilot_smoke.output_path',side_effect=lambda p:Path(p)):
            root=Path(tmp)
            args=SimpleNamespace(output=root/'smoke.json',model_path=root/REVISION,
                                 data=root/'missing-data.json',context_plan=root/'context.json')
            with self.assertRaises(RuntimeError):run(args)
            saved=read(args.output);original=args.output.read_bytes()
            self.assertFalse(saved['passed']);self.assertEqual(saved['phase'],'preflight')
            with self.assertRaises(FileExistsError):run(args)
            self.assertEqual(args.output.read_bytes(),original)

    def test_uncommitted_smoke_result_is_protected(self):
        from .pilot_smoke import run
        with tempfile.TemporaryDirectory() as tmp, \
            patch('experiments.linear_probe.pilot_smoke.output_path',side_effect=lambda p:Path(p)):
            path=Path(tmp)/'smoke.json';pending=Path(str(path)+'.tmp')
            pending.write_text('previous diagnostic')
            with self.assertRaises(FileExistsError):run(SimpleNamespace(output=path))
            self.assertFalse(path.exists())
            self.assertEqual(pending.read_text(),'previous diagnostic')


if __name__=='__main__':unittest.main()
