"""CPU-only tests of ramp guards, fixed splits and mocked paired Solver APIs."""
import copy
import json
import random
import string
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .common import seal, digest, problem_hash, allocate_words, LEVEL_WORDS, read, ROOT
from .pilot_data import select, validate_pilot, similarity
from .pilot import report, prepare_solver, solver_summary, main as pilot_main, SOLVER, SOLVER_REVISION
from .length_check import condition, failure_kind, previous_gate
from .submit_pilot import command


def synthetic_data():
    rows=[]
    for i in range(80):
        rng=random.Random(i)
        text='Problem '+''.join(rng.choices(string.ascii_letters,k=120))
        rows.append({'id':str(i),'query':text,'level':i%5+1,'solution':r'\boxed{1}',
                     'answer':'1','problem_hash':problem_hash(text)})
    final=seal({'records':[{'id':'final','query':'Reserved final evaluation question only'}]})
    return select(rows,final,{'dataset':'synthetic','split':'train'},seed=101), rows, final


def synthetic_evaluation(data):
    rows=[]
    for r in data['splits']['evaluation']:
        steps=['1. First. Hint: count','2. Second. Hint: combine']
        rows.append({'id':r['id'],'problem_hash':r['problem_hash'],'steps':steps,'steps_fingerprint':digest(steps),
            'teacher':{'credits':[30,70]},'probe':{'credits':[40,60],'raw':[41.,59.]},
            'budgets':{m:allocate_words(c,LEVEL_WORDS[r['level']],'constant')
                       for m,c in [('teacher',[30,70]),('probe',[40,60])]}})
    return seal({'kind':'evaluation','data_fingerprint':data['fingerprint'],'decay':'constant',
                  'summary':{},'records':rows})


class DataTests(unittest.TestCase):
    def test_fixed_problem_splits_and_external_benchmarks(self):
        data,rows,final=synthetic_data()
        self.assertEqual(data,select(list(reversed(rows)),final,{'dataset':'synthetic','split':'train'},seed=101))
        self.assertEqual({k:len(v) for k,v in data['splits'].items()},{'train':24,'validation':6,'evaluation':5})
        self.assertEqual({r['level'] for r in data['benchmarks']},set(range(1,6)))
        validate_pilot(data)
        changed=copy.deepcopy(data);changed['splits']['evaluation'][0]=changed['splits']['train'][0]
        with self.assertRaises(ValueError): validate_pilot(seal({k:v for k,v in changed.items() if k!='fingerprint'}))

    def test_overlap_duplicate_and_bad_level_excluded(self):
        data,rows,final=synthetic_data()
        rows += [{'id':'overlap','query':final['records'][0]['query'],'answer':'1','level':1},
                 {**rows[0],'id':'duplicate'}, {'id':'unknown','query':'unknown level','level':None,'answer':'1'}]
        result=select(rows,final,{'dataset':'synthetic','split':'train'},seed=101)
        reasons={r['reason'] for r in result['audit']['rejected']}
        self.assertTrue({'math500_normalized_overlap','source_normalized_duplicate','invalid_native_level'} <= reasons)
        self.assertEqual(similarity(r'Compute \left(\dfrac{1}{2}\right)',r'Compute (\frac{1}{2})'),1.)

    def test_prepared_real_manifest_is_cpu_auditable(self):
        path=ROOT/'experiments/linear_probe/runs/pilot_v1/data.json'
        if not path.exists(): self.skipTest('Prepared manifest not available')
        data=validate_pilot(read(path))
        final=read(ROOT/'experiments/llama_qwq/math500_manifest.json')
        forbidden={problem_hash(r['query']) for r in final['records']}
        selected=[*data['benchmarks'],*[r for split in data['splits'].values() for r in split]]
        self.assertFalse(forbidden & {r['problem_hash'] for r in selected})
        self.assertTrue(all('/train/' in r['id'] for r in selected))


class LengthTests(unittest.TestCase):
    def test_padding_changes_only_problem(self):
        class Tok:
            def apply_chat_template(self,messages,**kw): return list(range(sum(len(m['content']) for m in messages)//3))
        items=[condition(Tok(),n) for n in (1024,2048,4096)]
        self.assertEqual(len({i['messages'][0]['content'] for i in items}),1)
        self.assertTrue(all(abs(i['prompt_tokens']-i['target_tokens'])<=16 for i in items))

    def test_failure_categories(self):
        for text,kind in [('CUDA out of memory','oom'),('Model context exceeded','context_exceeded'),
                          ('Decomposition truncated','generation_truncated'),('Expected 2-5 subquestions','parse_failure'),
                          ('network problem','runtime_error')]:
            self.assertEqual(failure_kind(ValueError(text)),kind)

    def test_prior_failure_memory_and_plan_gates(self):
        plan={'fingerprint':'plan'}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'previous.json'
            body={'plan_fingerprint':'plan','target_tokens':2048,'status':'success','safe_to_extend':True,
                  'peak_reserved_bytes':16*2**30}
            path.write_text(json.dumps(seal(body)))
            previous_gate(plan,4096,path,21.)
            for change in ({'status':'oom'},{'peak_reserved_bytes':22*2**30},{'plan_fingerprint':'other'}):
                path.write_text(json.dumps(seal({**body,**change})))
                with self.assertRaises(ValueError): previous_gate(plan,4096,path,21.)
            with self.assertRaises(ValueError): previous_gate(plan,2048,None,21.)

    def test_failure_telemetry_preserves_actual_tokens(self):
        from .test_probe import CaptureTests
        from .capture import generate
        model,tok,_=CaptureTests().fake();trace={}
        with self.assertRaisesRegex(ValueError,'truncated'):
            generate(model,tok,[],max_new_tokens=5,diagnostics=trace)
        self.assertEqual(trace['prompt_tokens'],2)
        self.assertEqual(len(trace['generated_ids']),5)
        self.assertEqual(trace['forward_calls'],5)


class PilotTests(unittest.TestCase):
    def test_24_6_5_mock_teacher_train_compare_end_to_end(self):
        from .test_probe import rows_for
        from .cli import main as cli_main
        from .common import MODEL,REVISION,FEATURE
        data,_,_=synthetic_data();payloads=[]
        signature={'model':MODEL,'revision':REVISION,'feature_definition':FEATURE,
            'model_config_sha256':'synthetic','weight_index_sha256':'synthetic',
            'decomp_prompt_fingerprint':'synthetic','capture_implementation_sha256':'synthetic',
            'tokenizer_files_sha256':{},'decoding':{}}
        class Client:
            def __init__(self,**kw):
                self.models=SimpleNamespace(list=lambda:SimpleNamespace(data=[SimpleNamespace(id=MODEL)]))
                self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def create(self,**payload):
                payloads.append(payload)
                raw={'choices':[{'message':{'content':'{"1":{"credit":30},"2":{"credit":70}}'},'finish_reason':'stop'}],
                     'usage':{'prompt_tokens':20,'completion_tokens':5}}
                return SimpleNamespace(model_dump=lambda **kw:raw)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict('sys.modules',{'openai':SimpleNamespace(OpenAI=Client)}), \
             patch('experiments.linear_probe.common.output_path',side_effect=lambda p:Path(p)), \
             patch('experiments.linear_probe.cli.output_path',side_effect=lambda p:Path(p)):
            root=Path(tmp);(root/'data.json').write_text(json.dumps(data))
            (root/'teacher.json').write_text(json.dumps({'model':MODEL,'revision':REVISION,'temperature':.1,'top_p':.9}))
            for split in ('train','validation','evaluation'):
                rows=rows_for(data,split)
                for row in rows:
                    row.pop('teacher')
                    row['capture'].update(usage={'prompt_tokens':10,'completion_tokens':20,'latency_seconds':.02},
                        metrics={'generation_plus_feature_seconds':.03,'extra_forward_passes':0})
                doc=seal({'kind':'capture','data_fingerprint':data['fingerprint'],'split':split,
                          'config':signature,'records':rows})
                (root/f'{split}.json').write_text(json.dumps(doc))
                cli_main(['label','--data',str(root/'data.json'),'--capture',str(root/f'{split}.json'),
                    '--teacher-config',str(root/'teacher.json'),'--output',str(root/f'{split}_labeled.json')])
            self.assertEqual(len(payloads),35)
            self.assertTrue(all("['1. A Hint: a', '2. B Hint: b']" in p['messages'][1]['content'] for p in payloads))
            cli_main(['train','--data',str(root/'data.json'),'--train',str(root/'train_labeled.json'),
                '--validation',str(root/'validation_labeled.json'),'--output',str(root/'probe')])
            cli_main(['evaluate','--data',str(root/'data.json'),'--capture',str(root/'evaluation_labeled.json'),
                '--probe',str(root/'probe'),'--output',str(root/'evaluation_core.json')])
            result=report(data,read(root/'evaluation_core.json'))
            self.assertEqual(result['summary']['paired_problems'],5)
            self.assertTrue(all(len(r['step_differences'])==2 for r in result['records']))
            self.assertEqual(len(prepare_solver(data,result)['rows']),5)

    def test_credit_metrics_and_exact_budget_solver_prompts(self):
        from run_solver_compare import messages_for
        from .common import original_definitions
        data,_,_=synthetic_data(); evaluation=synthetic_evaluation(data)
        comparison=report(data,evaluation)
        self.assertEqual(comparison['summary']['step_micro_mae'],10.)
        self.assertEqual(comparison['summary']['step_micro_rmse'],10.)
        requests=prepare_solver(data,evaluation)
        self.assertEqual(sum(len(r['requests']) for r in requests['rows']),10)
        _,instruction,fmt=original_definitions()
        for row,source,ev in zip(requests['rows'],data['splits']['evaluation'],evaluation['records']):
            for m in ('teacher','probe'):
                payload=row['requests'][m]
                self.assertEqual(payload['messages'],messages_for({**source,'steps':row['steps']},'weighted_pab',ev['budgets'][m],instruction,fmt))
                self.assertEqual(payload['max_tokens'],8192)
        broken=copy.deepcopy(evaluation);broken['records'].pop()
        with self.assertRaises(ValueError): report(data,seal({k:v for k,v in broken.items() if k!='fingerprint'}))

    def test_mocked_solver_failure_resume_preserves_ten_pairs(self):
        data,_,_=synthetic_data(); requests=prepare_solver(data,synthetic_evaluation(data))
        calls=[]
        class Client:
            fail=True
            def __init__(self,**kw):
                self.models=SimpleNamespace(list=lambda:SimpleNamespace(data=[SimpleNamespace(id=SOLVER)]))
                self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
            def __enter__(self): return self
            def __exit__(self,*a): pass
            def create(self,**payload):
                calls.append(payload)
                if Client.fail:
                    Client.fail=False;raise ValueError('synthetic API failure')
                raw={'choices':[{'message':{'content':r'\boxed{1}'},'finish_reason':'stop'}],
                     'usage':{'prompt_tokens':12,'completion_tokens':3}}
                return SimpleNamespace(model_dump=lambda **kw:raw)
        tok=SimpleNamespace(apply_chat_template=lambda *a,**kw:[1]*20)
        modules={'openai':SimpleNamespace(OpenAI=Client),
                 'transformers':SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:tok))}
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules',modules), \
             patch('experiments.linear_probe.common.output_path',side_effect=lambda p:Path(p)), \
             patch('experiments.linear_probe.cli.output_path',side_effect=lambda p:Path(p)), \
             patch('experiments.linear_probe.pilot.output_path',side_effect=lambda p:Path(p)), \
             patch('experiments.linear_probe.pilot.evaluator_info',return_value={'fingerprint':'fake'}), \
             patch('experiments.linear_probe.pilot.evaluate_answer',return_value={'status':'correct'}):
            root=Path(tmp);(root/'requests.json').write_text(json.dumps(requests))
            (root/'server.json').write_text(json.dumps({'model':SOLVER,'revision':SOLVER_REVISION,
                'server_settings':{'dtype':'float16','tensor_parallel_size':4,'max_model_len':12288}}))
            args=['solver-run','--requests',str(root/'requests.json'),'--server-config',str(root/'server.json'),
                  '--tokenizer-path',str(root/SOLVER_REVISION),'--output',str(root/'results.json'),'--retries','0']
            with self.assertRaises(RuntimeError):pilot_main(args)
            self.assertEqual(len(calls),10)
            pilot_main([*args,'--resume'])
            self.assertEqual(len(calls),11)  # Only the single failed request is repeated.
            result=read(root/'results.json')
            self.assertEqual(result['summary']['methods']['teacher']['accuracy_all_5'],1.)
            self.assertEqual(result['summary']['methods']['probe']['accuracy_all_5'],1.)
            self.assertEqual(len(result['summary']['paired']),5)
            with self.assertRaises(FileExistsError):pilot_main(args)


class ResourceTests(unittest.TestCase):
    def test_user_node_ratio_policy_and_memory_headroom(self):
        from .resources import parse_nodes,limits,common_limits,validate_allocation
        nodes=parse_nodes('NodeName=example CPUTot=96 RealMemory=428032 Gres=gpu:normal:8(S:0-7) Partitions=batch_ce_ugrad AvailableFeatures=rtx3090')
        self.assertEqual(limits(nodes[0]),{'cpus_per_gpu':12,'ram_gb_per_gpu':49})
        self.assertEqual(common_limits(nodes,4)['cpus_per_gpu'],12)
        validate_allocation(nodes[0],4,48,196*1024)
        with self.assertRaises(ValueError):validate_allocation(nodes[0],1,13,49*1024)
        with self.assertRaises(ValueError):validate_allocation(nodes[0],1,12,51*1024)
        with self.assertRaises(ValueError):common_limits(nodes,1,constraint='not-advertised')
    def test_resource_scaling_and_print_only_command(self):
        one=command('capture','experiments/linear_probe/runs/pilot_v1',4,32)
        four=command('solver','experiments/linear_probe/runs/pilot_v1',4,32)
        self.assertIn('--cpus-per-task=4',one);self.assertIn('--mem=32G',one)
        self.assertIn('--cpus-per-task=16',four);self.assertIn('--mem=128G',four)
        self.assertIn('--time=02:00:00',four)
        self.assertFalse(any('nodelist' in x for x in four))
        with self.assertRaises(ValueError):command('length','new',4,32,target=4096)
        with self.assertRaises(ValueError):command('capture','new',0,32)


if __name__=='__main__':unittest.main()
