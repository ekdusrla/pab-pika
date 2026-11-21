"""CPU-only synthetic tests; no pretrained models, datasets, network or GPU."""
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from .common import (ROOT, FEATURE, digest, seal, prepare, validate_data, project_credits,
    parse_credits, decomp_messages, original_definitions, output_path, problem_hash, allocate_words)
from .capture import generate, step_positions
from .learning import fit_probe, predict, validate_capture, feature_rows
from .cli import checkpoint, capture_signature, main


def fixture():
    evaluation = seal({'records': [{'id': 'eval', 'query': 'EVAL problem', 'level': 1}]})
    source = {'provenance': {'dataset': 'external-math-train', 'revision': 'test'},
              'records': [{'id': str(i), 'problem': f'Train problem {i}', 'level': i % 5 + 1} for i in range(10)],
              'benchmarks': [{'id': f'b{i}', 'query': f'Benchmark {i}', 'level': i, 'solution': 'reference'} for i in range(1,6)]}
    return source, evaluation


def rows_for(data, split):
    rows = []
    for i, source in enumerate(data['splits'][split]):
        steps = ['1. A Hint: a', '2. B Hint: b']
        features = [{'0': [float(i), 1., 0.], '1': [float(i), 0., 1.]},
                    {'0': [float(i), 0., 1.], '1': [float(i), 1., 0.]}]
        rows.append({'id': source['id'], 'problem_hash': source['problem_hash'], 'status': 'success',
            'steps_fingerprint': digest(steps), 'capture': {'steps': steps, 'features': features, 'feature_definition': FEATURE},
            'teacher': {'credits': [30,70], 'steps_fingerprint': digest(steps)}})
    return rows


class ContractTests(unittest.TestCase):
    def test_projection_contract_and_extremes(self):
        for raw in ([20.,30.,50.], [-3.,0.], [0.,0.,0.], [1e308,1e308], [1e-300,2e-300]):
            result = project_credits(raw)
            encoded = json.dumps({str(i): {'credit': c} for i,c in enumerate(result['credits'],1)})
            self.assertEqual(parse_credits(encoded, len(raw)), result['credits'])
            for decay in ('constant','linear','polynomial','exponential','cosine'):
                budgets = allocate_words(result['credits'], 350, decay)
                self.assertEqual(sum(budgets), 350)
                self.assertTrue(all(x >= 0 for x in budgets))
        for raw in ([float('nan'),1.], [float('inf'),2.], [1.]):
            with self.assertRaises(ValueError): project_credits(raw)
        self.assertEqual(project_credits([0.,0.,0.])['credits'], [34,33,33])

    def test_exact_original_prompt(self):
        row = {'query': 'problem', 'level': 3}
        template = original_definitions()[0]['break_down']
        expected = [{'role': role, 'content': template[role].format(domain='math',problem='problem',level=3)} for role in ('system','user')]
        self.assertEqual(decomp_messages(row), expected)
        self.assertNotIn('<<<STEP>>>', str(expected))

    def test_teacher_receives_same_steps_and_external_benchmarks(self):
        from .common import assessment_messages
        row={'query':'question','level':3}
        steps=['1. First. Hint: first hint','2. Second. Hint: second hint']
        benchmark=[{'query':'external reference','level':2,'solution':'external solution'}]
        messages=assessment_messages(row,steps,benchmark)
        self.assertIn(str(steps),messages[1]['content'])
        self.assertIn('external solution',messages[0]['content'])
        with self.assertRaises(ValueError): parse_credits('{"1":{"credit":0},"2":{"credit":100}}',2)

    def test_disjoint_deterministic_manifest(self):
        source, evaluation = fixture()
        first = prepare(source,evaluation)
        self.assertEqual(first,prepare(source,evaluation))
        self.assertEqual(validate_data(first), first)
        self.assertFalse({r['id'] for r in first['splits']['train']} & {r['id'] for r in first['splits']['validation']})

    def test_overlap_and_tamper_rejected(self):
        for query in (' EVAL  problem ', 'ＥＶＡＬ problem'):
            source, evaluation = fixture(); source['records'][0]['problem'] = query
            with self.assertRaises(ValueError): prepare(source,evaluation)
        source, evaluation = fixture(); source['records'][1]['problem'] = source['records'][0]['problem']
        with self.assertRaises(ValueError): prepare(source,evaluation)
        source, evaluation = fixture(); data = prepare(source,evaluation)
        data['splits']['validation'].append(copy.deepcopy(data['splits']['train'][0]))
        with self.assertRaises(ValueError): validate_data(seal({k:v for k,v in data.items() if k!='fingerprint'}))

    def test_baseline_output_guard(self):
        with self.assertRaises(ValueError): output_path(ROOT/'experiments/llama_qwq/runs/math500_llama31_8b_v1/planner.json')
        with self.assertRaises(ValueError): output_path(ROOT/'experiments/linear_probe/runs/../../llama_qwq/planner.json')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'existing.json'; p.write_text(json.dumps(seal({'config': {'x':1}})))
            # checkpoint itself rejects destinations outside isolated runs, even on resume.
            with self.assertRaises(ValueError): checkpoint(SimpleNamespace(output=p,resume=True),{'x':1},{})

    def test_symlink_escape_guard(self):
        from . import common
        with tempfile.TemporaryDirectory() as tmp:
            fake=Path(tmp)/'experiment'; fake.mkdir()
            outside=Path(tmp)/'baseline'; outside.mkdir()
            (fake/'runs').symlink_to(outside,target_is_directory=True)
            with patch.object(common,'__file__',str(fake/'common.py')):
                with self.assertRaises(ValueError): output_path(fake/'runs/planner.json')


class CaptureTests(unittest.TestCase):
    def fake(self, stop=257):
        import torch
        text='1. A\nHint: first\n\n2. B\nHint: second'
        sequence=[ord(c) for c in text]+[stop]
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight=torch.nn.Parameter(torch.zeros(1)); self.index=0
                self.config=SimpleNamespace(num_hidden_layers=1,max_position_embeddings=2048)
                self.generation_config=SimpleNamespace(eos_token_id=[256,257])
            def forward(self,input_ids,**kw):
                if kw.get('past_key_values') is None: self.index=0
                logits=torch.full((1,1,258),-1000.)
                logits[0,0,sequence[self.index]]=1000.; self.index+=1
                h=input_ids[0,-1].float().reshape(1,1,1).repeat(1,1,3)
                return SimpleNamespace(logits=logits,hidden_states=(h,h+1),past_key_values='cached')
        class Tokenizer:
            def apply_chat_template(self,*a,**kw): return torch.tensor([[1,2]])
            def decode(self,ids,**kw): return ''.join(chr(i) for i in ids if i<256)
        return Model(),Tokenizer(),text

    def test_processed_token_alignment_eos_and_zero_extra_forward(self):
        model,tok,text=self.fake()
        result=generate(model,tok,[],layers=[1])
        self.assertEqual(result['text'],text)
        self.assertEqual(result['metrics']['extra_forward_passes'],0)
        self.assertEqual(result['metrics']['forward_calls'],len(text)+1)
        self.assertEqual(len(result['features']),2)
        for position, feature in zip(result['feature_positions'], result['features']):
            # Feature is h[token], not h[token-1].
            self.assertEqual(feature['1'][0],ord(text[position])+1)
        self.assertTrue(result['usage']['completion_tokens']>0)

    def test_truncated_and_malformed_rejected(self):
        model,tok,_=self.fake()
        with self.assertRaisesRegex(ValueError,'truncated'): generate(model,tok,[],max_new_tokens=5)
        with self.assertRaises(ValueError): step_positions('no numbered steps',['no'])

    def test_capture_control_is_separate_generation(self):
        from .capture import capture_pair
        model,tok,text=self.fake()
        result=capture_pair(model,tok,[],layers=[1])
        self.assertTrue(result['capture_control']['same_generated_ids'])
        self.assertEqual(result['capture_control']['metrics']['logical_generation_calls'],1)
        self.assertEqual(result['metrics']['logical_generation_calls'],1)
        self.assertIsNotNone(result['capture_control']['capture_minus_control_seconds'])


class LearningTests(unittest.TestCase):
    def test_scaler_fits_train_only_and_layer_selection(self):
        source,evaluation=fixture(); data=prepare(source,evaluation)
        tr,va=rows_for(data,'train'),rows_for(data,'validation')
        for row in va:
            for f in row['capture']['features']:
                f['0'][0]=10000.; f['1'][0]=10000.
        pipeline,meta=fit_probe(tr,va,[0,1])
        self.assertFalse(meta['validation_used_for_fit'])
        self.assertLess(pipeline['scaler'].mean_[0],100.)
        self.assertEqual(sum(predict(pipeline,va[0],meta['best_layer_idx'])['credits']),100)

    def test_missing_step_and_changed_features_rejected(self):
        source,evaluation=fixture(); data=prepare(source,evaluation)
        rows=rows_for(data,'train')
        doc=seal({'data_fingerprint':data['fingerprint'],'split':'train','records':rows})
        self.assertEqual(validate_capture(doc,data,'train'),rows)
        changed=copy.deepcopy(doc); changed['records'][0]['capture']['features'].pop()
        changed=seal({k:v for k,v in changed.items() if k!='fingerprint'})
        with self.assertRaises(ValueError): validate_capture(changed,data,'train')
        changed=seal({**{k:v for k,v in doc.items() if k!='fingerprint'},'records':rows[:-1]})
        with self.assertRaises(ValueError): validate_capture(changed,data,'train')

    def test_feature_archive_integrity(self):
        import numpy as np
        from .common import file_hash
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'f.npz'; np.savez_compressed(path, **{'3':np.ones((2,4))})
            row={'capture':{'features':{'path':str(path),'sha256':file_hash(path),'n_steps':2}}}
            self.assertEqual(len(feature_rows(row,[3])),2)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError): feature_rows(row,[3])


class IntegrationTests(unittest.TestCase):
    def test_synthetic_train_evaluate_export_and_legacy_solver_consumer(self):
        """Small synthetic Ridge fit, 500 synthetic feature pairs; no LLM training."""
        from .common import read_manifest, save_new, read, MODEL, REVISION
        from run_solver_compare import planner_rows, attach_plans
        source,_=fixture()
        manifest_path=ROOT/'experiments/llama_qwq/math500_manifest.json'
        manifest=read_manifest(manifest_path)
        data=prepare(source,manifest)
        signature={'model':MODEL,'revision':REVISION,'feature_definition':FEATURE,
            'model_config_sha256':'synthetic','weight_index_sha256':'synthetic',
            'decomp_prompt_fingerprint':'synthetic','capture_implementation_sha256':'synthetic',
            'tokenizer_files_sha256':{},'decoding':{}}
        teacher_settings={'model':MODEL,'revision':REVISION}
        def document(split):
            rows=rows_for(data,split)
            for row in rows:
                row['capture'].update(usage={'prompt_tokens':10,'completion_tokens':20,'latency_seconds':.02},
                    metrics={'generation_plus_feature_seconds':.03,'extra_forward_passes':0})
                row['teacher']['usage']={'prompt_tokens':30,'completion_tokens':40,'latency_seconds':.1}
                row['teacher_stage']={'status':'success','parsed':[30,70],'attempts':[]}
            return seal({'schema_version':1,'kind':'labeled_capture','data_fingerprint':data['fingerprint'],
                'split':split,'capture_config':signature,'config':{'teacher_settings':teacher_settings},'records':rows})
        # Restrict synthetic outputs to a TemporaryDirectory; production guard has dedicated tests.
        with tempfile.TemporaryDirectory() as tmp, patch('experiments.linear_probe.common.output_path',side_effect=lambda p:Path(p)), patch('experiments.linear_probe.cli.output_path',side_effect=lambda p:Path(p)):
            root=Path(tmp)
            save_new(root/'data.json',data)
            for split in ('train','validation','evaluation'):
                save_new(root/f'{split}.json',document(split))
            main(['train','--data',str(root/'data.json'),'--train',str(root/'train.json'),
                '--validation',str(root/'validation.json'),'--output',str(root/'probe')])
            # Rerunning a completed artifact must not overwrite it.
            with self.assertRaises(FileExistsError):
                main(['train','--data',str(root/'data.json'),'--train',str(root/'train.json'),
                    '--validation',str(root/'validation.json'),'--output',str(root/'probe')])
            main(['evaluate','--data',str(root/'data.json'),'--capture',str(root/'evaluation.json'),
                '--probe',str(root/'probe'),'--output',str(root/'report.json')])
            report=read(root/'report.json')
            self.assertEqual(report['summary']['paired_problems'],500)
            self.assertTrue(all(r['probe']['logical_generation_calls']==0 for r in report['records']))
            for method in ('probe','teacher'):
                target=root/f'{method}-planner.json'
                main(['export','--data',str(root/'data.json'),'--evaluation',str(root/'report.json'),
                    '--eval-manifest',str(manifest_path),'--method',method,'--output',str(target)])
                plans=planner_rows(read(target),manifest)
                consumer={'config':{'decay':'constant'},'comparisons':[{**r,'methods':{'weighted_pab':{'status':'pending'}}} for r in manifest['records']]}
                attach_plans(consumer,plans)
                self.assertTrue(all(sum(r['words_per_step'])>0 for r in consumer['comparisons']))
            # Probe-only evaluation makes no Teacher request and needs no Teacher labels.
            unlabeled=document('evaluation')
            unlabeled['kind']='capture'; unlabeled['config']=signature
            for row in unlabeled['records']:
                row.pop('teacher'); row.pop('teacher_stage')
            save_new(root/'unlabeled.json',seal({k:v for k,v in unlabeled.items() if k!='fingerprint'}))
            main(['evaluate','--data',str(root/'data.json'),'--capture',str(root/'unlabeled.json'),
                '--probe',str(root/'probe'),'--output',str(root/'probe-only.json')])
            self.assertEqual(read(root/'probe-only.json')['summary']['paired_problems'],0)
            self.assertIsNone(read(root/'probe-only.json')['summary']['credit_mae'])
            # Geometry-compatible but different prompt/model metadata is rejected.
            wrong=copy.deepcopy(unlabeled); wrong['capture_config']['decomp_prompt_fingerprint']='changed'
            save_new(root/'wrong.json',seal({k:v for k,v in wrong.items() if k!='fingerprint'}))
            with self.assertRaisesRegex(ValueError,'Incompatible feature'):
                main(['evaluate','--data',str(root/'data.json'),'--capture',str(root/'wrong.json'),
                    '--probe',str(root/'probe'),'--output',str(root/'wrong-report.json')])
            # Artifact tampering is rejected before deserialization.
            (root/'probe/probe.joblib').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Artifact/data changed'):
                main(['evaluate','--data',str(root/'data.json'),'--capture',str(root/'evaluation.json'),
                    '--probe',str(root/'probe'),'--output',str(root/'report2.json')])

    def test_import_is_weight_free(self):
        import subprocess, sys
        result=subprocess.run([sys.executable,'-B','-c',
            "import sys; import experiments.linear_probe.cli; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
            cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_checkpoint_resume_and_config_guard(self):
        from .common import read
        with tempfile.TemporaryDirectory() as tmp, patch('experiments.linear_probe.cli.output_path',side_effect=lambda p:Path(p)):
            target=Path(tmp)/'run.json'
            target.write_text(json.dumps(seal({'config':{'seed':42},'records':[]})))
            with self.assertRaises(FileExistsError): checkpoint(SimpleNamespace(output=target,resume=False),{'seed':42},{})
            self.assertEqual(checkpoint(SimpleNamespace(output=target,resume=True),{'seed':42},{})['records'],[])
            with self.assertRaises(ValueError): checkpoint(SimpleNamespace(output=target,resume=True),{'seed':43},{})
            Path(str(target)+'.tmp').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Uncommitted'): checkpoint(SimpleNamespace(output=target,resume=True),{'seed':42},{})

    def test_teacher_signature_allows_new_job_but_not_new_model_settings(self):
        from .cli import teacher_signature
        from .common import MODEL,REVISION
        first={'model':MODEL,'revision':REVISION,'temperature':.1,'top_p':.9,
            'slurm_job_id':'1','server_settings':{'dtype':'float16','tensor_parallel_size':1,'max_model_len':8192}}
        second=copy.deepcopy(first); second['slurm_job_id']='2'
        self.assertEqual(teacher_signature(first,42),teacher_signature(second,42))
        second['server_settings']['dtype']='bfloat16'
        self.assertNotEqual(teacher_signature(first,42),teacher_signature(second,42))


if __name__=='__main__':
    unittest.main()
