"""CPU-only qualification for the blocked preparation."""
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .audit import ARMS, DATA, OUT, PILOT, PROBE, model_identity, probe_metadata
from .analysis import accuracy_interval, collect, export, paired_summary, summarize
from .runner import Journal, PROTOCOL, allocations, capture_problem, experiment, journal, position_table, solver_payload
from ..common import FEATURE, LEVEL_WORDS, digest, file_hash, read, verify
from ..solver_y1_v1.analysis import budget_usage
from experiment_state import StopRequested, lock


class Response:
    def model_dump(self, mode):
        return {'choices': [{'message': {'content': 'Answer: $\\boxed{2}$.'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 12}}


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = verify(read(OUT / 'candidate_manifest.json'))
        cls.pilot = verify(read(PILOT))
        cls.train = verify(read(PROBE.parent / 'train_labeled.json'))['records']

    def test_candidate_and_model_identity(self):
        m = self.doc
        self.assertEqual(len(m['rows']), 500)
        self.assertEqual(len({r['id'] for r in m['rows']}), 500)
        self.assertEqual(len({r['problem_hash'] for r in m['rows']}), 500)
        self.assertEqual([sum(r['chunk']==i for r in m['rows']) for i in range(1,11)], [50]*10)
        self.assertEqual(m['untouched_count'], 0)
        self.assertTrue(all(not e['matches'] for e in m['screens'].values()))
        for p, sha in m['input_sha256'].items():
            self.assertEqual(file_hash(p), sha)
        for identity in m['model_identity'].values():
            self.assertEqual(model_identity(identity['model'], identity['revision']), identity)
        with self.assertRaisesRegex(ValueError, 'execution blocked'):
            experiment(OUT / 'candidate_manifest.json')

    def test_frozen_probe_without_fit(self):
        import joblib
        from importlib.metadata import version
        from ..learning import predict
        meta = probe_metadata()
        for name in ('scikit-learn', 'numpy', 'joblib'):
            self.assertEqual(version(name), meta['runtime'][name])
        pipeline = joblib.load(PROBE / 'probe.joblib')
        self.assertEqual(pipeline.n_features_in_, 4096)
        f = verify(read(OUT.parent/'pilot_v1/evaluation_recovered.json'))['records']
        gold = {r['id']:r['credits']['ridge'] for r in self.pilot['solver_requests']}
        for row in f:
            self.assertEqual(predict(pipeline, row, meta['best_layer_idx'])['credits'], gold[row['id']])

    def test_all_five_frozen_payloads(self):
        position = position_table(self.train)
        sources = {r['id']:r for r in verify(read(DATA))['splits']['evaluation']}
        for row in self.pilot['solver_requests']:
            source = sources[row['id']]
            cap = {'steps':row['steps'], 'ridge':{'credits':row['credits']['ridge']}}
            c, b = allocations(source, cap, row['credits']['teacher'], position)
            self.assertEqual(c, row['credits'])
            for arm in ARMS:
                budgets = b.get(arm, [])
                self.assertEqual(budgets, row['budgets_words'][arm])
                if arm != 'vanilla':
                    self.assertEqual(sum(budgets), LEVEL_WORDS[row['level']])
                self.assertEqual(solver_payload(source, row['steps'], arm, budgets, 42), row['requests'][arm])
        self.assertIsNone(budget_usage('a','length',2)['total_word_budget_compliant'])

    def test_request_resume_and_partial_failure(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            calls = []
            outcomes = iter((Response(), StopRequested(15), Response()))

            def create(**payload):
                calls.append(payload)
                value = next(outcomes)
                if isinstance(value, BaseException):
                    raise value
                return value

            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            a, b = {'id':'a'}, {'id':'b'}
            request = {'model':'mock'}
            self.assertTrue(Journal(path, a).api(request, client, lambda r:r['usage']))
            with self.assertRaises(StopRequested):
                Journal(path, b).api(request, client, lambda r:r['usage'])
            self.assertTrue(Journal(path, a).api(request, client, lambda r:r['usage']))
            self.assertTrue(Journal(path, b).api(request, client, lambda r:r['usage']))
            self.assertEqual(len(calls), 3)
            self.assertEqual([x['status'] for x in Journal(path,b).record['stage']['attempts']],
                             ['interrupted','success'])
            with self.assertRaises(ValueError):
                Journal(path,a).api({'model':'different'},client, lambda r:r)
            with lock(path/'writer'), self.assertRaises(RuntimeError), lock(path/'writer'):
                self.fail('Second writer acquired lock')

    def test_capture_resume_keeps_successful_control(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            r = {'query':'mock','level':1}
            j = Journal(path, {'capture':'mock'})
            result = dict(usage={'prompt_tokens':1,'completion_tokens':2,'latency_seconds':.3},
                          generated_ids=[1,2], steps=['1. X','2. Y'], feature_definition=FEATURE)
            with patch('experiments.linear_probe.safety.guarded_generate',
                       side_effect=[dict(status='success',result={**result,'features':None},stop_required=False),
                                   dict(status='interrupted',result=None,stop_required=True,prompt_tokens=1,
                                        completion_tokens=1,diagnostics={})]):
                with self.assertRaises(RuntimeError):
                    capture_problem(j,r,None,None,None,18)
            with patch('experiments.linear_probe.safety.guarded_generate',
                       return_value=dict(status='success',result={**result,'features':[{'18':[.5]},{'18':[.6]}]},
                                         stop_required=False)) as call:
                capture_problem(Journal(path, {'capture':'mock'}),r,None,None,None,18)
            self.assertEqual(call.call_count, 1)
            self.assertTrue(Journal(path, {'capture':'mock'}).record['stage']['result']['same_generated_ids'])

    def test_stats_and_incomplete_result(self):
        ci = accuracy_interval([0]*500)
        self.assertLess(ci[0], 0)
        self.assertGreater(ci[1], 0)
        self.assertEqual(paired_summary([-5, 0, 2])['negative'], 1)
        rows = [dict(id=str(i), level=1, arm=a, correct=(a=='ridge' or i==0),
                     finish_reason='stop', answer_status='correct' if (a=='ridge' or i==0) else 'incorrect',
                     solver_total_tokens=i+1) for i in range(3) for a in ARMS]
        s = summarize(rows, None)
        self.assertAlmostEqual(s['paired']['ridge_vs_teacher']['accuracy_difference_pp'], 200/3)
        self.assertIsNone(s['paired']['ridge_vs_teacher']['noninferiority'])
        with tempfile.TemporaryDirectory() as d, patch('experiments.linear_probe.solver_y3_500_v1.analysis.OUT', Path(d)):
            doc = {'fingerprint':'mock','rows':[{'id':'mock','chunk':1}]}
            conf = read(PROTOCOL)
            with self.assertRaisesRegex(ValueError,'Incomplete execution'):
                collect(doc,conf)
            for arm in ARMS:
                j = journal(Path(d)/'journal/mock/chunk-01',doc,conf,doc['rows'][0],arm)
                j.record['stage']['status']='blocked'
                j.save()
            doc['rows'][0].update(level=1, answer='2')
            gathered, provenance = collect(doc,conf)
            self.assertEqual((len(gathered), len(provenance)), (5,5))
            self.assertTrue(all(r['system_total_tokens'] is None and not r['correct'] for r in gathered))
            dest = Path(d) / 'out'
            export(dest, 42, gathered, provenance, 'mock', None)
            self.assertEqual(len(verify(read(dest / 'seed-42.json'))['rows']), 5)
            with (dest / 'seed-42.csv').open(encoding='utf-8-sig') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 5)
