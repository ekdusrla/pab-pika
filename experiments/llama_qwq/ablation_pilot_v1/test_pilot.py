"""CPU-only contract/resume tests; fake clients never open a socket."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot as p


class FakeResponse:
    def __init__(self, number):
        self.raw = {'id': 'fake-' + str(number), 'model': p.MODEL,
                    'choices': [{'finish_reason': 'stop', 'index': 0,
                                 'message': {'role': 'assistant', 'content': r'\boxed{2000}'}}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}}

    def model_dump(self, **kwargs):
        return copy.deepcopy(self.raw)


class FakeClient:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail
        self.chat = SimpleNamespace(completions=self)

    def create(self, **payload):
        self.calls.append(copy.deepcopy(payload))
        if self.fail:
            raise self.fail
        return FakeResponse(len(self.calls))


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.planner, cls.solver, _, _ = p.sources()
        cls.selection = p.select(cls.manifest)
        cls.requests = p.make_requests(cls.selection, cls.solver)
        cls.cp = p.initialize(cls.selection, cls.requests, cls.solver, {})

    def test_selection_only_uses_id_level(self):
        stripped = {'records': [{'id': r['id'], 'level': r['level']} for r in reversed(self.manifest['records'])]}
        self.assertEqual(p.select(stripped), self.selection)
        self.assertEqual(len(set(self.selection['ids'])), 100)
        self.assertEqual({l: sum(r['level'] == l for r in self.selection['records']) for l in p.COUNTS}, p.COUNTS)

    def test_exact_reuse_counts(self):
        self.assertEqual(self.requests['equal_reuse_count'], 10)
        self.assertEqual(self.requests['new_solver_calls'], 190)
        self.assertEqual(sum(t['provenance']['kind'] == 'new_generation'
                             for r in self.cp['rows'] for t in r['methods'].values()), 190)

    def test_only_decomposition_block_changes(self):
        rows = {r['id']: r for r in self.solver['comparisons']}
        for req in self.requests['rows']:
            r = rows[req['id']]
            old = req['payloads']['weighted_pab']
            for method in p.NEW:
                new = req['payloads'][method]
                self.assertEqual({k: v for k, v in new.items() if k != 'messages'},
                                 {k: v for k, v in old.items() if k != 'messages'})
                self.assertEqual(new['messages'][0], old['messages'][0])
                weighted = '\n\n'.join(s + p.budget_sentence(b) for s, b in zip(r['steps'], r['words_per_step']))
                expected = ('\n\n'.join(r['steps']) if method == 'plan_only' else
                            '\n\n'.join(s + p.budget_sentence(b) for s, b in zip(r['steps'], req['equal_budgets'])))
                self.assertEqual(new['messages'][1]['content'], old['messages'][1]['content'].replace(weighted, expected, 1))
            self.assertEqual(sum(req['equal_budgets']), r['words_budget'])
            self.assertLessEqual(max(req['equal_budgets']) - min(req['equal_budgets']), 1)

    def test_requests_reject_ambiguous_block(self):
        row = copy.deepcopy(self.solver['comparisons'][0])
        old = '\n\n'.join(s + p.budget_sentence(b) for s, b in zip(row['steps'], row['words_per_step']))
        row['methods']['weighted_pab']['request']['messages'][1]['content'] += old
        with self.assertRaises(ValueError):
            p.payload(row, 'plan_only')

    def test_checkpoint_reuse_cannot_be_modified(self):
        p.validate_checkpoint(self.cp, self.requests, self.solver)
        cp = copy.deepcopy(self.cp)
        cp['rows'][0]['methods']['vanilla']['result']['usage']['completion_tokens'] += 1
        with self.assertRaises(ValueError):
            p.validate_checkpoint(cp, self.requests, self.solver)

    def test_request_tampering_rejected(self):
        cp = copy.deepcopy(self.cp)
        cp['rows'][0]['methods']['plan_only']['request']['seed'] = 43
        with self.assertRaises(ValueError):
            p.validate_checkpoint(cp, self.requests, self.solver)

    def test_diagnostic_does_not_transform_sets_intervals(self):
        for text in [r'\boxed{{(2,\infty)}}', r'\boxed{\{1,2\}}', r'\boxed{{1,2}}', r'\boxed{[0,1)}']:
            ev = {'status': 'incorrect'}
            before = copy.deepcopy(ev)
            result = p.diagnostic(text, ev)
            self.assertIsNone(result['automatic_transformation'])
            self.assertIsNone(result['alternate_accuracy'])
            self.assertEqual(before, ev)
            self.assertEqual(result['content_sha256'], p.digest(text))

    def test_unbalanced_boxed_is_flagged(self):
        d = p.diagnostic(r'\boxed{1', {'status': 'undetermined'})
        self.assertIn('boxed_marker_without_balanced_extraction', d['flags'])

    def test_complete_fake_execution_then_resume_zero_calls(self):
        cp = copy.deepcopy(self.cp)
        client = FakeClient()
        saves = []
        with patch.object(p, 'evaluate_new', side_effect=lambda r, m: r['methods'][m].update(
                evaluation={'status': 'correct'}, evaluation_fingerprint=p.EVAL)), patch('builtins.print'):
            self.assertTrue(p.execute(cp, client, lambda: saves.append(1)))
            self.assertEqual(len(client.calls), 190)
            self.assertTrue(p.execute(cp, client, lambda: saves.append(1)))
            self.assertEqual(len(client.calls), 190)
        p.validate_checkpoint(cp, self.requests, self.solver)
        self.assertGreater(len(saves), 190)

    def test_unknown_previous_attempt_blocks_all_calls(self):
        cp = copy.deepcopy(self.cp)
        t = cp['rows'][0]['methods']['plan_only']
        t['status'] = 'running'
        t['attempts'] = [{'status': 'in_flight', 'usage': {}, 'request_fingerprint': p.digest(t['request'])}]
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            p.execute(cp, client, lambda: None)
        self.assertEqual(client.calls, [])

    def test_resume_flag_required_before_server_start(self):
        cp = copy.deepcopy(self.cp)
        t = cp['rows'][0]['methods']['plan_only']
        t['attempts'] = [{'status': 'interrupted', 'cost_unknown': True}]
        with self.assertRaisesRegex(ValueError, '--resume'):
            p.check_resume(cp, False, False)
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            p.check_resume(cp, True, False)
        p.check_resume(cp, True, True)

    def test_interruption_is_durably_recorded(self):
        t = copy.deepcopy(self.cp['rows'][0]['methods']['plan_only'])
        client = FakeClient(p.StopRequested(15))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint.json'
            with self.assertRaises(p.StopRequested):
                p.run_stage(t, t['request'], client, lambda: p.atomic_save(path, t), lambda result: {})
            saved = p.load_json(path)
            self.assertEqual(saved['status'], 'interrupted')
            self.assertEqual(saved['attempts'][0]['status'], 'interrupted')
            self.assertTrue(saved['attempts'][0]['cost_unknown'])
            self.assertIsNone(saved['attempts'][0]['usage']['completion_tokens'])

    def test_response_received_recovers_without_api_call(self):
        t = copy.deepcopy(self.cp['rows'][0]['methods']['plan_only'])
        response = FakeResponse(1).model_dump()
        result = {'raw_response': response, 'content': r'\boxed{2000}', 'reasoning_content': None,
                  'finish_reason': 'stop', 'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'latency_seconds': 1}}
        t.update(status='response_received', result=result, attempts=[
            {'number': 1, 'status': 'response_received', 'request_fingerprint': p.digest(t['request']), **result}])
        client = FakeClient()
        p.run_stage(t, t['request'], client, lambda: None, lambda result: {})
        self.assertEqual(client.calls, [])
        self.assertEqual(t['status'], 'success')

    def test_failed_api_cost_is_unknown_not_zero(self):
        t = copy.deepcopy(self.cp['rows'][0]['methods']['plan_only'])
        client = FakeClient(ValueError('synthetic API failure'))
        p.run_stage(t, t['request'], client, lambda: None, lambda result: {})
        cost = p.sum_usage([t], True)
        self.assertEqual(cost['completion_tokens']['missing_calls'], 1)
        self.assertTrue(t['attempts'][0]['cost_unknown'])

    def test_undetermined_does_not_replay_successful_response(self):
        cp = copy.deepcopy(self.cp)
        client = FakeClient()
        with patch.object(p, 'evaluate_new', side_effect=lambda r, m: r['methods'][m].update(
                evaluation={'status': 'undetermined'}, evaluation_fingerprint=p.EVAL)), patch('builtins.print'):
            self.assertTrue(p.execute(cp, client, lambda: None))
            self.assertTrue(p.execute(cp, client, lambda: None))
        self.assertEqual(len(client.calls), 190)

    def test_pipeline_costs_and_missing_denominators(self):
        report, rows, audit = p.analyze(self.cp, self.planner)
        self.assertEqual(len(rows), 400)
        self.assertEqual(report['methods']['plan_only']['denominator'], 100)
        self.assertIsNone(report['methods']['plan_only']['final_pipeline']['completion_tokens']['mean'])
        self.assertEqual(report['methods']['plan_only']['final_pipeline']['completion_tokens']['missing_calls'], 100)
        for r in rows:
            if r['method'] in ('vanilla', 'plan_only', 'equal_credit'):
                self.assertEqual(r['final_assessment_completion_tokens'], 0)
        self.assertEqual(report['methods']['weighted_pab']['new_execution_only']['calls'], 0)
        self.assertEqual(report['methods']['equal_credit']['new_execution_only']['calls'], 0)

    def test_plan_only_withholds_assessment_cost(self):
        plan = self.planner['stats'][0]
        self.assertEqual(p.planner_components(plan, 'plan_only'), [plan['stages']['decomposition']])
        self.assertEqual(len(p.planner_components(plan, 'weighted_pab')), 2)
        self.assertEqual(p.planner_components(plan, 'vanilla'), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
