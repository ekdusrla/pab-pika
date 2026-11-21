"""CPU-only prerequisite regression tests; synthetic checkpoints in /tmp."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from check_8b_planner import validate, MODEL, REVISION
from experiment_state import read_manifest, SCHEMA_VERSION

HERE = Path(__file__).resolve().parent


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pab_8b_preflight_')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'planner.json'
        self.manifest = HERE / 'math500_manifest.json'
        manifest = read_manifest(self.manifest)
        steps = ['1. First', '2. Second']
        self.document = {'schema_version': SCHEMA_VERSION, 'config': {'model': MODEL, 'revision': REVISION,
            'manifest_fingerprint': manifest['fingerprint']}, 'stats': [{**r, 'status': 'success',
            'steps': steps, 'credits': [50, 50], 'stages': {
                'decomposition': {'status': 'success', 'parsed': steps, 'result': {'content': '1. First\n2. Second'}},
                'assessment': {'status': 'success', 'parsed': [50,50], 'result': {'content': '{"1":{"credit":50},"2":{"credit":50}}'}}}}
            for r in manifest['records']]}

    def check(self):
        self.path.write_text(json.dumps(self.document))
        before = self.path.read_bytes()
        try:
            return validate(self.path, self.manifest)
        finally:
            self.assertEqual(before, self.path.read_bytes())

    def test_complete_readonly(self):
        self.assertEqual(self.check(), read_manifest(self.manifest)['fingerprint'])

    def test_missing(self):
        with self.assertRaisesRegex(ValueError, 'Run the 8B Planner first'):
            validate(self.path, self.manifest)

    def test_pending(self):
        self.document['stats'][0]['stages']['assessment']['status'] = 'pending'
        with self.assertRaisesRegex(ValueError, '499/500'): self.check()

    def test_duplicates(self):
        self.document['stats'][1] = copy.deepcopy(self.document['stats'][0])
        with self.assertRaisesRegex(ValueError, 'unique IDs'): self.check()

    def test_wrong_model(self):
        self.document['config']['model'] = '70B'
        with self.assertRaisesRegex(ValueError, 'model/revision'): self.check()

    def test_bad_credit(self):
        self.document['stats'][0]['credits'] = [49,50]
        with self.assertRaisesRegex(ValueError, 'steps/credits'): self.check()

    def test_raw_mismatch(self):
        self.document['stats'][0]['stages']['decomposition']['result']['content'] = '1. Changed\n2. Second'
        with self.assertRaisesRegex(ValueError, 'Raw response'): self.check()

    def test_manifest_mismatch(self):
        self.document['config']['manifest_fingerprint'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'frozen manifest'): self.check()


if __name__ == '__main__':
    unittest.main()
