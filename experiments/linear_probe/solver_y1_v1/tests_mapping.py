"""Compute-node CPU regression tests; synthetic minors are NOT new GPU observations."""
import unittest
from pathlib import Path
from ..common import read,verify
from .gpu_mapping import resolve_mapping,check_visible,requested_ids


class MappingTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]/'runs/solver_y1_v1'
        self.saved=verify(read(root/'launch_solver_221033_smoke.json'))
        self.inventory=self.saved['inventory'];self.visible=[d['uuid'] for d in self.saved['devices']]
        # Hypothetical physical minors test the parser; Job221033 did NOT record them.
        self.identity='\n'.join(f'{i}, GPU-{u}, {i+3}, 00000000:{i+1:02X}:00.0' for i,u in enumerate(self.visible))

    def test_observed_221033_failure_reproduced(self):
        self.assertEqual(self.saved['error']['message'],'Assigned GPU index absent from inventory')
        with self.assertRaisesRegex(ValueError,'Assigned GPU index absent'):
            resolve_mapping('3,4,5,6',self.inventory)

    def test_renumbered_inventory_with_independent_minors(self):
        result=resolve_mapping('3,4,5,6',self.inventory,self.identity)
        self.assertTrue(check_visible(result,self.visible)['passed'])
        self.assertEqual(result['method'],'slurm_numeric_id_to_device_minor_uuid')

    def test_direct_full_node_indices_still_work(self):
        identity=self.identity.replace(', 3,',', 0,').replace(', 4,',', 1,').replace(', 5,',', 2,').replace(', 6,',', 3,')
        self.assertTrue(check_visible(resolve_mapping('0-3',self.inventory,identity),self.visible)['passed'])

    def test_explicit_slurm_uuids(self):
        raw=','.join('GPU-'+u for u in self.visible)
        self.assertTrue(check_visible(resolve_mapping(raw,self.inventory,self.identity),self.visible)['passed'])

    def test_mismatched_visible_uuid_rejected(self):
        result=resolve_mapping('3-6',self.inventory,self.identity)
        with self.assertRaisesRegex(ValueError,'CUDA-visible UUIDs differ'):
            check_visible(result,self.visible[:3]+['wrong'])

    def test_unverifiable_minor_namespace_rejected(self):
        with self.assertRaisesRegex(ValueError,'authoritative GRES mapping needed'):
            resolve_mapping('4-7',self.inventory,self.identity)

    def test_bad_counts_duplicates_ranges_rejected(self):
        for raw in ('','0,1,2','0,1,1,2','0-4','6-3','0-GPU','GPU-x,1,2,3'):
            with self.subTest(raw=raw),self.assertRaises(ValueError):requested_ids(raw)

    def test_missing_or_corrupt_identity_rejected(self):
        for identity in ('',self.identity.replace(', 3,',', N/A,'),self.identity.replace(', 4,',', 3,'),
                         self.identity.replace('GPU-'+self.visible[0],'GPU-other')):
            with self.subTest(identity=identity),self.assertRaises(ValueError):
                resolve_mapping('3-6',self.inventory,identity)

    def test_original_safety_thresholds_unchanged(self):
        from .plan import PROFILES
        p=PROFILES['solver']
        self.assertEqual((p['gpu_count'],p['tp'],p['cpus'],p['ram_gib']),(4,4,16,64))
        self.assertEqual((p['gpu_name'],p['min_total_gib'],p['max_total_gib']),('NVIDIA GeForce RTX 3090',23,25))
        self.assertEqual((p['min_free_gib'],p['used_limit_gib']),(22.,22.5))


if __name__=='__main__':unittest.main()
