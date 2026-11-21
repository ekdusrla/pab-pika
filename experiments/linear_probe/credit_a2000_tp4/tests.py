"""Supplemental CPU-only tests; no CUDA, model loading or real-data fitting."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from ..resources import limits,validate_allocation
from ..credit_v1.teacher import payload_for as original_payload
from .audit import SERVER,POLICY,check_architecture
from .teacher import payload_for,expected_signature,check_server_config
from .gpu import check_device


class TP4Tests(unittest.TestCase):
    def test_model_partition_dimensions(self):
        config={'hidden_size':4096,'num_attention_heads':32,'num_key_value_heads':8,
                'intermediate_size':14336,'vocab_size':128256}
        self.assertEqual(check_architecture(config)['num_attention_heads']//4,8)
        config['num_key_value_heads']=3
        with self.assertRaises(ValueError):check_architecture(config)

    def test_request_is_identical_to_original_pab_credit(self):
        source={'query':'synthetic','level':1};row={'capture':{'steps':['1. a','2. b']}}
        data={'benchmarks':[{'query':'external','solution':'synthetic','level':1}]}
        self.assertEqual(payload_for(source,row,data),original_payload(source,row,data))
        self.assertEqual(expected_signature()['server']['tensor_parallel_size'],4)
        self.assertEqual(expected_signature()['distributed_executor_backend'],'mp')

    def test_policy_ratio_and_single_gpu_budget(self):
        node={'cpus':48,'gpus':8,'normal_gpus':8,'memory_mib':122174}
        self.assertEqual(limits(node),{'cpus_per_gpu':6,'ram_gb_per_gpu':14})
        validate_allocation(node,4,24,48*1024)
        with self.assertRaises(ValueError):validate_allocation(node,4,25,48*1024)
        with self.assertRaises(ValueError):validate_allocation(node,4,24,57*1024)

    def test_a2000_capacity_and_headroom(self):
        check_device(POLICY['gpu_name'],12*2**30,11*2**30)
        with self.assertRaises(ValueError):check_device(POLICY['gpu_name'],6*2**30,5*2**30)
        with self.assertRaises(ValueError):check_device(POLICY['gpu_name'],12*2**30,9*2**30)

    def test_tp1_or_changed_executor_cannot_mix_with_tp4(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'server.json';path.write_text(json.dumps(SERVER))
            check_server_config(path)
            changed=copy.deepcopy(SERVER);changed['server_settings']['tensor_parallel_size']=1
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):check_server_config(path)
            changed=copy.deepcopy(SERVER);changed['server_settings']['distributed_executor_backend']='ray'
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):check_server_config(path)


if __name__=='__main__':unittest.main()
