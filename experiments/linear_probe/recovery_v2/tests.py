"""CPU contracts: eligibility, accounting, strata and complete feature validation."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ..common import MODEL,REVISION,FEATURE,digest,file_hash,decomp_messages,seal,parse_steps
from ..capture import step_positions
from .audit import overlong,policy,inspect_capture
from .resume import classification,cost_calls,cost_summary


class RecoveryV2Tests(unittest.TestCase):
    def failure(self,count=6):
        text='\n\n'.join(f'{i}. Question {i}\nHint: purpose' for i in range(1,count+1))
        try:parse_steps(text)
        except ValueError as exc:error=str(exc)
        row={'id':'validation/problem','status':'error','attempts':[{'status':'parse_failure',
            'error':{'type':'ValueError','phase':'step_alignment','message':error},
            'diagnostics':{'generated_ids':[1]*20+[9]},'usage':{'prompt_tokens':681,'completion_tokens':21},
            'observations':{'finish_reason':'stop'},'memory':{'peak_reserved_bytes':16*2**30}}]}
        return row,SimpleNamespace(decode=lambda *a,**kw:text)

    def test_eligibility_is_not_hardcoded_to_validation_id(self):
        for n in (6,7,8):
            row,tok=self.failure(n)
            self.assertEqual(overlong(row,tok,{9})['step_count'],n)

    def test_no_repair_for_other_error_or_truncation_or_memory(self):
        row,tok=self.failure()
        for field,value in [('status','oom'),('error',{'type':'RuntimeError'}),
                            ('observations',{'finish_reason':'length'}),
                            ('memory',{'peak_reserved_bytes':22*2**30})]:
            bad=copy.deepcopy(row);bad['attempts'][0][field]=value
            with self.assertRaises(ValueError):overlong(bad,tok,{9})

    def test_policy_reuses_exact_instruction_and_one_repair(self):
        from run_planner import RECOVERY_INSTRUCTION
        self.assertEqual(policy()['instruction'],RECOVERY_INSTRUCTION)
        self.assertEqual(policy()['max_repair_attempts'],1)
        self.assertTrue(policy()['evaluation_frozen_before_first_generation'])
        self.assertEqual(policy()['reserved_limit_gib'],21.)

    def test_failure_strata(self):
        self.assertEqual(classification({'status':'success','attempts':[{'status':'success'}]}),'initial_success')
        self.assertEqual(classification({'status':'success','attempts':[{'status':'parse_failure'},{'status':'success'}]}),'repaired_success')
        self.assertEqual(classification({'status':'error','attempts':[{'status':'parse_failure'},
            {'status':'parse_failure','purpose':'overlong_recovery_v2'}]}),'repair_failed')

    def test_failed_control_is_not_double_counted(self):
        ctrl={'status':'parse_failure','prompt_tokens':10,'completion_tokens':20,'diagnostics':{}}
        row={'attempts':[{'status':'parse_failure','usage':{'prompt_tokens':10,'completion_tokens':20},
                         'capture_control_attempt':ctrl}]}
        self.assertEqual(len(cost_calls(row)),1)
        self.assertEqual(cost_summary(row)['actual_generation_calls'],1)
        self.assertEqual(cost_summary(row)['observed_completion_tokens'],20)
        row['attempts'][0]['capture_control_attempt']['status']='success'
        self.assertEqual(len(cost_calls(row)),2)
        row['attempts'].append({'calls':[{'role':'control','status':'success'},{'role':'capture','status':'success'}]})
        self.assertEqual(len(cost_calls(row)),4)

    def test_full_npz_integrity_and_fingerprint(self):
        import numpy as np
        text='1. First task\nHint: first\n\n2. Second task\nHint: second'
        ids=[ord(x) for x in text]+[128009]
        tok=SimpleNamespace(decode=lambda values,**kw:''.join(chr(x) for x in values if x!=128009),
                            apply_chat_template=lambda *a,**kw:[1,2])
        steps,positions=step_positions(text,[text[:i+1] for i in range(len(text))])
        problem={'id':'p','problem_hash':'hash','query':'synthetic only','level':3}
        data={'fingerprint':'data','splits':{'validation':[problem]}}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'f.npz'
            np.savez_compressed(path,**{str(i):np.ones((2,4096),dtype=np.float32) for i in range(33)})
            cap={'steps':steps,'feature_positions':positions,'text':text,'generated_ids':ids,'finish_reason':'stop',
                'usage':{'completion_tokens':len(ids)},'feature_definition':FEATURE,'prompt_ids_sha256':digest([1,2]),
                'features':{'path':str(path),'sha256':file_hash(path),'n_steps':2}}
            row={'id':'p','problem_hash':'hash','status':'success','capture':cap,'steps_fingerprint':digest(steps),
                 'messages':decomp_messages(problem),'memory':{'peak_reserved_bytes':16*2**30}}
            doc=seal({'split':'validation','data_fingerprint':'data','records':[row],
                'sessions':[{'generation_config':{'eos_token_id':[128009]}}],
                'config':{'model':MODEL,'revision':REVISION,'layers':list(range(33)),
                    'runtime':{'source_sha256':{},'baseline_source_sha256':{}},
                    'decoding':{'temperature':.1,'top_p':.9,'max_new_tokens':1024,'max_context':8192,
                                'dtype':'float16','attention':'eager','seed':42}}})
            self.assertEqual(inspect_capture(doc,data,'validation',tok)['success_count'],1)
            broken=copy.deepcopy(doc);broken['records'][0]['steps_fingerprint']='changed';broken.pop('fingerprint')
            with self.assertRaises(ValueError):inspect_capture(seal(broken),data,'validation',tok)
            with path.open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):inspect_capture(doc,data,'validation',tok)


if __name__=='__main__':unittest.main()
