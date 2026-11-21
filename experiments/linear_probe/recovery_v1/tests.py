"""Compute-node CPU tests of immutable protection and opt-in prompt repair."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ..common import digest,file_hash
from .audit import analyze_failure,verify_protected,protected_rows
from .recover import messages_for_recovery,prepare_document,policy
from run_planner import RECOVERY_INSTRUCTION


class RecoveryTests(unittest.TestCase):
    def test_full_decode_confirms_real_overlong_contract_violation(self):
        for suffix,count in [('precalculus/train/511',7),('counting_and_probability/train/60',6)]:
            text='\n\n'.join(f'{i}. Question {i}\nHint: purpose' for i in range(1,count+1))
            from ..common import parse_steps
            try:parse_steps(text)
            except ValueError as exc:message=str(exc)
            row={'id':'dataset/'+suffix,'attempts':[{'status':'parse_failure',
                'error':{'type':'ValueError','phase':'step_alignment','message':message},
                'diagnostics':{'generated_ids':[1]*20+[9]},'usage':{'prompt_tokens':100,'completion_tokens':21},
                'observations':{'finish_reason':'stop'},'memory':{'peak_reserved_bytes':16*2**30}}]}
            tok=SimpleNamespace(decode=lambda *args,**kw:text)
            result=analyze_failure(row,tok,{9})
            self.assertEqual(result['n_numbered_steps'],count)
            self.assertFalse(result['feature_validation_reached'])
            broken=copy.deepcopy(row);broken['attempts'][0]['memory']['peak_reserved_bytes']=22*2**30
            with self.assertRaises(ValueError):analyze_failure(broken,tok,{9})
            broken=copy.deepcopy(row);broken['attempts'][0]['observations']['finish_reason']='length'
            with self.assertRaises(ValueError):analyze_failure(broken,tok,{9})

    def test_original_prompt_is_retained_and_repair_is_explicit(self):
        from ..common import decomp_messages
        row={'query':'original problem','level':3}
        messages=messages_for_recovery(row,{'decoded_text':'seven-step original draft'})
        self.assertEqual(messages[:2],decomp_messages(row))
        self.assertEqual(messages[2],{'role':'assistant','content':'seven-step original draft'})
        self.assertEqual(messages[3],{'role':'user','content':RECOVERY_INSTRUCTION})
        self.assertEqual(policy()['reserved_limit_gib'],21.)

    def test_protected_row_and_npz_cannot_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'feature.npz';p.write_bytes(b'original opaque feature bytes')
            row={'id':'ok','status':'success','capture':{'features':{'path':str(p),'sha256':file_hash(p)}}}
            source={'records':[row]};protected=protected_rows(source)
            verify_protected(source,protected,True)
            changed=copy.deepcopy(source);changed['records'][0]['status']='error'
            with self.assertRaises(ValueError):verify_protected(changed,protected)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_protected(source,protected,True)

    def test_new_document_preserves_old_rows_and_original(self):
        source={'config':{'output':'old.json','seed':42},'fingerprint':'source-fingerprint',
                'records':[{'id':'ok','status':'success','capture':{'features':{'path':'old.npz','sha256':'sha'}}},
                           {'id':'bad','status':'error','attempts':[{'status':'parse_failure'}]}]}
        old=copy.deepcopy(source)
        approval={'source_checkpoint_sha256':'original-file-sha','fingerprint':'audit-fingerprint',
                  'protected_successes':protected_rows(source),'source_path':'old.json',
                  'immutable_source_snapshot':'snapshot.json','failures':[{'id':'bad'}]}
        proposed=prepare_document(source,approval,Path('new.json'))
        self.assertEqual(source,old)
        self.assertEqual(digest(proposed['records'][0]),digest(source['records'][0]))
        self.assertEqual(proposed['records'][1]['attempts'],source['records'][1]['attempts'])
        self.assertEqual(proposed['config']['output'],'new.json')
        self.assertIn('recovery_v1',proposed['config'])


if __name__=='__main__':unittest.main()
