"""Synthetic, CPU-only checks. Run ONLY in the supplied CPU allocation."""
import copy
import unittest
from ..common import allocate_words,project_credits
from ..resources import validate_allocation
from .plan import teacher_schedule,ARMS,PROFILES
from .client import planned,validate
from .analysis import compare,teacher


class FollowupTests(unittest.TestCase):
    def test_stability_schedule(self):
        ids=list('abcde');s=teacher_schedule(ids)
        self.assertEqual(sum(len(b['ids']) for blocks in s.values() for b in blocks),45)
        self.assertEqual([b['seed'] for b in s['warm'] if b['cohort']=='same_seed'],[42]*3)
        self.assertEqual([b['seed'] for b in s['warm'] if b['cohort']=='different_seed'],[43,44,45])
        self.assertEqual(len(s),4)

    def test_latin_order(self):
        orders=[ARMS[q:]+ARMS[:q] for q in range(5)]
        for col in zip(*orders):self.assertEqual(set(col),set(ARMS))

    def test_teacher_payload_is_copy(self):
        rows=[{'id':k,'prompt_tokens':1,'payload':{'seed':42},'n_steps':2,'steps_fingerprint':k} for k in 'abcde']
        p={'teacher_requests':rows,'teacher_schedule':teacher_schedule(list('abcde'))};before=copy.deepcopy(p)
        records=planned(p,'teacher','warm');self.assertEqual(len(records),30)
        self.assertEqual(p,before);self.assertEqual(records[5]['payload']['seed'],43)
        self.assertEqual(len(planned(p,'teacher','smoke')),1)

    def test_metrics_and_ties(self):
        m=compare([20,30,50],[20,30,50],350)
        self.assertEqual(m['credit_mae_pp'],0);self.assertEqual(m['budget_mae_words'],0)
        self.assertEqual(m['spearman'],1)
        self.assertIsNone(compare([50,50],[40,60],200)['spearman'])
        self.assertEqual(compare([40,60],[60,40],200)['pair_order_agreement_including_ties'],0)

    def test_allocation_uses_original(self):
        for n in range(2,6):
            c=project_credits([1]*n)['credits']
            for words in (200,250,350,450,600):self.assertEqual(sum(allocate_words(c,words,'constant')),words)

    def test_teacher_truncation_and_usage(self):
        r={'content':'x','finish_reason':'length','usage':{'prompt_tokens':100,'completion_tokens':2048,'latency_seconds':1}}
        with self.assertRaises(ValueError):validate(r,'teacher',{'n_steps':2})
        r['finish_reason']='stop';r['usage']['prompt_tokens']=None
        with self.assertRaises(ValueError):validate(r,'teacher',{'n_steps':2})

    def test_resource_profiles(self):
        u={'cpus':48,'gpus':8,'normal_gpus':8,'memory_mib':122174}
        y={'cpus':96,'gpus':8,'normal_gpus':8,'memory_mib':428743}
        for name,node in [('teacher',u),('solver',y)]:
            p=PROFILES[name];validate_allocation(node,4,p['cpus'],p['ram_gib']*1024)
        with self.assertRaises(ValueError):validate_allocation(u,4,48,48*1024)

    def test_analysis_five_units_not_fifteen(self):
        source=[{'id':k,'n_steps':2,'total_words':200,'original_teacher_credits':[40,60],
                 'frozen_predictions':{'probe':[40,60],'uniform':[50,50],'train_position_mean':[45,55]}} for k in 'abcde']
        records=[{'id':k,'cohort':c,'round':i,'seed':42,'stage':{'parsed':{'credits':[40,60]},'attempts':[]}}
                 for k in 'abcde' for c in ('same_seed','different_seed','restart_seed42') for i in range(3)]
        out=teacher({'teacher_requests':source},[{'records':records}])
        for c in out['cohorts'].values():
            self.assertEqual(c['independent_problem_units'],5);self.assertEqual(c['dependent_pair_comparisons'],15)
            self.assertEqual(c['pairwise_macro']['credit_mae_pp']['mean'],0)


if __name__=='__main__':unittest.main()
