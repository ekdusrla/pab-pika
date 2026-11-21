"""Compute-node CPU tests; no model, network, CUDA or research-data fitting."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ..common import assessment_messages,original_definitions,credit_schema,project_credits,allocate_words
from .teacher import payload_for,label_one
from .audit import capture_projection
from .train import fit_grid,ranks,spearman,comparison
from experiment_state import stage,attempt_cost


class CreditTests(unittest.TestCase):
    def test_exact_pab_assessment_request(self):
        source={'query':'synthetic problem','level':3};steps=['1. First','2. Second']
        row={'capture':{'steps':steps}};data={'benchmarks':[{'level':1,'query':'benchmark','solution':'answer'}]}
        payload=payload_for(source,row,data)
        self.assertEqual(payload['messages'],assessment_messages(source,steps,data['benchmarks']))
        template=original_definitions()[0]['evaluate_question_level_batch']
        benchmarks='\nLevel 1 Example:\nQ: benchmark\nA: answer\n'
        self.assertEqual(payload['messages'],[{'role':role,'content':template[role].format(
            domain='math',problem=source['query'],steps=steps,benchmarks=benchmarks)} for role in ('system','user')])
        self.assertEqual(payload['extra_body'],{'guided_json':credit_schema(2),'guided_decoding_backend':'outlines'})
        self.assertEqual((payload['temperature'],payload['top_p'],payload['seed'],payload['max_tokens']),(.1,.9,42,2048))

    def response(self,credits):
        content={str(i+1):{'reason':'synthetic','evaluated_level':1,'credit':c} for i,c in enumerate(credits)}
        body={'choices':[{'message':{'content':json.dumps(content)},'finish_reason':'stop'}],
              'usage':{'prompt_tokens':10,'completion_tokens':20}}
        return SimpleNamespace(model_dump=lambda **kw:body)

    def test_transient_retry_costs_are_preserved(self):
        class APIConnectionError(Exception):pass
        call=SimpleNamespace(create=unittest.mock.Mock(side_effect=[APIConnectionError(),self.response([40,60])]))
        client=SimpleNamespace(chat=SimpleNamespace(completions=call));target=stage();saved=[]
        with patch('experiment_state.time.sleep'):
            self.assertTrue(label_one(target,{'synthetic':True},2,client,lambda:saved.append(copy.deepcopy(target)),None))
        self.assertEqual(len(target['attempts']),2);self.assertEqual(target['attempts'][0]['status'],'error')
        self.assertEqual(target['parsed'],[40,60]);self.assertEqual(attempt_cost([target])['attempts'],2)
        self.assertTrue(any(s['status']=='response_received' for s in saved))

    def test_invalid_credit_is_not_silently_retried(self):
        call=SimpleNamespace(create=unittest.mock.Mock(return_value=self.response([49,50])))
        client=SimpleNamespace(chat=SimpleNamespace(completions=call));target=stage()
        self.assertFalse(label_one(target,{},2,client,lambda:None,None))
        with self.assertRaises(ValueError):label_one(target,{},2,client,lambda:None,None)
        self.assertEqual(call.create.call_count,1)

    def test_teacher_keys_do_not_change_capture_but_steps_do(self):
        row={'id':'p','problem_hash':'h','capture':{'steps':['1. a','2. b']},
             'steps_fingerprint':'s','messages':[{'role':'user','content':'original'}]}
        original=capture_projection(row);row['teacher']={'credits':[50,50]}
        self.assertEqual(capture_projection(row),original)
        row['capture']['steps'][0]='changed';self.assertNotEqual(capture_projection(row),original)

    def test_projection_budget_and_rank_contracts(self):
        credits=project_credits([-1.,0.,0.])['credits']
        self.assertEqual(sum(credits),100);self.assertTrue(all(type(c)is int and c>0 for c in credits))
        self.assertEqual(allocate_words([40,35,25],3,'constant'),[2,1,0])
        self.assertEqual(ranks([10,10,30]),[1.5,1.5,3.])
        self.assertIsNone(spearman([50,50],[40,60]));self.assertAlmostEqual(spearman([20,80],[30,70]),1.)

    def test_joint_grid_fits_only_train_and_comparison_is_paired(self):
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        def row(identifier,offset):
            return {'id':identifier,'steps_fingerprint':'f','teacher':{'credits':[30,70]},
                'capture':{'steps':['1. a','2. b'],'features':[
                    {'0':np.array([offset,1.,2.]),'1':np.array([offset,3.,4.])},
                    {'0':np.array([offset+3,2.,1.]),'1':np.array([offset+4,4.,3.])}]}}
        training=[row('t'+str(i),float(i)) for i in range(3)];validation=[row('v',20.)]
        fits=[];scaler_fit=StandardScaler.fit;ridge_fit=Ridge.fit
        def scaler(self,X,*a,**kw):fits.append(('scaler',X.shape));return scaler_fit(self,X,*a,**kw)
        def ridge(self,X,*a,**kw):fits.append(('ridge',X.shape));return ridge_fit(self,X,*a,**kw)
        with patch.object(StandardScaler,'fit',scaler),patch.object(Ridge,'fit',ridge):
            model,meta=fit_grid(training,validation,[0,1],[.1,1.])
        self.assertEqual(len(meta['selection_scores']),4);self.assertFalse(meta['validation_used_for_fit'])
        self.assertTrue(all(shape==(6,3) for _,shape in fits));self.assertEqual(len(fits),8)
        report=comparison(validation,model,meta['best_layer_idx'],{'splits':{'validation':[{'id':'v','level':1}]}},'validation')
        self.assertEqual(report['summary']['problems'],1)
        for budgets in report['records'][0]['budgets'].values():self.assertEqual(sum(budgets),200)


if __name__=='__main__':unittest.main()
