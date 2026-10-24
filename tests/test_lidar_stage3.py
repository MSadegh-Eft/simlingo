import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from simlingo_training.lidar.evaluation import (rouge_l, structured_answer,
    dreamer_decision, group_metrics, retention_gate, evaluate_requests, language_metrics)
from tools.lidar_stage3.profile_pipeline import compare_tensors
from simlingo_training.lidar.numerics import cumulative_waypoints,native_grid_positions


class EvaluationContracts(unittest.TestCase):
    def test_route_group_balance_and_population_binding(self):
        rows=[{'uid':str(i),'group':g,'metrics':{'error':v}} for i,(g,v) in enumerate([('a',1.),('a',1.),('a',1.),('b',5.)])]
        metric=group_metrics(rows)['error']
        self.assertEqual(metric['value'],3.)
        self.assertEqual(metric['samples'],4)
        self.assertEqual(metric['groups'],2)
        self.assertEqual(metric,group_metrics(list(reversed(rows)))['error'])
        rows[0]['uid']='different'
        self.assertNotEqual(metric['population_sha256'],group_metrics(rows)['error']['population_sha256'])

    def test_qa_parser_abstains_on_prose_and_negated_claims(self):
        self.assertEqual(structured_answer('A: Yes.'),('category','yes'))
        self.assertIsNone(structured_answer('It is not green.'))
        self.assertIsNone(structured_answer('Yes, but no.'))
        self.assertEqual(structured_answer('A: 20-40 metres.'),('distance_bin','20-40'))
        self.assertEqual(structured_answer('A: No observed vehicle; this does not establish that the lane is clear.'),('category','no'))
        self.assertEqual(structured_answer('A: 4.2 m/s.'),('number',4.2,'m/s'))

    def test_dreamer_grammar_does_not_accept_substring(self):
        self.assertEqual(dreamer_decision('Following the given instruction. Waypoints:'),'follow')
        self.assertEqual(dreamer_decision('Ignore instruction as it leads to collision. Waypoints:'),'refuse')
        self.assertIsNone(dreamer_decision('I cannot confirm following the given instruction.'))

    def test_rouge_l_uses_sequence_not_set_overlap(self):
        self.assertEqual(rouge_l('turn left now','turn left now'),100.)
        self.assertEqual(rouge_l('now left turn','turn left now'),100./3)
        self.assertEqual(rouge_l('','go'),0.)

    def test_retention_requires_every_metric_and_same_population(self):
        baseline={k:{'value':v,'samples':10,'groups':2,'population_sha256':'same'} for k,v in [
            ('route_ade',1.),('speed_ade',2.),('qa_structured_accuracy',80.),
            ('dreamer_decision_accuracy',90.),('commentary_rouge_l',70.)]}
        self.assertTrue(retention_gate(baseline,baseline)['passed'])
        candidate={k:dict(v) for k,v in baseline.items()}
        candidate['route_ade']['value']=1.021
        self.assertFalse(retention_gate(candidate,baseline)['passed'])
        candidate={k:dict(v) for k,v in baseline.items()}
        candidate['qa_structured_accuracy']['value']=77.9
        self.assertFalse(retention_gate(candidate,baseline)['passed'])
        candidate={k:dict(v) for k,v in baseline.items()}
        candidate['speed_ade']['population_sha256']='other'
        self.assertFalse(retention_gate(candidate,baseline)['passed'])
        del candidate['commentary_rouge_l']
        self.assertFalse(retention_gate(candidate,baseline)['passed'])

    def test_evaluator_conditions_driving_on_generated_text_and_restores_mode(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.tokenizer=SimpleNamespace(decode=lambda ids,**_: 'Following the given instruction. Waypoints:')
                self.received=None
            def generate_ids(self,sample,limit):
                self.assert_eval=not self.training and not torch.is_grad_enabled()
                return torch.tensor([[7,8]])
            def driving(self,sample,generated):
                self.received=generated
                return {'route':sample.route[None]+1,'speed_wps':sample.waypoints[None]},None
        model=Model(); sample=SimpleNamespace(uid='a',ids=torch.tensor([1,2,3]),question_ids=torch.tensor([1]),
            answer_mask=torch.tensor([False,True,True]),language_active=True,driving_active=True,
            waypoints=torch.zeros(10,2),route=torch.zeros(20,2))
        sample.to=lambda _:sample
        class Dataset:
            rows=[{'group':'route'}]
            def __getitem__(self,_): return sample
        request={'index':0,'task':'dreamer','replay':False,'seed':42,'position':0,'uid':'a'}
        with patch('torch.autocast',return_value=nullcontext()),patch('torch.cuda.synchronize'):
            result=evaluate_requests(model,Dataset(),{'dreamer':[request]})
        self.assertTrue(model.training)
        self.assertTrue(model.assert_eval)
        self.assertEqual(model.received.tolist(),[[7,8]])
        self.assertAlmostEqual(result['metrics']['route_ade']['value'],2**.5,places=6)
        self.assertEqual(result['metrics']['speed_ade']['value'],0.)

    def test_nonfinite_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            group_metrics([{'uid':'a','group':'g','metrics':{'ade':float('nan')}}])

    def test_reference_chat_header_is_not_scored_as_answer_content(self):
        self.assertEqual(language_metrics('dreamer','Following the given instruction. Waypoints:',
            'assistant\nFollowing the given instruction. Waypoints:')['dreamer_decision_accuracy'],100.)
        self.assertEqual(rouge_l('Follow the route.','assistant\nFollow the route.'),100.)
        metric=language_metrics('qa','A: Yes, the vehicle is ahead.','assistant\nA: Yes, the vehicle is ahead.')
        self.assertEqual(metric['qa_structured_accuracy'],100.)
        self.assertEqual(metric['qa_exact_match'],100.)

    def test_resume_comparison_rejects_one_bad_tensor(self):
        expected={'a':torch.tensor([0.,1.]),'b':torch.tensor([2.])}
        actual={'a':torch.tensor([1e-8,1.]),'b':torch.tensor([2.])}
        result=compare_tensors(actual.items(),expected,atol=1e-6,rtol=1e-5)
        self.assertFalse(result['bitwise']); self.assertTrue(result['within_tolerance'])
        actual['b']=torch.tensor([2.1])
        result=compare_tensors(actual.items(),expected,atol=1e-6,rtol=1e-5)
        self.assertFalse(result['within_tolerance'])
        self.assertAlmostEqual(result['max_absolute_error'],.1,places=6)

    def test_fp32_waypoint_scan_forward_and_reverse_sum_adjoint(self):
        increments=torch.randn(2,20,2,dtype=torch.float64,requires_grad=True)
        weights=torch.randn_like(increments)
        output=cumulative_waypoints(increments)
        torch.testing.assert_close(output,increments.cumsum(1),rtol=1e-12,atol=1e-12)
        gradient=torch.autograd.grad((output*weights).sum(),increments)[0]
        torch.testing.assert_close(gradient,weights.flip(1).cumsum(1).flip(1),rtol=1e-12,atol=1e-12)
        low=torch.tensor([[[256.],[1.],[1.]]],dtype=torch.bfloat16)
        accurate=cumulative_waypoints(low)
        self.assertEqual(accurate.dtype,torch.float32)
        self.assertEqual(accurate.flatten().tolist(),[256.,257.,258.])

    def test_native_position_grid_identity_matches_bicubic(self):
        embedding=SimpleNamespace(image_size=28,patch_size=14)
        positions=torch.randn(1,4,3,requires_grad=True)
        actual=native_grid_positions(embedding,positions,2,2)
        original=torch.nn.functional.interpolate(positions.reshape(1,2,2,3).permute(0,3,1,2),size=(2,2),
            mode='bicubic',align_corners=False).reshape(1,3,4).permute(0,2,1)
        self.assertTrue(torch.equal(actual,original))
        self.assertIs(actual,positions)
        with self.assertRaises(ValueError): native_grid_positions(embedding,positions,4,4)


if __name__=='__main__': unittest.main()
