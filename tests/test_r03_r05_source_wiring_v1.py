"""Offline thin-preparer wiring tests. No files in formal runs and no model API."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import prepare_r03_r05_refinement_v1 as prep
from jev_mem import r05_refinement_v1 as core
from jev_mem import source_support_stack_v1 as stack
from jev_mem import source_append_r05 as old
from jev_mem.evidence_exchange_r01 import digest


class MemoryReader:
    def pointer(self,p):return {'path':str(p),'sha256':digest(str(p))}


def fixture():
    inputs={'envelopes':[{'session_id':'session_1','observation_date':'2023-06-01',
        'source_turn_ids':[f'D1:{i}' for i in range(14)],
        'messages':[{'content':f'Alice: boots hiking item {i}; complete source sentence.'} for i in range(14)]}]}
    turns=old.source_turns(inputs,{'path':'fixture.json','sha256':'a'*64})
    scope={'session_id':'session_1','source_turn_ids':[t['source_turn_id'] for t in turns],
        'historical_turn_ids':[],'existing_memory_ids':[]}
    facts=[{'id':f'f{i}','text':f'Alice likes boots hiking item {i}.','observation_date':'2023-06-01'} for i in range(20)]
    raw={f['id']:{**f,'input_scope':deepcopy(scope)} for f in facts}
    supports={}
    for f in facts:
        plan=stack.build_support_plan(f,scope,turns,conversation=0)
        supports[f['id']]={'plan':plan,'scores':{k:.9 for r in plan['requests'] for k in r['key_to_window_index']},
            'status':'complete','unit_ref':{'path':'support-unit'},'result_refs':[]}
    job={'query_id':'c0_q0','conversation':0,'question':'boots hiking','reference_date':None,
        'reference':'FORBIDDEN_GOLD_QA','selected_facts':deepcopy(facts),'selected_ids':[f['id'] for f in facts],
        'bank':{'path':'bank'},'selection':{'path':'selection'}}
    b={'original_jobs':[job],'pools':{'c0_q0':{'original_r03':deepcopy(facts)}}}
    m={'protocol_sha256':'p'*64,'pool_inputs':{'path':'pools'},'protocol':{'frozen_source':{'path':'frozen'}}}
    refined={'c0_q0':{'records':list(reversed(deepcopy(facts))),
        'decision':{'source':'mock-r03-decision'},'set_unit':{'path':'set-unit'},'score_refs':[]}}
    return m,b,{0:{'turns':turns,'rawfacts':raw}},{0:supports},refined


def rows_for(requests,status='complete',value=.9):
    return {r['request_id']:{'status':status,'scores':{k:value for k in r['signature']['body']['questions']}
        if status=='complete' else None,'result_ref':{'path':r['request_id']+'.response'}} for r in requests}


class SourceWiringTest(unittest.TestCase):
    def setUp(self):
        self.m,self.b,self.sources,self.supports,self.refined=fixture()
        self.reader=MemoryReader();self.run=Path('memory-only-no-formal-run')

    def multi(self):
        with patch.object(prep,'_resolved_r03',return_value=self.refined),patch.object(prep,'support_results',return_value=self.supports):
            return list(prep.source_phase_values(self.reader,self.run,'multi_compare',self.m,self.b,self.sources))

    def nextphase(self,phase,previous,requests,status='complete',value=.9):
        refs=[({'path':u['unit_id']+'.unit'},u) for u,_ in previous]
        with patch.object(prep,'phase_results',return_value=rows_for(requests,status,value)),patch.object(prep,'units',return_value=refs):
            return list(prep.source_phase_values(self.reader,self.run,phase,self.m,self.b,self.sources))

    def test_four_source_upstreams_multi_then_closure_then_rank_gold_free(self):
        multi=self.multi();mu,mreq=multi[0]
        self.assertEqual(mu['records']['original'],self.b['pools']['c0_q0']['original_r03'])
        self.assertEqual(mu['records']['refined'],self.refined['c0_q0']['records'])
        self.assertEqual(len(mreq),2)
        closure=self.nextphase('closure',multi,mreq);cu,creq=closure[0]
        self.assertEqual(set(cu['locators']),set(prep.SOURCE_ARMS))
        for arm,key,is_multi in prep.source_arm_specs():
            expected=cu['multi_resolved'][key]['locator'] if is_multi else mu['locators'][key]
            self.assertEqual(cu['locators'][arm],expected)
            self.assertTrue(all(plan['base']==base for plan,base in
                zip(cu['plans'][arm],expected['candidates'])))
            self.assertEqual(cu['records'][key],mu['records'][key])
        rank=self.nextphase('rank',closure,creq);ru,rreq=rank[0]
        for arm,key,_ in prep.source_arm_specs():
            sig=core.build_rank_request('boots hiking',None,cu['records'][key],ru['locators'][arm]['candidates'])
            actual=next((r['signature'] for r in rreq if r['request_id']==ru['request_ids'][arm]),None)
            self.assertEqual(sig,actual)
            self.assertTrue(ru['locators'][arm]['closure']['enabled'])
        for req in mreq+creq+rreq:
            self.assertNotIn('FORBIDDEN_GOLD_QA',old.canonical(req['signature']))
            self.assertNotIn('reference"',old.canonical(req['signature']))

    def test_complete_sevenarm_views_final20_and_distinct_two_attachments(self):
        multi=self.multi();closure=self.nextphase('closure',multi,multi[0][1])
        rank=self.nextphase('rank',closure,closure[0][1]);ru,rreq=rank[0]
        with patch.object(prep,'base',return_value=(self.m,self.b,self.sources)),\
            patch.object(prep,'_resolved_r03',return_value=self.refined),\
            patch.object(prep,'phase_results',return_value=rows_for(rreq)),\
            patch.object(prep,'units',return_value=[({'path':'rank.unit'},ru)]):
            selections=list(prep.expected_selections(self.reader,self.run))
        self.assertEqual([arm for _,arm,_ in selections],list(prep.ARMS))
        for _,arm,rec in selections:
            self.assertEqual(len(rec['selected_facts']),20)
            self.assertEqual(len(set(rec['selected_ids'])),20)
            if arm in prep.SOURCE_ARMS:
                key=next(k for a,k,_ in prep.source_arm_specs() if a==arm)
                upstream=ru['records'][key]
                for f,original in zip(rec['selected_facts'],upstream):
                    self.assertEqual(f['id'],original['id']);self.assertEqual(f['observation_date'],original['observation_date'])
                    self.assertTrue(f['text'].startswith(original['text']))
                self.assertLessEqual(len(rec['attachments']),2)
                self.assertEqual(len({c['packet_id'] for c in rec['attachments']}),len(rec['attachments']))
                self.assertEqual(len({c['origin']['anchor_slot'] for c in rec['attachments']}),len(rec['attachments']))

    def test_failed_multi_and_closure_keep_original_unknown_blocks(self):
        multi=self.multi();mu,mreq=multi[0]
        closure=self.nextphase('closure',multi,mreq,status='failed');cu,creq=closure[0]
        for key in ('original','refined'):
            self.assertEqual(cu['multi_resolved'][key]['locator']['candidates'],mu['locators'][key]['candidates'])
            self.assertTrue(all(d['comparison_probability'] is None for d in cu['multi_resolved'][key]['decisions']))
        rank=self.nextphase('rank',closure,creq,status='failed');ru,_=rank[0]
        for arm in prep.SOURCE_ARMS:
            self.assertEqual(ru['locators'][arm]['candidates'],cu['locators'][arm]['candidates'])
        with self.assertRaises(ValueError):self.nextphase('closure',multi,mreq,status='unknown')

    def test_failed_final_rank_retains_upstream20(self):
        multi=self.multi();closure=self.nextphase('closure',multi,multi[0][1],status='failed')
        rank=self.nextphase('rank',closure,closure[0][1],status='failed');ru,rreq=rank[0]
        with patch.object(prep,'base',return_value=(self.m,self.b,self.sources)),\
            patch.object(prep,'_resolved_r03',return_value=self.refined),\
            patch.object(prep,'phase_results',return_value=rows_for(rreq,'failed')),\
            patch.object(prep,'units',return_value=[({'path':'rank.unit'},ru)]):
            selections=list(prep.expected_selections(self.reader,self.run))
        for _,arm,rec in selections:
            if arm not in prep.SOURCE_ARMS:continue
            key=next(k for a,k,_ in prep.source_arm_specs() if a==arm)
            self.assertEqual(rec['selected_facts'],ru['records'][key])
            self.assertEqual(rec['attachments'],[])


if __name__=='__main__':unittest.main()
