"""Draft post-terminal Judge compatibility tests; synthetic model responses only."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jev_mem.judge_terminal_label_compat_v1 import derive_label, eligible_raw_response
from tools.replay_plain_judge_compat_v1 import replay_source, InvalidArtifact
from tests.test_summarize_mem0_regroup_plain_qa_v1 import Recording
from tools import run_mem0_regroup_plain_qa_v1 as driver
from tools.run_sdk20_modules import save_new, verify


class TerminalLabelTests(unittest.TestCase):
    def test_strict_valid_outcomes_and_raw_whitespace_unchanged(self):
        for text,value in [('[[yes]]',True),(' \n[[no]]\n\t',False)]:
            row=derive_label(text)
            self.assertEqual(row['status'],'strict_valid');self.assertIs(row['correct'],value)
            self.assertEqual(row['raw'],text);self.assertFalse(row['raw_modified'])

    def test_terminal_unanimous_labels_accept_both_outcomes(self):
        cases=[('The response contains the reference.\n[[yes]]',True),
               ('Only a subset is covered.\n\n [[no]] \n',False),
               ('The explanation uses [[yes]] consistently.\n[[yes]]',True)]
        for text,value in cases:
            row=derive_label(text)
            self.assertEqual(row['status'],'compat_valid');self.assertIs(row['correct'],value)
            self.assertEqual(row['raw'],text)

    def test_missing_final_label_extra_tokens_and_quoted_conflict_rejected(self):
        bad=['yes','The answer is [[yes]]','[[yes]] because correct','[[yes]]\nmore explanation',
             '{"label":"[[yes]]"}','[[YES]]','```\n[[yes]]\n```','',
             'The prompt says [[no]], but I conclude:\n[[yes]]','[[yes]]\n[[no]]']
        for text in bad:
            with self.subTest(text=text):
                row=derive_label(text)
                self.assertEqual(row['status'],'rejected');self.assertIsNone(row['correct']);self.assertIsNone(row['label'])

    def test_raw_envelope_is_conservative_and_never_combines_thoughts(self):
        text='Explanation.\n[[yes]]'
        raw={'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':text}]}}]}
        self.assertEqual(eligible_raw_response(raw,text,http_status=200),(True,None))
        signed=copy.deepcopy(raw);signed['candidates'][0]['content']['parts'][0]['thoughtSignature']='opaque-provider-metadata'
        self.assertEqual(eligible_raw_response(signed,text,http_status=200),(True,None))
        signed['candidates'][0]['content']['parts'][0]['thoughtSignature']=None
        self.assertFalse(eligible_raw_response(signed,text,http_status=200)[0])
        altered=[]
        x=copy.deepcopy(raw);x['candidates'][0]['finishReason']='MAX_TOKENS';altered.append(x)
        x=copy.deepcopy(raw);x['candidates']*=2;altered.append(x)
        x=copy.deepcopy(raw);x['candidates'][0]['content']['parts'][0]['thought']=False;altered.append(x)
        x=copy.deepcopy(raw);x['candidates'][0]['content']['parts'].insert(0,{'thought':True,'text':'reason'});altered.append(x)
        for x in altered:self.assertFalse(eligible_raw_response(x,text,http_status=200)[0])
        self.assertFalse(eligible_raw_response(raw,text,http_status=201)[0])
        self.assertFalse(eligible_raw_response(raw,text+'changed',http_status=200)[0])


class SourceReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.q={'id':'c0_q0','question':'Where?','reference':'Paris','category':1}
        self.m={'protocol_sha256':'mock','reference_dates':{'0':'2023-06-30'}}
        self.arm=driver.ARMS[0]
        self.bank={'facts':[{'id':f'f{i}','text':f'Paris {i}','observation_date':None} for i in range(20)]}
        self.api=Recording(self.root/'conv_0/api',{'conversation':0,'architecture':driver.SCHEMA})
        for target in ('requests.Session.post','socket.create_connection','jev_mem.providers.load_credentials'):
            p=patch(target,side_effect=AssertionError('No real API'));p.start();self.addCleanup(p.stop)

    def source(self,text):
        self.api.shared['judge']=text
        p=self.root/'conv_0/selection'/self.arm/'c0_q0.json'
        save_new(p,{**driver.identity(0,self.arm,self.q,self.m),'status':'complete','bank_sha256':'mock',
                    'selected_ids':[f['id'] for f in self.bank['facts']],'error':None})
        final=driver.qa_one(self.root,0,self.arm,self.q,self.m,self.bank,'mock',self.api,lambda *a:.75)
        row=verify(final['judge']['path'])
        return row,row['api_events'][0]['context']

    def replay(self,row,context):
        return replay_source(self.api.root,row,context,self.api.shared['ledger'],question='Where?',reference='Paris',reader_text='Paris')

    def test_http_raw_replay_accepts_terminal_no_without_touching_source(self):
        row,context=self.source('The details are insufficient.\n[[no]]')
        original=copy.deepcopy(row)
        proof=self.replay(row,context)
        self.assertEqual(row,original)
        self.assertEqual(proof['interpretation']['status'],'compat_valid')
        self.assertIs(proof['interpretation']['correct'],False)
        self.assertEqual(proof['original_status'],'failed');self.assertEqual(proof['model_calls'],0)

    def test_original_strict_success_is_inherited_unchanged(self):
        row,context=self.source('[[yes]]')
        proof=self.replay(row,context)
        self.assertEqual(proof['interpretation']['status'],'strict_valid')
        self.assertEqual(proof['interpretation']['raw'],row['result']['raw'])

    def test_conflicting_original_response_stays_failed(self):
        row,context=self.source('The prompt mentions [[no]].\n[[yes]]')
        proof=self.replay(row,context)
        self.assertEqual(proof['interpretation']['status'],'rejected');self.assertIsNone(proof['interpretation']['correct'])

    def test_source_row_and_request_drift_or_other_failure_rejected(self):
        row,context=self.source('Explanation\n[[yes]]')
        row['raw_invalid_label']='changed'
        with self.assertRaises(InvalidArtifact):self.replay(row,context)
        row['error']['type']='RuntimeError'
        with self.assertRaises(InvalidArtifact):self.replay(row,context)


if __name__=='__main__':unittest.main()
