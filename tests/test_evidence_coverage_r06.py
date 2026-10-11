import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy
from jev_mem import evidence_coverage_r06 as pure
from tools import prepare_evidence_coverage_r06 as prep

SPEC=json.loads(prep.RESEARCH_PROTOCOL.read_bytes())


def records(n=80):
    return [{'id':f'id{i}','text':f'Actor{i} planned event{i}.','observation_date':None}for i in range(n)]


class PureCoverageTests(unittest.TestCase):
    def test_empty_and_original_duplicate_facets(self):
        q='Who did what?'
        self.assertEqual(pure.parse_facets('{"subrequirements":[]}',q)['facets'],[q])
        result=pure.parse_facets(json.dumps({'subrequirements':['  WHO did WHAT? ','A  relationship?','a relationship?']}),q)
        self.assertEqual(result,{'subrequirements':['A  relationship?'],'facets':[q,'A  relationship?'],
            'raw_count':3,'duplicate_count':2,'m':2})

    def test_json_strict_no_repairs_or_duplicate_keys(self):
        malformed=['```json\n{"subrequirements":[]}\n```','{"subrequirements":[]} extra',
          '{"subrequirements":[],"subrequirements":[]}','{"subrequirements":[],"answer":"x"}',
          '{"subrequirements":[true]}','{"subrequirements":[""]}',
          '{"subrequirements":["a","b","c","d"]}','["a"]']
        for text in malformed:
            with self.subTest(text=text),self.assertRaises((ValueError,TypeError)):
                pure.parse_facets(text,'Q?')

    def test_unicode_bytes_and_total_byte_limit(self):
        with self.assertRaises(ValueError):pure.parse_facets(json.dumps({'subrequirements':['中'*171]},ensure_ascii=False),'Q?')
        with self.assertRaises(ValueError):pure.parse_facets(' '*10+'{"subrequirements":[]'+ ' '*2050+'}', 'Q?')
        self.assertEqual(pure.parse_facets('  {"subrequirements":["'+ 'a'*512+'"]}  ','Q?')['m'],2)

    def test_frozen_response_model_finish_thought_handling(self):
        raw={'modelVersion':'gemini-3.8-flash','candidates':[{'finishReason':'STOP','content':{'parts':[
          {'text':'private ignored','thought':True},{'text':'{"subrequirements":[]}'}]}}]}
        self.assertEqual(pure.parse_facet_response(raw,'Q?')['m'],1)
        for mutation in ({'modelVersion':'other'},{'candidates':[{'finishReason':'MAX_TOKENS'}]}):
            with self.assertRaises(RuntimeError):pure.parse_facet_response({**raw,**mutation},'Q?')

    def test_signature_whitelists_and_literal_payload(self):
        q='Who? {input_json} " reference gold';sig=pure.build_facet_request(q,None,SPEC)
        self.assertEqual(sig['body']['generationConfig'],{'maxOutputTokens':8192,'thinkingConfig':{'thinkingLevel':'low'}})
        self.assertNotIn('systemInstruction',sig['body'])
        self.assertEqual(sig['operation'],'generateContent')
        text=sig['body']['contents'][0]['parts'][0]['text'];prefix=SPEC['facet_generation']['prompt'].split('{input_json}')[0]
        self.assertEqual(json.loads(text[len(prefix):]),{'question':q,'reference_date':''})
        fs=records(20);fs[0]['text']='Literal {observation_date} and {text}.'
        request=pure.build_request('Q?',None,'Need?',fs,SPEC)
        self.assertEqual(set(request['body']['state']),{'question','reference_date','requirement'})
        self.assertIn(fs[0]['text'],request['body']['questions']['f00']['instructions'])
        self.assertEqual(len(request['body']['questions']),20)

    def test_complementary_feature_gain_and_trace(self):
        matrix=[[.2,0.0]for _ in range(80)];matrix[0]=[.9,0.0];matrix[20]=[.89,.99]
        result=pure.choose_coverage(matrix)
        self.assertEqual(result['greedy_indices'][:2],[20,0])
        self.assertEqual(result['covered'],[.9,.99]);self.assertEqual(result['zero_fill_count'],18)
        self.assertEqual(result['indices'],sorted(set(result['greedy_indices'])))
        self.assertEqual(len(result['indices']),20)

    def test_exact_ties_originalQ_then_poolindex(self):
        matrix=[[0.0,0.0]for _ in range(80)];matrix[0]=[.2,.8];matrix[1]=[.8,.2];matrix[2]=[.8,.2]
        result=pure.choose_coverage(matrix)
        self.assertEqual(result['greedy_indices'][:2],[1,0])
        self.assertEqual(result['greedy_indices'][2],2)

    def test_all_zero_keeps_exact_baseline(self):
        result=pure.choose_coverage([[0.0]*4 for _ in range(80)])
        self.assertEqual(result['indices'],list(range(20)))
        self.assertEqual(result['zero_fill_count'],20)

    def test_one_facet_acts_like_scalar_membership_pool_order(self):
        matrix=[[i/80]for i in range(80)]
        d=pure.choose_coverage(matrix)
        self.assertEqual(d['indices'],list(range(60,80)))
        self.assertEqual(d['greedy_indices'],list(range(79,59,-1)))

    def test_full_matrix_numeric_and_shape_required(self):
        invalid=[[[0.0]for _ in range(19)],[[0.0,True]for _ in range(20)],
          [[float('nan')]for _ in range(20)],[[1.1]for _ in range(20)],
          [[0.0]*5 for _ in range(20)],[[0.0]for _ in range(19)]+[[0.0,0.0]]]
        for matrix in invalid:
            with self.subTest(matrix=matrix[:1]),self.assertRaises(ValueError):pure.choose_coverage(matrix)

    def test_nonadditive_max_not_score_sum(self):
        matrix=[[.8,.0],[.7,.0],[.0,.6]]+[[0.,0.]for _ in range(17)]
        result=pure.choose_coverage(matrix)
        self.assertEqual(result['greedy_indices'][:2],[0,2])
        self.assertEqual(result['marginal_gains'][2]['delta_sum'],0.0)

    def test_cheap_column_normalization_and_zero_column(self):
        fs=records(20)
        with patch.object(pure,'stable_bm25',side_effect=[[2.,4.]+[0.]*18,[0.]*20]):
            matrix,raw=pure.cheap_matrix('Q?',None,fs,['Q?','Need?'])
        self.assertEqual(matrix[:2],[[.5,0.],[1.,0.]])
        self.assertEqual(raw[0][:2],[2.,4.])
        with patch.object(pure,'stable_bm25',return_value=[-1.]*20),self.assertRaises(ValueError):
            pure.cheap_matrix('Q?',None,fs,['Q?'])

    def test_unknown_and_missing_are_not_zero(self):
        fs=records(20)
        with self.assertRaises(ValueError):pure.choose_jev(fs,[{'f00':.8}])
        with self.assertRaises(ValueError):pure.choose_jev(fs,None,score_status='unknown')
        with self.assertRaises(ValueError):pure.choose_jev(fs,[{}],score_status='failed')
        d=pure.choose_jev(fs,None,score_status='failed')
        self.assertEqual(pure.selected_facts(fs,d['pool_selected_indices']),fs)
        with self.assertRaises(ValueError):pure.choose_cheap('Q?',None,fs,['Q?'],facet_status='failed')

    def test_raw_noul_exact_model_allkeys_and_type(self):
        sig=pure.build_request('Q?',None,'Q?',records(20),SPEC)
        raw={'model':'jev-1.13.0','answers':{k:{'type':'noul','noul':.5}for k in sig['body']['questions']}}
        self.assertEqual(len(pure.parse_scores(raw,sig)),20)
        for changed in ({**raw,'model':'other'}, {**raw,'answers':{'f00':{'type':'noul','noul':.5}}}):
            with self.assertRaises(ValueError):pure.parse_scores(changed,sig)

    def test_null_date_metadata_not_changed(self):
        fs=records(20);original=deepcopy(fs)
        columns=[{f'f{i:02d}':i/20 for i in range(20)}]
        d=pure.choose_jev(fs,columns)
        self.assertEqual(pure.selected_facts(fs,d['pool_selected_indices']),original)
        self.assertEqual(fs,original)


class PrepSeamTests(unittest.TestCase):
    def test_phase_ledger_projection_all_contexts_and_append(self):
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'api_calls.jsonl';stage='evidence_coverage_facet_r06'
            original=[{'stage':stage,'request_hash':'h1','context':{'query_id':'c0_q1'}},
                {'stage':'wrong_stage','request_hash':'h1','context':{'query_id':'rogue'}}]
            path.write_text(''.join(json.dumps(v)+'\n'for v in original),encoding='utf-8')
            binding=prep.ledger_projection(prep.Reader(),path,stage,['h1'])
            self.assertEqual(binding['selected_row_count'],2)
            with path.open('a',encoding='utf-8')as f:f.write(json.dumps({'stage':'evidence_coverage_score_r06','request_hash':'new'})+'\n')
            self.assertEqual(binding,prep.ledger_projection(prep.Reader(),path,stage,['h1']))
            with path.open('a',encoding='utf-8')as f:f.write(json.dumps({'stage':stage,'request_hash':'unknown_hash'})+'\n')
            self.assertNotEqual(binding,prep.ledger_projection(prep.Reader(),path,stage,['h1']))
            path.write_text(json.dumps(original[0]),encoding='utf-8')
            with self.assertRaises(ValueError):prep.ledger_projection(prep.Reader(),path,stage,['h1'])

    def test_audit_gate_rechecks_immutable_intent_and_phase_ledger(self):
        with tempfile.TemporaryDirectory()as directory:
            run=Path(directory);codes={'tools/score_evidence_coverage_r06.py':'auditor_hash'}
            manifest=prep.save_new(run/'run_manifest.json',{'phase':'facet'})
            outcome=prep.save_new(run/'facet_outcome.json',{'status':'terminal'})
            controller=prep.save_new(run/'execution_intent_facet.json',{'phase':'facet'})
            request=prep.save_new(run/'facet_requests/c0_q1.json',{'request_id':'c0_q1','query_id':'c0_q1','request_hash':'h'})
            response=prep.save_new(run/'facet_responses/c0_q1.json',{'status':'failed'})
            stage='evidence_coverage_facet_r06';ledger=run/'api_calls.jsonl'
            ledger.write_text(json.dumps({'stage':stage,'request_hash':'h'})+'\n',encoding='utf-8')
            audit={'schema':prep.AUDIT_SCHEMA,'phase':'facet','status':'passed','manifest':manifest,'outcome':outcome,
              'response_refs':[{**response,'request_id':'c0_q1','query_id':'c0_q1','request_ref':request}],
              'unknown':[],'known_terminal_failures':[],'requests_count':1,
              'input_refs':[manifest,outcome,controller,request,response],
              'ledger_projection':prep.ledger_projection(prep.Reader(),ledger,stage,['h']),
              'codepins':{'path':str((prep.ROOT/'tools/score_evidence_coverage_r06.py').resolve()),'sha256':'auditor_hash'}}
            prep.save_new(run/'reviews/facet_terminal_audit_v1.json',audit)
            with patch.object(prep,'_codes',return_value=codes):
                prep._audit_gate(prep.Reader(),run,'facet',manifest,[request])
                with ledger.open('a',encoding='utf-8')as f:f.write(json.dumps({'stage':'evidence_coverage_score_r06','request_hash':'new'})+'\n')
                prep._audit_gate(prep.Reader(),run,'facet',manifest,[request])
                with ledger.open('a',encoding='utf-8')as f:f.write(json.dumps({'stage':'wrong_stage','request_hash':'h'})+'\n')
                with self.assertRaises(ValueError):prep._audit_gate(prep.Reader(),run,'facet',manifest,[request])
                ledger.write_text(json.dumps({'stage':stage,'request_hash':'h'})+'\n',encoding='utf-8')
                Path(controller['path']).write_text('{"phase":"changed"}',encoding='utf-8')
                with self.assertRaises(ValueError):prep._audit_gate(prep.Reader(),run,'facet',manifest,[request])

    def test_real_old_pool_reader_and_proof_contract(self):
        self.assertTrue(callable(prep.prior._inputs));self.assertTrue(callable(prep.prior._validate_qa_manifest))
        r=prep.Reader();value=r.read(prep.REUSE_QA_MANIFEST,sealed=False)
        prep.prior._validate_qa_manifest(value)
        self.assertEqual(value['baseline_reuse_proof'],r.raw_pointer(prep.FROZEN_REUSE_PROOF))
        self.assertEqual(prep.PROTOCOL_SHA256,SPEC and pure.PROTOCOL_SHA256)

    def test_request_record_exact_dispatcher_seam(self):
        from tools import score_evidence_coverage_r06 as score
        old={'query_id':'c2_q7','conversation':2,'question':'Q?','reference_date':None}
        for phase,signature,j,req in [('facet',pure.build_facet_request('Q?',None,SPEC),None,None),
          ('jev',pure.build_request('Q?',None,'Need?',records(20),SPEC),1,'Need?')]:
            row=prep._request_value(signature,old,'psha',facet_index=j,requirement=req)
            self.assertEqual(score.validate_record(row,Path(row['request_id']+'.json'),phase),len(pure.serialized_signature(signature)))
        self.assertEqual(row['reference_date'],None)

    def test_content_bound_tamper_and_audit_weak_counts_rejected(self):
        with tempfile.TemporaryDirectory()as directory:
            run=Path(directory);p=run/'reviews/facet_terminal_audit_v1.json'
            prep.save_new(p,{'schema':prep.AUDIT_SCHEMA,'phase':'facet','status':'passed',
                'manifest':{'path':'none','sha256':'none'},'unknown':[],'requests_count':0,'codepins':{}})
            with patch.object(prep,'_codes',return_value={'tools/score_evidence_coverage_r06.py':'h'}),self.assertRaises(ValueError):
                prep._audit_gate(prep.Reader(),run,'facet',{'path':'none','sha256':'none'},[])
            value=json.loads(p.read_bytes());value['status']='other';p.write_text(json.dumps(value),encoding='utf-8')
            with self.assertRaises(ValueError):prep.Reader().read(p)

    def test_complete_facet_result_raw_replay_and_failure(self):
        with tempfile.TemporaryDirectory()as directory:
            run=Path(directory);old={'query_id':'c0_q1','conversation':0,'question':'Q?','reference_date':None}
            signature=pure.build_facet_request('Q?',None,SPEC)
            req=prep.save_new(run/'facet_requests/c0_q1.json',prep._request_value(signature,old,'psha'))
            response={'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'{"subrequirements":[]}'}]}}]}
            rawp=run/'api_cache/raw.json';rawp.parent.mkdir();rawp.write_text(json.dumps({'request':signature,'response':response}),encoding='utf-8')
            rawref={'path':str(rawp.resolve()),'sha256':prep.file_sha(rawp)}
            row={'schema':prep.FACET_SCORE_SCHEMA,'phase':'facet','request_id':'c0_q1',
              'query_id':'c0_q1','conversation':0,'protocol_sha256':'psha','request_hash':pure.request_hash(signature),
              'actual_request_hash':pure.request_hash(signature),'status':'complete','raw':rawref,
              'facet_parse':pure.parse_facet_response(response,'Q?')}
            resultpath=run/'facet_responses/c0_q1.json';prep.save_new(resultpath,row)
            result=prep._result(prep.Reader(),run,req,phase='facet',spec=SPEC)
            self.assertEqual(result['parsed']['m'],1)
            self.assertEqual(prep._compiled_item(old,{'proposal':{},'request':req},
                {'candidate_records':records(20)},result,[])['cheap_decision']['facet_count'],1)


if __name__=='__main__':unittest.main()
