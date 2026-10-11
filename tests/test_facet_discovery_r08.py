"""Independent no-network R08 tests with hand-constructed discovery outcomes.

The expected lists are written explicitly; they are not recomputed with the
implementation's greedy/fusion helpers. Production APIs are never constructed.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jev_mem import facet_discovery_r08 as x
from jev_mem import evidence_selection_r03 as old

def vector(first,second=0):return [first,second]+[0.]*766
def rankrow(i,r,c=.1):return {'id':f'i{i}','rank':r,'cosine':c,'bank_ordinal':i}
def records():
    return [{'id':f'i{i}','text':f'Unchanged memory {i}',
             'observation_date':'2023-01-02' if i==79 else None}for i in range(80)]
def pool_fixture():
    return ([f'i{i}'for i in range(80)], [f'i{i}'for i in range(20)],
            [f'i{i}'for i in range(20,60)],
            [f'i{i}'for i in range(60)]+[f'i{i}'for i in range(100,120)])

class Discovery(unittest.TestCase):
    def test_normalization_copies_real_dimension_and_rejects_non_numeric(self):
        source=vector(3.,4.);before=source.copy();out=x.normalize_vector(source)
        self.assertEqual(source,before);self.assertIsNot(out,source)
        self.assertAlmostEqual(out[0],.6);self.assertAlmostEqual(out[1],.8)
        out[0]=0.;self.assertEqual(source[0],3.)
        for bad in ([],[1.]*767,vector(0.),vector(True),vector(float('nan')),
                    vector(float('inf')),vector('3')):
            with self.subTest(bad_type=type(bad[0]).__name__ if bad else 'empty'):
                with self.assertRaises((ValueError,TypeError)):x.normalize_vector(bad)

    def test_cosine_keeps_negative_and_breaks_ties_by_original_bank_ordinal(self):
        facts=[{'id':'z','embedding':vector(3.)},
               {'id':'a','embedding':vector(6.)},
               {'id':'n','embedding':vector(-3.)}]
        queries=[vector(5.),vector(-2.)];before=copy.deepcopy((facts,queries))
        ranks=x.rank_facets(facts,queries)
        self.assertEqual([r['id']for r in ranks[0]],['z','a','n'])
        self.assertEqual([r['id']for r in ranks[1]],['n','z','a'])
        self.assertEqual([r['bank_ordinal']for r in ranks[0]],[0,1,2])
        self.assertEqual([r['cosine']for r in ranks[0]],[1.,1.,-1.])
        self.assertEqual((facts,queries),before)

    def test_round_robin_advances_duplicate_and_native_rows_without_stalling(self):
        ranks=[[rankrow(1,1),rankrow(80,2),rankrow(81,3),rankrow(84,4)],
               [rankrow(80,1),rankrow(82,2),rankrow(81,3),rankrow(83,4)]]
        before=copy.deepcopy(ranks)
        found=x.round_robin(ranks,[f'i{i}'for i in range(80)],
                            [f'i{i}'for i in range(60)],limit=4)
        self.assertEqual([r['id']for r in found],['i80','i82','i81','i83'])
        self.assertEqual([r['facet_index']for r in found],[1,2,1,2])
        self.assertEqual(ranks,before)

    def test_exhausted_facet_does_not_prevent_other_list_progress(self):
        found=x.round_robin([[rankrow(80,1)],[],[rankrow(81,1),rankrow(82,2)]],
                            [f'i{i}'for i in range(80)],[],limit=20)
        self.assertEqual([r['id']for r in found],['i80','i81','i82'])
        self.assertEqual([r['facet_index']for r in found],[1,3,3])

    def test_zero_discovery_exact_original_pool_and_never_original_native_filler(self):
        original,baseline,others,native=pool_fixture()
        p=x.build_pool(original,baseline,others,native,[])
        self.assertEqual(p['pool_ids'],original)
        self.assertEqual(p['added_ids'],[]);self.assertEqual(p['displaced_ids'],[])
        self.assertFalse(set(p['pool_ids'])&{f'i{i}'for i in range(100,120)})

    def test_old_lexical_discovery_can_change_order_without_membership_expansion(self):
        original,baseline,others,native=pool_fixture()
        discoveries=[dict(rankrow(65,1),facet_index=0),dict(rankrow(80,2),facet_index=1)]
        before=copy.deepcopy((original,baseline,others,native,discoveries))
        p=x.build_pool(original,baseline,others,native,discoveries)
        expected=[f'i{i}'for i in range(60)]+['i65','i80']+[
            'i60','i61','i62','i63','i64','i66','i67','i68','i69','i70',
            'i71','i72','i73','i74','i75','i76','i77','i78']
        self.assertEqual(p['pool_ids'],expected)
        self.assertEqual(p['added_ids'],['i80']);self.assertEqual(p['displaced_ids'],['i79'])
        self.assertEqual(p['fill_count'],18);self.assertEqual(len(set(expected)),80)
        self.assertEqual((original,baseline,others,native,discoveries),before)
        same=x.build_pool(original,baseline,others,native,[dict(rankrow(65,1),facet_index=0)])
        self.assertEqual(set(same['pool_ids']),set(original))
        self.assertNotEqual(same['pool_ids'],original)

    def test_bad_protected_prefix_and_non80_pool_are_not_silently_repaired(self):
        original,baseline,others,native=pool_fixture()
        for op,bp,np,vp in ((original[:-1],baseline,others,native),
                          (original,baseline[::-1],others,native),
                          (original,baseline,others[::-1],native),
                          (original[:-1]+[original[0]],baseline,others,native)):
            with self.subTest(length=len(op)):
                with self.assertRaises(ValueError):x.build_pool(op,bp,np,vp,[])

class ExactModelContract(unittest.TestCase):
    def test_full_R03_request_literal_is_two_fields_and_context_is_only_original_question(self):
        spec=json.loads((ROOT/'research/jev_autoresearch_2026-10-04/R03_selection_protocol_v2.json').read_bytes())
        r=records();actual=x.build_request('Where did A plan to go?','2023-02-03',r,spec)
        self.assertEqual(actual,old.build_request('Where did A plan to go?','2023-02-03',r,spec))
        self.assertEqual(set(actual['body']['state']),{'query'})
        self.assertEqual(set(actual['body']['questions']),{f'f{i:02d}'for i in range(80)})
        for q in actual['body']['questions'].values():
            literal=json.loads(q['instructions'].split('Memory record (JSON):\n')[1])
            self.assertEqual(set(literal),{'text','observation_date'})
            self.assertNotIn('id',literal)
        self.assertEqual(x.request_hash(actual),old.request_hash(actual))

    def test_all_zero_scores_remain_valid_and_top20_restores_pool_order(self):
        r=records();scores={f'f{i:02d}':0. for i in range(80)}
        self.assertEqual(x.choose_jev(r,scores)['pool_selected_indices'],list(range(20)))
        scores['f79']=.99
        self.assertEqual(x.choose_jev(r,scores)['pool_selected_indices'],[*range(19),79])
        for invalid in (True,float('nan'),None):
            bad=scores.copy();bad['f78']=invalid
            with self.assertRaises(ValueError):x.choose_jev(r,bad)

    def test_same_HTTP_body_with_changed_IDs_cannot_claim_producer_provenance(self):
        spec=json.loads((ROOT/'research/jev_autoresearch_2026-10-04/R03_selection_protocol_v2.json').read_bytes())
        original=records();modified=copy.deepcopy(original);modified[0]['id']='replacement-same-text'
        self.assertEqual(x.build_request('Q',None,original,spec),x.build_request('Q',None,modified,spec))
        with self.assertRaises(ValueError):
            x.same_score_provenance([r['id']for r in modified],[r['id']for r in original])
        self.assertTrue(x.same_score_provenance([r['id']for r in original],[r['id']for r in original]))

class EmbeddingBatchContracts(unittest.TestCase):
    def test_query_signature_exact_task_dimension_order_and_size_not_document(self):
        sig=x.embedding_signature(['first','second'])
        self.assertEqual(sig['operation'],'batchEmbedContents')
        self.assertEqual([r['content']['parts'][0]['text']for r in sig['body']['requests']],['first','second'])
        for row in sig['body']['requests']:
            self.assertEqual(row['taskType'],'RETRIEVAL_QUERY');self.assertEqual(row['outputDimensionality'],768)
        self.assertNotEqual(x.request_hash(sig),x.request_hash(x.embedding_signature(['second','first'])))
        for texts in ([],['x']*33,['x'*513],[True]):
            with self.assertRaises(ValueError):x.embedding_signature(texts)

    def test_batch_parse_enforces_count_and_every_vector_without_partial_fill(self):
        raw={'embeddings':[{'values':vector(3.,4.)},{'values':vector(-1.)}]}
        before=copy.deepcopy(raw);values=x.parse_embeddings(raw,2)
        self.assertEqual(raw,before);self.assertEqual(values[1][0],-1.)
        with self.assertRaises(ValueError):x.parse_embeddings(raw,1)
        for bad in (vector(True),vector(0.),vector(float('nan')),[1.]*769):
            response=copy.deepcopy(raw);response['embeddings'][1]['values']=bad
            with self.assertRaises(ValueError):x.parse_embeddings(response,2)

    def test_numeric_qid_order_firstseen_dedup_and_no_cross_conversation_batch(self):
        items=[{'query_id':'c0_q10','conversation':0,'status':'complete','facets':['Q','beta','shared']},
               {'query_id':'c1_q0','conversation':1,'status':'complete','facets':['Q','shared']},
               {'query_id':'c0_q2','conversation':0,'status':'complete','facets':['Q','shared','alpha']},
               {'query_id':'c0_q1','conversation':0,'status':'complete','facets':['Q']},
               {'query_id':'c0_q3','conversation':0,'status':'failed','facets':None}]
        before=copy.deepcopy(items);b=x.pack_batches(items)
        self.assertEqual(items,before);self.assertEqual(len(b),2)
        self.assertEqual(b[0]['conversation'],0);self.assertEqual(b[0]['texts'],['shared','alpha','beta'])
        self.assertEqual(b[0]['consumers'][0],[{'query_id':'c0_q2','facet_index':1},
                                             {'query_id':'c0_q10','facet_index':2}])
        self.assertEqual(b[1]['conversation'],1);self.assertEqual(b[1]['texts'],['shared'])
        self.assertEqual(x.pack_batches([items[3]]),[])
        with self.assertRaises(ValueError):x.pack_batches([{**items[3],'status':'unknown'}])

    def test_conversation_batch32_remainder_mapping_stable_under_input_reordering(self):
        items=[{'query_id':f'c0_q{i}','conversation':0,'status':'complete',
                'facets':['originalQ',f'derived{i}']}for i in range(34)]
        batches=x.pack_batches(list(reversed(items)))
        self.assertEqual([len(b['texts'])for b in batches],[32,2])
        self.assertEqual(batches[0]['texts'],[f'derived{i}'for i in range(32)])
        self.assertEqual(batches[1]['texts'],['derived32','derived33'])
        self.assertEqual(batches,x.pack_batches(items))

if __name__=='__main__':unittest.main()
