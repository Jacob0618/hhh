"""Independent lifecycle negatives; never production archive/delete/model calls."""
from copy import deepcopy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import archive_hypermem_native_v2 as a
from tools import run_five_baselines_reviewed_chain_v7 as chain


def payload(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
def digest(b):return hashlib.sha256(b).hexdigest()


class View:
    def __init__(self):self.values={}
    def put(self,p,value):self.values[str(p)]=payload(value);return {'path':str(p),'sha256':self.sha(p)}
    def bytes(self,p):return self.values[str(p)]
    def read(self,p):return json.loads(self.bytes(p))
    def sha(self,p):return digest(self.bytes(p))


class SummaryContract(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=a.ROOT/'research');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.old=self.root/'old';self.new=self.root/'new'
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(a,'OLD',self.old));self.stack.enter_context(patch.object(a,'NEW',self.new))
        self.view=View();self.review=self.root/'result_review.json'
        qs=[{'query_id':f'fixture_{i}'} for i in range(1540)]
        old_inv={str(self.old/'identity.txt'):digest(b'original immutable fixture')}
        original={'questions':qs}
        parent=self.view.put(self.old/'run_manifest.json',original)
        proof=self.view.put(self.root/'proof.json',{'questions':qs,'original_inventory':old_inv})
        core_review={'schema':a.core.REVIEW_SCHEMA,'status':'passed','proof_sha256':proof['sha256'],
                     'code_sha256':{},'upstream_sha256':{},'specification':{},'evidence_sha256':{}}
        cr=self.view.put(self.root/'core_review.json',core_review)
        self.view.put(self.new/'run_manifest.json',{'schema':a.core.SCHEMA,'original_run':str(self.old),'new_run':str(self.new),
            'questions':qs,'parent_manifest':parent,'terminal_proof':proof,'review':cr,'code_sha256':{},'upstream_sha256':{},'specification':{}})
        self.view.put(self.old/'status.json',{'stage':'needs_review'})
        self.counts={'slots':1540,'reader_valid':1539,'judge_valid':1538,'missing_reader':1,'missing_judge':2}
        self.summary={'schema':'five-baseline-result-v1','method':'hypermem','audit_status':'no_invalid_artifacts',
                      'firstpass':deepcopy(self.counts),'effective':deepcopy(self.counts)}
        self.original_inv=old_inv;self.update()

    def update(self):
        sb=self.view.put(self.new/'results/summary.json',self.summary)
        self.view.put(self.new/'status.json',{'stage':'completed_awaiting_result_review','summary_sha256':sb['sha256']})
        self.view.put(self.review,{'schema':a.RESULT_SCHEMA,'status':'passed','method':'hypermem','summary_sha256':sb['sha256'],
            'reviewed_run_inventories':{'original':self.original_inv,'continuation':{str(self.new/'results/summary.json'):sb['sha256']}}})

    def test_fixed_slots_do_not_require_all_qa_valid(self):
        actual=a.result_review(self.review,self.view)
        self.assertEqual(actual[2]['effective']['slots'],1540)
        self.assertEqual(actual[2]['effective']['judge_valid'],1538)
        self.assertEqual(actual[2]['effective']['missing_judge'],2)

    def test_boolean_invalid_count_and_wrong_denominators_rejected(self):
        baseline=deepcopy(self.summary)
        for phase in ('firstpass','effective'):
            for field,value in (('reader_valid',True),('judge_valid',1541),('reader_valid',-1),('judge_valid',1538.0),
                                ('slots',1539),('missing_reader',0),('missing_judge',1)):
                with self.subTest(phase=phase,field=field,value=value):
                    self.summary=deepcopy(baseline);self.summary[phase][field]=value;self.update()
                    self.assertRaises(ValueError,a.result_review,self.review,self.view)

    def test_missing_inventory_and_wrong_original_status_rejected(self):
        self.view.put(self.old/'status.json',{'stage':'completed'})
        self.assertRaises(ValueError,a.result_review,self.review,self.view)
        self.view.put(self.old/'status.json',{'stage':'needs_review'})
        review=self.view.read(self.review);review['reviewed_run_inventories'].pop('original');self.view.put(self.review,review)
        self.assertRaises(ValueError,a.result_review,self.review,self.view)


class CertificateContract(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=a.ROOT/'research');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.view=View();self.codes={'fixture_source.py':'a'*64}
        self.patcher=patch.object(a,'code_hashes',return_value=self.codes);self.patcher.start();self.addCleanup(self.patcher.stop)
        self.result=self.root/'result.json';self.tool=self.root/'tool.json';self.cert=self.root/'cert.json';self.br=self.root/'bank_review.json'
        rb=self.view.put(self.result,{'fixture':'actual-result'});tb=self.view.put(self.tool,{'fixture':'tool-review'})
        output={}
        for name in a.OUTPUTS:
            p=a.NEW/'results'/name;output[str(p)]=self.view.put(p,{'fixture':name})['sha256']
        self.c={'schema':a.CERT_SCHEMA,'status':'passed','scope':'actual_predelete_bytes_and_unmodified_frozen_full_audits',
                'code_sha256':self.codes,'inventory_before_after_equal':True,'database_engine_opened':False,
                'retrieval_search_reexecuted':False,'model_calls':0,'credential_reads':0,'postdelete_scope':a.SCOPE,
                'result_review':rb,'tool_review':tb,'output_sha256':output}
        cb=self.view.put(self.cert,self.c)
        self.b={'schema':a.BANK_REVIEW_SCHEMA,'status':'passed','scope':'independently_verified_predelete_bank_bytes_and_declared_postdelete_scope',
                'postdelete_scope':a.SCOPE,'code_sha256':self.codes,'certificate':cb,'result_review':rb,'tool_review':tb}
        self.view.put(self.br,self.b)

    def change(self,field,value):
        c=deepcopy(self.c);c[field]=value;b=deepcopy(self.b);b['certificate']=self.view.put(self.cert,c);self.view.put(self.br,b)

    def test_exact_certificate_binds_all_four_outputs(self):
        self.assertEqual(a.verify_certificate(self.view,self.br,self.result,self.tool),self.c)
        outputs=deepcopy(self.c['output_sha256']);outputs.pop(next(iter(outputs)))
        self.change('output_sha256',outputs)
        self.assertRaises(ValueError,a.verify_certificate,self.view,self.br,self.result,self.tool)

    def test_rebound_invalid_scope_or_bool_call_counts_still_rejected(self):
        for key,value in (('model_calls',False),('credential_reads',False),('database_engine_opened',True),
                          ('retrieval_search_reexecuted',True),('inventory_before_after_equal',False),('postdelete_scope','index search replayed')):
            with self.subTest(key=key):
                self.change(key,value);self.assertRaises(ValueError,a.verify_certificate,self.view,self.br,self.result,self.tool)

    def test_result_or_code_changed_and_changed_output_rejected(self):
        self.view.put(self.result,{'fixture':'changed'})
        self.assertRaises(ValueError,a.verify_certificate,self.view,self.br,self.result,self.tool)
        self.view.put(self.result,{'fixture':'actual-result'})
        self.view.put(a.NEW/'results/summary.json',{'fixture':'different'})
        self.assertRaises(ValueError,a.verify_certificate,self.view,self.br,self.result,self.tool)

    def test_live_certificate_view_rejects_foreign_path_before_read(self):
        foreign=Path(a.ROOT.anchor)/'not_the_workspace'/'untrusted_certificate.json'
        for name in ('read','sha','bytes'):
            with self.subTest(operation=name),patch.object(Path,'read_bytes',side_effect=AssertionError('Foreign file read forbidden')),\
                    patch.object(Path,'read_text',side_effect=AssertionError('Foreign file read forbidden')),\
                    patch.object(Path,'open',side_effect=AssertionError('Foreign file read forbidden')):
                self.assertRaises(ValueError,getattr(a.seal.LiveView(),name),foreign)


class RetentionContract(unittest.TestCase):
    def test_unique_text_and_original_raw_can_never_be_disposed(self):
        for owner in (a.OLD,a.NEW):
            for suffix in ('native/c2/cache/audit.json','native/c2/cache/error.txt','native/c2/sessions/session_1/source.json',
                           'native/c2/embedding_evidence/.integrity/document_'+'a'*64+'.json.sha256','api_method/api_cache/raw.json'):
                self.assertIsNone(a.disposition(owner/'work'/suffix))
        self.assertRaises(ValueError,a.disposition,a.ROOT/'research/extra/cache/hypergraph_bm25_index_conv_0.pkl')

    def test_crc_valid_fake_disposal_or_missing_blob_rejected(self):
        for variant in ('unique_text_false','numeric_wrong_reason','numeric_retained_integer','retained_without_blob'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory(dir=a.ROOT/'research') as td:
                root=Path(td);p=a.NEW/'work/native/c0/cache/hypergraph_bm25_index_conv_0.pkl'
                reason=a.disposition(p)
                row={'path':str(p),'sha256':digest(b'bytes'),'size':5,'retained':False,'disposal_reason':reason}
                if variant=='unique_text_false':row['path']=str(a.NEW/'work/native/c0/cache/audit.json')
                elif variant=='numeric_wrong_reason':row['disposal_reason']='anything'
                elif variant=='numeric_retained_integer':row['retained']=0
                else:row={'path':str(a.NEW/'work/native/c0/cache/audit.json'),'sha256':digest(b'bytes'),'size':5,'retained':True,'member':'blobs/'+digest(b'bytes')}
                mp=root/'manifest.json';mp.write_bytes(payload({'schema':a.SCHEMA,'records':[row]}));zp=root/'archive.zip'
                with zipfile.ZipFile(zp,'x') as z:z.writestr('archive_manifest.json',mp.read_bytes())
                self.assertRaises(ValueError,a.Resolver,zp,mp)


class StrongDispatchContract(unittest.TestCase):
    def test_every_actual_predecessor_verifier_required_for_exact_next_method(self):
        for method in ('vmem','prism'):
            expected=['amem','simplemem','hypermem']+(['vmem'] if method=='prism' else [])
            refs={n:{kind:a.ROOT/'research/synthetic_not_written'/n/(kind+'.json') for kind in ('plan','receipt','completion')} for n in expected}
            calls=[]
            with ExitStack() as stack:
                stack.enter_context(patch.object(chain,'completion_paths',return_value=refs))
                stack.enter_context(patch.object(chain,'binding',side_effect=lambda p:{'path':str(p),'sha256':digest(str(p).encode())}))
                stack.enter_context(patch.object(chain,'assert_stopped'))
                for module,name in ((chain.previous.previous.archive,'amem'),(chain.previous.simple,'simplemem'),
                                    (chain.hyper,'hypermem'),(chain.previous.previous.frozen.lifecycle,'vmem')):
                    stack.enter_context(patch.object(module,'verify_completion',side_effect=lambda *args,n=name,**kw:(calls.append((n,args)) or {'status':'passed'})))
                actual=chain.strong_predecessors(method)
                self.assertEqual([n for n,args in calls],expected)
                self.assertEqual(set(actual),set(expected))
                if method=='prism':self.assertEqual(calls[-1],('vmem',('vmem',)))

    def test_failed_new_hyper_completion_never_falls_back_to_legacy_receipt(self):
        refs={n:{'completion':a.ROOT/'research/synthetic_not_written'/n/'completion.json'} for n in ('amem','simplemem','hypermem')}
        with patch.object(chain,'completion_paths',return_value=refs),patch.object(chain,'binding',side_effect=lambda p:{'path':str(p),'sha256':'a'*64}),\
             patch.object(chain,'assert_stopped'),patch.object(chain.previous.previous.archive,'verify_completion',return_value={'status':'passed'}),\
             patch.object(chain.previous.simple,'verify_completion',return_value={'status':'passed'}),\
             patch.object(chain.hyper,'verify_completion',return_value={'status':'needs_review'}) as hyper,\
             patch.object(chain.previous.previous.frozen.lifecycle,'verify_completion') as legacy:
            self.assertRaisesRegex(ValueError,'HyperMem two-root',chain.strong_predecessors,'vmem')
            hyper.assert_called_once_with();legacy.assert_not_called()

    def test_all_unknown_or_earlier_methods_rejected(self):
        for method in ('amem','simplemem','hypermem','unknown'):
            self.assertRaises(ValueError,chain.completion_paths,method)
            self.assertRaises(ValueError,chain.pristine,method,prepared=False)


if __name__=='__main__':unittest.main(verbosity=2)
