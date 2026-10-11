"""Offline recovery tests: only temporary outputs; no model/network dispatch."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tools import recover_gemini38_fact_tag_v1 as r

class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review=r.ROOT/'research/gemini38_retest_2026-10-03/tag_firstpass_source_seal_61.json'
        cls.snapshot,cls.fact,cls.context,cls.report=r.firstpass_snapshot(r.SOURCE_RUN,cls.review)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.ns=Path(self.temp.name)/'once';self.posts=[]
        self.patches=[patch.object(r,'PRODUCTION_NAMESPACE',self.ns),patch.object(r,'firstpass_snapshot',return_value=(copy.deepcopy(self.snapshot),copy.deepcopy(self.fact),self.context,copy.deepcopy(self.report)))]
        for p in self.patches:p.start();self.addCleanup(p.stop)
    def prepare(self):return r.prepare(r.SOURCE_RUN,self.ns,self.review)
    def api(self,root,m,bad=None):
        api=r.CachedLegacyAPI(root,retries=0,cache_roots=(),keys={'TYPESAFE_API_KEY':'offline-dummy'},context=m['original_context'])
        owner=self
        class Session:
            def post(self,url,headers,json,timeout):
                owner.posts.append(copy.deepcopy(json))
                ans={k:{'noul':0.4}for k in json['questions']}
                if bad=='bool':ans['episodic']['noul']=True
                if bad=='missing':ans.pop('episodic')
                response={'answers':ans,'model':'jev-1.13.0','usage':{'input_tokens':100,'output_tokens':10}}
                return SimpleNamespace(ok=True,status_code=200,headers={},json=lambda:response)
        api.local.session=Session();return api
    def execute(self,bad=None):
        with patch('jev_mem.providers._pace_jev',lambda:None):return r.execute(self.ns,api_factory=lambda root,m:self.api(root,m,bad))
    def replace(self,name,doc):
        p=self.ns/name;p.unlink();r.save_new(p,{k:v for k,v in doc.items()if k!='content_sha256'})
    def test_prepare_check_and_unique_namespace(self):
        self.prepare();m,_,_,_=r.check(self.ns);self.assertEqual(m['policy']['retries'],0);self.assertFalse((self.ns/'api').exists())
        with self.assertRaises(ValueError):r.prepare(r.SOURCE_RUN,self.ns.parent/'other',self.review)
        with self.assertRaises(FileExistsError):self.prepare()
    def test_exact_once_success_raw_and_metadata(self):
        self.prepare();row=self.execute();m,got,_=r.validate_recovery(self.ns)
        self.assertEqual(self.posts,[m['original_signature']['body']]);self.assertEqual(row,got)
        self.assertEqual(got['original_request_hash'],r.REQUEST_HASH);self.assertEqual(got['fact_tag_record']['protocol_sha256'],self.snapshot['original_protocol_sha256'])
        self.assertEqual(len(got['fact_tag_record']['result']['fine_scores']),6)
        with self.assertRaises(ValueError):self.execute()
        self.assertEqual(len(self.posts),1)
    def test_invalid_response_retained_failed_no_repeat(self):
        self.prepare()
        with self.assertRaises(Exception):self.execute('bool')
        self.assertEqual(r.verify(self.ns/'outcome.json')['status'],'failed');self.assertFalse((self.ns/'result.json').exists())
        self.assertTrue((self.ns/'api/api_cache'/(r.REQUEST_HASH+'.json')).exists())
        with self.assertRaises(Exception):r.validate_recovery(self.ns)
        with self.assertRaises(ValueError):self.execute()
        self.assertEqual(len(self.posts),1)
    def test_missing_score_retained_without_fill(self):
        self.prepare()
        with self.assertRaises(Exception):self.execute('missing')
        self.assertEqual(len(self.posts),1);self.assertEqual(r.verify(self.ns/'outcome.json')['status'],'failed')
    def test_raw_result_reseal_forgery_rejected(self):
        self.prepare();self.execute();result=r.verify(self.ns/'result.json');result['fact_tag_record']['result']['fine_scores']['temporal_constraint']=0.9
        self.replace('result.json',result);out=r.verify(self.ns/'outcome.json');out['result']=r.pointer(self.ns/'result.json');self.replace('outcome.json',out)
        with self.assertRaises(ValueError):r.validate_recovery(self.ns)
    def test_missing_ended_or_duplicate_attempt_rejected(self):
        self.prepare();self.execute();result=r.verify(self.ns/'result.json');p=Path(result['api_ledger']['path']);row=json.loads(p.read_text());row.pop('timestamp_utc');p.write_text(json.dumps(row)+'\n')
        result['api_ledger']=r.pointer(p);self.replace('result.json',result);out=r.verify(self.ns/'outcome.json');out['api_ledger']=result['api_ledger'];out['result']=r.pointer(self.ns/'result.json');self.replace('outcome.json',out)
        with self.assertRaises(ValueError):r.validate_recovery(self.ns)
    def test_wrong_retries_api_rejected_before_post(self):
        self.prepare()
        with self.assertRaises(ValueError):r.execute(self.ns,api_factory=lambda root,m:SimpleNamespace(retries=2,cache_roots=(),context=m['original_context']))
        self.assertEqual(self.posts,[]);self.assertTrue((self.ns/'execute_intent.json').exists())
        with self.assertRaises(ValueError):self.execute()
    def test_guard_and_exact_ten_scores(self):
        guard=r.ExactJudge(self.snapshot['original_signature'],response={})
        with self.assertRaises(ValueError):guard.request('gemini','gemini-3.8-flash',{},stage='jev_score',operation='systemone')
        result=r.tag_fact({**self.fact,'source_context':self.context},r.Capture());r.strict_result(result,r.FACT_ID)
        for value in (True,float('nan'),-0.1,1.1):
            altered=copy.deepcopy(result);altered['coarse_scores']['episodic']=value
            with self.assertRaises(ValueError):r.strict_result(altered,r.FACT_ID)
        result['fine_scores']['extra']=0.1
        with self.assertRaises(ValueError):r.strict_result(result,r.FACT_ID)
    def test_stage_access_execution_idle_validate_retrieve_only(self):
        source=Path(self.temp.name)/'source';lock=source/'retrieval/active_stage.lock';lock.parent.mkdir(parents=True)
        for stage in ('tag','unknown','retrieve'):
            lock.write_text(json.dumps({'stage':stage,'pid':123}),encoding='utf8')
            with self.assertRaises(ValueError):r._stage_access(source)
            if stage=='retrieve':r._stage_access(source,True)
            else:
                with self.assertRaises(ValueError):r._stage_access(source,True)
        lock.write_text(json.dumps({'stage':'retrieve','pid':True}),encoding='utf8')
        with self.assertRaises(ValueError):r._stage_access(source,True)
    def test_attempt_missing_end_unknown_and_nan(self):
        valid={'timestamp_utc':'2026-10-04T00:00:00+00:00','latency_ms':1.0};r._ended(valid)
        for row in ({'latency_ms':1.0},{'timestamp_utc':'2026-10-04T00:00:00','latency_ms':1.0},{'timestamp_utc':valid['timestamp_utc'],'latency_ms':float('nan')}):
            with self.assertRaises(ValueError):r._ended(row)

if __name__=='__main__':unittest.main()
