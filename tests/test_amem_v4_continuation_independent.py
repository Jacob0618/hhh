"""Independent zero-network v4 admission boundaries using retained raw evidence."""
from pathlib import Path
import copy
import tempfile
import unittest
from unittest.mock import patch
from tools import run_five_baselines_amem_native_v4 as v4
from tools.baselines_common_v1 import read,sha,BaselineFatal

class V4AdmissionIndependentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.new=Path(self.temp.name)/'v4'
        self.raw=v4.SOURCE/'work/api_method/api_cache'/(v4.FAILED_KEY+'.json')
        self.signature=read(self.raw)['request'];self.context={'method':'amem','conversation':2,'phase':'build'}
        self.entry={'mode':'inherit','request_hash':v4.FAILED_KEY,'signature':self.signature,'context':self.context,
            'stage':'amem_evolution','raw_path':str(self.raw),'raw_sha256':sha(self.raw)}
        self.proof={'conversations':{str(i):{'mode':'rebuild' if i==2 else 'inherit','prefix':[self.entry] if i==2 else []} for i in range(10)}}
        self.patches=[patch.object(v4,'NEW',self.new),patch.object(v4.controller,'NEW',self.new),
            patch.object(v4.controller,'_PROOF',self.proof),
            patch('jev_mem.providers.load_credentials',side_effect=AssertionError('credentials forbidden')),
            patch('requests.sessions.Session.request',side_effect=AssertionError('network forbidden')),
            patch('socket.socket.connect',side_effect=AssertionError('network forbidden'))]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()
    def api(self,context=None,run_dir=None,keys=True):
        kw={'context':context or self.context}
        if keys:kw['keys']={'gemini':'offline-placeholder'}
        return v4.NativeAPI(run_dir or self.new/'work/api_method',**kw)
    def test_real_failure_import_keeps_original_bytes_and_emits_quote_journal(self):
        api=self.api();body=self.signature['body']
        system=body.get('systemInstruction',{}).get('parts',[{}])[0].get('text')
        result=api.generate(body['contents'][0]['parts'][0]['text'],system=system,json_output=True,stage='amem_evolution')
        self.assertEqual(api.position,1)
        target=api.run_dir/'api_cache'/self.raw.name
        self.assertEqual(target.read_bytes(),self.raw.read_bytes())
        journal=read(api.run_dir/'parser_lineage'/self.raw.name)
        self.assertEqual(journal['derived_text_utf8_sha256'],'48f83282d5521f19b2fa00fa686faf6f258a9943c853cbd0ac344a4460a183d3')
        self.assertEqual(len(journal['changes']),14)
        self.assertFalse(list((api.run_dir/'request_intents').glob('*.json')))
        self.assertFalse(list((api.run_dir/'replacements').glob('*.json')))
        self.assertIn('new_tags_neighborhood',result)
    def test_replacement_and_historical_nonprefix_request_hard_stop(self):
        api=self.api();self.entry['mode']='replace_once'
        with self.assertRaises(BaselineFatal):api.request('gemini','gemini-2.5-flash',self.signature['body'],stage='amem_evolution')
        self.assertEqual(api.position,0)
        self.assertFalse(list((api.run_dir/'replacements').glob('*.json')))
        self.entry['mode']='inherit';api.position=1
        with self.assertRaises(BaselineFatal):api.request('gemini','gemini-2.5-flash',self.signature['body'],stage='amem_evolution')
        self.assertFalse(list((api.run_dir/'request_intents').glob('*.json')))
    def test_wrong_root_inherited_and_missing_proof_block_before_credentials(self):
        for kw in ({'run_dir':self.new/'wrong'}, {'context':{**self.context,'conversation':5}}):
            with self.assertRaises(BaselineFatal):self.api(keys=False,**kw)
        with patch.object(v4.controller,'_PROOF',None),self.assertRaises(BaselineFatal):self.api(keys=False)
        self.assertFalse(self.new.exists())

class V4QualificationIndependentTests(unittest.TestCase):
    def test_actual_scoped_qualification_and_unknown_or_retry_negative(self):
        import json
        actual=v4._qualification(v4.DEFAULT_QUALIFICATION)
        self.assertEqual(actual['retry_limit'],0)
        self.assertFalse(actual['launch_admitted'])
        with tempfile.TemporaryDirectory(prefix='v4_qualification_independent_',dir=v4.ROOT/'runtime') as directory:
            path=Path(directory)/'qualification.json'
            tampered=copy.deepcopy(actual);tampered['retry_limit']=1
            path.write_text(json.dumps(tampered),encoding='utf-8')
            with self.assertRaises(ValueError):v4._qualification(path)
            support=read(actual['review_path']);support['unknown_or_inflight_requests']=[{'request_hash':'unknown'}]
            sp=Path(directory)/'support.json';sp.write_text(json.dumps(support),encoding='utf-8')
            tampered=copy.deepcopy(actual);tampered.update(review_path=str(sp),review_sha256=sha(sp))
            path.write_text(json.dumps(tampered),encoding='utf-8')
            with self.assertRaises(ValueError):v4._qualification(path)

if __name__=='__main__':unittest.main()
