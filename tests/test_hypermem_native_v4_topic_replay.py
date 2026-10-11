"""Author tiny pinned topic-pipeline replay; synthetic answers, no provider.

This exercises actual original create/update metadata, multi-topic ordering and
same-text updates. It is not the real ten-conversation prefix preflight.
"""
from contextlib import ExitStack
from copy import deepcopy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import requests

from tests.test_baselines_hypermem_v1 import FakeAPI,conversation
from tools import baselines_hypermem_native_v4 as adapter
from tools import audit_hypermem_continuation_v4 as audit
from tools.baselines_common_v1 import read,save_new,sha
from jev_mem import providers
from jev_mem.sdk20_cached_api import digest

ROOT=Path(__file__).resolve().parents[1]

class ActualTopicReplay(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(dir=ROOT/'research',prefix='tiny_topic_v4_')
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name);self.bank=self.root/'native'
        self.records={};self.caller={}
        outer=self
        class Tiny(FakeAPI):
            def __init__(self):super().__init__(boundary_every=2);self.matches=0
            def generate(self,prompt,**kwargs):
                value=super().generate(prompt,**kwargs)
                frame=inspect.currentframe().f_back;call=frame.f_locals['path'];request=frame.f_locals['request']
                actual=frame.f_back;stage=kwargs['stage']
                if stage=='hypermem_topic_match':
                    self.matches+=1
                    # Create a second topic, then update both on later episodes.
                    # Both topic titles/summaries are intentionally identical.
                    for row in value['results']:row['match']=self.matches!=1
                sig=audit.signature(request);key=digest(sig);raw=self_root/'raw'/(key+'.json')
                record={'request':sig,'stage':stage,'response':{'candidates':[{'finishReason':'STOP',
                    'content':{'parts':[{'text':json.dumps(value)}]}}]}}
                if raw.exists():outer.assertEqual(read(raw),record)
                else:save_new(raw,record)
                outer.records[key]={'signature':sig,'raw':audit.binding(raw),'raw_value':record}
                if call.name.startswith('topic_'):
                    outer.caller[call.name]={'local':adapter._Generation._local_record(actual.f_locals),'stage':stage}
                    if stage=='hypermem_episode_roles':outer.caller[call.name]['local_episodes']=[e.to_dict() for e in actual.f_locals['episodes']]
                return value
        self_root=self.root
        with patch.object(providers,'load_credentials',side_effect=AssertionError('No credentials')),\
             patch.object(requests.Session,'request',side_effect=AssertionError('No network')),\
             patch.object(requests.Session,'post',side_effect=AssertionError('No network')):
            self.state=adapter.build(conversation(12),self.bank,Tiny())
        self.proof={'hash_environment':audit.admission.hash_environment(),'compatibility_cases':{}}

    def replay(self):return audit.replay_topic_callers(self.bank,self.records,self.proof)

    def test_actual_multitopic_same_text_updates_keep_complete_local_metadata_and_source_order(self):
        result=self.replay();self.assertEqual(result['status'],'passed_complete')
        self.assertTrue(result['raw_derived_only']);self.assertFalse(result['topic_journals_used_as_inputs'])
        self.assertEqual(set(result['calls']),set(self.caller))
        nonadjacent=[];seen={};by_topic={}
        for name,row in result['calls'].items():
            self.assertEqual(row['local'],self.caller[name]['local'])
            if row['stage']=='hypermem_episode_roles':
                topic=row['local_topic'];tid=topic['topic_id']
                self.assertEqual(topic,row['local']['topic'])
                self.assertEqual(row['local_episodes'],self.caller[name]['local_episodes'])
                self.assertEqual({f'episode_{i+1}':e['event_id'] for i,e in enumerate(row['local_episodes'])},row['simple_to_real'])
                source=row['topic_generation_source'];sourcevalue=read(source['path'])
                self.assertIn(sourcevalue['request']['stage'],('hypermem_topic_create','hypermem_topic_update'))
                self.assertLess(int(Path(source['path']).name.split('_')[1]),int(name.split('_')[1]))
                by_topic.setdefault(tid,[]).append(deepcopy(topic))
                if int(name.split('_')[1])-1!=int(Path(source['path']).name.split('_')[1]):nonadjacent.append(name)
                if tid in seen and topic['episode_ids']!=seen[tid]['episode_ids']:
                    self.assertEqual(topic['title'],seen[tid]['title']);self.assertEqual(topic['summary'],seen[tid]['summary'])
                seen[tid]=topic
        self.assertEqual(len(by_topic),2);self.assertTrue(nonadjacent)
        self.assertTrue(all(len(rows)>2 and len(rows[-1]['episode_ids'])>len(rows[0]['episode_ids']) for rows in by_topic.values()))

    def test_synchronized_native_parsed_seal_tamper_cannot_override_original_raw(self):
        path=next((self.bank/'calls').glob('topic_*hypermem_topic_update.json'))
        value=read(path);value['parsed_response']['summary']='Changed recorded value'
        path.write_text(json.dumps(value),encoding='utf8')
        with self.assertRaisesRegex(adapter.HyperMemError,'exact raw-derived value'):self.replay()

    def test_missing_raw_or_modified_complete_episode_source_is_rejected(self):
        path=next((self.bank/'calls').glob('topic_*.json'));key=digest(audit.signature(read(path)['request']))
        original=self.records.pop(key)
        with self.assertRaisesRegex(adapter.HyperMemError,'exact raw producer'):self.replay()
        self.records[key]=original
        episodes=read(self.bank/'episodes.json');episodes[0]['participants'].append('foreign-participant')
        (self.bank/'episodes.json').write_text(json.dumps(episodes),encoding='utf8')
        checkpoint=read(self.bank/'episode_stage.json');checkpoint['artifacts']['episodes.json']=sha(self.bank/'episodes.json')
        adapter._runtime._seal_path(self.bank/'episode_stage.json').unlink()
        adapter._runtime._write_json(self.bank/'episode_stage.json',checkpoint);adapter._runtime._seal(self.bank/'episode_stage.json')
        with self.assertRaisesRegex(adapter.HyperMemError,'full prompt/order'):self.replay()

if __name__=='__main__':unittest.main()
