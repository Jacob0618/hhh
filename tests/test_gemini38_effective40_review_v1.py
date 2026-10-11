"""Pure offline contract mutations; no production prepare/export/SQL/API."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import review_gemini38_effective40_v1 as r


def exported():
    ci,arm=3,'boundary_multiscale'; mid,text,n,count=r.KNOWN[(ci,arm)]
    uid=f'dated_regroup_c{ci}_{arm}'
    facts=[{'id':mid,'text':text,'user_id':uid,'embedding':[1.0]+[0.0]*767}]
    facts += [{'id':str(i),'text':'Some source fact.','user_id':uid,'embedding':[1.0]+[0.0]*767} for i in range(count-1)]
    return {'schema':r.EXPORT_SCHEMA,'conversation':ci,'arm':arm,'protocol_sha256':'protocol',
            'source_dataset_sha256':r.SOURCE_HASH,'original_write_status':'failed','all_planned_groups_persisted':True,
            'reviewed_export_status':'reviewed_with_native_limitations','entity_links_complete':False,'strict_passed':False,
            'facts':facts,'user_id':uid,'orphan_fact_ids':[mid],'journal_files':[{}]*n}


def fake_core(target, *args):
    # Names come from isolated inspector globals, not this test module.
    return {'bank':index_audit.pointer(target[2]/'bank.json'),
            'inputs':index_audit.pointer(target[2]/'group_inputs.json')}


class Contracts(unittest.TestCase):
    def setUp(self): r.idx.HASHES.clear()
    def test_exact_export_and_normal_bank_delegation(self):
        self.assertEqual(len(r.validate_fact_bank(exported(),3,'boundary_multiscale','protocol')),177)
        with patch.object(r.retrieval,'_fact_bank',return_value=['original']) as f:
            self.assertEqual(r.validate_fact_bank({'schema':'old'},0,'session','p'),['original'])
            f.assert_called_once()
    def test_export_reject_mutations(self):
        for key,value in [('complete',True),('complete',False),('entity_links_complete',True),('strict_passed',True),
                          ('all_planned_groups_persisted',False),('original_write_status','complete'),
                          ('protocol_sha256','wrong'),('source_dataset_sha256','wrong'),('orphan_fact_ids',[])]:
            b=exported();b[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):r.validate_fact_bank(b,3,'boundary_multiscale','protocol')
    def test_partial_and_unknown_orphan_refused(self):
        for mutate in (lambda b:b['facts'].pop(),lambda b:b['facts'][0].update(text='changed'),
                       lambda b:b['facts'][0].update(embedding=[1.0]),lambda b:b['journal_files'].pop(),
                       lambda b:b.update(orphan_fact_ids=['unknown'])):
            b=exported();mutate(b)
            with self.assertRaises(ValueError):r.validate_fact_bank(b,3,'boundary_multiscale','protocol')
        with self.assertRaises(ValueError):r.validate_fact_bank(exported(),2,'boundary_multiscale','protocol')
    def test_exact40_population(self):
        rows=[{'conversation':ci,'arm':a}for ci,a in r.EXPECTED]
        self.assertEqual(len(r._slots(rows,'bad')),40)
        for bad in (rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):r._slots(bad,'bad')
    def test_path_namespace_and_full_pointer(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'ok';p.write_text('one');ref=r.idx.pointer(p)
            self.assertEqual(r.checked(ref),p.resolve())
            with self.assertRaises(ValueError):r.checked({**ref,'more':True})
            p.write_text('two')
            with self.assertRaises(ValueError):r.checked(ref)
            with self.assertRaises(ValueError):r.inside(Path(td).parent/'outside',td)
    def test_inspector_pointer_mapping_is_exact_and_globals_unmodified(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);(folder/'group_inputs.json').write_text('{}')
            target=(3,'boundary_multiscale',folder,{}, {},exported())
            ref={'path':str(folder/'newexport.json'),'sha256':'unpublished'}
            previous=vars(r.old.strict).get('index_audit')
            with patch.object(r.old.strict,'inspect_bank',fake_core):
                result=r.inspect_effective_bank(target,{}, {},{},None,None,bank_pointer=ref)
                self.assertEqual(result['bank'],ref)
                self.assertEqual(result['inputs'],r.idx.pointer(folder/'group_inputs.json'))
                self.assertIs(vars(r.old.strict).get('index_audit'),previous)
            with self.assertRaises(ValueError):r.inspect_effective_bank((0,'session',folder,{}, {},{}),{}, {},{},None,None,bank_pointer=ref)
    def test_inventory_no_wal_symlink_and_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            native=Path(td);(native/'history.sqlite').write_bytes(b'closed');(native/'x').write_bytes(b'raw')
            inventory=r.full_native_inventory(native);self.assertEqual(set(inventory),{'history.sqlite','x'})
            (native/'history.sqlite-wal').write_bytes(b'active')
            with self.assertRaises(ValueError):r.full_native_inventory(native)
    def test_real_readonly_os_lock_and_live_pid_refusal(self):
        import portalocker
        with tempfile.TemporaryDirectory() as td:
            native=Path(td);(native/'qdrant').mkdir();(native/'qdrant/.lock').write_bytes(b'0');(native/'history.sqlite').write_bytes(b'closed')
            state={'status':'complete','finished_utc':'now','pid':1}
            with patch.object(r.recovery,'pid_alive',return_value=False):
                with r.closed_native(native,state) as inventory:self.assertIn('history.sqlite',inventory)
                with (native/'qdrant/.lock').open('rb') as h:
                    portalocker.lock(h,portalocker.LOCK_EX|portalocker.LOCK_NB)
                    try:
                        with self.assertRaises(portalocker.exceptions.LockException):
                            with r.closed_native(native,state):pass
                    finally:portalocker.unlock(h)
            with patch.object(r.recovery,'pid_alive',return_value=True):
                with self.assertRaises(ValueError):
                    with r.closed_native(native,state):pass
    def test_exact_known_diagnosis_rejects_drift(self):
        ci,arm=3,'boundary_multiscale';mid,text,groups,facts=r.KNOWN[(ci,arm)]
        state={'status':'failed','error':r.ERROR,'completed_groups':groups,'expected_groups':groups,'facts':facts}
        out={'status':'failed','error':r.ERROR};p={'path':'parent','sha256':'sha'};inp={'path':'input','sha256':'sha'}
        proof={'schema':'gemini38-terminal-c3-entity-overwrite-readonly-audit-v1','state':state,'outcome':out,
            'parent_manifest':p,'inputs':inp,'source_sha_unchanged':True,'source_mutations':0,'model_calls':0,'bank_json_exists':False,
            'native_inventory_before':{'a\\b':'h'},'native_inventory_after':{'a\\b':'h'},
            'orphans':[{'fact':{'id':mid,'text':text}}],
            'source_generation_add_chain':{'groups_verified':groups,'facts_verified':facts,'raw_refs':[]},
            'fact_bm25_exact':{'compared':facts,'exact_matches':facts}}
        r.validate_known_defect(ci,arm,state,out,proof,{'a/b':'h'},p,inp)
        for key,value in [('source_sha_unchanged',False),('bank_json_exists',True),('model_calls',1),('native_inventory_after',{}),('orphans',[])]:
            bad=copy.deepcopy(proof);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):r.validate_known_defect(ci,arm,state,out,bad,{'a/b':'h'},p,inp)
        bad=copy.deepcopy(state);bad['completed_groups']=3
        with self.assertRaises(ValueError):r.validate_known_defect(ci,arm,bad,out,proof,{'a/b':'h'},p,inp)
    def test_missing_gate_failclosed_no_old_gate_fallback(self):
        with tempfile.TemporaryDirectory() as td, patch.object(r.recovery,'closed_parent_lock',return_value=contextlib.nullcontext()), \
             patch.object(r.writer,'manifest',return_value={'protocol_sha256':'p'}),patch.object(r.retrieval,'verify_new_config'):
            with self.assertRaises(FileNotFoundError):r.admit_gate(td)
    def test_recovery_prefix_full_request_ledger_lineage_and_order(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);native=folder/'native'
            (native/'api_cache').mkdir(parents=True);(native/'cache_lineage').mkdir()
            prefix=[];ledger=[];events=[]
            for i in range(18):
                key=f'{i:064x}';signature={'provider':'gemini','model':'gemini-3.8-flash',
                    'operation':'generateContent','body':{'complete_user_request':f'original {i}'}}
                original=folder/f'original_{i}.json';raw=native/'api_cache'/f'{key}.json'
                value={'request':signature,'response':{'preserved':'entire response'}}
                original.write_text(json.dumps(value));raw.write_bytes(original.read_bytes())
                ref=r.idx.pointer(original);ctx={'conversation':3,'method':'session','session_id':f'session_{i}'}
                row={'request_hash':key,'signature':signature,'context':ctx,'stage':'mem0_extract','raw':ref,
                    'intent':{'path':'originalintent','sha256':str(i)},'original_attempt':{'attempt':1}}
                prefix.append(row)
                lineage={'source_path':ref['path'],'source_sha256':ref['sha256'],'target_path':str(raw.resolve()),
                    'target_sha256':ref['sha256'],'request_hash':key,'original_intent':row['intent'],
                    'original_physical_attempt':row['original_attempt']}
                (native/'cache_lineage'/f'{key}.json').write_text(json.dumps(lineage))
                ledger.append({'request_hash':key,'cache_hit':True,'context':ctx,'stage':'mem0_extract',
                    'provider':'gemini','model':'gemini-3.8-flash','operation':'generateContent'})
                events.append({'request_hash':key,'event':'request_returned','source':'inherited_exact_request',
                    'context':ctx,'stage':'mem0_extract','original_raw':ref,'consumed_raw':r.idx.pointer(raw)})
            def publish():
                (native/'api_calls.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in ledger))
                (folder/'recovery_request_events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
                r.idx.HASHES.clear()
            publish();r.validate_recovery_prefix(folder,{'prefix_requests':prefix})
            for mutate in ('ledger_context','physical_instead_cache','order','signature'):
                saved=copy.deepcopy((prefix,ledger,events))
                if mutate=='ledger_context':ledger[0]['context']={'wrong':'consumer'}
                elif mutate=='physical_instead_cache':ledger[0]['cache_hit']=False
                elif mutate=='order':events[0],events[1]=events[1],events[0]
                else:prefix[0]['signature']['body']['complete_user_request']='different request same response'
                publish()
                with self.subTest(mutate=mutate),self.assertRaises(ValueError):r.validate_recovery_prefix(folder,{'prefix_requests':prefix})
                prefix,ledger,events=saved


import contextlib
if __name__=='__main__':unittest.main()
