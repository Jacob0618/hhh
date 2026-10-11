"""Second-reviewer v4 gates. Temporary fixtures only; never production CLI.

No full-source preflight or controller-shaped subprocess is used here.
"""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

from tools import run_five_baselines_hypermem_native_v4 as entry
from tools import hypermem_prefix_api_v4 as prefix
from tools import audit_hypermem_continuation_v4 as audit
from jev_mem import providers


class FixedGrantBinding(unittest.TestCase):
    """Synthetic root publication tests, not evidence of human permission."""
    def setup_grant(self,root,stack):
        grant=root/'human_authorization_v1.json';review=root/'independent_candidate_review_v1.json'
        proof_path=root/'terminal_candidate_v1.json';proposal=root/'proposal.json';code=root/'candidate.py'
        code.write_text('# isolated synthetic source',encoding='utf-8')
        proof_path.write_text('{"synthetic":true}',encoding='utf-8')
        code_map={str(code):prefix.sha(code)}
        rv={'schema':entry.REVIEW_SCHEMA,'status':'passed_implementation_and_zero_api_preflight_only',
            'production_launch_authorized':False,'code_sha256':code_map}
        review.write_text(json.dumps(rv),encoding='utf-8')
        rows=[]
        for ci in (2,3,6,9):
            rows.append({'conversation':ci,'request_hash':str(ci)*64,'signature':{'synthetic_full_signature':ci},
                'context':{'method':'hypermem','conversation':ci,'phase':'build'},'stage':'hypermem_episode_roles',
                'original_raw':{'source_path':str(root/f'old_raw_{ci}.json'),'observed_sha256':'a'*64}})
        compatible=[{'conversation':ci,'request_hash':str(ci)*64} for ci in (0,1,4,5,7,8)]
        proposal.write_text(json.dumps({'targets':rows+compatible}),encoding='utf-8')
        refs={'proposal':prefix.binding(proposal)}
        replacements=[{k:v for k,v in r.items() if k!='original_raw'} | {
            'raw':{'path':r['original_raw']['source_path'],'sha256':r['original_raw']['observed_sha256']},'logical_limit':1} for r in rows]
        value={'schema':'hypermem-native-v4-explicit-human-grant-v1','status':'authorized_by_explicit_user_instruction',
            'issued_by':'root_after_trusted_user_confirmation','source_bindings':refs,'logical_limit_per_key':1,
            'old_consumed_key':prefix.admission.CONSUMED_OLD_KEY,'old_consumed_key_permission':False,
            'human_confirmation':{'actor':'user','source':'direct_user_message','message_id':'synthetic_fixture_only',
                'text':'synthetic fixture, no real authorization','text_sha256':prefix.adapter.text_sha('synthetic fixture, no real authorization')},
            'terminal_proof':prefix.binding(proof_path),'implementation_review':prefix.binding(review),
            'code_sha256':code_map,'replacements':replacements,
            'accepted_compatibility_request_hashes':sorted(r['request_hash'] for r in compatible)}
        proof={'source_bindings':refs,'replacement_cases':{r['request_hash']:{k:v for k,v in r.items() if k!='logical_limit'} for r in replacements},
            'compatibility_cases':{r['request_hash']:r for r in compatible}}
        for attr,val in (('HUMAN_GRANT_PATH',grant),('REVIEW_PATH',review),('PROOF_PATH',proof_path)):
            stack.enter_context(patch.object(prefix,attr,val))
        stack.enter_context(patch.object(prefix.admission,'fixed_bindings',return_value=refs))
        return grant,review,proof_path,code,value,proof

    def test_exact_four_request_grant_and_all_stale_bindings(self):
        with tempfile.TemporaryDirectory(dir=entry.ROOT/'research') as tmp,ExitStack() as stack:
            grant,review,proof_path,code,value,proof=self.setup_grant(Path(tmp),stack)
            grant.write_text(json.dumps(value),encoding='utf-8')
            self.assertEqual(prefix.require_human_grant(prefix.binding(grant),proof),value)
            original=deepcopy(value)
            edits=[lambda v:v.__setitem__('old_consumed_key_permission',True),
                lambda v:v.__setitem__('logical_limit_per_key',True),
                lambda v:v['replacements'].pop(),
                lambda v:v['replacements'].append(deepcopy(v['replacements'][0])),
                lambda v:v['replacements'][0].__setitem__('request_hash',prefix.admission.CONSUMED_OLD_KEY),
                lambda v:v['replacements'][0]['context'].__setitem__('conversation',9),
                lambda v:v['replacements'][0]['signature'].__setitem__('changed',True),
                lambda v:v['accepted_compatibility_request_hashes'].pop(),
                lambda v:v['human_confirmation'].__setitem__('text','different text'),
                lambda v:v['terminal_proof'].__setitem__('sha256','0'*64),
                lambda v:v['implementation_review'].__setitem__('sha256','0'*64)]
            for i,edit in enumerate(edits):
                v=deepcopy(original);edit(v);grant.write_text(json.dumps(v),encoding='utf-8')
                with self.subTest(edit=i),self.assertRaises(prefix.BaselineFatal):prefix.require_human_grant(None,proof)
            grant.write_text(json.dumps(original),encoding='utf-8');old_binding=prefix.binding(grant)
            grant.write_text(json.dumps(original,indent=2),encoding='utf-8')
            with self.assertRaisesRegex(prefix.BaselineFatal,'single fixed'):prefix.require_human_grant(old_binding,proof)
            for path in (proof_path,review,code):
                old=path.read_bytes();path.write_bytes(old+b' ')
                try:
                    with self.subTest(changed=path.name),self.assertRaises(prefix.BaselineFatal):prefix.require_human_grant(None,proof)
                finally:path.write_bytes(old)


class InactiveMain(unittest.TestCase):
    def forbidden(self,*args,**kwargs):
        raise AssertionError('No grant must fail before a production operation')

    def guards(self,stack,root):
        stack.enter_context(patch.object(prefix,'HUMAN_GRANT_PATH',root/'absent_human_authorization.json'))
        stack.enter_context(patch.object(entry,'NEW',root/'production_candidate'))
        stack.enter_context(patch.object(entry,'read',side_effect=self.forbidden))
        stack.enter_context(patch.object(entry,'save_new',side_effect=self.forbidden))
        stack.enter_context(patch.object(entry.admission,'validate_proof',side_effect=self.forbidden))
        stack.enter_context(patch.object(entry.chain,'strong_predecessors',side_effect=self.forbidden))
        stack.enter_context(patch.object(entry.chain.base,'status',side_effect=self.forbidden))
        stack.enter_context(patch.object(providers,'load_credentials',side_effect=self.forbidden))
        stack.enter_context(patch.object(entry.BaselineAPI,'__init__',side_effect=self.forbidden))
        stack.enter_context(patch.object(Path,'mkdir',side_effect=self.forbidden))

    def test_actual_three_main_routes_without_grant_never_read_manifest_or_write(self):
        with tempfile.TemporaryDirectory(dir=entry.ROOT/'research') as tmp,ExitStack() as stack:
            root=Path(tmp);self.guards(stack,root)
            for command in ('prepare','check','run'):
                with self.subTest(command=command),patch.object(sys,'argv',['independent-candidate-fixture',command]):
                    with self.assertRaisesRegex(prefix.BaselineFatal,'No explicit human authorization'):
                        entry.main()
            self.assertEqual(list(root.iterdir()),[])

    def test_no_grant_reentry_preserves_existing_status_intents_and_manifest(self):
        with tempfile.TemporaryDirectory(dir=entry.ROOT/'research') as tmp,ExitStack() as stack:
            root=Path(tmp);new=root/'production_candidate';new.mkdir()
            for name in ('run_manifest.json','execution_intent.json','continuation_execution_intent.json','status.json'):
                (new/name).write_bytes(('retained '+name).encode())
            before={str(p):p.read_bytes() for p in new.iterdir()}
            self.guards(stack,root)
            stack.enter_context(patch.dict(entry._RUN_STATE,{'owned':True}))
            for command in ('prepare','check','run'):
                with self.subTest(command=command),patch.object(sys,'argv',['independent-candidate-fixture',command]):
                    with self.assertRaisesRegex(prefix.BaselineFatal,'No explicit human authorization'):
                        entry.main()
            self.assertEqual({str(p):p.read_bytes() for p in new.iterdir()},before)

    def test_direct_production_entry_functions_have_same_first_barrier(self):
        with tempfile.TemporaryDirectory(dir=entry.ROOT/'research') as tmp,ExitStack() as stack:
            root=Path(tmp);self.guards(stack,root)
            actions=(lambda:entry.prepare(root/'irrelevant_review.json',root/'irrelevant_proof.json'),
                lambda:entry.manifest(),lambda:entry.run())
            for action in actions:
                with self.assertRaisesRegex(prefix.BaselineFatal,'No explicit human authorization'):action()
            self.assertEqual(list(root.iterdir()),[])


class ExecutionOwnership(unittest.TestCase):
    def test_real_main_restores_deep_guard_and_only_owned_run_can_write_failure(self):
        changes=((entry.chain.guard,'active_controllers'),(entry.chain.guard,'assert_stopped'),
            (entry.chain.previous.simple,'_stop'))
        before=[getattr(m,k) for m,k in changes]
        for owned in (False,True):
            def run(**kwargs):
                self.assertIs(entry.chain.guard.assert_stopped,entry.admission.assert_stopped)
                entry._RUN_STATE['owned']=owned
                raise ValueError('independent bounded run sentinel')
            with self.subTest(owned=owned),patch.object(entry,'require_activation',return_value={'fixture':True}),\
                    patch.object(entry,'run',side_effect=run),patch.object(entry.chain.base,'status') as status,\
                    patch.object(sys,'argv',['independent-entry-fixture','run']):
                with self.assertRaisesRegex(ValueError,'independent bounded run sentinel'):entry.main()
                self.assertEqual(status.call_count,int(owned))
                if owned:self.assertEqual(status.call_args.args,(entry.NEW,'needs_review'))
            self.assertEqual([getattr(m,k) for m,k in changes],before)
        entry._RUN_STATE['owned']=False

    def test_ticket_is_single_use_with_sha_attestations_and_no_recursive_old_inputs(self):
        with tempfile.TemporaryDirectory(dir=entry.ROOT/'research') as tmp,ExitStack() as stack:
            root=Path(tmp);old=root/'old';new=root/'new';old.mkdir();new.mkdir()
            (old/'run_manifest.json').write_text('{}')
            predecessors={}
            for method in ('amem','simplemem'):
                predecessors[method]={}
                for kind in ('completion','receipt','plan'):
                    p=root/(method+'_'+kind+'.json');p.write_text('{}')
                    predecessors[method][kind]=p
            accepted={n:{k:entry.binding(p) for k,p in paths.items()} for n,paths in predecessors.items()}
            m={'human_grant':{'fixture':True},'review':{'path':str(root/'review.json')},
                'terminal_proof':{'path':str(root/'proof.json')},'hash_environment':{}}
            (new/'run_manifest.json').write_text(json.dumps(m))
            stack.enter_context(patch.object(entry,'OLD',old));stack.enter_context(patch.object(entry,'NEW',new))
            stack.enter_context(patch.object(entry,'require_activation',return_value=m['human_grant']))
            stack.enter_context(patch.object(entry,'check_review',return_value={}))
            stack.enter_context(patch.object(entry.admission,'check_hash_environment',return_value=None))
            stack.enter_context(patch.object(entry.chain,'completion_paths',return_value=predecessors))
            ticket=entry.GateTicket(m,{},accepted)
            with self.assertRaisesRegex(ValueError,'Wrong/reused'):ticket.consume('vmem')
            self.assertEqual([p.name for p in new.iterdir()],['run_manifest.json'])
            changed=predecessors['amem']['plan'];changed.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'Accepted predecessor changed'):ticket.consume('hypermem')
            self.assertFalse(ticket.used);changed.write_text('{}')
            ticket.consume('hypermem')
            intent=entry.read(new/'continuation_execution_intent.json')
            record=intent['predecessor_acceptance']
            self.assertFalse(record['new_scientific_input']);self.assertFalse(record['recursive_rearchive_claimed'])
            self.assertEqual(record['verified_objects'],{n:{k+'_sha256':v['sha256'] for k,v in rows.items()} for n,rows in accepted.items()})
            self.assertNotIn(str(root),json.dumps(record))
            before=(new/'continuation_execution_intent.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'Wrong/reused'):ticket.consume('hypermem')
            with self.assertRaisesRegex(ValueError,'Fresh unstarted'):entry.GateTicket(m,{},accepted).consume('hypermem')
            self.assertEqual((new/'continuation_execution_intent.json').read_bytes(),before)


class RolePostReplayBinding(unittest.TestCase):
    """Tests the full-object comparison after replay, not the replay algorithm.

    The replay object is explicit synthetic data. Matching resealed journals
    cannot change any field of that previously derived object, even fields not
    shown in the formatter's first five episode IDs.
    """
    def fixture(self,root):
        folder=root/'isolated';native=folder/'work/native/c0';native.mkdir(parents=True)
        p=native/'calls/topic_000024_hypermem_episode_roles.json'
        ids=[str(uuid.uuid5(uuid.NAMESPACE_URL,'independent-v4-local-'+str(i))) for i in range(7)]
        mapping={f'episode_{i+1}':v for i,v in enumerate(ids)}
        local=[{'event_id':v,'subject':'same title','summary':'same summary','original_data':[{'dia_id':f'D1:{i}'}]} for i,v in enumerate(ids)]
        topic={'topic_id':'topic_'+str(uuid.uuid5(uuid.NAMESPACE_URL,'topic')),'title':'same title','summary':'same summary',
            'keywords':['same'],'timestamp':'2026-10-09T00:00:00','participants':['A','B'],'user_id_list':['A_0','B_0'],
            'episode_ids':ids,'metadata':{'retained':'complete'}}
        request={'prompt':'synthetic already-replayed full prompt','response_format':{'type':'json_object'},
            'stage':'hypermem_episode_roles','json_output':True,'max_output':16384,'system':None}
        original={'episode_roles':[{'episode_id':v,'role':'developing','weight':.7} for v in ids],'coherence_score':.8}
        derived,pure=audit.adapter.represent_episode_ids(original,mapping)
        value={'request':request,'raw_response':original,'parsed_response':derived}
        manifest={'terminal_proof':{'path':str(root/'synthetic_proof.json'),'sha256':'d'*64}}
        (folder/'run_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        rr={'signature':audit.signature(request),'raw':{'path':str(folder/'work/api_method/api_cache/synthetic.json'),'sha256':'a'*64}}
        proof={'compatibility_cases':{},'role_cases':[],'producer_registry':{},'replacement_cases':{}}
        journal={**pure,'native_call_path':str(p),'native_request':request,'local_topic':deepcopy(topic),'local_episodes':deepcopy(local)}
        capture={'simple_to_real':mapping,'local_topic':deepcopy(topic),'local_episodes':deepcopy(local)}
        return native,p,value,original,rr,proof,manifest,journal,capture

    def seal_journal(self,values):
        native,p,value,original,rr,proof,m,j,capture=values
        evidence={k:v for k,v in j.items() if k!='authorization'};key=audit.digest(rr['signature'])
        j['authorization']={'schema':'hypermem-v4-role-representation-authorization-v1','status':'authorized','scope':'new_exact_bijection_response',
            'manifest':audit.binding(native.parents[2]/'run_manifest.json'),'terminal_proof':m['terminal_proof'],'request_hash':key,'signature_sha256':key,
            'context':{'method':'hypermem','conversation':0,'phase':'build'},'stage':'hypermem_episode_roles','raw':rr['raw'],
            'native_request_sha256':audit.adapter._runtime._sha(value['request']),'evidence_sha256':audit.adapter._runtime._sha(evidence),
            'original_parsed_sha256':j['original_parsed_sha256'],'derived_parsed_sha256':j['derived_parsed_sha256'],'model_regeneration_permitted':False}
        path=native/'representations'/p.name;path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(j),encoding='utf-8')
        seal=audit.adapter._runtime._seal_path(path);seal.parent.mkdir(exist_ok=True)
        seal.write_text(audit.sha(path),encoding='ascii')

    def verify(self,values):
        native,p,value,original,rr,proof,_,_,capture=values
        with patch.dict(audit._TOPIC_REPLAY,{str(native):{p.name:capture}},clear=True):
            audit.verify_representation(native,p,value,original,rr,proof,0)

    def test_equal_complete_objects_pass_but_resealed_hidden_metadata_changes_fail(self):
        with tempfile.TemporaryDirectory(dir=entry.ROOT/'research') as tmp:
            values=self.fixture(Path(tmp));self.seal_journal(values);self.verify(values)
            pristine=deepcopy(values[7])
            edits=[lambda j:j['local_topic'].__setitem__('topic_id','another'),
                lambda j:j['local_topic'].__setitem__('timestamp','2026-10-10T00:00:00'),
                lambda j:j['local_topic'].__setitem__('participants',['B','A']),
                lambda j:j['local_topic'].__setitem__('user_id_list',['wrong']),
                lambda j:j['local_topic']['episode_ids'].__setitem__(6,'hidden-after-five'),
                lambda j:j['local_topic']['metadata'].__setitem__('retained','changed'),
                lambda j:j['local_episodes'][6]['original_data'][0].__setitem__('dia_id','foreign'),
                lambda j:j['local_episodes'].reverse()]
            for i,edit in enumerate(edits):
                with self.subTest(edit=i):
                    values[7].clear();values[7].update(deepcopy(pristine));edit(values[7]);self.seal_journal(values)
                    with self.assertRaisesRegex(ValueError,'Complete actual caller-local objects'):self.verify(values)

    def test_no_replay_or_unknown_old_transformation_is_not_waived(self):
        with tempfile.TemporaryDirectory(dir=entry.ROOT/'research') as tmp:
            values=self.fixture(Path(tmp));self.seal_journal(values)
            native,p,value,original,rr,proof,*_=values
            with patch.dict(audit._TOPIC_REPLAY,{},clear=True):
                with self.assertRaisesRegex(ValueError,'Unadmitted response transformation'):
                    audit.verify_representation(native,p,value,original,rr,proof,0)
            proof['producer_registry'][audit.digest(rr['signature'])]={'synthetic':'unqualified historical response'}
            with self.assertRaisesRegex(ValueError,'Unknown old role transformation'):self.verify(values)


if __name__=='__main__':unittest.main(verbosity=2)
