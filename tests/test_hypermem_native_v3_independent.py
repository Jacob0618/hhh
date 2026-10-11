"""Independent representation-domain checks; no model, database, or production artifacts."""
from copy import deepcopy
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import uuid

from tools import baselines_hypermem_native_v3 as a


class RepresentationDomain(unittest.TestCase):
    def setUp(self):
        self.ids=[str(uuid.uuid5(uuid.NAMESPACE_URL,f'independent-domain-{i}')) for i in range(3)]
        self.mapping={f'episode_{i+1}':v for i,v in enumerate(self.ids)}
        self.value={'episode_roles':[{'episode_id':v,'role':'developing','weight':.4,'rationale':{'keep':[None,True,2.0]}} for v in self.ids[::-1]],'coherence_score':.9,'reasoning':'unchanged','extra':{'string':'exact'}}

    def test_empty_native_alias_success_domain_is_not_restricted(self):
        value={'episode_roles':[],'coherence_score':.8}
        a._CompatibleGeneration._validate(value,'hypermem_episode_roles',{'simple_to_real':{}})
        derived,journal=a.represent_episode_ids(value,{})
        self.assertIs(derived,value)
        self.assertIsNone(journal)

    def test_generation_changes_are_exactly_two_hooks_and_other_ast_is_identical(self):
        tree=ast.parse(a._derived_source)
        removed=[]
        class Hooks(ast.NodeTransformer):
            def visit_Assign(self,node):
                value=node.value
                if isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute) and value.func.attr=='_represent':
                    removed.append(node)
                    return None
                return self.generic_visit(node)
        stripped=Hooks().visit(tree)
        self.assertEqual(len(removed),2)
        self.assertEqual(ast.dump(stripped,include_attributes=False),ast.dump(ast.parse(a._original),include_attributes=False))
        for node in removed:
            self.assertEqual([t.id for t in node.targets],['value'])
            self.assertEqual(ast.unparse(node.value),"self._represent(value, stage, frame.f_locals, path, request, cached_native='cached' in locals())")

    def test_future_response_same_exact_contract_needs_no_known_uuid_allowlist(self):
        before=deepcopy(self.value)
        result,journal=a.represent_episode_ids(self.value,self.mapping)
        self.assertEqual(self.value,before)
        self.assertEqual([r['episode_id'] for r in result['episode_roles']],['episode_3','episode_2','episode_1'])
        self.assertEqual({k:v for k,v in result.items() if k!='episode_roles'},{k:v for k,v in before.items() if k!='episode_roles'})
        for new,old in zip(result['episode_roles'],before['episode_roles']):
            self.assertEqual({k:v for k,v in new.items() if k!='episode_id'},{k:v for k,v in old.items() if k!='episode_id'})
        result['episode_roles'][0]['rationale']['keep'].append('derived-only')
        self.assertEqual(self.value,before)
        self.assertTrue(journal['inverse_equal'])

    def test_unknown_local_ids_and_noncanonical_or_nonsequential_mapping_refused(self):
        maps=[{'episode_1':self.ids[0],'episode_2':self.ids[1],'episode_3':str(uuid.uuid4())},
              {f'episode_{i+1}':v.upper() for i,v in enumerate(self.ids)},
              {'episode_01':self.ids[0],'episode_2':self.ids[1],'episode_3':self.ids[2]},
              {'episode_1':self.ids[0],'episode_2':self.ids[0],'episode_3':self.ids[2]}]
        for mapping in maps:
            with self.subTest(mapping=mapping),self.assertRaises(a.HyperMemError):
                a.represent_episode_ids(self.value,mapping)

    def test_other_stage_never_receives_role_representation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'research') as d:
            api=SimpleNamespace(authorize_episode_representation=lambda *args,**kw:self.fail('other stage cannot authorize'))
            gen=a._Generation(api,Path(d),'fixture')
            for stage in ('hypermem_topic_match','hypermem_fact_roles','hypermem_boundary'):
                self.assertIs(gen._represent(self.value,stage,{},Path(d)/'unused',{}),self.value)
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_real_generation_must_use_actual_caller_episodes_before_authority(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'research') as d:
            api=SimpleNamespace(generate=lambda *args,**kw:deepcopy(self.value),authorize_episode_representation=lambda *args,**kw:self.fail('invalid caller cannot authorize'))
            gen=a._Generation(api,Path(d),'topic')
            async def _assign_episode_roles_and_weights():
                simple_to_real=self.mapping
                episodes=[SimpleNamespace(event_id=v,to_dict=lambda:{}) for v in self.ids[::-1]]
                topic=SimpleNamespace(to_dict=lambda:{'topic_id':'fixture'})
                return await gen.generate('fixture',response_format={'type':'json_object'})
            coro=_assign_episode_roles_and_weights()
            try:
                with self.assertRaises(a.HyperMemError):coro.send(None)
            finally:coro.close()
            call=json.loads((Path(d)/'calls/topic_000001_hypermem_episode_roles.json').read_text())
            self.assertEqual(call['raw_response'],self.value)
            self.assertEqual(call['status'],'failed')
            self.assertFalse((Path(d)/'representations').exists())


if __name__=='__main__':unittest.main()
