"""Stage-queue admission/stop guards. All processes and model dispatch mocked."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_planned_pipeline_v1 as queue
from tests.test_sdk20_plain_qa_v1 import job


class IndependentPipelineReview(unittest.TestCase):
    def cohort(self):
        template=job();jobs=[]
        for arm in queue.H_ARMS:
            for i in range(1540):
                ci=i%10;qid=f'c{ci}_q{i}'
                jobs.append({**template,'job_id':arm+'/'+qid,'conversation':ci,'method':arm,'query_id':qid})
        return {'schema':queue.planned.SCHEMA,'jobs':jobs,'protocol_sha256':'independent_protocol',
                'phase_kind':'firstpass','phase_id':'sdk20_plain_qa_firstpass_v1',
                'source_manifest':{'path':'independent_source','sha256':'independent_digest'}}

    def test_all_stage_files_exist_but_unknown_is_not_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest=self.cohort();refs={}
            for stage in ('reader','judge','compat','qa'):
                path=root/(stage+'.json');queue.save_new(path,{'stage':stage,'status':'unknown'})
                refs[stage]=queue.pointer(path)
            result={'total':4620,'terminal':True,'protocol_sha256':manifest['protocol_sha256'],
                    'outcomes':[],'results':[{**j,**refs} for j in manifest['jobs']]}
            with self.assertRaises(ValueError):queue.validate_h_terminal(result,manifest)

    def test_local_controller_failure_blocks_even_complete_count(self):
        manifest=self.cohort()
        result={'total':4620,'terminal':True,'protocol_sha256':manifest['protocol_sha256'],
                'results':manifest['jobs'],'outcomes':[{'local_errors':[{'type':'disk_failed'}]}]}
        with self.assertRaises(ValueError):queue.validate_h_terminal(result,manifest)

    def test_missing_or_duplicate_h_slot_cannot_replace_4620(self):
        manifest=self.cohort();rows=copy.deepcopy(manifest['jobs']);rows[-1]=rows[0]
        with self.assertRaises(ValueError):
            queue.validate_h_terminal({'total':4620,'terminal':True,
                'protocol_sha256':manifest['protocol_sha256'],'results':rows},manifest)

    def test_exit_zero_and_two_blank_complete_arms_are_insufficient(self):
        rows={str(ci):{'status':'complete','arms':[{'status':'complete','outputs':{}},
                    {'status':'complete','outputs':{}}]} for ci in range(10)}
        state={'status':'complete','requested_conversations':list(range(10)),'outcomes':rows}
        with (patch.object(queue,'read',return_value=state),patch.object(queue,'verify',return_value=state),
              patch.object(queue,'pointer',return_value={'fixture':True})):
            with self.assertRaises(ValueError):queue.verify_step('boundary')

    def test_fixed_queue_stops_at_development_and_has_no_full_writer_or_qa(self):
        steps=queue.steps()
        self.assertEqual([s[0] for s in steps],['boundary','global_prepare_c1','global_prepare_c2',
            'global_score','writer_build_c1','writer_build_c2','writer_development'])
        self.assertEqual(steps[-1][2][-3:],['--conversations','0','1'])
        self.assertFalse(any('qa' in s[0] or 'expansion' in s[0] for s in steps))

    def test_h_pid_replacement_requires_review_without_any_c_child(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);h=root/'h';h.mkdir();run=root/'queue';run.mkdir()
            refs={key:{'path':str(root/key),'sha256':'code'} for key in ('h_manifest','h_execution','c_manifest')}
            manifest={**refs,'h_process_identity':{'pid':123,'creation_ticks':1},
                'code_sha256':{p:'code' for p in queue.CODE},'steps':[list(s) for s in queue.steps()]}
            def pointer(path):return {'path':str(Path(path).resolve()),'sha256':'code'}
            with (patch.object(queue,'H',h),patch.object(queue,'QUEUE',run),
                  patch.object(queue,'verify',return_value=manifest),patch.object(queue,'pointer',side_effect=pointer),
                  patch.object(queue,'sha',return_value='code'),patch.object(queue,'durable_intent'),
                  patch.object(queue,'publish_json') as publish,
                  patch.object(queue.parallel,'process_identity',return_value={'pid':123,'creation_ticks':2}),
                  patch.object(queue.subprocess,'Popen') as child):
                with self.assertRaises(RuntimeError):queue.run()
            child.assert_not_called()
            self.assertEqual(publish.call_args.args[1]['status'],'needs_review')

    def test_complete_4620_with_valid_no_passes_without_recovery_dispatch(self):
        manifest=self.cohort();lookup={};results=[]
        def pointer(path):return {'path':str(Path(path).resolve()),'sha256':'mock-file-hash'}
        for j in manifest['jobs']:
            ident=queue.planned.identity(j,manifest);paths=queue.planned.stage_files(queue.H,j)
            refs={s:pointer(p) for s,p in paths.items()};results.append({**j,**refs})
            event={'error_type':None,'cache_sha256':'mock-file-hash','cache_path':'mock-raw.json'}
            for s,p in paths.items():
                lookup[str(p)]={**ident,'stage':s,'status':'strict_valid' if s=='compat' else 'complete',
                     'error':None,'result':{'correct':False},'api_events':[event]}
            lookup[str(paths['judge'])]['reader_sha256']=refs['reader']['sha256']
            lookup[str(paths['compat'])]['strict']=refs['judge']
            lookup[str(paths['qa'])].update(reader_status='complete',judge_status='complete',
                                          **{s:refs[s] for s in ('reader','judge','compat')})
        out={'total':4620,'terminal':True,'protocol_sha256':manifest['protocol_sha256'],
             'results':results,'outcomes':[{'local_errors':[]} for _ in range(10)]}
        with (patch.object(queue,'pointer',side_effect=pointer),
              patch.object(queue,'verify',side_effect=lambda p:lookup[str(p)]),
              patch.object(queue.planned,'known_failure',side_effect=AssertionError('Valid NO cannot trigger retry'))):
            self.assertTrue(queue.validate_h_terminal(out,manifest))

    def test_pid_created_after_intent_cannot_bind_waiter(self):
        intent={'pid':123,'protocol_sha256':'p','created_utc':'2026-10-03T00:00:00+00:00'}
        ticks=int((1790985601+11644473600)*10000000)
        with patch.object(queue.parallel,'process_identity',return_value={'pid':123,'creation_ticks':ticks}):
            with self.assertRaisesRegex(ValueError,'created after original intent'):
                queue.h_identity_for_intent(intent,{'protocol_sha256':'p'})

    def test_exact_closed_c_controller_passes_duplicate_arm_does_not(self):
        def ptr(path):return {'path':str(Path(path).resolve()),'sha256':'mock-code'}
        controller=queue.C/'parallel_controller_v1/boundary';cp=ptr(controller/'manifest.json')
        manifest={'schema':queue.parallel.VERSION,'parent':ptr(queue.C/'manifest.json'),
            'code_sha256':{p:'mock-code' for p in queue.parallel.FILES},
            'policy':{'stage':'boundary','arms':list(queue.boundary.variants.ARMS),'conversations':list(range(10))}}
        outcomes={};terminals={}
        def outputs(run,stage,arm,ci):return {'grouping':ptr(queue.C/arm/'boundary'/f'conv_{ci}'/'grouping.json')}
        for ci in range(10):
            arms=[]
            for arm in queue.boundary.variants.ARMS:
                row={'status':'complete','conversation':ci,'stage':'boundary','arm':arm,
                     'controller_manifest':cp,'finished_utc':'closed','outputs':outputs(None,None,arm,ci)}
                arms.append(row);terminals[str(controller/'conversations'/f'conv_{ci}'/arm/'terminal.json')]=row
            outcomes[str(ci)]={'status':'complete','conversation':ci,'arms':arms}
        state={'status':'complete','stage':'boundary','finished_utc':'closed','controller_manifest':cp,
            'arms':list(queue.boundary.variants.ARMS),'requested_conversations':list(range(10)),'outcomes':outcomes}
        def verify(path):return manifest if Path(path).name=='manifest.json' else terminals[str(path)]
        with (patch.object(queue,'read',return_value=state),patch.object(queue,'verify',side_effect=verify),
              patch.object(queue,'pointer',side_effect=ptr),patch.object(queue,'sha',return_value='mock-code'),
              patch.object(queue.parallel,'output_pointers',side_effect=outputs)):
            self.assertEqual(queue.verify_step('boundary'),ptr(controller/'parent_state.json'))
            state['outcomes']['0']['arms'][1]=state['outcomes']['0']['arms'][0]
            with self.assertRaisesRegex(ValueError,'both complete declared arms'):queue.verify_step('boundary')

    def test_all_seven_stages_serial_then_awaiting_review_no_eighth_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);h=root/'h';h.mkdir();run=root/'queue';run.mkdir()
            (h/'run_outcome.json').write_text('{}',encoding='utf-8');seen=[]
            class Child:
                pid=123
                def wait(self):seen.append('waited');return 0
            def spawn(args,**kwargs):
                self.assertTrue(not seen or seen[-1]=='waited')
                seen.append(args);return Child()
            with (patch.object(queue,'H',h),patch.object(queue,'QUEUE',run),
                  patch.object(queue,'verify',return_value={}),patch.object(queue,'read',return_value={}),
                  patch.object(queue,'validate_queue_bindings'),patch.object(queue,'validate_h_terminal',return_value=True),
                  patch.object(queue,'pointer',return_value={'fixture':'proof'}),patch.object(queue,'verify_step',return_value={'fixture':'proof'}),
                  patch.object(queue,'verify_lock'),patch.object(queue.boundary,'manifest'),
                  patch.object(queue.subprocess,'Popen',side_effect=spawn) as child):
                result=queue.run()
            self.assertEqual(child.call_count,7);self.assertEqual(result['status'],'awaiting_development_review')
            self.assertEqual([s['stage'] for s in result['steps']],[s[0] for s in queue.steps()])
            self.assertEqual(len(list(run.glob('*/intent.json'))),7)
            self.assertTrue((run/'execution_intent.json').exists())


if __name__=='__main__':unittest.main()
