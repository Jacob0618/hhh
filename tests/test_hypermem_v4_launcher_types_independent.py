"""JSON scalar-type negatives added after the initial launcher review."""
from copy import deepcopy
from unittest.mock import patch
import unittest
from tests import test_hypermem_v4_launcher_independent as original_fixture


class GrantTypes(unittest.TestCase):
    def test_boolean_limits_reject_before_any_launch_write(self):
        for field,bad in (('logical_limit_per_key',True),('row_logical_limit',True),
                ('logical_limit_per_key','1'),('row_logical_limit',1.0),('row_conversation','2'),
                ('row_conversation',2.0),('old_permission',0),('review_permission',0)):
            fixture=original_fixture.Launcher()
            self.addCleanup(fixture.doCleanups)
            save=original_fixture.write
            def write(path,value):
                value=deepcopy(value)
                if path.name=='human_authorization_v1.json':
                    if field=='logical_limit_per_key':value[field]=bad
                    elif field=='row_logical_limit':value['replacements'][0]['logical_limit']=bad
                    elif field=='row_conversation':value['replacements'][0]['conversation']=bad
                    elif field=='old_permission':value['old_consumed_key_permission']=bad
                if path.name=='independent_candidate_review_v1.json' and field=='review_permission':
                    value['production_launch_authorized']=bad
                return save(path,value)
            with self.subTest(field=field,value=bad),patch.object(original_fixture,'write',side_effect=write):
                root,run,launch,outcome=fixture.fixture('valid')
                self.assertNotEqual(outcome['outcome'],'success','Malformed JSON scalar was accepted')
                self.assertFalse((launch/'launch_intent.json').exists())
                self.assertFalse((root/'intercepted.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
