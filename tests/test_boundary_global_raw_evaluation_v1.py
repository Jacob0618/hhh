"""Independent ledger damage diagnostics: no providers or source rewrites."""
import copy
import json
import unittest
from tools import audit_boundary_global_raw_v1 as a

class GlobalRawReview(unittest.TestCase):
    def test_damage_is_exposed_not_repaired(self):
        row={'request_hash':'bound','status':200}
        data=(json.dumps(row)+'\n\n}\n').encode()
        rows,blank,bad=a.ledger_rows(data)
        self.assertEqual(rows,[(1,row)])
        self.assertEqual(blank,[2]);self.assertEqual(bad,[{'line':3,'length':1,'error_type':'JSONDecodeError'}])
        self.assertEqual(data,(json.dumps(row)+'\n\n}\n').encode())

    def test_valid_json_nonobject_is_still_bad_ledger(self):
        rows,blank,bad=a.ledger_rows(b'[]\nnull\n1\n')
        self.assertEqual(rows,[]);self.assertEqual(blank,[]);self.assertEqual(len(bad),3)

    def test_transport_omitted_cache_flag_is_original_valid_shape(self):
        expected={'context':{'conversation':2},'usage':{'input_tokens':10}}
        row={'context':expected['context'],'provider':'jev','model':a.original.grouping.MODEL,
            'operation':'systemone','stage':'global_segment_same_event','attempt':1,'status':'transport_error'}
        self.assertFalse(a.ledger_mismatch(row,expected))
        row.update(status=200,cache_hit=False,usage=expected['usage'])
        self.assertFalse(a.ledger_mismatch(row,expected))
        for key,value in (('context',{'conversation':3}),('usage',{'input_tokens':11}),('attempt',True),('cache_hit',True)):
            mutated=copy.deepcopy(row);mutated[key]=value
            self.assertTrue(a.ledger_mismatch(mutated,expected),key)

if __name__=='__main__':unittest.main()
