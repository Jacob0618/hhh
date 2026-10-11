import unittest
from jev_mem.fullconv_llm_segmentation_v1 import parse_partition, request_body, response_text, MODEL

def turns():
    return [{'source_turn_id':f'D{i//3+1}:{i%3+1}','session_id':f'session_{i//3+1}',
             'speaker':'A','observation_date':f'2023-01-0{i//3+1}','text':f'turn {i}'} for i in range(6)]

class PartitionTests(unittest.TestCase):
    def test_cross_date_group_and_original_coverage(self):
        x=parse_partition('0 1 E001\n2 2 E002\n3 4 E001\n5 5 E003',turns())
        self.assertEqual(len(x['segments']),4);self.assertEqual(len(x['groups']),3)
        self.assertIsNone(x['groups'][0]['observation_date'])
        self.assertEqual(x['groups'][0]['source_order_indices'],[0,1,3,4])
    def test_bad_ranges_and_schema(self):
        for text in ('0 3 E001\n4 5 E002','0 1 E001\n3 5 E002','0 2 E001\n2 5 E002',
                     '0 2 E001','0 2 E001\n3 6 E002','```\n0 2 E001\n3 5 E002\n```'):
            with self.subTest(text=text),self.assertRaises(ValueError):parse_partition(text,turns())
    def test_no_json_mode_or_qa(self):
        b=request_body(turns());self.assertNotIn('responseMimeType',b['generationConfig'])
        self.assertIn('turn 5',b['contents'][0]['parts'][0]['text'])
    def test_response_finish_and_model(self):
        r={'modelVersion':MODEL,'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'0 2 E001\n3 5 E002'}]}}]}
        self.assertIn('E002',response_text(r))
        r['candidates'][0]['finishReason']='MAX_TOKENS'
        with self.assertRaises(ValueError):response_text(r)

if __name__=='__main__':unittest.main()
