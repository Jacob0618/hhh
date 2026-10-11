"""Offline invariants: never initialize an API client or read credentials."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_mem.global_segment_grouping import (
    prepare_candidates, pair_key, pair_state, judge_pair, complete_link_groups,
)
from jev_mem.sparse_topic_pipeline import _group_record


def source(n=4):
    raw=[{"source_turn_id":f"D{i+1}:1","session_id":f"session_{i+1}","speaker":"speaker",
          "observation_date":f"2026-09-{i+1:02d}","text":f"complete original turn {i}",
          "role":"user","caption":f"original caption {i}"} for i in range(n)]
    segments=[]
    for i in range(n):
        s=_group_record(raw,[i],f"segment_{i:04d}")
        s["segment_id"]=s["group_id"]
        segments.append(s)
    vectors=[[1.]+[0.]*767 for _ in raw]
    return raw,segments,vectors


def plan(n=4,k=1):
    _,segments,vectors=source(n)
    return prepare_candidates(0,segments,vectors,top_k=k)


class Judge:
    def __init__(self,value=.9): self.value=value;self.seen=[]
    def score_many(self,state,criteria,stage):
        self.seen.append(json.loads(state))
        return {k:self.value for k in criteria}


class GlobalGroupingTests(unittest.TestCase):
    def test_symmetric_global_knn_union_no_pair_duplicates(self):
        p=plan(4,1)
        # All vectors tie: future segments select segment0. Symmetric union retains
        # their proposals even though segment0's own sole nearest neighbor is1.
        self.assertEqual({e['pair_id'] for e in p['candidate_pairs']},
                         {pair_key(0,1),pair_key(0,2),pair_key(0,3)})
        self.assertEqual(p['cosine_matrix'],[[1.]*4 for _ in range(4)])
        self.assertEqual(p['coverage']['segments_with_initial_candidates'],4)
        self.assertEqual(p['cross_conversation_comparisons'],0)

    def test_full_turn_fields_are_visible_and_pair_symmetric(self):
        p=plan(3)
        self.assertEqual(pair_state(p,0,2),pair_state(p,2,0))
        j=Judge();row=judge_pair(p,0,2,j)
        self.assertEqual(j.seen[0]['segments'][0]['turns'],p['segments'][0]['turns'])
        self.assertEqual(j.seen[0]['segments'][1]['turns'],p['segments'][2]['turns'])
        self.assertTrue(row['full_turns_visible'])
        self.assertEqual(row['truncated_turns'],0)

    def test_topk_all_when_small_and_one_segment_zero_calls(self):
        p=plan(1,5)
        self.assertEqual(p['candidate_pairs'],[])
        g=complete_link_groups(p,{})
        self.assertEqual(len(g['groups']),1)
        p=plan(3,5)
        self.assertEqual(len(p['candidate_pairs']),3)

    def test_complete_link_prevents_positive_chain_merge(self):
        p=plan(3,5)
        scores={pair_key(0,1):.95,pair_key(1,2):.9,pair_key(0,2):.2}
        g=complete_link_groups(p,scores)
        self.assertEqual([x['segment_ids'] for x in g['groups']],
                         [['segment_0000','segment_0001'],['segment_0002']])
        self.assertEqual(g['trace']['merge_decisions'][-1]['action'],'keep_separate_observed_negative')

    def test_missing_pair_budget_is_unknown_not_negative(self):
        p=plan(3,1)
        scores={e['pair_id']:.9 for e in p['candidate_pairs']}
        g=complete_link_groups(p,scores,supplemental_budget=0)
        self.assertEqual(len(g['groups']),2)
        self.assertEqual(g['trace']['merge_decisions'][-1]['action'],'keep_separate_unknown_budget')
        self.assertNotIn(pair_key(1,2),g['trace']['all_observed_pair_scores'])

    def test_complete_link_supplement_once_full_dates_order_partition(self):
        p=plan(3,1);calls=[]
        def extra(i,j): calls.append((i,j));return .8
        g=complete_link_groups(p,{e['pair_id']:.9 for e in p['candidate_pairs']},supplemental=extra)
        self.assertEqual(calls,[(1,2)])
        self.assertEqual(len(g['groups']),1)
        merged=g['groups'][0]
        self.assertIsNone(merged['observation_date'])
        self.assertTrue(merged['cross_date']);self.assertTrue(merged['cross_session'])
        self.assertEqual(merged['turns'],[s['turns'][0] for s in p['segments']])
        self.assertEqual(len(merged['complete_link_pair_ids']),3)
        self.assertFalse(g['trace']['writer_capacity_checked'])

    def test_pair_failure_not_zero_and_invalid_scores_rejected(self):
        p=plan(3,1)
        def fails(i,j): raise TimeoutError('unknown')
        with self.assertRaises(TimeoutError):
            complete_link_groups(p,{e['pair_id']:.9 for e in p['candidate_pairs']},supplemental=fails)
        for value in (None,True,float('nan'),float('inf'),-1,1.1):
            with self.assertRaises(ValueError):judge_pair(p,0,1,Judge(value))

    def test_initial_unknown_cannot_be_imputed(self):
        p=plan(3,1)
        with self.assertRaises(ValueError):complete_link_groups(p,{})

    def test_qa_fields_and_misaligned_vectors_rejected(self):
        _,segments,vectors=source(3)
        segments[0]['turns'][0]['gold']='forbidden'
        with self.assertRaises(ValueError):prepare_candidates(0,segments,vectors)
        _,segments,vectors=source(3)
        with self.assertRaises(ValueError):prepare_candidates(0,segments,vectors[:-1])

    def test_input_guard_never_truncates_and_inputs_not_mutated(self):
        _,segments,vectors=source(3);before=deepcopy(segments)
        prepare_candidates(0,segments,vectors)
        self.assertEqual(before,segments)
        with self.assertRaises(ValueError):prepare_candidates(0,segments,vectors,max_pair_characters=20)


if __name__=='__main__':unittest.main()
