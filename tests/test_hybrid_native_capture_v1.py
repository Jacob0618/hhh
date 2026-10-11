"""Network-free native scoring, capture restoration and same-pool ablation tests."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.hybrid_native_capture_v1 import capture_hybrid_native, build_hybrid_pool, select_hybrid_arms
from jev_mem.routing import TYPE_KEYS
from jev_mem.sdk20_type_adapter import _digest
from jev_mem.selection import COARSE_CRITERIA, FINE_CRITERIA

spec = importlib.util.spec_from_file_location("offline_hybrid_upstream_scoring", ROOT / ".venv/Lib/site-packages/mem0/utils/scoring.py")
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)


def row(i, score):
    return SimpleNamespace(id=f"id{i}", score=score, payload={"user_id": "u", "data": f"fact {i}"})


class Store:
    def __init__(self, dense, lexical):
        self.dense, self.lexical, self.calls = dense, lexical, []

    def search(self, query, vectors, top_k=5, filters=None):
        self.calls.append(("dense", top_k, deepcopy(filters)))
        return self.dense

    def keyword_search(self, query, top_k=5, filters=None):
        self.calls.append(("keyword", top_k, deepcopy(filters)))
        if isinstance(self.lexical, Exception):
            raise self.lexical
        return self.lexical


class Memory:
    def __init__(self, lexical=None):
        self.facts = [{"id": f"id{i}", "text": f"fact {i}", "observation_date": "2023-05-08"} for i in range(160)]
        dense = [row(i, .99 - i*.004) for i in range(80)]
        self.vector_store = Store(dense, [row(i, 160-i) for i in range(80, 160)] if lexical is None else lexical)
        self.module = SimpleNamespace(score_and_rank=scoring.score_and_rank,
                                      _payload_is_expired=lambda p: bool(p.get("expired")))
        self.public = None

    def search(self, query, *, top_k=20, filters=None, threshold=.1, rerank=False, explain=False, reference_date=None, show_expired=False):
        dense = self.vector_store.search(query=query.strip(), vectors=[.1,.2], top_k=max(top_k*4,60), filters=filters)
        lexical = self.vector_store.keyword_search(query="lemma " + query, top_k=max(top_k*4,60), filters=filters)
        raw = [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in dense]
        bm25 = {str(r.id): r.score/1000 for r in lexical or [] if r.score > 0}
        selected = self.module.score_and_rank(raw, bm25, {}, threshold, top_k, explain)
        self.public = {"results": selected}
        return self.public


def capture(memory):
    with capture_hybrid_native(memory, query="test query", filters={"user_id":"u"}, bank_facts=memory.facts,
                               _scorer_module=memory.module) as captured:
        result = memory.search("test query", top_k=20, filters={"user_id":"u"}, rerank=False)
        assert result is memory.public
    expected = [{"id": r["id"], "score": r["score"], "text": memory.facts[int(r["id"][2:])]["text"],
                 "observation_date":"2023-05-08"} for r in result["results"]]
    return captured, expected


def pool(memory):
    captured, expected = capture(memory)
    return build_hybrid_pool(captured, memory.facts, bank_sha256="a"*64, expected_native20=expected,
                             expected_native_capture=captured["native_capture"])


def labels(facts):
    return {f["id"]: {"coarse_scores": {k: .3 for k in COARSE_CRITERIA},
                       "fine_scores": {k: .3 for k in FINE_CRITERIA}} for f in facts}


class HybridTests(unittest.TestCase):
    def test_one_native_search_preserves_objects_and_restores_class_methods(self):
        m = Memory()
        originals = m.module.score_and_rank
        before = deepcopy(m.facts)
        captured, _ = capture(m)
        self.assertIs(m.module.score_and_rank, originals)
        self.assertNotIn("search", vars(m))
        self.assertNotIn("search", vars(m.vector_store))
        self.assertNotIn("keyword_search", vars(m.vector_store))
        self.assertEqual(m.vector_store.calls, [("dense",80,{"user_id":"u"}),("keyword",80,{"user_id":"u"})])
        self.assertEqual(captured["native_capture"]["base20_ids"], [f"id{i}" for i in range(20)])
        self.assertEqual(m.facts, before)
        self.assertEqual(captured["lexical_legal_ids"], [f"id{i}" for i in range(80,100)])

    def test_equal_rrf_retains_both_front20_and_reports_physical160_logical100(self):
        p = pool(Memory())
        self.assertEqual(len(p["pool_ids"]),80)
        self.assertTrue(set(f"id{i}" for i in list(range(20))+list(range(80,100))).issubset(p["pool_ids"]))
        self.assertEqual(p["budget"]["native_backend_returned"]+p["budget"]["keyword_backend_returned"],160)
        self.assertEqual(p["budget"]["fusion_retained_slots"],100)
        self.assertEqual(p["budget"]["additional_backend_queries"],0)
        self.assertEqual(p["pool_ids"][:4],["id0","id80","id1","id81"])

    def test_overlap_is_id_only_and_shared_items_have_two_contributions(self):
        m=Memory([row(i,50-i) for i in range(20)])
        p=pool(m)
        self.assertEqual(p["budget"]["union_unique_ids"],80)
        self.assertEqual(p["ranking"][0]["fused_score_fraction"],"2/61")
        self.assertEqual(p["pool_ids"][:20], [f"id{i}" for i in range(20)])

    def test_empty_channel_is_supported_but_unavailable_and_error_are_distinct(self):
        m=Memory([])
        p=pool(m)
        self.assertEqual(p["pool_ids"], [f"id{i}" for i in range(80)])
        for lexical, status in ((None,"unavailable"),(OSError("backend"),"error")):
            m=Memory([]);m.vector_store.lexical=lexical
            with self.assertRaises((RuntimeError,OSError)):
                with capture_hybrid_native(m,query="test query",filters={"user_id":"u"},bank_facts=m.facts,_scorer_module=m.module) as c:
                    m.search("test query",filters={"user_id":"u"})
            self.assertEqual(c["status"],"failed")
            self.assertEqual(c["channels"]["keyword"]["status"],status)
            self.assertNotIn("search",vars(m))
            self.assertNotIn("keyword_search",vars(m.vector_store))

    def test_invalid_scope_or_payload_text_fails_without_extra_query(self):
        for key,value in (("user_id","other"),("data","corrupted")):
            m=Memory();m.vector_store.lexical[0].payload[key]=value
            with self.assertRaises(ValueError): capture(m)
            self.assertEqual(len(m.vector_store.calls),2)
            self.assertNotIn("search",vars(m))

    def test_nonpositive_and_expired_keyword_are_filtered_without_backfill(self):
        rows=[row(80,0),row(81,20),row(82,-1),row(83,18)]
        rows[1].payload["expired"]=True
        m=Memory(rows)
        c,_=capture(m)
        self.assertEqual(c["lexical_legal_ids"],["id83"])
        self.assertEqual([r["reason"] for r in c["lexical_excluded"]],["nonpositive","expired","nonpositive"])

    def test_historical_replay_and_hash_binding_fail_closed(self):
        m=Memory();c,expected=capture(m)
        expected[0]["observation_date"]="2030-01-01"
        with self.assertRaises(ValueError): build_hybrid_pool(c,m.facts,bank_sha256="a"*64,expected_native20=expected,expected_native_capture=c["native_capture"])
        c["query"]="drift"
        with self.assertRaises(ValueError): build_hybrid_pool(c,m.facts,bank_sha256="a"*64,expected_native20=expected,expected_native_capture=c["native_capture"])

    def test_complete_native_tail_drift_is_rejected_even_when_front20_match(self):
        m=Memory();c,expected=capture(m)
        old=deepcopy(c["native_capture"])
        old["eligible_ids"][70],old["eligible_ids"][71]=old["eligible_ids"][71],old["eligible_ids"][70]
        old["eligible"][70],old["eligible"][71]=old["eligible"][71],old["eligible"][70]
        old["capture_sha256"]=_digest({k:v for k,v in old.items() if k!="capture_sha256"})
        with self.assertRaises(ValueError):
            build_hybrid_pool(c,m.facts,bank_sha256="a"*64,expected_native20=expected,expected_native_capture=old)

    def test_original_query_whitespace_is_retained_while_backend_uses_native_trim(self):
        m=Memory()
        with capture_hybrid_native(m,query=" test query ",filters={"user_id":"u"},bank_facts=m.facts,_scorer_module=m.module) as c:
            m.search(" test query ",filters={"user_id":"u"})
        self.assertEqual(c["query"]," test query ")
        self.assertEqual(c["channels"]["dense"]["query"],"test query")

    def test_native_short_pool_is_not_padded(self):
        m=Memory([]);m.vector_store.dense=m.vector_store.dense[:25]
        p=pool(m)
        self.assertEqual(len(p["pool_ids"]),25)
        self.assertEqual(p["budget"]["native_backend_returned"],25)
        arms=select_hybrid_arms(p,labels(m.facts),{k:.5 for k in TYPE_KEYS})
        self.assertTrue(all(len(a["selected_ids"])==20 for a in arms.values()))

    def test_search_drift_and_second_call_are_rejected_before_backend_work(self):
        for kwargs in ({"rerank":True},{"threshold":.2},{"top_k":60}):
            m=Memory()
            with self.assertRaises(ValueError):
                with capture_hybrid_native(m,query="test query",filters={"user_id":"u"},bank_facts=m.facts,_scorer_module=m.module):
                    m.search("test query",filters={"user_id":"u"},**kwargs)
            self.assertEqual(m.vector_store.calls,[])
        m=Memory()
        with self.assertRaises(RuntimeError):
            with capture_hybrid_native(m,query="test query",filters={"user_id":"u"},bank_facts=m.facts,_scorer_module=m.module):
                m.search("test query",filters={"user_id":"u"})
                m.search("test query",filters={"user_id":"u"})
        self.assertEqual(len(m.vector_store.calls),2)

    def test_same_pool_scores_retained_and_only_relation_weight_is_masked(self):
        m=Memory();p=pool(m);t=labels(m.facts)
        for fid in p["pool_ids"]:
            t[fid]["fine_scores"]["explicit_relation"]=(int(fid[2:])%7)/7
        q={key:0.0 for key in TYPE_KEYS};q["explicit_relation"]=1.0
        before=deepcopy((p,t,q))
        arms=select_hybrid_arms(p,t,q)
        h0,h1,h2=[arms[k] for k in ("hybrid_rank","hybrid_types","hybrid_types_no_relation")]
        self.assertEqual(h2["selected_ids"],h0["selected_ids"])
        self.assertNotEqual(h1["selected_ids"],h0["selected_ids"])
        self.assertEqual((p,t,q),before)
        for arm in arms.values():
            self.assertEqual(arm["trace"]["pool_sha256"],p["pool_sha256"])
            self.assertEqual(arm["trace"]["pool_ids"],p["pool_ids"])
            self.assertEqual(len(set(arm["selected_ids"])),20)
            self.assertEqual(arm["trace"]["fact_scores"][p["pool_ids"][0]]["explicit_relation"],t[p["pool_ids"][0]]["fine_scores"]["explicit_relation"])
        self.assertEqual(h1["trace"]["original_query_scores"],h2["trace"]["original_query_scores"])
        self.assertEqual(h2["trace"]["fallback_reason"],"zero_query_weights")

    def test_missing_labels_and_mutated_pool_are_rejected(self):
        m=Memory();p=pool(m);t=labels(m.facts);q={key:.5 for key in TYPE_KEYS}
        del t[p["pool_ids"][0]]["fine_scores"]["explicit_relation"]
        with self.assertRaises(ValueError): select_hybrid_arms(p,t,q)
        p["pool_ids"][0]="bad"
        with self.assertRaises(ValueError): select_hybrid_arms(p,labels(m.facts),q)


if __name__=="__main__": unittest.main()
