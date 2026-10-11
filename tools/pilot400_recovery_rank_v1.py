"""Stable query-term summation for NEW recovered-bank queries only.

Existing failed QA must use recorded selections instead. The old routing module
is loaded into a separate module namespace; frozen code/globals are not mutated.
"""
import importlib.util
import math
from collections import Counter
from pathlib import Path
from jev_mem.pilot400 import reader_fact

VERSION='rank20-sorted-bm25-query-terms-v1'
SOURCE=Path(__file__).resolve().parents[1]/'jev_mem/routing.py'
spec=importlib.util.spec_from_file_location('jev_mem._recovery_routing_private',SOURCE)
engine=importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

def stable_bm25(query,texts):
    documents=[Counter(engine._tokens(text)) for text in texts]
    if not documents:
        return []
    lengths=[sum(d.values()) for d in documents]
    mean=sum(lengths)/len(documents)
    frequencies=Counter(term for d in documents for term in d)
    terms=sorted(set(engine._tokens(query)))
    scores=[]
    for document,length in zip(documents,lengths):
        score=0.0
        for term in terms:
            frequency=document.get(term,0)
            if not frequency:
                continue
            df=frequencies[term]
            idf=math.log(1+(len(documents)-df+.5)/(df+.5))
            norm=engine.BM25_K1*(1-engine.BM25_B+engine.BM25_B*length/mean)
            score+=idf*frequency*(engine.BM25_K1+1)/(frequency+norm)
        scores.append(score)
    return scores

engine.bm25_scores=stable_bm25

def retrieve(question,facts,vector):
    pool=engine.build_candidate_pool(question,facts,{},vector,top_per_channel=20,cap=60)
    if len(pool)<20:
        raise ValueError('Fewer than 20 unique facts')
    selected=[reader_fact(f) for f in pool[:20]]
    return {'pool':pool,'selected':selected,'trace':{'version':VERSION,'selector':'rank',
        'candidate_count':len(pool),'selected_ids':[f['id'] for f in selected],
        'final_fact_count':20,'max_pool':60,'top_per_channel':20,'new_model_calls':0}}
