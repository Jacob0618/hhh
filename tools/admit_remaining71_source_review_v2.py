"""Materialize a source-reviewed subset without changing generated claims.

All generated banks/responses remain immutable. Only the active fact ID set is
filtered, with every excluded object and assistant-review reason retained.
"""
from __future__ import annotations
import copy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.pilot400_members_only_writer_v2 import OUT,VERSION,read,sha,save_once
from tools.audit_pilot400_members_only_v2 import audit
from jev_mem.pilot400 import fingerprint,COHORT_SHA256
from jev_mem.data import SOURCE_HASH

REVIEW=ROOT/'research/pilot400_remaining71_semantic_review_v2_2026-09-28.json'

def filter_bank(bank,review,review_path,original_path):
    rejected=review['excluded_fact_ids']; accepted=review['accepted_fact_ids']
    if (len(rejected)!=len(set(rejected)) or len(accepted)!=len(set(accepted))
            or set(rejected)&set(accepted)):
        raise ValueError('Review accepted/rejected identity collision')
    reviews={r['fact_id']:r for r in review['fact_reviews']}
    if len(reviews)!=len(review['fact_reviews']) or set(reviews)!=set(accepted)|set(rejected):
        raise ValueError('Review does not enumerate every new candidate exactly once')
    if (any(reviews[i]['status']!='exclude' for i in rejected)
            or any(reviews[i]['status']!='accept_with_stated_source_limits' for i in accepted)):
        raise ValueError('Review decisions conflict with identity lists')
    fresh={f['id'] for g in bank['groups'] if g['kind']==VERSION for f in g['facts']}
    old={f['id'] for g in bank['groups'] if g['kind']!=VERSION for f in g['facts']}
    if old&set(rejected) or not fresh<=set(reviews):
        raise ValueError('Review changes an inherited record or omits a new candidate')
    result=copy.deepcopy(bank)
    result.update(schema='pilot400-source-reviewed-active-bank-v2',
        generated_bank_path=str(original_path),generated_bank_sha256=sha(original_path),
        source_review_path=str(review_path),source_review_sha256=sha(review_path),
        groups_semantics='Immutable generated group records, including quarantined facts; active retrieval uses facts only.',
        facts=[copy.deepcopy(f) for f in bank['facts'] if f['id'] not in set(rejected)],
        source_review_quarantine=[{'fact':copy.deepcopy(f),'review':reviews[f['id']]}
            for f in bank['facts'] if f['id'] in set(rejected)],
        independent_assistant_review=True,requires_independent_review=False,
        review_is_not_human_gold=True)
    if [f for f in bank['facts'] if f['id'] not in set(rejected)]!=result['facts']:
        raise ValueError('Active facts changed beyond reviewed exclusion')
    return result

def main():
    review=read(REVIEW)
    for entry in review['source_bindings']:
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Semantic review source changed')
    provenance=audit(OUT)
    if provenance['status']!='passed':raise ValueError('Generation audit did not pass')
    original_banks={ci:read(OUT/'banks'/('conv_'+str(ci))/'bank_overlay.json') for ci in (1,7)}
    new_ids={f['id'] for b in original_banks.values() for g in b['groups'] if g['kind']==VERSION for f in g['facts']}
    if set(review['accepted_fact_ids'])|set(review['excluded_fact_ids'])!=new_ids:
        raise ValueError('Review population differs from all newly generated facts')
    if len(review['excluded_fact_ids'])!=7 or len(review['accepted_fact_ids'])!=106:
        raise ValueError('Unexpected completed review population')
    banks={}
    for ci,bank in original_banks.items():
        original=OUT/'banks'/('conv_'+str(ci))/'bank_overlay.json'
        output=original.with_name('bank_overlay_admitted.json')
        filtered=filter_bank(bank,review,REVIEW,original)
        save_once(output,filtered)
        banks[str(ci)]={'admitted':True,'bank_path':output.relative_to(OUT).as_posix(),
            'bank_sha256':sha(output),'facts':len(filtered['facts']),
            'generated_bank_sha256':sha(original),'quarantined':len(filtered['source_review_quarantine'])}
    gate={'schema':'pilot400-recovery-v2-bank-admission','source_dataset_sha256':SOURCE_HASH,
        'cohort_sha256':COHORT_SHA256,'review_path':REVIEW.relative_to(ROOT).as_posix(),
        'review_sha256':sha(REVIEW),'banks':banks,
        'review_scope':'Independent assistant source-only review of all113 new facts; inherited records unchanged.',
        'not_human_gold':True,'all_generated_source_records_preserved':True,
        'admission_tool_sha256':sha(__file__)}
    save_once(OUT/'bank_admission.json',gate)
    print(banks)

if __name__=='__main__':main()
