"""Publish a bound development/full gate only after offline and semantic review.

No model, query, gold or selection is used. This never changes Writer statuses
or repairs native banks; incomplete or deficient banks are rejected.
"""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_gemini38_writer_v1 as writer
from tools import mem0_regroup_firstpass_review_v2 as idx
from tools.run_sdk20_modules import save_new, now


def publish(run, audit_path, semantic_path):
    run = Path(run).resolve()
    with writer.grouped.writer_process_lock(run):
        m = writer.manifest(run)
        audit = idx.verify(audit_path)
        semantic = idx.verify(semantic_path)
        if (audit.get('schema') != 'gemini38-terminal-source-index-audit-v1' or audit.get('strict_passed') is not True
            or audit.get('exceptions') != [] or audit.get('protocol_sha256') != m['protocol_sha256']
            or audit.get('writer_manifest') != idx.pointer(run/'run_manifest.json')):
            raise ValueError('Strict bound source/index audit is required; no inherited exceptions')
        convs = audit.get('conversations')
        if convs not in ([0,1], list(range(10))):
            raise ValueError('Only development8/full40 admission supported')
        expected = {(ci,a) for ci in convs for a in writer.ARMS}
        if len(audit['banks']) != len(expected) or {(b['conversation'],b['arm']) for b in audit['banks']} != expected:
            raise ValueError('Incomplete structural audit population')
        if (semantic.get('schema') != 'gemini38-writer-source-semantic-review-v1'
            or semantic.get('status') != 'reviewed_with_limitations'
            or semantic.get('protocol_sha256') != m['protocol_sha256']
            or semantic.get('structural_audit') != idx.pointer(audit_path)
            or semantic.get('qa_gold_used') is not False
            or len(semantic.get('banks', [])) != len(expected)
            or {(b['conversation'],b['arm']) for b in semantic['banks']} != expected):
            raise ValueError('Every bank requires a source-only semantic review with explicit limitations')
        for path, hashed in audit['source_file_hashes'].items():
            if idx.sha(path) != hashed:
                raise ValueError('Audited source changed before gate publication')
        indexed = {(b['conversation'],b['arm']):b for b in semantic['banks']}
        rows = []
        for row in audit['banks']:
            ci, arm = row['conversation'],row['arm']
            bank = idx.bound(row['bank'])
            expected_bank = run/f'conv_{ci}'/arm/'bank.json'
            if (row['bank'] != idx.pointer(expected_bank) or bank.get('complete') is not True
                or (bank.get('conversation'),bank.get('arm'),bank.get('protocol_sha256')) != (ci,arm,m['protocol_sha256'])):
                raise ValueError('Bank path or scientific identity differs from this admission')
            review = indexed[(ci,arm)]
            if (row.get('strict_passed') is not True or not row.get('source_generation_persistence_passed')
                or row.get('orphan_fact_ids') or review.get('bank') != row['bank'] or not review.get('notes')
                or review.get('blocking_implementation_error') is not False):
                raise ValueError('Unreviewed semantic/implementation error or native deficit')
            # Source sampling must cover every declared high-risk target plus
            # first and final group. It is not claimed as per-fact human gold.
            required = {x['journal']['path'] for x in row.get('semantic_review_targets', [])}
            required.update(row['journals'][i]['journal']['path'] for i in (0,-1))
            refs = {x['path']:x for x in bank['journal_files']}
            reviewed = review.get('reviewed_journals', [])
            if not required <= {p['path'] for p in reviewed}:
                raise ValueError('Required source/date/empty/history review targets missing')
            for p in reviewed:
                if refs.get(p['path']) != p:
                    raise ValueError('Semantic review cites a journal outside this bank')
                idx.bound(p)
            folder = run/f'conv_{ci}'/arm
            state = idx.read(folder/'write_status.json'); out = idx.verify(folder/'write_outcome.json')
            if (state.get('status') != 'complete' or out.get('status') != 'complete' or state['bank'] != row['bank']
                or out['bank'] != row['bank'] or idx.native_inventory(folder/'native') != row['native_inventory']):
                raise ValueError('Bank/native terminal changed')
            rows.append({'conversation':ci, 'arm':arm, 'bank':row['bank'], 'write_status':idx.pointer(folder/'write_status.json'),
                         'outcome':idx.pointer(folder/'write_outcome.json'), 'native_dir':row['native_dir'],
                         'native_inventory':row['native_inventory'], 'source_date_audit':idx.pointer(audit_path),
                         'index_audit':idx.pointer(audit_path), 'exceptions':[],
                         'checks':{k:True for k in ('source_dates','generation_chain','fact_dense','fact_bm25','entity_dense','entity_links')}})
        path = run/('development_review.json' if convs == [0,1] else 'all_bank_review.json')
        record = {'schema':'gemini38-development-bank-admission-v1' if convs == [0,1] else 'gemini38-complete-bank-admission-v1',
                  'status':'passed', 'created_utc':now(), 'protocol_sha256':m['protocol_sha256'],
                  'writer_manifest':idx.pointer(run/'run_manifest.json'),
                  'banks':[r['bank'] for r in rows] if convs == [0,1] else rows,
                  'source_date_audits':[idx.pointer(audit_path)], 'index_audits':[idx.pointer(audit_path)],
                  'semantic_review':idx.pointer(semantic_path), 'exceptions':[], 'model_calls':0,
                  'semantic_scope':'Assistant source review; all declared sampled targets, not human gold or correctness guarantee.'}
        save_new(path,record)
        return {'gate':str(path), 'banks':len(rows), 'model_calls':0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--semantic-review', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(publish(a.run_dir,a.audit,a.semantic_review)))


if __name__ == '__main__':
    main()
