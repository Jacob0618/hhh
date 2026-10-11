"""Read-only health snapshots for R07 numeric continuation and its frozen QA."""
from collections import Counter
from pathlib import Path
from types import FunctionType
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import snapshot_write_omission_r07 as old


def snapshot_numeric(run):
    r = old.snapshot_omission(run)
    if not r.get('exists'):
        return r
    run = Path(run).resolve()
    r['retrieval_banks'] = {}
    for folder in sorted((run/'retrieval').glob('conv_*/*')):
        if not folder.is_dir() or folder.name not in ('cheap', 'jev'):
            continue
        item = {'selections': 0, 'statuses': {}, 'latest_mtime': 0}
        statuses = Counter()
        for path in (folder/'selections').glob('*.json'):
            try:
                row = old.read(path)
                statuses[row.get('status','missing')] += 1
                item['selections'] += 1
                if path.stat().st_mtime >= item['latest_mtime']:
                    item.update(latest_mtime=path.stat().st_mtime, query_id=row.get('query_id'))
            except Exception as ex:
                r['read_errors'].append({'path':str(path),'error_type':type(ex).__name__})
        item['statuses'] = dict(statuses)
        for name in ('retrieval_outcome.json','retrieval_intent.json'):
            p = folder/name
            if p.exists():
                try: item[name] = old.read(p)
                except Exception as ex: r['read_errors'].append({'path':str(p),'error_type':type(ex).__name__})
        r['retrieval_banks'][str(folder.relative_to(run))] = item
    return r

_scope = {**vars(old), 'snapshot_omission': snapshot_numeric}
main = FunctionType(old.main.__code__, _scope, 'main', old.main.__defaults__, old.main.__closure__)

if __name__ == '__main__': main()
