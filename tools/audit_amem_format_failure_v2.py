"""Offline, exact original A-Mem prefix admission; never initializes a provider."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_five_baselines_gemini25_v1 as base
from tools import baselines_amem_v1 as amem
from tools.baselines_common_v1 import read, sha, save_new, now
from jev_mem.gemini25_ablation_provider_v1 import generation_body, response_text
from jev_mem.sdk20_cached_api import digest

OLD = base.RUN / 'amem'
MANIFEST_SHA = '6a3814763964c86df747cefdd4dbd3eccbbc9b3329472ad75aa792e4598b5381'

def audit():
    assert sha(OLD / 'run_manifest.json') == MANIFEST_SHA
    base.manifest('amem')
    outcome = read(OLD / 'firstpass_outcome.json')
    assert len(outcome['outcomes']) == 10
    assert {r['conversation'] for r in outcome['outcomes']} == set(range(10))
    assert all(r['status'] == 'needs_review' for r in outcome['outcomes'])
    assert read(OLD / 'status.json')['stage'] == 'needs_review'
    assert not (OLD / 'results/firstpass').exists()
    assert not (OLD / 'results/recovery').exists()
    api = OLD / 'work/api_method'
    ledger = [json.loads(x) for x in (api / 'api_calls.jsonl').read_text(encoding='utf-8').splitlines()]
    events = [json.loads(x) for x in (api / 'logical_events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(ledger) == len(events) == 40
    assert all(r.get('status') == 200 and r.get('attempt') == 1 and not r.get('cache_hit') for r in ledger)
    assert all(not r.get('error_type') and r.get('source') == 'new_request' for r in events)
    records, prefixes = {}, {}
    for ci in range(10):
        native = OLD / 'work/native' / f'c{ci}'
        state = read(native / 'checkpoint.json')
        assert state['completed_notes'] == 1 and len(state['notes']) == 1
        assert not state.get('build_complete') and state['completed_sessions'] == []
        expected_names = [f'note_{n:06d}_{stage}.json' for n in (1, 2) for stage in ('amem_analysis', 'amem_evolution')]
        assert {p.name for p in (native / 'calls').glob('*.json')} == set(expected_names)
        prefixes[str(ci)] = []
        for name in expected_names:
            p = native / 'calls' / name
            call = read(p)
            req = call['request']
            assert call['request_sha256'] == amem._sha(req)
            signature = {'provider': 'gemini', 'model': 'gemini-2.5-flash', 'operation': 'generateContent',
                'body': generation_body(req['prompt'], system=req['system'], json_output=True, max_output=16384)}
            key = digest(signature)
            raw_path, intent_path = api / 'api_cache' / (key + '.json'), api / 'request_intents' / (key + '.json')
            raw, intent = read(raw_path), read(intent_path)
            assert raw['request'] == intent['signature'] == signature
            assert intent['request_hash'] == key and intent['stage'] == req['stage']
            context = {'method': 'amem', 'conversation': ci, 'phase': 'build'}
            assert intent['context'] == context
            physical = [r for r in ledger if r['request_hash'] == key]
            logical = [r for r in events if r['request_hash'] == key]
            assert len(physical) == len(logical) == 1
            assert physical[0]['context'] == logical[0]['context'] == context
            assert physical[0]['stage'] == logical[0]['stage'] == req['stage']
            assert logical[0]['cache_sha256'] == sha(raw_path)
            assert Path(logical[0]['cache_path']).resolve() == raw_path.resolve()
            assert json.loads(response_text(raw['response'])) == call['raw_response']
            assert raw['response'].get('modelVersion') in (None, 'gemini-2.5-flash')
            value = call['raw_response']
            schema = req['response_format']['json_schema']['schema']
            failure = name == 'note_000002_amem_evolution.json'
            if failure:
                assert call['status'] == 'failed' and call['error'] == 'response.suggested_connections[0]: expected integer'
                assert value['suggested_connections'] == ['0']
                repaired = {**value, 'suggested_connections': [0]}
                amem._validate_json(repaired, schema)
                assert {k:v for k,v in repaired.items() if k != 'suggested_connections'} == {k:v for k,v in value.items() if k != 'suggested_connections'}
            else:
                assert call['status'] == 'success' and call['parsed_response'] == value
                amem._validate_json(value, schema)
            assert key not in records
            records[key] = {'raw_path':str(raw_path.resolve()), 'raw_sha256':sha(raw_path),
                'intent_sha256':sha(intent_path), 'context':context, 'stage':req['stage'],
                'native_call_path':str(p.resolve()), 'native_call_sha256':sha(p),
                'representation_failure':failure}
            prefixes[str(ci)].append(key)
    assert {p.stem for p in (api / 'api_cache').glob('*.json')} == set(records)
    assert {p.stem for p in (api / 'request_intents').glob('*.json')} == set(records)
    assert not list((api / 'cache_lineage').glob('*.json'))
    assert not list(OLD.rglob('*.tmp'))
    inventory = {str(p.relative_to(OLD)):sha(p) for p in sorted(OLD.rglob('*')) if p.is_file()}
    return {'schema':'amem-format-prefix-admission-v2', 'status':'passed', 'created_utc':now(),
        'original_run':str(OLD), 'manifest_sha256':MANIFEST_SHA, 'records':records,
        'prefixes':prefixes, 'original_inventory':inventory, 'physical_http200':40,
        'known_representation_failures':10, 'unknown':0, 'qa_calls':0,
        'scope':'Exact original four-generation prefix per conversation; no partial memory copy; original raw immutable.'}

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    value=audit();save_new(args.output,value)
    print(json.dumps({'status':value['status'],'physical_http200':40,'failed_representations':10,'output':str(args.output)}))
