"""Offline terminal evidence for the versioned A-Mem native continuation.

This observer never instantiates an API, reads credentials, edits source runs,
or authorizes an unknown request. It refuses a live/incomplete source.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_five_baselines_format_v2 as previous
from tools import baselines_amem_v1 as amem
from tools import baselines_amem_v2 as amem2
from tools import baselines_amem_v3 as amem3
from tools.baselines_common_v1 import read, sha, save_new, now
from jev_mem.gemini25_ablation_provider_v1 import generation_body, response_text
from jev_mem.sdk20_cached_api import digest

SOURCE = previous.NEW
ANCESTOR = previous.OLD
EXPECTED_MANIFEST = '49f565cf7cc97abc80c3ce9998975804e3fd5d93407f9c00912e17936357e713'


def process_evidence():
    command = ("Get-CimInstance Win32_Process | Where-Object { "
               "$_.Name -in @('python.exe','pythonw.exe') -and "
               "$_.CommandLine -match '(run_five_baselines_gemini25_v1|run_five_baselines_format_v2|run_five_baselines_amem_native_v3)\\.py' "
               "} | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 4 -Compress")
    result = subprocess.run(['powershell', '-NoProfile', '-Command', command],
                            check=True, capture_output=True, text=True)
    found = json.loads(result.stdout) if result.stdout.strip() else []
    found = [found] if isinstance(found, dict) else found
    # Inspect executable identities, not a status file's historical PID.
    if found:
        raise ValueError('A baseline runner process is still live; terminal admission refused')
    return {'checked_utc': now(), 'matching_python_processes': found, 'process_stopped': True}


def inventory(root, allowed_temporaries=None):
    allowed_temporaries = allowed_temporaries or {}
    files = sorted(p for p in root.rglob('*') if p.is_file())
    if any(p.is_symlink() for p in files):
        raise ValueError('Symlink source artifact cannot be admitted')
    temporary = {str(p.relative_to(root)): sha(p) for p in files if p.suffix == '.tmp'}
    if temporary != allowed_temporaries:
        raise ValueError('Unqualified temporary source artifacts')
    return {str(p.relative_to(root)): sha(p) for p in files}


def lines(path):
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        raise ValueError('Incomplete terminal JSONL: ' + str(path))
    return [json.loads(x) for x in data.decode('utf-8').splitlines() if x.strip()]


def certify_failure(ci, entry, call, qualification, qualification_path):
    rec = qualification['certificates'][str(ci)]
    if (qualification.get('status') != 'passed' or rec.get('status') != 'passed'
            or rec['original_raw_sha256'] != entry['raw_sha256']
            or rec['request_hash'] != entry['request_hash']):
        raise ValueError('Independent failure classification binding differs')
    if (rec['context'] != entry['context'] or rec['stage'] != entry['stage']
            or rec['native_call_sha256'] != entry['native_call_sha256']
            or Path(rec['native_call_path']).resolve() != Path(entry['native_call_path']).resolve()
            or sha(Path(rec['review_path'])) != rec['review_sha256']):
        raise ValueError('Independent original context/call/review binding differs')
    kind, retry = rec['kind'], rec['retry_limit']
    if kind not in ('literal_tab_in_json_string', 'native_open_string_action', 'known_complete_invalid_json'):
        raise ValueError('Unreviewed failure class')
    if retry != (1 if kind == 'known_complete_invalid_json' else 0):
        raise ValueError('Invalid single-retry policy')
    raw = read(entry['raw_path'])['response']
    candidates = raw.get('candidates', [])
    if len(candidates) != 1 or candidates[0].get('finishReason') != 'STOP':
        raise ValueError('Failure is not one complete STOP candidate')
    parts = candidates[0].get('content', {}).get('parts', [])
    if len(parts) != 1 or set(parts[0]) != {'text'}:
        raise ValueError('Failure is not one plain text part')
    text = response_text(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        if kind == 'native_open_string_action':
            raise ValueError('Native action exception must already be syntactically valid')
        if kind == 'known_complete_invalid_json' and '\t' in text:
            raise ValueError('Mixed format defects need a separate explicit classification')
    else:
        if kind != 'native_open_string_action' or call['request']['stage'] != 'amem_evolution':
            raise ValueError('Valid syntax cannot receive a format retry')
        if not any(x not in ('strengthen', 'update_neighbor') for x in parsed['actions']):
            raise ValueError('Expected native open-string action case absent')
    if kind != 'known_complete_invalid_json':
        parsed, journal = amem3.decode_generation_json(read(entry['raw_path']), raw_artifact_sha256=entry['raw_sha256'])
        if (kind == 'literal_tab_in_json_string') != bool(journal['changes']):
            raise ValueError('Lossless JSON representation classification differs')
        req = call['request']
        neighbors_text = req['prompt'].split('The nearest neighbors memories:', 1)[-1].split('Based on this information', 1)[0]
        neighbors = [int(x) for x in re.findall(r'memory index:(\d+)', neighbors_text)]
        ordinal = int(Path(entry['native_call_path']).name.split('_')[1])
        parsed, _ = amem2.normalize_evolution(parsed, memory_count=ordinal - 1, neighbor_indices=neighbors)
        amem._validate_json(parsed, req['response_format']['json_schema']['schema'])
    entry['mode'] = 'replace_once' if retry else 'inherit'
    cert = {'status': 'passed', 'kind': kind, 'review_path': str(qualification_path.resolve()),
            'review_sha256': sha(qualification_path), 'original_raw_sha256': entry['raw_sha256'],
            'retry_limit': retry}
    entry['failure_certificate'] = cert
    return cert


def audit(qualification_path):
    process = process_evidence()
    if sha(SOURCE / 'run_manifest.json') != EXPECTED_MANIFEST:
        raise ValueError('Unexpected source manifest')
    previous.install('amem')
    manifest = previous.manifest('amem')
    status = read(SOURCE / 'status.json')
    if status['stage'] != 'needs_review':
        raise ValueError('Expected terminal failed source status')
    outcome = read(SOURCE / 'firstpass_outcome.json')
    rows = outcome['outcomes']
    if len(rows) != 10 or {r['conversation'] for r in rows} != set(range(10)):
        raise ValueError('Ten unique terminal conversations required')
    if any(r['status'] not in ('terminal', 'needs_review') for r in rows):
        raise ValueError('A conversation has not ended')
    if (SOURCE / 'qa_recovery_plan.json').exists() or (SOURCE / 'results/recovery').exists():
        raise ValueError('Source QA recovery already exists; protocol requires re-review')
    qualification = read(qualification_path)
    if qualification.get('status') != 'passed':
        raise ValueError('Independent failure qualifications required')
    allowed_temporary = {}
    for ci, rec in qualification['certificates'].items():
        if rec['kind'] == 'known_local_checkpoint_io':
            relative = f'work/native/c{ci}/checkpoint.json.tmp'
            temp_path = SOURCE / relative
            if rec.get('checkpoint_tmp_sha256') != sha(temp_path):
                raise ValueError('Local IO temporary evidence differs')
            allowed_temporary[str(Path(relative))] = rec['checkpoint_tmp_sha256']
    before = inventory(SOURCE, allowed_temporary)
    original = previous.proof()
    ancestors = inventory(ANCESTOR)
    if ancestors != original['original_inventory']:
        raise ValueError('Original v1 prefix changed')
    raw_audit = previous.audit_requests(SOURCE)
    if raw_audit['status'] != 'passed':
        raise ValueError('Terminal producer/consumer audit failed: ' + repr(raw_audit['errors']))
    api = SOURCE / 'work/api_method'
    events = lines(api / 'logical_events.jsonl')
    event_index = defaultdict(list)
    for event in events:
        event_index[(event['context'].get('conversation'), event['context'].get('phase'),
                     event['stage'], event['request_hash'])].append(event)
    conversations, calls_seen = {}, Counter()
    successful_calls = failure_calls = inherited_qa = 0
    for ci in range(10):
        row = next(r for r in rows if r['conversation'] == ci)
        native = SOURCE / f'work/native/c{ci}'
        checkpoint = read(native / 'checkpoint.json')
        calls = list((native / 'calls').glob('note_*.json'))
        calls.sort(key=lambda p: (int(p.name.split('_')[1]), 0 if p.name.endswith('_analysis.json') else 1))
        completed = checkpoint['completed_notes']
        expected = [(i, s) for i in range(1, completed + 1) for s in ('amem_analysis', 'amem_evolution')]
        failed = row['status'] == 'needs_review'
        qualified = qualification['certificates'].get(str(ci), {})
        local_io = failed and qualified.get('kind') == 'known_local_checkpoint_io'
        if failed:
            if (SOURCE / f'results/c{ci}/build.json').exists() or (SOURCE / f'results/c{ci}/retrieval').exists():
                raise ValueError('Only independently qualified build failures are covered')
            remain = len(calls) - len(expected)
            if remain not in (1, 2) or (local_io and remain != 2):
                raise ValueError('Failure is not exactly the next note prefix')
            expected += [(completed + 1, s) for s in ('amem_analysis', 'amem_evolution')[:remain]]
        if len(calls) != len(expected):
            raise ValueError('Build native call count differs')
        prefix, cert = [], None
        for idx, (path, (ordinal, stage)) in enumerate(zip(calls, expected)):
            if path.name != f'note_{ordinal:06d}_{stage}.json':
                raise ValueError('Native request ordering differs')
            call = read(path); req = call['request']
            if call['request_sha256'] != amem._sha(req) or req['stage'] != stage:
                raise ValueError('Native request signature differs')
            if (req['system'] != 'You must respond with a JSON object.' or req['json_output'] is not True
                    or req['max_output'] != 16384):
                raise ValueError('Original method generation contract differs')
            sig = {'provider': 'gemini', 'model': 'gemini-2.5-flash', 'operation': 'generateContent',
                   'body': generation_body(req['prompt'], system=req['system'], json_output=True, max_output=16384)}
            key = digest(sig); raw_path = api / 'api_cache' / (key + '.json'); raw = read(raw_path)
            context = {'method': 'amem', 'conversation': ci, 'phase': 'build'}
            consumers = event_index[(ci, 'build', stage, key)]
            calls_seen[(ci, stage, key)] += 1
            if (raw['request'] != sig or not consumers or calls_seen[(ci, stage, key)] > len(consumers)
                    or any(e['context'] != context or e['cache_sha256'] != sha(raw_path) for e in consumers)):
                raise ValueError('Native call does not bind an original full-signature consumer')
            entry = {'request_hash': key, 'signature': sig, 'context': context, 'stage': stage,
                     'raw_path': str(raw_path), 'raw_sha256': sha(raw_path),
                     'native_call_path': str(path), 'native_call_sha256': sha(path), 'mode': 'inherit'}
            if call['status'] == 'failed':
                if not failed or local_io or idx != len(calls) - 1:
                    raise ValueError('Failure must be the last consumed prefix response')
                cert = certify_failure(ci, entry, call, qualification, qualification_path)
                failure_calls += 1
            else:
                if call['status'] != 'success':
                    raise ValueError('Unknown native call status')
                parsed, json_journal = amem3.decode_generation_json(raw, raw_artifact_sha256=entry['raw_sha256'])
                if json_journal['changes']:
                    raise ValueError('Previously strict success now requires a representation change')
                if parsed != call['raw_response'] or amem._sha(parsed) != call['raw_response_payload_sha256']:
                    raise ValueError('Successful original response was altered')
                if stage == 'amem_evolution':
                    neighbors_text = req['prompt'].split('The nearest neighbors memories:', 1)[-1].split('Based on this information', 1)[0]
                    neighbors = [int(x) for x in re.findall(r'memory index:(\d+)', neighbors_text)]
                    value, norm = amem2.normalize_evolution(parsed, memory_count=ordinal - 1, neighbor_indices=neighbors)
                    if norm != call['normalization']:
                        raise ValueError('Original decimal derivation changed')
                    if any(x not in ('strengthen', 'update_neighbor') for x in value['actions']):
                        raise ValueError('Prior success unexpectedly had unknown action')
                else:
                    value = parsed
                amem._validate_json(value, req['response_format']['json_schema']['schema'])
                if value != call['parsed_response']:
                    raise ValueError('Successful original parsed value differs')
                successful_calls += 1
            prefix.append(entry)
        if failed:
            if local_io:
                last = prefix[-1]
                failure_path = SOURCE / f'results/c{ci}/failure.json'
                tmp_path = native / 'checkpoint.json.tmp'
                temporary = read(tmp_path)
                if (qualified.get('status') != 'passed' or qualified.get('retry_limit') != 0
                        or qualified.get('original_raw_sha256') != last['raw_sha256']
                        or qualified.get('request_hash') != last['request_hash']
                        or qualified.get('failure_sha256') != sha(failure_path)
                        or sha(Path(qualified['review_path'])) != qualified['review_sha256']
                        or temporary['completed_notes'] != completed + 1
                        or len(temporary['notes']) != completed + 1
                        or temporary['input_sha256'] != checkpoint['input_sha256']
                        or temporary['embeddings_sha256'] != sha(native / 'cache/embeddings.npy')
                        or temporary['memory_sha256'] != amem._sha({k:temporary[k] for k in ('notes','provenance','corpus','evo_cnt')})
                        or read(failure_path)['error']['type'] != 'PermissionError'):
                    raise ValueError('Local persistence failure certificate/evidence differs')
                cert = {'status': 'passed', 'kind': 'known_local_checkpoint_io',
                        'review_path': str(qualification_path.resolve()), 'review_sha256': sha(qualification_path),
                        'original_raw_sha256': last['raw_sha256'], 'retry_limit': 0,
                        'checkpoint_tmp_sha256': sha(tmp_path), 'failure_sha256': sha(failure_path)}
                last['failure_certificate'] = cert
            if cert is None:
                raise ValueError('Build failure missing independent qualification')
            conversations[str(ci)] = {'mode': 'rebuild', 'prefix': prefix, 'failure_certificate': cert}
        else:
            qs = [q for q in manifest['questions'] if q['conversation'] == ci]
            if not checkpoint.get('build_complete') or read(SOURCE / f'results/c{ci}/build.json')['status'] != 'complete':
                raise ValueError('Inherited conversation build incomplete')
            if {r['query_id'] for r in row['results']} != {q['query_id'] for q in qs} or len(row['results']) != len(qs):
                raise ValueError('Inherited query identity inventory differs')
            for q in qs:
                previous.base.replay_qa_leaf(SOURCE, 'amem', q, SOURCE / 'results/firstpass' / q['query_id'])
            inherited_qa += len(qs)
            conversations[str(ci)] = {'mode': 'inherit', 'prefix': [], 'inherited_queries': len(qs)}
    all_build = Counter((e['context']['conversation'], e['stage'], e['request_hash']) for e in events if e['context'].get('phase') == 'build')
    if all_build != calls_seen:
        raise ValueError('Unaccounted build producer/consumer')
    if set(qualification['certificates']) != {ci for ci, rec in conversations.items() if rec['mode'] == 'rebuild'}:
        raise ValueError('Independent failure cohort differs from final terminal cohort')
    after = inventory(SOURCE, allowed_temporary)
    if before != after or inventory(ANCESTOR) != ancestors:
        raise ValueError('Source changed during terminal audit')
    process_evidence()
    return {'schema': 'amem-native-continuation-admission-v3', 'status': 'passed', 'created_utc': now(),
            'source_run': str(SOURCE), 'source_manifest_sha256': EXPECTED_MANIFEST, 'source_inventory': after,
            'ancestor_run': str(ANCESTOR), 'ancestor_manifest_sha256': previous.MANIFEST_SHA, 'ancestor_inventory': ancestors,
            'original_prefix_proof': {'path': str(previous.PROOF), 'sha256': sha(previous.PROOF)},
            'source_terminal': {**process, 'outcomes_sha256': sha(SOURCE / 'firstpass_outcome.json'),
                                'status_sha256': sha(SOURCE / 'status.json')},
            'conversations': conversations, 'raw_audit': raw_audit,
            'successful_build_calls_replayed': successful_calls, 'known_failed_build_calls': failure_calls,
            'inherited_firstpass_queries_replayed': inherited_qa,
            'failure_qualification': {'path': str(qualification_path.resolve()), 'sha256': sha(qualification_path)},
            'preserved_temporary_artifacts': allowed_temporary,
            'paid_calls': 0, 'credential_reads': 0, 'source_modifications': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualification', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Admission output must be new')
    result = audit(args.qualification)
    save_new(args.output, result)
    print(json.dumps({k: result[k] for k in ('status', 'successful_build_calls_replayed', 'known_failed_build_calls',
                                           'inherited_firstpass_queries_replayed', 'paid_calls')}))
