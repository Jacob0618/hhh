"""Bounded, synthetic native-topic replay checks; no production admission.

The child name is not a controller name. A timeout is a test failure, never
permission to continue. All native files are made by the pinned tiny FakeAPI
fixture in a new temporary research directory.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def bounded_child():
    print('child imports', file=sys.stderr, flush=True)
    from tools import audit_hypermem_continuation_v4 as audit
    from tests.test_baselines_hypermem_v1 import FakeAPI, conversation
    from jev_mem import providers
    import requests
    started = time.monotonic()
    print('imports complete', file=sys.stderr, flush=True)
    errors = {}
    with tempfile.TemporaryDirectory(prefix='v4-topic-independent-', dir=ROOT/'research') as tmp:
        native = Path(tmp)/'native'
        def forbidden(*args, **kwargs):
            raise AssertionError('Synthetic audit may not use provider or credentials')
        with patch.object(providers, 'load_credentials', side_effect=forbidden), \
                patch.object(requests.sessions.Session, 'request', side_effect=forbidden), \
                redirect_stdout(io.StringIO()):
            print('tiny build', file=sys.stderr, flush=True)
            audit.adapter.build(conversation(), native, FakeAPI())
            print('tiny build complete', file=sys.stderr, flush=True)
            paths = sorted((native/'calls').glob('topic_*.json'))
            assert len(paths) == 2, [p.name for p in paths]
            originals = {p: p.read_bytes() for p in paths}
            pristine = {}
            for p in paths:
                n = audit.read(p)
                sig = audit.signature(n['request'])
                raw = Path(tmp)/(p.stem + '_synthetic_sdk.json')
                rv = {'request': sig, 'response': {'candidates': [{'finishReason': 'STOP',
                    'content': {'parts': [{'text': json.dumps(n['raw_response'], ensure_ascii=False)}]}}]}}
                raw.write_text(json.dumps(rv, ensure_ascii=False), encoding='utf-8')
                pristine[audit.digest(sig)] = {'signature': sig, 'raw': audit.binding(raw), 'raw_value': rv}
            proof = {'compatibility_cases': {}, 'replacement_cases': {}, 'role_cases': [],
                'producer_registry': {}, 'hash_environment': {}}
            # Startup identity has its own independent tests. This fixture tests
            # real pinned algorithm/error propagation only, not that admission.
            with patch.object(audit.admission, 'check_hash_environment', return_value=None):
                print('positive replay', file=sys.stderr, flush=True)
                positive = audit.replay_topic_callers(native, pristine, proof)
                print('positive replay complete', file=sys.stderr, flush=True)
                assert positive['status'] == 'passed_complete'
                assert len(positive['calls']) == 2
                roles = [v for v in positive['calls'].values() if v['stage'] == 'hypermem_episode_roles']
                assert len(roles) == 1 and roles[0]['local_topic'] and roles[0]['local_episodes']
                key = audit.digest(audit.signature(audit.read(paths[0])['request']))
                for case in ('prompt', 'signature', 'missing_producer', 'bad_raw_json', 'missing_call', 'extra_call'):
                    print('case ' + case, file=sys.stderr, flush=True)
                    records = deepcopy(pristine)
                    for p, data in originals.items():
                        p.write_bytes(data)
                    extra = native/'calls/topic_000003_hypermem_episode_roles.json'
                    if extra.exists():
                        extra.unlink()
                    if case == 'prompt':
                        value = audit.read(paths[0]); value['request']['prompt'] += '\nchanged'
                        value['request_sha256'] = audit.adapter._runtime._sha(value['request'])
                        paths[0].write_text(json.dumps(value), encoding='utf-8')
                    elif case == 'signature':
                        records[key]['signature']['body']['contents'][0]['parts'][0]['text'] += '\nchanged'
                    elif case == 'missing_producer':
                        del records[key]
                    elif case == 'bad_raw_json':
                        records[key]['raw_value']['response']['candidates'][0]['content']['parts'][0]['text'] = '{bad json'
                    elif case == 'missing_call':
                        paths[-1].unlink()
                    elif case == 'extra_call':
                        extra.write_bytes(originals[paths[-1]])
                    tick = time.monotonic()
                    try:
                        audit.replay_topic_callers(native, records, proof)
                    except (Exception, audit.adapter._runtime.HyperMemError) as exc:
                        errors[case] = {'type': type(exc).__name__, 'message': str(exc),
                            'elapsed_seconds': round(time.monotonic()-tick, 6)}
                    else:
                        raise AssertionError('Malformed replay was accepted: ' + case)
            audit.adapter._runtime._INDEXES.pop(str(native), None)
    print(json.dumps({'status': 'passed', 'scope': 'tiny_synthetic_native_replay_bounded_errors',
        'positive_topic_calls': 2, 'negative_cases': errors, 'model_calls': 0,
        'elapsed_seconds': round(time.monotonic()-started, 6)}))


class BoundedTopicReplay(unittest.TestCase):
    def test_true_native_pipeline_and_six_corruptions_finish_before_timeout(self):
        env = dict(os.environ, PYTHONHASHSEED='780', PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--bounded-child'],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['positive_topic_calls'], 2)
        self.assertEqual(set(report['negative_cases']),
            {'prompt', 'signature', 'missing_producer', 'bad_raw_json', 'missing_call', 'extra_call'})
        self.assertEqual(report['model_calls'], 0)
        print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    if '--bounded-child' in sys.argv:
        bounded_child()
    else:
        unittest.main(verbosity=2)
