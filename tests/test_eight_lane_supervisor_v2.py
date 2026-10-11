"""No-network checks for recoverable and persistent Windows write failures."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("supervisor_v2", ROOT / "tools/run_eight_lane_supervisor_v2.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_replace_retry_keeps_prior_snapshot(tmp_path):
    path = tmp_path / "status.json"
    path.write_text('{"old": true}', encoding="utf-8")
    calls = []

    def replace(source, destination):
        calls.append(1)
        if len(calls) < 3:
            assert json.loads(destination.read_text()) == {"old": True}
            raise PermissionError("temporary sharing denial")
        source.replace(destination)

    mod.atomic_json(path, {"new": True}, replace=replace, sleeper=lambda _: None)
    assert len(calls) == 3
    assert json.loads(path.read_text()) == {"new": True}


def test_persistent_denial_preserves_old_and_failed_payload(tmp_path):
    path = tmp_path / "status.json"
    path.write_text('{"old": true}', encoding="utf-8")

    def denied(*_):
        raise PermissionError("persistent denial")

    try:
        mod.atomic_json(path, {"new": True}, replace=denied, sleeper=lambda _: None, attempts=2)
    except PermissionError:
        pass
    else:
        raise AssertionError("Persistent denial must propagate")
    assert json.loads(path.read_text()) == {"old": True}
    assert len(list(tmp_path.glob("*.tmp"))) == 1


class StatusWriteTests(unittest.TestCase):
    def test_temporary_denial(self):
        with tempfile.TemporaryDirectory() as directory:
            test_replace_retry_keeps_prior_snapshot(Path(directory))

    def test_persistent_denial(self):
        with tempfile.TemporaryDirectory() as directory:
            test_persistent_denial_preserves_old_and_failed_payload(Path(directory))


if __name__ == "__main__":
    unittest.main()
