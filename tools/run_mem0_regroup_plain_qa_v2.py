"""Source-admission v2 only; execute the frozen v1 retrieval and plain-text QA.

Each spawned process installs the same narrow source binding. No frozen source
file is edited. Original failures and reviewed exports/recoveries remain distinct.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_mem0_regroup_plain_qa_v1 as frozen
from tools import mem0_regroup_plain_qa_sources_v2 as sources

SOURCE_PROTOCOL = "reviewed-native-export-and-eof-recovery-qa-sources-v2"
_BASE_WORKER = frozen.worker
_ACTIVE = False
EXTRA_CODE = ("tools/run_mem0_regroup_plain_qa_v2.py", "tools/mem0_regroup_plain_qa_sources_v2.py", *sources.VALIDATOR_CODE)


@contextmanager
def source_binding():
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError("QA source binding cannot be nested")
    original = frozen.sources, frozen.worker, frozen.CODE
    _ACTIVE = True
    frozen.sources, frozen.worker = sources, worker
    frozen.CODE = tuple(dict.fromkeys((*original[2], *EXTRA_CODE)))
    try:
        yield
    finally:
        frozen.sources, frozen.worker, frozen.CODE = original
        _ACTIVE = False


def prepare(run_dir, gate_path, config_path=sources.CONFIG):
    with source_binding():
        return frozen.prepare(run_dir, gate_path, config_path)


def manifest(run_dir, *, recheck_sources=False, conversation=None):
    with source_binding():
        value = frozen.manifest(run_dir, recheck_sources=recheck_sources, conversation=conversation)
    if any(name not in value["code_sha256"] for name in EXTRA_CODE):
        raise ValueError("Manifest lacks the explicit v2 source adapter binding")
    gate = sources.verify(value["gate"]["path"])
    if gate.get("schema") != sources.GATE_SCHEMA:
        raise ValueError("QA source admission is not the reviewed v2 policy")
    return value


def worker(run_dir, ci):
    # This named top-level function is what ProcessPoolExecutor pickles on
    # Windows. Child imports do not inherit a parent's runtime module binding.
    manifest(run_dir, conversation=ci)
    with source_binding():
        return _BASE_WORKER(run_dir, ci)


def run(run_dir):
    manifest(run_dir, recheck_sources=True)
    with source_binding():
        return frozen.run(run_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "check", "run"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.gate is None:
            parser.error("prepare requires --gate for the reviewed20-bank source policy")
        value = prepare(args.run_dir, args.gate)
    elif args.phase == "check":
        value = {"protocol_sha256": manifest(args.run_dir, recheck_sources=True)["protocol_sha256"],
                 "source_protocol": SOURCE_PROTOCOL, "model_calls": 0}
    else:
        value = run(args.run_dir)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
