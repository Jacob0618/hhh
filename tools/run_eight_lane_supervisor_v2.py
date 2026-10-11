"""Eight-lane supervisor with bounded Windows status replacement retries.

Scheduling only: reuse the frozen v1 job/cohort/adoption policy, never replay an
epoch, and never alter or terminate a QA worker. Use a freshly reconciled handoff.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eight_lane_supervisor as base


def atomic_json(path, value, *, replace=None, sleeper=time.sleep, attempts=40):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    replace = replace or (lambda source, destination: source.replace(destination))
    for attempt in range(attempts):
        try:
            replace(temp, path)
            return
        except PermissionError as error:
            # Readers/antivirus can briefly deny Windows replacement. Preserve
            # the previous complete JSON and retry this same prepared payload.
            if getattr(error, "winerror", None) not in (None, 5, 32, 33) or attempt + 1 == attempts:
                raise
            sleeper(min(0.05 * (attempt + 1), 0.5))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True)
    args = parser.parse_args(argv)
    path = Path(args.handoff).resolve()
    handoff = base.read_json(path)
    mutex = base.RootMutex(handoff["root"])
    try:
        if (path.parent / "global_status.json").exists():
            raise RuntimeError("This epoch is already used; reconcile a new handoff before restarting")
        from process_control_windows import WinProcess
        supervisor = base.Supervisor(path, process_factory=WinProcess, writer=atomic_json)
        supervisor.global_state["scheduler_revision"] = "v2_status_replace_retry"
        supervisor.global_state["resumed_from_epoch"] = handoff.get("resumed_from_epoch")
        return supervisor.run()
    finally:
        mutex.close()


if __name__ == "__main__":
    raise SystemExit(main())
