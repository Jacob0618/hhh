"""Independent recovery overlay; plan/cache-only never initialize credentials."""
from __future__ import annotations
import argparse
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_mem import recovery_v1 as recovery


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "cache-only", "execute"))
    parser.add_argument("--seal", type=Path, default=ROOT / "research/main_batch_firstpass_seal_v1/seal.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/technical_recovery_v1")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args(argv)
    store = recovery.SealStore(args.seal)
    transport = None
    if args.mode == "execute":
        if args.review is None:
            parser.error("execute requires --review binding exact seal, plan, and recovery code")
        recovery.verify_review(args.review, store, args.output)
        from tools.check_experiment_health import inventory
        running = inventory()
        # inventory() returns a process list; any old QA/scheduler is a blocker.
        if running:
            raise RuntimeError("Original experiment process still present; recovery refused")
        api = None
        init_lock = threading.Lock()

        def transport(signature, stage, proof):
            # Lazy initialization occurs only after namespace validation and
            # the durable attempt marker. Offline modes never reach this.
            nonlocal api
            with init_lock:
                if api is None:
                    from jev_mem.providers import ModelAPI
                    api = ModelAPI(args.output / "transport", retries=0,
                                   context={"phase": "technical_recovery_v1"})
            client = api.scoped(**proof["consumer"], recovery_kind=proof["kind"])
            observed = []
            original_log = client._log
            def audit_log(row):
                observed.append(dict(row))
                original_log(row)
            client._log = audit_log
            try:
                return client.request(signature["provider"], signature["model"], signature["body"],
                                      operation=signature["operation"], stage=stage, cache=False)
            except Exception as error:
                # An explicit non-2xx reply is known failure. A transport or
                # post-send local exception remains conservatively unknown.
                if observed and isinstance(observed[-1].get("status"), int) and observed[-1]["status"] >= 400:
                    raise recovery.RecoveryStop("failed", str(error)) from error
                raise

    result = recovery.run(store, args.output, args.mode, args.workers, transport=transport)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
