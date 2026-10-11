"""Offline source/data snapshots only. No Mem0 imports, models, or network calls."""

import ast
import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "mem0"
MEM0_COMMIT = "94c3fe9f238f3dbf29c9ce98643bd71eb13077cd"
BENCH_COMMIT = "4b61c5d31b9c668a12b4f5e78064248a02c82d2b"
DATA_SOURCE = Path(r"C:\Users\86139\Desktop\jev\data\locomo10.json")
DATA_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prompt_dir = ROOT / "prompts"
    data_dir = ROOT / "data"
    prompt_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    manifest = {"status": "preparation_only", "mem0_commit": MEM0_COMMIT,
                "benchmark_commit": BENCH_COMMIT, "snapshots": []}
    specs = [
        (UPSTREAM / "mem0/configs/prompts.py", "mem0_prompts_source.py", {
            "ADDITIVE_EXTRACTION_PROMPT": "active_additive_system.txt",
            "AGENT_CONTEXT_SUFFIX": "conditional_agent_suffix.txt",
        }, MEM0_COMMIT),
        (UPSTREAM / "evaluation/benchmarks/locomo/prompts.py", "locomo_prompts_source.py", {
            "ANSWER_GENERATION_PROMPT": "upstream_locomo_answer_template.txt",
            "JUDGE_SYSTEM_PROMPT": "upstream_locomo_judge_system.txt",
            "_JUDGE_TEMPLATE": "upstream_locomo_judge_template.txt",
        }, BENCH_COMMIT),
    ]
    for source, copy_name, constants, commit in specs:
        shutil.copyfile(source, prompt_dir / copy_name)
        manifest["snapshots"].append({"kind": "verbatim_source", "source": str(source),
                                      "snapshot": copy_name, "sha256": digest(source), "commit": commit})
        tree = ast.parse(source.read_text(encoding="utf-8"))
        found = set()
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in constants:
                    value = ast.literal_eval(node.value)
                    assert isinstance(value, str)
                    path = prompt_dir / constants[target.id]
                    path.write_bytes(value.encode("utf-8"))
                    found.add(target.id)
                    manifest["snapshots"].append({"kind": "exact_string_literal", "symbol": target.id,
                        "source": str(source), "start_line": node.lineno, "end_line": node.end_lineno,
                        "snapshot": path.name, "sha256": digest(path), "commit": commit})
        assert found == set(constants), (source, found, constants)
    # Preserve license alongside upstream-derived material.
    shutil.copyfile(UPSTREAM / "LICENSE", prompt_dir / "LICENSE.mem0")
    bench_licenses = list((UPSTREAM / "evaluation").glob("LICENSE*"))
    for path in bench_licenses:
        if path.is_file():
            shutil.copyfile(path, prompt_dir / ("benchmark_" + path.name))
    (prompt_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    assert digest(DATA_SOURCE) == DATA_SHA256, "Unexpected local dataset bytes"
    target = data_dir / "locomo10.json"
    if target.exists():
        assert digest(target) == DATA_SHA256, "Refusing to overwrite a different project dataset"
    else:
        shutil.copyfile(DATA_SOURCE, target)
    assert digest(target) == DATA_SHA256
    source_data = json.loads(target.read_text(encoding="utf-8"))
    rows = []
    for index, item in enumerate(source_data):
        conv = item["conversation"]
        sessions = [v for k, v in conv.items() if re.fullmatch(r"session_\d+", k) and isinstance(v, list)]
        turns = [turn for session in sessions for turn in session]
        rows.append({"conversation_index": index, "sample_id": item.get("sample_id"),
                     "sessions": len(sessions), "raw_turns": len(turns),
                     "nonempty_text_turns": sum(bool(str(t.get("text", "")).strip()) for t in turns),
                     "included_questions": sum(int(qa["category"]) in [1, 2, 3, 4] for qa in item["qa"])})
    stats = {"status": "preparation_only", "local_copy": str(target), "source_sha256": digest(target),
             "selection_manifest": str(ROOT / "research/locomo_1540_manifest.json"),
             "conversations": rows,
             "totals": {key: sum(row[key] for row in rows) for key in ["sessions", "raw_turns", "nonempty_text_turns", "included_questions"]},
             "note": "Turn counts are static source counts, not calls executed. Benchmark also appends existing image captions/search text."}
    assert stats["totals"]["included_questions"] == 1540
    (data_dir / "local_copy_manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prompt_snapshots": len(manifest["snapshots"]), "data_sha256": digest(target),
                      "totals": stats["totals"], "model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
