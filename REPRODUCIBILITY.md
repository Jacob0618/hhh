# Reproduction and continuation

This repository is a curated continuation snapshot, not a new experiment and not a complete raw-evidence backup. The original workspace is `D:\jev mem` on Windows. No credentials, model weights, virtual environments, caches or active databases are included.

## 1. Verify the delivered bytes

```text
git submodule update --init --recursive
python tools/verify_repository_snapshot.py
```

The helper uses Python 3.11 standard library and local Git only. It verifies the file manifest and pinned submodule HEADs. It does not import provider code, load credentials, make network/model requests, recompute judgments, or certify full scientific replay. The snapshot manifest excludes itself to avoid a circular hash. Frozen files use `.gitattributes` with text conversion disabled.

`docs/RESULTS.md` maps manuscript and baseline claims to accepted summary and per-question files. These enable numerical inspection. The larger archived raw requests, native memory/source and original full audits are only catalogued in `docs/local_only_artifacts.json`; copy them separately from the original machine if full raw replay is required. Missing raw is never silently replaced by regenerated outputs.

## 2. Data and upstream sources

`data/locomo10.json` is the preserved LoCoMo copy, SHA-256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`. `data/locomo_source.json` records the upstream URL and pinned revision. The experiment selects categories 1–4: 1540 questions, 10 conversations and 272 sessions. See `data/local_copy_manifest.json` and the newly exported `data/questions_1540.json`; the latter is a packaging-time projection of the unchanged dataset, not an old experimental receipt. Source material is not a newly collected user dataset.

Mem0 and its evaluation submodule, A-Mem, V-Mem and PRISM retain their own upstream licenses and pinned Git history. SimpleMem is a vendored source snapshot with its license and provenance. Do not treat a submodule checkout or a source file's presence as a completed experiment. Repository publication does not relicense third-party materials; retain their notices.

One historical metric loader additionally expects `research/sources/locomo_evaluation_cbfbc1.py`. That standalone upstream source is not bundled here. Its pinned source is `https://raw.githubusercontent.com/snap-research/locomo/cbfbc1dba6bc53d00625212a0f22d55ffee7c1fc/task_eval/evaluation.py`, SHA-256 `8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd`. The local source records no explicit license found at that pinned root/task_eval; no broader license is asserted. If separately obtained for the original scorer, verify these exact bytes. The repository byte checker does not need this file; a full historical scoring replay does.

## 3. Environment and portability

The accepted Windows baseline runtime used CPython 3.11.14. `configs/installed_versions.txt`, `pyproject.toml` and selected baseline dependency records are observed configurations, not a complete portable lockfile. The existing `pyproject.toml` contains a local Mem0 file URL and D-drive runtime/cache paths. It is preserved byte-for-byte for provenance.

Several frozen scripts contain absolute paths, source hashes and one-shot launch guards. An ordinary clone does not make all historical launchers portable. To reproduce the original layout use `D:\jev mem`; otherwise make separately reviewed path/config adaptations, do not overwrite frozen evidence or pretend original hash checks still apply. Install the pinned upstream dependencies in a new environment; do not copy the old virtual environment.

API credentials must be configured separately. The local Windows workflow reads the user's existing DPAPI vault; provider code also documents environment-variable fallbacks where supported. No vault/secret is distributed. Offline file checking needs neither credentials nor a GPU.

## 4. Manuscript

The editable manuscript is `paper_acl2027_2026-10-07/main.tex`, with its preserved compiled `main.pdf`. The small current source ZIP can be uploaded to Overleaf; see the paper README for compilation. Local Tectonic binaries and rendered page caches are omitted. The paper's versioned validator references original evidence paths and writes outputs: read it before use; the repository integrity checker is the safe read-only starting point.

## 5. Run state

Read `docs/CONTINUATION.md` before any paid operation. A-Mem and SimpleMem are complete; HyperMem was cancelled and cleaned; V-Mem and PRISM are unstarted. The timer remains paused. Do not run the old five-method chain unchanged, invent a HyperMem completion receipt, restore deleted caches, or reuse consumed recovery quotas. Publication itself authorizes no new model experiment.
