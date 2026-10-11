"""Verify the published file snapshot offline. Does not import experiment code."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'docs/repository_snapshot.json').read_text(encoding='utf-8'))
    errors = []
    for row in manifest['files']:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root):
            errors.append({'path': row['path'], 'error': 'outside checkout'})
        elif not path.is_file():
            errors.append({'path': row['path'], 'error': 'missing'})
        elif path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            errors.append({'path': row['path'], 'error': 'size/hash mismatch'})
    missing_submodules = []
    for row in manifest.get('submodules', []):
        path = root / row['path']
        if not (path / '.git').exists():
            missing_submodules.append(row['path'])
            continue
        result = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, check=False)
        if result.returncode or result.stdout.strip() != row['commit']:
            errors.append({'path': row['path'], 'error': 'submodule revision mismatch'})
    print(json.dumps({'status': 'passed' if not errors and not missing_submodules else 'incomplete',
                      'checked_files': len(manifest['files']), 'errors': errors,
                      'missing_submodules': missing_submodules,
                      'scope': 'Byte integrity only; no inference or scientific raw replay.'},
                     ensure_ascii=False, indent=2))
    return 0 if not errors and not missing_submodules else 1


if __name__ == '__main__':
    raise SystemExit(main())
