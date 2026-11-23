#!/usr/bin/env python3
"""Manifest (size, sha256, source) of the large inputs staged in STAGE -> config/large_inputs.json.

The staged directory holds symlinks; copy it to another cluster with `rsync -L` (follows the links) and check
it there with `make_manifest.py --check`.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

STAGE = Path(os.environ.get('DES_DECADE_INPUTS', '/global/cfs/cdirs/m5099/des_decade_run_inputs'))
MANIFEST = Path(__file__).resolve().parent.parent/'config'/'large_inputs.json'


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


if '--check' in sys.argv:
    manifest = json.loads(MANIFEST.read_text())
    for key, entry in manifest['files'].items():
        path = STAGE/entry['file']
        ok = path.exists() and path.stat().st_size == entry['size'] and sha256(path) == entry['sha256']
        print(f"{key:16s} {'OK' if ok else 'MISMATCH/MISSING'}  {path}")
else:
    files = {}
    for path in sorted(STAGE.iterdir()):
        files[path.stem] = dict(file=path.name, size=path.stat().st_size, sha256=sha256(path),
                                source=os.path.realpath(path))
        print(path.name, files[path.stem]['size'], flush=True)
    MANIFEST.write_text(json.dumps(dict(_doc='Large inputs staged outside git (STAGE, default '
                                              '/global/cfs/cdirs/m5099/des_decade_run_inputs; override with '
                                              'DES_DECADE_INPUTS). Check with data_prep/make_manifest.py --check.',
                                        files=files), indent=1) + '\n')
    print('wrote', MANIFEST)
