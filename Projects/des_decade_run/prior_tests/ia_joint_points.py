#!/usr/bin/env python3
"""Noise-budget metrics (ia_joint_eval.metrics, production 2% rule) at every row of IA fit scans.

Rows: the 'grid' lists (and 'best') of paper_fit_outliers.py JSON outputs, or a plain list of points; each needs
f_c, f_sat, eta_c, eta_s, A_c, A_s.  Output per row: worst-bin invalid fraction (max over rotations) and min kappa per
sample variant, and 'accepted' per variant (invalid <= 2% in every bin and rotation).

usage: ia_joint_points.py OUTDIR RESULT.json NPROC INPUT.json [INPUT.json ...]
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ia_joint_eval as E  # noqa: E402

out, result, nproc = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rows = []
for f in sys.argv[4:]:
    d = json.loads(Path(f).read_text())
    for r in (d['grid'] if isinstance(d, dict) else d):
        rows.append(dict(r, source=Path(f).name))
rots = [0, 1, 2, 3]
E.G['rots'] = rots
E.G['geo'], E.G['bases'] = {}, {}
for s in E.SAMPLES:
    for r in rots:
        geo, meta = E.load_geometry(out/f'geom_{s}_b0_r{r}.npz')
        E.G['geo'][(s, r)] = (geo, meta)
        E.G['bases'][(s, r)] = E.load_bases(out, s, 0, r, meta['F_C'], meta['F_SAT'])
E.G['draws'] = {k: np.array([float(r[k]) for r in rows]) for k in E.PAR}
E.log(len(rows), 'rows')
with mp.get_context('fork').Pool(nproc) as pool:
    res = dict(pool.imap_unordered(E.work, range(len(rows)), chunksize=4))
for i, r in enumerate(rows):
    rr = res[i]                                         # (rot, variant, bin, metric)
    r['invalid'] = {v: float(rr[:, iv, :, 0].max()) for iv, v in enumerate(E.VARIANTS)}
    r['kappa'] = {v: float(rr[:, iv, :, 1].min()) for iv, v in enumerate(E.VARIANTS)}
    r['accepted'] = {v: bool(rr[:, iv, :, 0].max() <= E.LEVEL) for iv, v in enumerate(E.VARIANTS)}
result.write_text(json.dumps(rows, indent=1, default=float) + '\n')
E.log('saved', result)
