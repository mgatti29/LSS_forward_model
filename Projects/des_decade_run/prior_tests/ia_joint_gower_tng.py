#!/usr/bin/env python3
"""Noise-budget metrics (ia_joint_eval.metrics) at the IA best fits of the Gower St TNG-cosmology run
(Projects/IA_non_gaussian/cheap_model_validation/gower_tng_driver.sh, paper_fit_des_gower_tng/), and along A_c through
them.  Same outputs as ia_joint_special.py.

usage: ia_joint_gower_tng.py OUTDIR  -> OUTDIR/eval_gower_tng.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ia_joint_eval as E  # noqa: E402

FIT = Path('/pscratch/sd/m/mgatti/IA_non_gaussian/tng300/cheap_validation/DES/paper_fit_des_gower_tng')
out = Path(sys.argv[1])
rots = [0, 1, 2, 3]
E.G['rots'] = rots
E.G['geo'], E.G['bases'] = {}, {}
for s in E.SAMPLES:
    for r in rots:
        geo, meta = E.load_geometry(out/f'geom_{s}_b0_r{r}.npz')
        E.G['geo'][(s, r)] = (geo, meta)
        E.G['bases'][(s, r)] = E.load_bases(out, s, 0, r, meta['F_C'], meta['F_SAT'])
fits = {'Gower unconstrained': 'fit_outliers.json', 'Gower prior box': 'prior_box.json'}
pts = {}
for name, f in fits.items():
    b = json.loads((FIT/f).read_text())['best']
    pts[name] = {k: float(b[k]) for k in E.PAR}
for name in list(pts):
    for a in np.arange(0.5, 1.76, 0.125):
        pts[f'{name} line A_c={a:.3f}'] = dict(pts[name], A_c=float(a))
names = ['invalid', 'kappa', 'excess', 'sp_invalid', 'ratio']
res = {}
for name, d in pts.items():
    E.G['draws'] = {k: np.array([v]) for k, v in d.items()}
    _, rr = E.work(0)                                   # (rot, variant, bin, metric)
    res[name] = dict(params=d, **{f'{m}_worstbin_per_rot': {v: (rr[:, iv, :, k].min(1) if m == 'kappa' else rr[:, iv, :, k].max(1)).tolist()
                                                          for iv, v in enumerate(E.VARIANTS)} for k, m in enumerate(names)})
    E.log(name, 'worst-bin invalid (max over rot) / min kappa:',
          {v: (round(float(rr[:, iv, :, 0].max()), 4), round(float(rr[:, iv, :, 1].min()), 3)) for iv, v in enumerate(E.VARIANTS)})
(out/'eval_gower_tng.json').write_text(json.dumps(res, indent=1, default=float) + '\n')
