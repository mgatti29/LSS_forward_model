#!/usr/bin/env python3
"""All budget metrics (ia_joint_eval.metrics) at the physically motivated IA points and along A_c through them.

Points: the TNG300 fits of config/ia_priors.json and the IA draws stored with the test mocks (r0-r3).  Lines: A_c
scaled from 0.25 to 2.75 at the other parameters of each TNG fit (A_s fixed).  For each point, rotation, variant and
bin: invalid fraction, kappa, excess variance fraction, superpixel-budget fraction, mean ratio.

usage: ia_joint_special.py OUTDIR  -> OUTDIR/eval_special.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ia_joint_eval as E  # noqa: E402

out = Path(sys.argv[1])
rots = [0, 1, 2, 3]
E.G['rots'] = rots
E.G['geo'], E.G['bases'] = {}, {}
for s in E.SAMPLES:
    for r in rots:
        geo, meta = E.load_geometry(out/f'geom_{s}_b0_r{r}.npz')
        E.G['geo'][(s, r)] = (geo, meta)
        E.G['bases'][(s, r)] = E.load_bases(out, s, 0, r, meta['F_C'], meta['F_SAT'])
TNG = {'TNG outlier fit': dict(A_c=2.566, eta_c=-1.0, A_s=0.0193, eta_s=0.0, f_c=0.5, f_sat=0.8),
       'TNG chi2 (DES footprint) fit': dict(A_c=1.71, eta_c=-1.0, A_s=0.025, eta_s=0.0, f_c=0.7, f_sat=0.6)}
pts = dict(TNG)
for r in rots:
    st = E.G['geo'][('DESY6', r)][1]['stored_ia']
    pts[f'stored draw r{r}'] = {k: float(st[k]) for k in E.PAR}
for name, d in TNG.items():
    for a in np.arange(0.25, 2.76, 0.25):
        pts[f'{name} line A_c={a:.2f}'] = dict(d, A_c=float(a))
names = ['invalid', 'kappa', 'excess', 'sp_invalid', 'ratio']
res = {}
for name, d in pts.items():
    E.G['draws'] = {k: np.array([v]) for k, v in d.items()}
    _, rr = E.work(0)                                   # (rot, variant, bin, metric)
    res[name] = dict(params=d, **{f'{m}_worstbin_per_rot': {v: (rr[:, iv, :, k].min(1) if m == 'kappa' else rr[:, iv, :, k].max(1)).tolist()
                                                          for iv, v in enumerate(E.VARIANTS)} for k, m in enumerate(names)},
                     per_bin_rot0={v: {m: rr[0, iv, :, k].tolist() for k, m in enumerate(names)} for iv, v in enumerate(E.VARIANTS)})
    E.log(name, {v: (round(float(rr[:, iv, :, 0].max()), 4), round(float(rr[:, iv, :, 2].max()), 5), round(float(rr[:, iv, :, 3].max()), 5))
                 for iv, v in enumerate(E.VARIANTS)})
(out/'eval_special.json').write_text(json.dumps(res, indent=1, default=float) + '\n')
