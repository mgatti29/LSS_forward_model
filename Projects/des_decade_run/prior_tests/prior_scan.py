#!/usr/bin/env python3
"""Which halo-IA prior ranges keep the shear-shear IA contribution within the NLA prior envelope?

Uses the bases of ia_prior_2pt.py (TNG300, DES Y6-like, package sign convention).  For each candidate prior
box: percentiles over draws of chi2(dC/sigma_DES), max |dC|/sigma, max |dC|/C_GG and of the per-bin mean
var_IA / var_noise, compared with the NLA prior A, eta in [-2, 2].

output: prior_tests/out/prior_scan.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ia_prior_2pt as base
from ia_prior_2pt import v

N_DRAW = 20000
CANDIDATES = {
    'user (A_s +-2)': dict(A_c=(-2, 2), A_s=(-2, 2), eta_c=(-2, 2), eta_s=(-.5, .5)),
    'A_s +-0.1': dict(A_c=(-2, 2), A_s=(-.1, .1), eta_c=(-2, 2), eta_s=(-.5, .5)),
    'A_s 0 (centrals only)': dict(A_c=(-2, 2), A_s=(0, 0), eta_c=(-2, 2), eta_s=(0, 0)),
    'A_s 0, eta_c [-2,0]': dict(A_c=(-2, 2), A_s=(0, 0), eta_c=(-2, 0), eta_s=(0, 0)),
    'A_s +-0.05, eta_c [-2,0]': dict(A_c=(-2, 2), A_s=(-.05, .05), eta_c=(-2, 0), eta_s=(-.5, .5)),
    'A_s +-0.05, eta_c [-2,1]': dict(A_c=(-2, 2), A_s=(-.05, .05), eta_c=(-2, 1), eta_s=(-.5, .5)),
    'A_s +-0.05, eta_c [-1,1]': dict(A_c=(-2, 2), A_s=(-.05, .05), eta_c=(-1, 1), eta_s=(-.5, .5)),
    'A_s +-0.05, A_c +-1.5, eta_c [-2,1]': dict(A_c=(-1.5, 1.5), A_s=(-.05, .05), eta_c=(-2, 1), eta_s=(-.5, .5)),
    'adopted (A_c [-2,2.5], A_s [0,0.03], eta_c [-2,0])': dict(A_c=(-2, 2.5), A_s=(0, .03), eta_c=(-2, 0), eta_s=(-.5, .5)),
    'A_s +-0.02, eta_c [-2,0]': dict(A_c=(-2, 2), A_s=(-.02, .02), eta_c=(-2, 0), eta_s=(-.5, .5)),
    'A_s [0,0.03], eta_c [-2,0]': dict(A_c=(-2, 2), A_s=(0, .03), eta_c=(-2, 0), eta_s=(-.5, .5)),
    'A_s [0,0.05], A_c [0,3], eta_c [-2,0]': dict(A_c=(0, 3), A_s=(0, .05), eta_c=(-2, 0), eta_s=(-.5, .5)),
}


def main():
    rng = np.random.default_rng(5)
    definition = v.survey_definition(base.SURVEY)
    target = np.load(v.WORK/base.SURVEY/'target.npz')
    weights = target['counts']/target['counts'].sum(1, keepdims=True)
    _, g_m = v.frame_fields('model', weights)
    sigma, gg, noise = base.knox_sigma(g_m, definition)
    bases = {}
    for f_c in base.FC_NODES:
        for f_sat in base.FSAT_NODES:
            path = v.WORK/base.SURVEY/f'model_hod_sim_fc_{f_c}_fsat_{f_sat}_prior.npz'
            if path.exists():
                bases[(f_c, f_sat)] = base.HaloBasis(path, g_m)
    keys = list(bases)
    eta_c_nodes, eta_s_nodes = bases[keys[0]].eta_c, bases[keys[0]].eta_s
    pct = lambda x: [round(float(y), 3) for y in np.percentile(x, [50, 90, 99, 100])]
    out = {'populations': [list(k) for k in keys], 'percentiles': [50, 90, 99, 100]}

    # NLA reference
    nla = np.load(v.WORK/base.SURVEY/'nla_prior.npz')
    A = rng.uniform(-2, 2, N_DRAW)
    k = rng.integers(0, len(nla['eta']), N_DRAW)
    dC = np.zeros((N_DRAW, len(base.PAIRS), len(sigma[0])))
    for i in range(len(nla['eta'])):
        gi = np.array([base.bp(g_m[a], nla['alm_E'][b, i]) + base.bp(nla['alm_E'][a, i], g_m[b]) for a, b in base.PAIRS])
        ii = np.array([base.bp(nla['alm_E'][a, i], nla['alm_E'][b, i]) for a, b in base.PAIRS])
        use = k == i
        dC[use] = A[use, None, None]*gi[None] + (A[use]**2)[:, None, None]*ii[None]
    s = base.summarise(dC, sigma, gg)
    out['NLA A,eta in [-2,2]'] = {name: pct(x) for name, x in s.items()}
    print('NLA', out['NLA A,eta in [-2,2]'], flush=True)

    for name, box in CANDIDATES.items():
        A_c = rng.uniform(*box['A_c'], N_DRAW)
        A_s = rng.uniform(*box['A_s'], N_DRAW) if box['A_s'][1] > box['A_s'][0] else np.full(N_DRAW, box['A_s'][0])
        ok_c = np.flatnonzero((eta_c_nodes >= box['eta_c'][0] - 1e-9) & (eta_c_nodes <= box['eta_c'][1] + 1e-9))
        ok_s = np.flatnonzero((eta_s_nodes >= box['eta_s'][0] - 1e-9) & (eta_s_nodes <= box['eta_s'][1] + 1e-9))
        kc = rng.choice(ok_c, N_DRAW)
        ks = rng.choice(ok_s, N_DRAW)
        pop = rng.integers(0, len(keys), N_DRAW)
        dC = np.zeros((N_DRAW, len(base.PAIRS), len(sigma[0])))
        ratio = np.zeros((N_DRAW, 4))
        for p, key in enumerate(keys):
            for i in ok_c:
                for j in ok_s:
                    use = (pop == p) & (kc == i) & (ks == j)
                    if use.any():
                        dC[use] = bases[key].predict(A_c[use], A_s[use], i, j)
                        ratio[use] = bases[key].white(A_c[use], A_s[use], i, j)/noise[None]
        s = base.summarise(dC, sigma, gg)
        entry = {k_: pct(x) for k_, x in s.items()}
        entry['var_ratio_max_bin'] = pct(ratio.max(1))
        entry['frac_var_ratio_gt_0.5'] = float(np.mean(ratio.max(1) > 0.5))
        entry['frac_var_ratio_gt_1'] = float(np.mean(ratio.max(1) > 1))
        entry['box'] = box
        out[name] = entry
        print(name, {k_: entry[k_] for k_ in ('chi2', 'max_sigma', 'max_frac_gg', 'var_ratio_max_bin', 'frac_var_ratio_gt_1')},
              flush=True)
    (base.OUT/'prior_scan.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
