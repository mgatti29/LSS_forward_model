#!/usr/bin/env python3
"""Acceptance of the halo-IA noise budget over the prior, from ia_acceptance_scan.py outputs (several b, r).

A draw (A_c, eta_c, A_s, eta_s, f_c, f_sat) is accepted in a configuration if A_c^2 + (A_s/S0)^2 < rho*^2, with
rho*^2 the minimum over samples and bins of the scanned critical radius, interpolated linearly in phi, eta_c,
eta_s and bilinearly in (f_c, f_sat) (in log rho*^2).  Monte Carlo over the nominal and candidate priors.

usage: analyse_ia_acceptance.py PREFIX [PREFIX ...]   (PREFIX of PREFIX.json / PREFIX_rho2.npz)
output: prior_tests/out/ia_acceptance.json, ia_acceptance.png
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = Path(__file__).resolve().parent
NOMINAL = dict(A_c=(-2, 2.5), eta_c=(-2, 0), A_s=(0, 0.03), eta_s=(-.5, .5), f_c=(.2, .7), f_sat=(.3, .8))
TNG = {'TNG outlier fit': dict(A_c=2.566, eta_c=-1.0, A_s=0.0193, eta_s=0.0, f_c=0.5, f_sat=0.8),
       'TNG chi2 (DES footprint) fit': dict(A_c=1.71, eta_c=-1.0, A_s=0.025, eta_s=0.0, f_c=0.7, f_sat=0.6)}
N_MC = 400000


class Config:
    def __init__(self, prefix):
        self.res = json.loads(Path(prefix + '.json').read_text())
        z = np.load(prefix + '_rho2.npz')
        r = self.res
        self.levels = r['LEVELS']
        self.S0 = r['S0']
        self.samples = r['samples']
        fc, fs = r['F_C'], r['F_SAT']
        shape = (len(fc), len(fs), len(r['ETA_C']), len(r['ETA_S']), len(r['PHI']), len(self.levels))
        self.rho2 = {s: np.full(shape, np.nan) for s in self.samples}
        self.rho2_bin = {s: {} for s in self.samples}
        for k in z.files:
            s, key = k.split('|')
            f1, f2 = map(float, key.split('_'))
            i, j = fc.index(f1), fs.index(f2)
            arr = z[k]                                    # (ntomo, eta_c, eta_s, phi, level)
            self.rho2[s][i, j] = arr.min(0)
            self.rho2_bin[s][(i, j)] = arr
        self.rho2['all'] = np.min([self.rho2[s] for s in self.samples], axis=0)
        grid = (fc, fs, r['ETA_C'], r['ETA_S'], r['PHI'])
        self.interp = {s: [RegularGridInterpolator(grid, np.log(np.maximum(v[..., l], 1e-12)), bounds_error=False,
                                                    fill_value=None) for l in range(len(self.levels))]
                       for s, v in self.rho2.items()}
        # A_eff coefficients (ntomo) per node, eta: a_c, a_s
        self.ac, self.as_ = {}, {}
        for key, nd in r['nodes'].items():
            i, j = fc.index(nd['f_c']), fs.index(nd['f_sat'])
            for s in self.samples:
                if s in nd:
                    self.ac.setdefault(s, np.zeros((len(fc), len(fs), len(r['ETA_C']), r['ntomo'][s])))[i, j] = nd[s]['a_c']
                    self.as_.setdefault(s, np.zeros((len(fc), len(fs), len(r['ETA_S']), r['ntomo'][s])))[i, j] = nd[s]['a_s']
        self.a_interp = {s: (RegularGridInterpolator((fc, fs, r['ETA_C']), self.ac[s], bounds_error=False, fill_value=None),
                             RegularGridInterpolator((fc, fs, r['ETA_S']), self.as_[s], bounds_error=False, fill_value=None))
                         for s in self.ac}

    def accept(self, d, sample='all', level=0.02):
        u = d['A_s']/self.S0
        phi = np.arctan2(u, np.abs(d['A_c']))
        pts = np.stack([d['f_c'], d['f_sat'], d['eta_c'], d['eta_s'], phi], -1)
        rho2 = np.exp(self.interp[sample][self.levels.index(level)](pts))
        return d['A_c']**2 + u**2 < rho2

    def A_eff(self, d, sample):
        ic, is_ = self.a_interp[sample]
        return (d['A_c'][:, None]*ic(np.stack([d['f_c'], d['f_sat'], d['eta_c']], -1))
                + d['A_s'][:, None]*is_(np.stack([d['f_c'], d['f_sat'], d['eta_s']], -1)))


def draw(rng, prior, n=N_MC):
    return {k: rng.uniform(*prior[k], n) if np.ptp(prior[k]) > 0 else np.full(n, prior[k][0]) for k in prior}


def main():
    configs = [Config(p) for p in sys.argv[1:]]
    rng = np.random.default_rng(1)
    out = {'configs': [f"b{c.res['b']}_r{c.res['r']}" for c in configs], 'validation': []}
    for c in configs:
        out['validation'] += c.res['validation']

    d = draw(rng, NOMINAL)
    acc = np.array([c.accept(d) for c in configs])                 # (config, draw)
    out['nominal'] = dict(acceptance_per_config=acc.mean(1).tolist(), acceptance_mean=float(acc.mean()),
                          acceptance_all_configs=float(acc.all(0).mean()))
    out['nominal_per_sample'] = {s: float(np.mean([c.accept(d, s) for c in configs]))
                                 for s in configs[0].samples}
    out['nominal_levels'] = {str(L): float(np.mean([c.accept(d, level=L) for c in configs])) for L in configs[0].levels}
    # partial dependence: acceptance vs each parameter (marginalised over the others, nominal prior)
    pd_ = {}
    for k, (lo, hi) in NOMINAL.items():
        edges = np.linspace(lo, hi, 11)
        x = d[k] if k != 'A_c' else d[k]
        idx = np.clip(np.digitize(x, edges) - 1, 0, 9)
        pd_[k] = dict(centres=(0.5*(edges[1:] + edges[:-1])).tolist(),
                      acceptance=[float(acc[:, idx == i].mean()) for i in range(10)])
    out['partial_dependence'] = pd_

    # |A_c| bound for >= 99 % acceptance (other parameters at their nominal priors)
    bounds = {}
    for B in np.arange(0.5, 2.51, 0.05):
        pr = dict(NOMINAL, A_c=(-B, B))
        dd = draw(rng, pr, 200000)
        a = np.array([c.accept(dd) for c in configs])
        bounds[f'{B:.2f}'] = dict(mean=float(a.mean()), worst_config=float(a.mean(1).min()))
    out['A_c_bound_scan'] = bounds
    ok = [float(b) for b, v in bounds.items() if v['worst_config'] >= 0.99]
    out['A_c_bound_99'] = max(ok) if ok else None

    # candidate rectangular / joint priors
    cands = {
        'nominal': NOMINAL,
        '|A_c|<1': dict(NOMINAL, A_c=(-1, 1)),
        '|A_c|<1.25': dict(NOMINAL, A_c=(-1.25, 1.25)),
        '|A_c|<1.5': dict(NOMINAL, A_c=(-1.5, 1.5)),
        'A_c [-1, 1.5]': dict(NOMINAL, A_c=(-1, 1.5)),
        'A_c [-1.5, 2.5], f_sat [0.6, 0.8]': dict(NOMINAL, A_c=(-1.5, 2.5), f_sat=(.6, .8)),
    }
    out['candidates'] = {}
    for name, pr in cands.items():
        dd = draw(rng, pr, 200000)
        a = np.array([c.accept(dd) for c in configs])
        row = dict(prior={k: list(v) for k, v in pr.items()}, mean=float(a.mean()), worst_config=float(a.mean(1).min()))
        for s in configs[0].ac:
            ae = np.array([c.A_eff(dd, s) for c in configs]).mean(0)          # (draw, ntomo)
            row[f'A_eff_{s}_percentiles_1_5_50_95_99'] = np.percentile(ae, [1, 5, 50, 95, 99], axis=0).round(3).tolist()
        out['candidates'][name] = row

    # joint constraint: K = A_c^2 (1 - f_sat) ((1+z_p)/1.67)^(2 eta_c) style, fit on the critical radius at phi=0
    # (reported as the acceptance of the cut A_c^2 (1 - f_sat) < K over the nominal prior)
    joint = {}
    for K in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        keep = d['A_c']**2*(1 - d['f_sat']) < K
        joint[str(K)] = dict(prior_volume_kept=float(keep.mean()), acceptance_inside=float(acc[:, keep].mean()),
                             worst_config_inside=float(acc[:, keep].mean(1).min()))
    out['joint_Ac2_1mfsat'] = joint

    # TNG-like points
    out['TNG'] = {}
    for name, p in TNG.items():
        dd = {k: np.array([v]) for k, v in p.items()}
        row = dict(params=p, accepted_per_config=[bool(c.accept(dd)[0]) for c in configs])
        for s in configs[0].ac:
            row[f'A_eff_{s}'] = np.mean([c.A_eff(dd, s)[0] for c in configs], axis=0).round(3).tolist()
        # largest accepted multiple of the TNG amplitudes
        lam = np.linspace(0, 1.5, 151)
        dl = {k: np.full(len(lam), v) for k, v in p.items()}
        dl['A_c'] = p['A_c']*lam
        dl['A_s'] = p['A_s']*lam
        a = np.array([c.accept(dl) for c in configs]).all(0)
        row['max_accepted_amplitude_scale'] = float(lam[a].max()) if a.any() else 0.
        out['TNG'][name] = row

    (HERE/'out').mkdir(exist_ok=True)
    (HERE/'out'/'ia_acceptance.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(json.dumps({k: out[k] for k in ('nominal', 'nominal_per_sample', 'nominal_levels', 'A_c_bound_99')}, indent=1))
    for name, row in out['candidates'].items():
        print(name, 'mean', round(row['mean'], 4), 'worst config', round(row['worst_config'], 4))
    for name, row in out['TNG'].items():
        print(name, row)
    print('joint', json.dumps(joint))
    for row in out['validation']:
        print('validation', row['sample'], row['A_c'], 'emulated', round(row['emulated_worst'], 4), 'exact', row['exact_worst'])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(pd_), figsize=(3*len(pd_), 2.8), sharey=True)
    for ax, (k, v) in zip(axes, pd_.items()):
        ax.plot(v['centres'], v['acceptance'], 'o-')
        ax.set_xlabel(k)
        ax.axhline(0.99, color='k', lw=.5, ls='--')
    axes[0].set_ylabel('acceptance (nominal prior)')
    fig.tight_layout()
    fig.savefig(HERE/'out'/'ia_acceptance.png', dpi=110)


if __name__ == '__main__':
    main()
