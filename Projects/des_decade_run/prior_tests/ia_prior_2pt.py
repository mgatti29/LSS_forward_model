#!/usr/bin/env python3
"""Halo-IA prior vs NLA prior at the shear two-point level (TNG300, DES Y6-like, full-sky Knox errors for DES).

For every tomographic pair (s, t) the IA contribution to the shear-shear bandpowers is

    dC_st = G_s I_t + I_s G_t + I_s I_t   (white self term of I_t I_t removed)

with G the observed-convention shear of the TNG300 model frame and I the IA field of
  * the virtual-halo model (cheap_ia_validation.py hod, files model_hod_sim_fc_<f_c>_fsat_<f_sat>_prior.npz:
    unit-amplitude central and satellite fields on grids of eta_c, eta_s), polynomial in A_c, A_s;
  * NLA (nla_tng_fields.py, nla_prior.npz), linear in A.
The TNG300 galaxies themselves (target) are shown as the reference.

Prior draws: amplitudes continuous, redshift slopes and (f_c, f_sat) on the stored nodes (exact predictions).
sigma: Gaussian (Knox) error of the DES Y6 shear-shear bandpowers (f_sky, sigma_e, n_eff of the catalogue).
var_IA / var_noise: halo IA white variance over shape-noise variance per bin (mean over the sky).

usage: ia_prior_2pt.py [--n-draw N] [--as-max X ...]
output: Projects/des_decade_run/prior_tests/out/ia_prior_2pt.{json,png}
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model/Projects/IA_non_gaussian')
sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model/Projects/IA_non_gaussian/cheap_model_validation')
import cheap_ia_validation as v

SURVEY = 'DES'
OUT = Path(__file__).resolve().parent/'out'
FC_NODES = (0.2, 0.45, 0.7)
FSAT_NODES = (0.3, 0.55, 0.8)
PRIOR_HALO = dict(A_c=(-2., 2.), A_s=(-2., 2.), eta_c=(-2., 2.), eta_s=(-0.5, 0.5))
PRIOR_NLA = dict(A=(-2., 2.), eta=(-2., 2.))
# best fits on TNG300 DES (outlier objective, final; and chi2 DES-footprint fit)
BEST = {'final (f_c .5, f_sat .8)': dict(f_c=0.5, f_sat=0.8, A_c=2.566, A_s=0.0193, eta_c=-1.0, eta_s=0.0),
        'chi2 (f_c .7, f_sat .6)': dict(f_c=0.7, f_sat=0.6, A_c=1.71, A_s=0.025, eta_c=-1.0, eta_s=0.0)}
PAIRS = [(s, t) for s in range(4) for t in range(s, 4)]
# the halo marks and the TNG300 shapes have the opposite E-mode sign to the package lensing/NLA convention
# (sign_check.py); multiply their IA fields by HALO_SIGN so that positive amplitudes are physical alignments
HALO_SIGN = -1.0


def bp(a, b):
    return v.bandpower(hp.alm2cl(a, b))


def knox_sigma(g, definition):
    """Gaussian error of C_st = <e_s e_t>, e = G + noise (IA neglected in the covariance)."""
    f_sky = definition['area_deg2']/41252.96
    noise = np.array(definition['sigma_e'])**2/(np.array(definition['n_eff_arcmin2'])*(180*60/np.pi)**2)
    n_modes = v.modes_per_band()*f_sky
    C = {(s, t): bp(g[s], g[t]) + (noise[s] if s == t else 0.) for s in range(4) for t in range(s, 4)}
    get = lambda s, t: C[(min(s, t), max(s, t))]
    sigma = np.array([np.sqrt((get(s, s)*get(t, t) + get(s, t)**2)/n_modes) for s, t in PAIRS])
    gg = np.array([bp(g[s], g[t]) for s, t in PAIRS])
    return sigma, gg, noise


class HaloBasis:
    """Bandpowers of one (f_c, f_sat) population: predictions are polynomials in A_c, A_s."""

    def __init__(self, path, g):
        m = np.load(path)
        C, S = m['alm_central'], m['alm_satellite']
        self.eta_c, self.eta_s = m['eta_c'], m['eta_s']
        nc, ns = len(self.eta_c), len(self.eta_s)
        self.gC = HALO_SIGN*np.array([[[bp(g[a], C[b, k]) for k in range(nc)] for b in range(4)] for a in range(4)])
        self.gS = HALO_SIGN*np.array([[[bp(g[a], S[b, j]) for j in range(ns)] for b in range(4)] for a in range(4)])
        self.CC = {p: np.array([bp(C[p[0], k], C[p[1], k]) for k in range(nc)]) for p in PAIRS}
        self.SS = {p: np.array([bp(S[p[0], j], S[p[1], j]) for j in range(ns)]) for p in PAIRS}
        self.CS = {(a, b): np.array([[bp(C[a, k], S[b, j]) for j in range(ns)] for k in range(nc)])
                   for a in range(4) for b in range(4)}
        self.self_c, self.self_s = m['self_central'], m['self_satellite']      # (bin, eta)
        self.f_res = m['f_res']

    def predict(self, A_c, A_s, kc, ks):
        """dC (npair, nband) for arrays A_c, A_s (ndraw,) at node indices kc, ks."""
        out = []
        for a, b in PAIRS:
            value = (A_c[:, None]*(self.gC[a, b, kc] + self.gC[b, a, kc])[None]
                     + A_s[:, None]*(self.gS[a, b, ks] + self.gS[b, a, ks])[None]
                     + (A_c**2)[:, None]*self.CC[(a, b)][kc][None]
                     + (A_c*A_s)[:, None]*(self.CS[(a, b)][kc, ks] + self.CS[(b, a)][kc, ks])[None]
                     + (A_s**2)[:, None]*self.SS[(a, b)][ks][None])
            if a == b:
                value = value - ((A_c**2)*self.self_c[a, kc] + (A_s**2)*self.self_s[a, ks])[:, None]
            out.append(value)
        return np.stack(out, axis=1)

    def white(self, A_c, A_s, kc, ks):
        """IA white power per bin (ndraw, 4)."""
        return (A_c**2)[:, None]*self.self_c[:, kc][None] + (A_s**2)[:, None]*self.self_s[:, ks][None]


def summarise(dC, sigma, gg):
    r = dC/sigma[None]
    return dict(chi2=np.sum(r**2, axis=(1, 2)), max_sigma=np.max(np.abs(r), axis=(1, 2)),
                max_frac_gg=np.max(np.abs(dC/gg[None]), axis=(1, 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-draw', type=int, default=20000)
    parser.add_argument('--as-max', type=float, nargs='+', default=[2.0, 0.5, 0.2, 0.1])
    parser.add_argument('--ac-max', type=float, default=2.0)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(11)

    definition = v.survey_definition(SURVEY)
    target = np.load(v.WORK/SURVEY/'target.npz')
    weights = target['counts']/target['counts'].sum(1, keepdims=True)
    d_m, g_m = v.frame_fields('model', weights)
    d_t, g_t = v.frame_fields('target', weights)
    sigma, gg, noise = knox_sigma(g_m, definition)
    nb = len(v.ELL_EDGES) - 1
    ell = np.sqrt(v.ELL_EDGES[:-1]*v.ELL_EDGES[1:])

    # reference: TNG300 galaxies (target frame)
    I_t = target['alm_E']
    ref = []
    for a, b in PAIRS:
        value = HALO_SIGN*(bp(g_t[a], I_t[b]) + bp(I_t[a], g_t[b])) + bp(I_t[a], I_t[b])
        ref.append(value - (target['self_power'][a] if a == b else 0.))
    ref = np.array(ref)

    # NLA prior
    nla = np.load(v.WORK/SURVEY/'nla_prior.npz')
    nla_alm, nla_eta = nla['alm_E'], nla['eta']
    nla_gi = {k: [] for k in range(len(nla_eta))}
    A = rng.uniform(*PRIOR_NLA['A'], args.n_draw)
    k_nla = rng.integers(0, len(nla_eta), args.n_draw)
    dC_nla = np.zeros((args.n_draw, len(PAIRS), nb))
    for k in range(len(nla_eta)):
        use = k_nla == k
        gi = np.array([bp(g_m[a], nla_alm[b, k]) + bp(nla_alm[a, k], g_m[b]) for a, b in PAIRS])
        ii = np.array([bp(nla_alm[a, k], nla_alm[b, k]) for a, b in PAIRS])
        nla_gi[k] = (gi, ii)
        dC_nla[use] = A[use, None, None]*gi[None] + (A[use]**2)[:, None, None]*ii[None]
    s_nla = summarise(dC_nla, sigma, gg)
    # NLA corners
    corners_nla = {}
    for A0, k in itertools.product((-2., 2.), (0, len(nla_eta) - 1)):
        gi, ii = nla_gi[k]
        corners_nla[f'A={A0:+.0f},eta={nla_eta[k]:+.1f}'] = A0*gi + A0**2*ii

    # halo prior: populations on nodes
    bases = {}
    for f_c, f_sat in itertools.product(FC_NODES, FSAT_NODES):
        path = v.WORK/SURVEY/f'model_hod_sim_fc_{f_c}_fsat_{f_sat}_prior.npz'
        if path.exists():
            bases[(f_c, f_sat)] = HaloBasis(path, g_m)
            print('basis', f_c, f_sat, flush=True)
    keys = list(bases)
    any_basis = bases[keys[0]]
    results = {'ell': ell.tolist(), 'pairs': PAIRS, 'sigma': sigma.tolist(), 'gg': gg.tolist(),
               'noise_power': noise.tolist(), 'populations': [list(k) for k in keys],
               'eta_c_nodes': any_basis.eta_c.tolist(), 'eta_s_nodes': any_basis.eta_s.tolist(),
               'nla_eta_nodes': nla_eta.tolist(), 'reference_tng': summarise_dict(ref, sigma, gg)}
    pct = lambda x: np.percentile(x, [50, 90, 99, 100]).tolist()
    results['nla'] = {name: pct(x) for name, x in s_nla.items()}
    results['halo'] = {}
    halo_envelopes = {}
    for as_max in args.as_max:
        A_c = rng.uniform(-args.ac_max, args.ac_max, args.n_draw)
        A_s = rng.uniform(-as_max, as_max, args.n_draw)
        kc = rng.integers(0, len(any_basis.eta_c), args.n_draw)
        ks = rng.integers(0, len(any_basis.eta_s), args.n_draw)
        pop = rng.integers(0, len(keys), args.n_draw)
        dC = np.zeros((args.n_draw, len(PAIRS), nb))
        ratio = np.zeros((args.n_draw, 4))
        for p, key in enumerate(keys):
            for i in range(len(any_basis.eta_c)):
                for j in range(len(any_basis.eta_s)):
                    use = (pop == p) & (kc == i) & (ks == j)
                    if use.any():
                        dC[use] = bases[key].predict(A_c[use], A_s[use], i, j)
                        ratio[use] = bases[key].white(A_c[use], A_s[use], i, j)/noise[None]
        s = summarise(dC, sigma, gg)
        entry = {name: pct(x) for name, x in s.items()}
        entry['var_ratio_max_bin'] = pct(ratio.max(1))
        entry['frac_var_ratio_gt'] = {str(x): float(np.mean(ratio.max(1) > x)) for x in (0.25, 0.5, 1.0)}
        # which parameters drive the largest contributions
        top = s['chi2'] > np.percentile(s['chi2'], 99)
        entry['top1pct'] = dict(abs_A_c=float(np.mean(np.abs(A_c[top]))), abs_A_s=float(np.mean(np.abs(A_s[top]))),
                                eta_c=float(np.mean(any_basis.eta_c[kc[top]])),
                                f_sat=float(np.mean([keys[q][1] for q in pop[top]])),
                                f_c=float(np.mean([keys[q][0] for q in pop[top]])))
        results['halo'][f'A_s_max={as_max}'] = entry
        halo_envelopes[as_max] = dC
        print(f'A_s max {as_max}:', json.dumps(entry), flush=True)
    # NLA envelope and best fits
    best = {}
    for name, b in BEST.items():
        key = (b['f_c'], b['f_sat'])
        if key in bases:
            kc = int(np.argmin(np.abs(any_basis.eta_c - b['eta_c'])))
            ks = int(np.argmin(np.abs(any_basis.eta_s - b['eta_s'])))
            best[name] = bases[key].predict(np.array([b['A_c']]), np.array([b['A_s']]), kc, ks)[0]
            results[f'best {name}'] = summarise_dict(best[name], sigma, gg)
            results[f'best {name}']['var_ratio'] = (bases[key].white(np.array([b['A_c']]), np.array([b['A_s']]),
                                                                     kc, ks)[0]/noise).tolist()
    (OUT/'ia_prior_2pt.json').write_text(json.dumps(results, indent=1) + '\n')

    # figure: 95% envelopes of dC / sigma per pair
    fig, axes = plt.subplots(4, 4, figsize=(15, 12), sharex=True)
    for ax in axes.flat:
        ax.set_visible(False)
    colors = {2.0: 'C3', 0.5: 'C1', 0.2: 'C2', 0.1: 'C0'}
    for p, (a, b) in enumerate(PAIRS):
        ax = axes[b, a]
        ax.set_visible(True)
        lo, hi = np.percentile(dC_nla[:, p]/sigma[p], [2.5, 97.5], axis=0)
        ax.fill_between(ell, lo, hi, color='0.7', alpha=0.6, label='NLA prior 95%')
        for as_max, dC in halo_envelopes.items():
            lo, hi = np.percentile(dC[:, p]/sigma[p], [2.5, 97.5], axis=0)
            ax.plot(ell, lo, color=colors.get(as_max, 'k'), lw=1)
            ax.plot(ell, hi, color=colors.get(as_max, 'k'), lw=1, label=f'halo prior 95%, |A_s|<{as_max}')
        ax.plot(ell, ref[p]/sigma[p], 'k', lw=2, label='TNG300 galaxies')
        for name, value in best.items():
            ax.plot(ell, value[p]/sigma[p], ls='--', lw=1.2, label=f'halo best {name}')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xscale('log')
        ax.set_title(f'bins {a+1}{b+1}', fontsize=9)
        if a == 0:
            ax.set_ylabel(r'$\Delta C^{ee}_\ell/\sigma_{\rm DES}$')
        if b == 3:
            ax.set_xlabel(r'$\ell$')
    axes[0, 0].legend(fontsize=7, loc='lower left')
    fig.suptitle('IA contribution to DES Y6 shear-shear: halo-IA prior vs NLA prior (TNG300)')
    fig.tight_layout()
    fig.savefig(OUT/'ia_prior_2pt.png', dpi=110)
    print('saved', OUT/'ia_prior_2pt.png')


def summarise_dict(dC, sigma, gg):
    r = dC/sigma
    return dict(chi2=float(np.sum(r**2)), max_sigma=float(np.max(np.abs(r))),
                max_frac_gg=float(np.max(np.abs(dC/gg))))


if __name__ == '__main__':
    main()
