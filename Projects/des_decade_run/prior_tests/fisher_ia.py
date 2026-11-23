#!/usr/bin/env python3
"""Does the halo-IA model open the posterior more than NLA?  Fisher test on TNG300 DES Y6-like shear-shear.

Data vector: the 10 tomographic shear-shear bandpowers (cheap_ia_validation ell bins, 30 < ell < 2000) with the
Gaussian (Knox) DES Y6 covariance.  Model:
    C_st(a, theta) = a^2 GG_st + a GI_st(theta) + II_st(theta)
a = lensing amplitude (proxy for S8: GG ~ a^2, GI ~ a, II independent of a).
IA models (package sign convention, ia_prior_2pt.HALO_SIGN):
    NLA   (A, eta)                                    nla_prior.npz, eta nodes 0.5 apart
    halo  (A_c, eta_c, A_s, eta_s, f_c, f_sat)        model_hod_sim_fc_<f_c>_fsat_<f_sat>_prior.npz
Fiducials: best fit of each model to the TNG300 galaxies (on the stored nodes).  Derivatives: analytic in the
amplitudes, central differences on the eta nodes, and on the (f_c, f_sat) population nodes around the centre
node (0.45, 0.55).  Flat priors enter as Gaussian priors of width (range)/sqrt(12).
Reports the marginalised sigma(a) for: a alone, NLA, halo marks only (f_c, f_sat fixed), full halo.

output: prior_tests/out/fisher_ia.json
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ia_prior_2pt as base
from ia_prior_2pt import v

PRIOR_WIDTH = {'A': 4., 'eta': 4., 'A_c': 4., 'eta_c': 2., 'A_s': 0.1, 'eta_s': 1., 'f_c': 0.5, 'f_sat': 0.5}
FC0, FSAT0 = 0.45, 0.55


def flat(x):
    return np.asarray(x).reshape(-1)


def main():
    definition = v.survey_definition(base.SURVEY)
    target = np.load(v.WORK/base.SURVEY/'target.npz')
    weights = target['counts']/target['counts'].sum(1, keepdims=True)
    d_t, g_t = v.frame_fields('target', weights)
    _, g_m = v.frame_fields('model', weights)
    sigma, gg, noise = base.knox_sigma(g_m, definition)
    gg_t = np.array([base.bp(g_t[a], g_t[b]) for a, b in base.PAIRS])
    I_t = target['alm_E']
    gi_true = base.HALO_SIGN*np.array([base.bp(g_t[a], I_t[b]) + base.bp(I_t[a], g_t[b]) for a, b in base.PAIRS])
    ii_true = np.array([base.bp(I_t[a], I_t[b]) - (target['self_power'][a] if a == b else 0.) for a, b in base.PAIRS])
    icov = 1/flat(sigma)**2                       # diagonal Knox covariance (pairs x bands)
    chi2 = lambda r: float(np.sum(flat(r)**2*icov))
    truth = gi_true + ii_true

    # NLA pieces: GI and II per eta node
    nla = np.load(v.WORK/base.SURVEY/'nla_prior.npz')
    eta_n = nla['eta']
    nla_gi = np.array([[base.bp(g_m[a], nla['alm_E'][b, k]) + base.bp(nla['alm_E'][a, k], g_m[b])
                        for a, b in base.PAIRS] for k in range(len(eta_n))])
    nla_ii = np.array([[base.bp(nla['alm_E'][a, k], nla['alm_E'][b, k]) for a, b in base.PAIRS]
                       for k in range(len(eta_n))])

    def nla_model(a, A, k):
        return a**2*gg + a*A*nla_gi[k] + A**2*nla_ii[k]

    # NLA fiducial: best (A, eta node) to the TNG truth (a = 1)
    best = None
    for k in range(len(eta_n)):
        for A in np.linspace(-4, 4, 801):
            c = chi2(nla_model(1., A, k) - gg - truth)
            if best is None or c < best[0]:
                best = (c, A, k)
    _, A0, k0 = best
    k0 = int(np.clip(k0, 1, len(eta_n) - 2))
    h_eta = eta_n[k0 + 1] - eta_n[k0]
    d_nla = {'a': 2*gg + A0*nla_gi[k0], 'A': nla_gi[k0] + 2*A0*nla_ii[k0],
             'eta': (nla_model(1., A0, k0 + 1) - nla_model(1., A0, k0 - 1))/(2*h_eta)}

    # halo pieces
    bases = {}
    for f_c, f_sat in itertools.product(base.FC_NODES, base.FSAT_NODES):
        path = v.WORK/base.SURVEY/f'model_hod_sim_fc_{f_c}_fsat_{f_sat}_prior.npz'
        if path.exists():
            bases[(f_c, f_sat)] = base.HaloBasis(path, g_m)
    centre = bases[(FC0, FSAT0)]
    eta_c_n, eta_s_n = centre.eta_c, centre.eta_s

    def halo_ia(B, A_c, A_s, kc, ks, a=1.):
        """IA part with the lensing amplitude a on the GI terms."""
        # split GI (linear in the amplitudes) from II (quadratic): evaluate at +-(A_c, A_s)
        full = B.predict(np.array([A_c]), np.array([A_s]), kc, ks)[0]
        minus = B.predict(np.array([-A_c]), np.array([-A_s]), kc, ks)[0]
        gi = 0.5*(full - minus)
        ii = 0.5*(full + minus)
        return a*gi + ii

    # halo fiducial: best (A_c, A_s) for each (eta_c, eta_s) node at the centre population
    best = None
    for kc in range(len(eta_c_n)):
        for ks in range(len(eta_s_n)):
            for A_c in np.linspace(0, 4, 81):
                for A_s in (0., 0.01, 0.02, 0.03, 0.05):
                    c = chi2(halo_ia(centre, A_c, A_s, kc, ks) - truth)
                    if best is None or c < best[0]:
                        best = (c, A_c, A_s, kc, ks)
    chi2_halo, Ac0, As0, kc0, ks0 = best
    kc0 = int(np.clip(kc0, 1, len(eta_c_n) - 2))
    ks0 = int(np.clip(ks0, 1, len(eta_s_n) - 2))
    eps = 1e-3

    def hm(B=centre, A_c=Ac0, A_s=As0, kc=kc0, ks=ks0, a=1.):
        return a**2*gg + halo_ia(B, A_c, A_s, kc, ks, a)

    d_halo = {
        'a': (hm(a=1 + eps) - hm(a=1 - eps))/(2*eps),
        'A_c': (hm(A_c=Ac0 + eps) - hm(A_c=Ac0 - eps))/(2*eps),
        'A_s': (hm(A_s=As0 + eps) - hm(A_s=As0 - eps))/(2*eps),
        'eta_c': (hm(kc=kc0 + 1) - hm(kc=kc0 - 1))/(eta_c_n[kc0 + 1] - eta_c_n[kc0 - 1]),
        'eta_s': (hm(ks=ks0 + 1) - hm(ks=ks0 - 1))/(eta_s_n[ks0 + 1] - eta_s_n[ks0 - 1]),
        'f_c': (hm(B=bases[(0.7, FSAT0)]) - hm(B=bases[(0.2, FSAT0)]))/0.5,
        'f_sat': (hm(B=bases[(FC0, 0.8)]) - hm(B=bases[(FC0, 0.3)]))/0.5,
    }

    def sigma_a(derivs, names):
        D = np.array([flat(derivs[n]) for n in names])
        F = (D*icov) @ D.T
        for i, n in enumerate(names):
            if n in PRIOR_WIDTH:
                F[i, i] += 12./PRIOR_WIDTH[n]**2
        C = np.linalg.inv(F)
        return float(np.sqrt(C[0, 0])), {n: float(np.sqrt(C[i, i])) for i, n in enumerate(names)}

    out = {'fiducial_nla': dict(A=float(A0), eta=float(eta_n[k0])),
           'fiducial_halo': dict(A_c=float(Ac0), A_s=float(As0), eta_c=float(eta_c_n[kc0]), eta_s=float(eta_s_n[ks0]),
                                 f_c=FC0, f_sat=FSAT0, chi2_vs_truth=float(chi2_halo)),
           'n_data': int(icov.size), 'prior_widths': PRIOR_WIDTH}
    out['fiducial_nla']['chi2_vs_truth'] = chi2(nla_model(1., A0, k0) - gg - truth)
    cases = {'a only (no IA freedom)': (d_halo, ['a']),
             'NLA (A, eta)': (d_nla, ['a', 'A', 'eta']),
             'halo marks (A_c, eta_c, A_s, eta_s)': (d_halo, ['a', 'A_c', 'eta_c', 'A_s', 'eta_s']),
             'halo full (+ f_c, f_sat)': (d_halo, ['a', 'A_c', 'eta_c', 'A_s', 'eta_s', 'f_c', 'f_sat']),
             'halo A_c, eta_c only': (d_halo, ['a', 'A_c', 'eta_c'])}
    for name, (derivs, names) in cases.items():
        s, all_s = sigma_a(derivs, names)
        out[name] = dict(sigma_a=s, sigmas=all_s)
        print(f'{name:40s} sigma(a) = {s:.4f}   ', {k: round(x, 4) for k, x in all_s.items()}, flush=True)
    print('fiducials', out['fiducial_nla'], out['fiducial_halo'], flush=True)
    # bias of a when the truth is TNG300 (a = 1): one Gauss-Newton step from each model's best fit
    data = gg + truth
    for name, derivs, names, model in (('NLA (A, eta)', d_nla, ['a', 'A', 'eta'], nla_model(1., A0, k0)),
                                       ('halo full (+ f_c, f_sat)', d_halo,
                                        ['a', 'A_c', 'eta_c', 'A_s', 'eta_s', 'f_c', 'f_sat'], hm())):
        D = np.array([flat(derivs[n]) for n in names])
        F = (D*icov) @ D.T
        shift = np.linalg.solve(F, (D*icov) @ flat(data - model))
        out[name]['bias_a_vs_TNG'] = float(shift[0])
        out[name]['bias_a_over_sigma'] = float(shift[0]/out[name]['sigma_a'])
        print(f'{name}: bias in a when the truth is TNG300 = {shift[0]:+.4f} ({shift[0]/out[name]["sigma_a"]:+.2f} sigma)', flush=True)
    (base.OUT/'fisher_ia.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
