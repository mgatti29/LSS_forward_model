"""Pure NLA (nla_over_D=False, default since 2026-11-23) in lensing.make_WL_sample and bfd_utils.make_BFD_sample.

  legacy  nla_over_D=True is bit-identical to the 2026-11-23 backup (previous IA term sum n IA / D);
  pure    the NLA contribution e(A=1) - e(A=0) (same seeds) does not depend on the source-clustering bias:
          identical for b = 1.6 and b = 0 (D = 1), maps and (WL) catalogue; the previous term does depend on b
          (reported as the size of the removed -b delta IA term).
WL: synthetic catalogue NSIDE 32, inverse_density SC with unit corrections.  BFD: real N3 bin 3, count estimator,
NSIDE 1024, SC v2 Poisson.  Writes tests/out/test_nla_pure_<part>.json.  Usage: python tests/test_nla_pure.py [wl|bfd|all]
"""
import copy
import json
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np
import pyccl as ccl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
from LSS_forward_model import bfd_utils as new_bfd                 # noqa: E402
from LSS_forward_model import lensing as new_wl                    # noqa: E402
from test_ia_over_D import load_backup, same                       # noqa: E402

COSMO = {'cosmo_pyccl': ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)}
res = {}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def nla_params(A, b, nt):
    return dict(Omega_m=0.3, A_IA=A, eta_IA=0.5, A_IA2=0., bias_TA=np.zeros(nt), bias_sc=np.full(nt, b),
                bias_sc_slope=np.zeros(nt))


def test_wl():
    legacy = load_backup('lensing')
    nside, npix = 32, hp.nside2npix(32)
    rng = np.random.default_rng(5)
    nshell = 3
    fields = {'density': np.clip(rng.normal(0, 0.3, (nshell, npix)), -0.6, None),
              'gamma': rng.normal(0, 0.02, (nshell, 1, npix)) + 1j*rng.normal(0, 0.02, (nshell, 1, npix)),
              'IA_shear': rng.normal(0, 0.02, (nshell, 1, npix)) + 1j*rng.normal(0, 0.02, (nshell, 1, npix))}
    ngal = np.array([[0.2, 0.5, 0.3], [0.1, 0.3, 0.6]])
    zeff = np.linspace(0.3, 1.0, nshell)
    n = 60000
    ra, dec = rng.uniform(0, 90, n), np.degrees(np.arcsin(rng.uniform(0, 1, n)))
    cats = {t: dict(ra=ra[t::2], dec=dec[t::2], e1=rng.normal(0, 0.26, n//2), e2=rng.normal(0, 0.26, n//2),
                    w=rng.uniform(0.5, 1.5, n//2)) for t in range(2)}
    SC = {'corr_variance_fit': [lambda b: 1.0]*2, 'coeff_kurtosis_fit': [lambda b: 0.0]*2,
          'A_corr_fit': [lambda b: 1.0]*2}
    kw = dict(include_SC=True, SC_corrections=SC, noise_model='inverse_density', IA_model='NLA', do_catalog=True)

    def run(mod, A, b, **extra):
        sp = dict(dm=np.array([1.01, 0.98]), rot=0, delta_rot=0, **nla_params(A, b, 2))
        np.random.seed(11)
        return mod.make_WL_sample(ngal, zeff, COSMO, sp, nside, fields, cats, **kw, **extra)

    r = res['wl'] = {}
    old, cold = run(legacy, 1.0, 1.6)
    cur, ccur = run(new_wl, 1.0, 1.6, nla_over_D=True)
    r['legacy_bit_identical'] = bool(all(same(old[t], cur[t]) and same(cold[t], ccur[t]) for t in (0, 1)))
    log('WL nla_over_D=True bit-identical:', r['legacy_bit_identical'])
    for label, extra in (('pure', {}), ('old', dict(nla_over_D=True))):
        m1, c1 = run(new_wl, 1.0, 1.6, **extra)
        m0, c0 = run(new_wl, 0.0, 1.6, **extra)
        n1, d1 = run(new_wl, 1.0, 0.0, **extra)
        n0, d0 = run(new_wl, 0.0, 0.0, **extra)
        dev, rel, cdev = 0., 0., 0.
        for t in (0, 1):
            mask = m1[t]['var_'] > 0
            for k in ('g1_map', 'g2_map', 'e1', 'e2'):
                a = (m1[t][k] - m0[t][k])[mask]
                b = (n1[t][k] - n0[t][k])[mask]
                dev = max(dev, float(np.max(np.abs(a - b))))
                rel = max(rel, float(np.sqrt(np.mean((a - b)**2)/np.mean(b**2))))
            for k in ('e1', 'e2'):
                a = c1[t][k] - c0[t][k]
                b = d1[t][k] - d0[t][k]
                ok = np.isfinite(a) & np.isfinite(b)
                cdev = max(cdev, float(np.max(np.abs(a - b)[ok])))
        r[label] = dict(IA_term_b16_minus_b0_maxabs=dev, IA_term_b_dependence_rms_rel=rel, catalogue_maxabs=cdev)
        log('WL', label, json.dumps(r[label]))
    dflt, _ = run(new_wl, 1.0, 1.6)
    expl, _ = run(new_wl, 1.0, 1.6, nla_over_D=False)
    r['default_is_pure'] = bool(all(same(dflt[t], expl[t]) for t in (0, 1)))
    log('WL default pure:', r['default_is_pure'])


def test_bfd():
    import test_bfd_count as tbc
    legacy = load_backup('bfd_utils')
    stage, NSIDE, NPIX = tbc.STAGE, tbc.NSIDE, tbc.NPIX
    meta = json.loads((stage/'meta_BFD_N3.json').read_text())
    masks = np.load(stage/'mask_BFD_N3.npz')
    log('loading N3 catalogue')
    t = 3
    cat = np.load(stage/'cat_BFD_N3.npy', allow_pickle=True).item()[t]
    Rbar = np.array(meta['bins'][str(t)]['Rbar_code_sign'])
    rng = np.random.default_rng(300)
    fields, ngal, zeff, _ = tbc.fake_inputs(t, rng, cat)
    fields['IA_shear'] = [[0.02*(rng.standard_normal(NPIX) + 1j*rng.standard_normal(NPIX))] for _ in zeff]
    lam = np.zeros(NPIX)
    lam[masks[f'pix_{t}']] = masks[f'lam_{t}']

    def run(mod, A, b, rot=1, **extra):
        p = dict(rot=rot, delta_rot=0, dm=np.array([1.02]), **nla_params(A, b, 1))
        maps, _ = mod.make_BFD_sample(ngal, zeff, COSMO, p, NSIDE, fields, {0: cat}, include_SC=True, do_noise=True,
                                      decompose_shear=True, rng=np.random.default_rng(7), lam_maps=[lam],
                                      estimator='count', Rbar=[Rbar], nmin=3, IA_model='NLA', **extra)
        return maps[0]

    r = res['bfd_bin3'] = {}
    old = run(legacy, 1.0, 1.6)
    cur = run(new_bfd, 1.0, 1.6, nla_over_D=True)
    r['legacy_bit_identical'] = bool(same(old, cur))
    log('BFD nla_over_D=True bit-identical:', r['legacy_bit_identical'])
    del old, cur
    for rot in (1, 2):                                   # rot 2 mirrors the sky (e2 sign)
        for label, extra in (('pure', {}), ('old', dict(nla_over_D=True))):
            m1, m0 = run(new_bfd, 1.0, 1.6, rot, **extra), run(new_bfd, 0.0, 1.6, rot, **extra)
            n1, n0 = run(new_bfd, 1.0, 0.0, rot, **extra), run(new_bfd, 0.0, 0.0, rot, **extra)
            M = m1['mask']
            q = {}
            for k in ('g1_map', 'g2_map', 'e1', 'e2'):
                a, b = (m1[k] - m0[k])[M], (n1[k] - n0[k])[M]
                q[f'{k}_maxabs'] = float(np.max(np.abs(a - b)))
                q[f'{k}_rms_rel'] = float(np.sqrt(np.mean((a - b)**2)/np.mean(b**2)))
            r[f'{label}_rot{rot}'] = q
            log('BFD', label, 'rot', rot, json.dumps(q))
            del m1, m0, n1, n0


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('wl', 'all'):
        test_wl()
    if what in ('bfd', 'all'):
        test_bfd()
    out = HERE/'out'/f'test_nla_pure_{what}.json'
    out.write_text(json.dumps(res, indent=1) + '\n')
    log('wrote', out)
