"""Validation of the count-normalised BFD estimator (bfd_utils.make_bfd_map_count, make_BFD_sample(estimator='count')).

Runs on the real N3 catalogue, bins 0 and 3 (each passed as a one-bin sample), synthetic shell fields.
Checks:
  1. data map: mask == stored N3 mask, N-weighted mean == stored global estimator (meta 'global_shear_retained');
  2. rotated noise: measured per-component variance == predicted V cR / N^2 (global and in N bins);
  3. make_BFD_sample count path: e - e_base == dm gamma exactly (unit response); e1n on the same mask;
     SC normalisation conserves the total noise variance on the mask (no IA); halo IA + cap: s^2 var_IA <= var_noise,
     total white variance (noise + capped IA) conserved per pixel;
  4. legacy 'ml' path bit-identical to the 2026-11-20 backup (subset of rows).
Writes tests/out/test_bfd_count.json.  Usage: python tests/test_bfd_count.py
"""
import json
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from LSS_forward_model import bfd_utils as bu                      # noqa: E402

STAGE = Path('/global/cfs/cdirs/m5099/des_decade_run_inputs')
BACKUP = Path('/pscratch/sd/m/mgatti/des_decade_test/code_backup_20261120/bfd_utils.py')
NSIDE = 1024
NPIX = hp.nside2npix(NSIDE)
res = {}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def legacy_module():
    src = BACKUP.read_text().replace('from .', 'from LSS_forward_model.')
    path = HERE/'out'/'_backup_bfd_utils_20261120.py'
    path.write_text(src)
    sys.path.insert(0, str(path.parent))
    import _backup_bfd_utils_20261120 as legacy
    return legacy


def fake_inputs(t, rng, cat):
    """One-bin sample with 3 shells: smooth Gaussian gamma, density, halo-IA numerator on the rotated footprint."""
    nshell = 3
    lmax = 3 * NSIDE - 1
    cl = np.zeros(lmax + 1)
    ell = np.arange(lmax + 1)
    cl[2:] = 1e-9 * (ell[2:] / 100.0) ** -1.2
    fields = {'gamma': [], 'density': []}
    for i in range(nshell):
        g1, g2 = [hp.synfast(cl, NSIDE, verbose=False) if 'verbose' in hp.synfast.__code__.co_varnames
                  else hp.synfast(cl, NSIDE) for _ in range(2)]
        fields['gamma'].append([g1 + 1j * g2])
        fields['density'].append(np.clip(0.3 * rng.standard_normal(NPIX), -0.9, None))
    ngal = np.array([[0.2, 0.5, 0.3]])
    zeff = np.array([0.3, 0.5, 0.7])
    # halo IA numerator (sim frame): white complex marks, numerator_sq consistent with them
    num = 0.004 * (rng.standard_normal(NPIX) + 1j * rng.standard_normal(NPIX))
    halo = {'numerator': [num], 'numerator_sq': [np.abs(num) ** 2 * rng.uniform(1.0, 3.0, NPIX)]}
    return fields, ngal, zeff, halo


def params(dm=1.02):
    return dict(rot=1, delta_rot=0, bias_sc=np.array([1.3]), bias_sc_slope=np.array([0.5]), dm=np.array([dm]),
                Omega_m=0.3, A_IA=0., eta_IA=0., A_IA2=0., bias_TA=np.zeros(1))


def main():
    meta = json.loads((STAGE/'meta_BFD_N3.json').read_text())
    masks = np.load(STAGE/'mask_BFD_N3.npz')
    log('loading N3 catalogue')
    cats = np.load(STAGE/'cat_BFD_N3.npy', allow_pickle=True).item()
    legacy = legacy_module()
    for t in (0, 3):
        r = res[f'bin{t}'] = {}
        cat = cats[t]
        mb = meta['bins'][str(t)]
        Rbar = np.array(mb['Rbar_code_sign'])
        # 1. data map
        e1, e2, n, mask, V = bu.make_bfd_map_count(cat, Rbar, nside=NSIDE)
        pix_ref = masks[f'pix_{t}']
        r['mask_equal_N3'] = bool(np.array_equal(np.flatnonzero(mask), np.sort(pix_ref)))
        r['n_gal_equal_N3'] = bool(np.array_equal(n[np.sort(pix_ref)], masks[f'n_gal_{t}'][np.argsort(pix_ref)]))
        gmean = [float(np.sum((n * e1)[mask]) / n[mask].sum()), float(np.sum((n * e2)[mask]) / n[mask].sum())]
        r['global_mean_map'] = gmean
        r['global_shear_retained_meta'] = mb['global_shear_retained']
        log(t, 'mask', r['mask_equal_N3'], r['n_gal_equal_N3'], 'mean', gmean, 'meta', mb['global_shear_retained'])
        # 2. rotated noise variance vs prediction
        Rinv = np.linalg.inv(-Rbar)
        cR = 0.5 * np.sum(Rinv ** 2)
        vpred = np.zeros(NPIX)
        vpred[mask] = V[mask] * cR / n[mask] ** 2
        meas = []
        for k in range(4):
            a1, a2, n2, m2, V2 = bu.make_bfd_map_count(cat, Rbar, nside=NSIDE, random_rotate=True,
                                                       rng=np.random.default_rng([t, k]))
            assert np.array_equal(m2, mask) and np.allclose(V2, V, rtol=1e-10, atol=0)
            meas.append(0.5 * (a1 ** 2 + a2 ** 2))
        meas = np.mean(meas, 0)
        r['noise_var_measured_over_pred'] = float(meas[mask].sum() / vpred[mask].sum())
        nb = {}
        for lo, hi in [(3, 6), (6, 12), (12, 25), (25, 50), (50, 10 ** 6)]:
            s = mask & (n >= lo) & (n < hi)
            nb[f'{lo}-{hi}'] = float(meas[s].sum() / vpred[s].sum())
        r['noise_var_ratio_by_N'] = nb
        r['V_rotation_invariant'] = True
        log(t, 'noise var measured/pred', r['noise_var_measured_over_pred'], nb)
        del meas, a1, a2

        # 3. make_BFD_sample, count path
        rng = np.random.default_rng(100 + t)
        fields, ngal, zeff, halo = fake_inputs(t, rng, cat)
        lam = np.zeros(NPIX)
        lam[masks[f'pix_{t}']] = masks[f'lam_{t}']
        one = {0: cat}
        for label, kw in [('nla_sc', dict(IA_model='NLA')),
                          ('halo_cap', dict(IA_model='halo', halo_IA=halo, ia_cap_alpha=1.0))]:
            p = params()
            maps, _ = bu.make_BFD_sample(ngal, zeff, None, p, NSIDE, fields, one, include_SC=True, do_noise=True,
                                         decompose_shear=True, rng=np.random.default_rng(7), lam_maps=[lam],
                                         estimator='count', Rbar=[Rbar], nmin=3, **kw)
            m = maps[0]
            M = m['mask']
            q = r[label] = {}
            q['mask_equal'] = bool(np.array_equal(M, mask) and np.array_equal(m['mask_noise'], mask))
            q['unit_response_max_abs'] = float(max(np.abs(m['e1'] - m['e1_base'] - 1.02 * m['g1_map'])[M].max(),
                                                   np.abs(m['e2'] - m['e2_base'] - 1.02 * m['g2_map'])[M].max()))
            v = m['var_noise_data']
            sc2 = (m['q_noise_scale_map'] / np.where(m['ia_noise_factor'] > 0, m['ia_noise_factor'], 1)) ** 2
            q['sc_cv'] = m['sc_cv']
            if label == 'nla_sc':
                q['sc_total_variance_conserved'] = float(np.sum((sc2 * v)[M]) / np.sum(v[M]))
                # realised: rotated noise map variance vs target
                en = 0.5 * (m['e1n'] ** 2 + m['e2n'] ** 2)
                q['noise_realised_over_target'] = float(en[M].sum() / np.sum((m['q_noise_scale_map'] ** 2 * v)[M]))
            else:
                vn, vi, cap = m['ia_var_noise'], m['ia_var_IA'], m['ia_cap_factor']
                q['frac_capped'] = float(np.mean(cap[M] < 1))
                q['max_capped_ratio'] = float(np.max((cap ** 2 * vi / np.where(vn > 0, vn, np.inf))[M]))
                tot = m['q_noise_scale_map'] ** 2 * v + cap ** 2 * vi
                q['white_total_over_budget_global'] = float(tot[M].sum() / vn[M].sum())
                q['white_total_over_budget_pixel_maxdev'] = float(np.max(np.abs(tot[M] / vn[M] - 1)))
            log(t, label, json.dumps(q))
            del maps, m

        # 4. legacy ml path bit-identical (subset of rows)
        sub = {k: np.asarray(v)[:2_000_000] for k, v in cat.items()}
        p = params()
        a, _ = bu.make_BFD_sample(ngal, zeff, None, params(), NSIDE, fields, {0: sub}, include_SC=False, do_noise=True,
                                  rng=np.random.default_rng(3), IA_model='halo', halo_IA=halo)
        b, _ = legacy.make_BFD_sample(ngal, zeff, None, params(), NSIDE, fields, {0: sub}, include_SC=False,
                                      do_noise=True, rng=np.random.default_rng(3), IA_model='halo', halo_IA=halo)
        r['legacy_ml_bit_identical'] = bool(all(np.array_equal(a[0][k], b[0][k]) for k in ('e1', 'e2', 'e1n', 'e2n', 'mask')))
        log(t, 'legacy bit-identical', r['legacy_ml_bit_identical'])
        del a, b, sub
    (HERE/'out'/'test_bfd_count.json').write_text(json.dumps(res, indent=1) + '\n')
    log('done')


if __name__ == '__main__':
    main()
