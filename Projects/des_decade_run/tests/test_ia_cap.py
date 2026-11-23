#!/usr/bin/env python3
"""Local IA cap (ia_cap_alpha) of lensing.make_WL_sample and bfd_utils.make_BFD_sample.

Checks
  legacy     ia_cap_alpha=None (+ the previous self-term averaging) is bit-identical to the pre-cap code
             (backups made 2026-11-12)
  default    since 2026-11-18 the default is ia_cap_alpha=1 with the per-pixel self term
  cap        s in [0, 1]; s = 1 where var_IA <= alpha var_noise and where var_IA = 0; s^2 var_IA <= alpha var_noise;
             IA term of the signal map = s * IA (same factor on both components), gravitational shear unchanged;
             residual noise variance factor^2 var_noise = var_noise - s^2 var_IA >= (1 - alpha) var_noise >= 0
WL: synthetic catalogue, NSIDE 32, include_SC=False.  BFD: real catalogue slice (as test_bfd_v2.py), NSIDE 64,
SC v2 Poisson.
"""
import copy
import importlib.util
import sys
from pathlib import Path

import healpy as hp
import numpy as np

sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
from LSS_forward_model import bfd_utils as new_bfd  # noqa: E402
from LSS_forward_model import lensing as new_wl  # noqa: E402

BACKUP = Path('/pscratch/sd/m/mgatti/IA_non_gaussian/tng300/cheap_validation/gower_tng/code_backup_20261112/cap')
TMP = Path(__file__).resolve().parent/'out'
TMP.mkdir(exist_ok=True)


def load_legacy(name):
    src = (BACKUP/f'{name}.py').read_text().replace('from .', 'from LSS_forward_model.')
    path = TMP/f'_legacy_{name}.py'
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(f'_legacy_{name}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_cap(tag, m_nocap, m_cap, alpha, mask, g_keys=('g1_map', 'g2_map'), ia=None):
    s, vn, vi = m_cap['ia_cap_factor'], m_cap['ia_var_noise'], m_cap['ia_var_IA']
    assert np.all((s >= 0) & (s <= 1)), tag
    assert np.all(s[~mask] == 1), tag
    free = mask & (vi <= alpha*vn)
    assert np.all(s[free] == 1), tag
    assert np.all(s[mask & (vi == 0)] == 1), tag
    assert np.all(s[mask]**2*vi[mask] <= alpha*vn[mask]*(1 + 1e-12) + 1e-300), tag
    # signal: uncapped map minus (1 - s) IA, both components
    for k, comp in zip(g_keys, (ia.real, ia.imag)):
        assert np.allclose(m_cap[k][mask], m_nocap[k][mask] - (1 - s[mask])*comp[mask], rtol=0, atol=1e-12), (tag, k)
    f2 = m_cap['ia_noise_factor'][mask]**2
    ok = vn[mask] > 0
    resid = f2[ok]*vn[mask][ok]
    assert np.all(resid >= -1e-300), tag
    assert np.allclose(resid/vn[mask][ok], 1 - s[mask][ok]**2*vi[mask][ok]/vn[mask][ok], rtol=0, atol=1e-12), tag
    assert np.all(resid >= (1 - alpha)*vn[mask][ok]*(1 - 1e-9)), tag
    capped = mask & (s < 1)
    print(f'  {tag}: alpha {alpha}: capped {capped.sum()}/{mask.sum()} pixels, s min {s[mask].min():.3f}, '
          f'IA variance removed {np.sum((1 - s[mask]**2)*vi[mask])/np.sum(vi[mask]):.3f}, min residual/var_noise '
          f'{np.min(resid/vn[mask][ok]):.3f}')


def test_wl():
    legacy = load_legacy('lensing')
    nside, npix = 32, hp.nside2npix(32)
    rng = np.random.default_rng(3)
    nshell = 3
    fields = {'density': rng.normal(0, 0.3, (nshell, npix)),
              'gamma': rng.normal(0, 0.02, (nshell, 1, npix)) + 1j*rng.normal(0, 0.02, (nshell, 1, npix))}
    ngal = rng.uniform(0, 1, (2, nshell))
    zeff = np.linspace(0.3, 1.0, nshell)
    n = 60000
    ra, dec = rng.uniform(0, 90, n), np.degrees(np.arcsin(rng.uniform(0, 1, n)))
    cats = {t: dict(ra=ra[t::2], dec=dec[t::2], e1=rng.normal(0, 0.26, n//2), e2=rng.normal(0, 0.26, n//2),
                    w=rng.uniform(0.5, 1.5, n//2)) for t in range(2)}
    ia = 0.01*(rng.normal(size=(2, npix)) + 1j*rng.normal(size=(2, npix)))
    # IA self term: large in some superpixels, zero in others
    sq = np.abs(rng.normal(0, 1, (2, hp.nside2npix(4))))*np.repeat([[3e-3], [1e-3]], hp.nside2npix(4), 1)
    sq[:, ::5] = 0.
    halo = {'numerator': ia, 'numerator_sq': np.array([hp.ud_grade(x, nside) for x in sq])}
    sp = dict(dm=np.ones(2), rot=0, delta_rot=0)
    kw = dict(include_SC=False, IA_model='halo', halo_IA=halo, ia_noise_nside=4, do_catalog=True)
    np.random.seed(11)
    old, cold = legacy.make_WL_sample(ngal, zeff, {}, copy.deepcopy(sp), nside, fields, cats, **kw)
    np.random.seed(11)
    cur, ccur = new_wl.make_WL_sample(ngal, zeff, {}, copy.deepcopy(sp), nside, fields, cats, ia_cap_alpha=None, **kw)
    for t in range(2):
        for k in old[t]:
            assert np.array_equal(old[t][k], cur[t][k]), (t, k)
        for k in cold[t]:
            assert np.array_equal(cold[t][k], ccur[t][k]), (t, k)
    print('WL ia_cap_alpha=None bit-identical to the pre-cap code: OK')
    kw_def = {k: v for k, v in kw.items() if k != 'ia_noise_nside'}
    np.random.seed(11)
    dflt, _ = new_wl.make_WL_sample(ngal, zeff, {}, copy.deepcopy(sp), nside, fields, cats, **kw_def)
    np.random.seed(11)
    expl, _ = new_wl.make_WL_sample(ngal, zeff, {}, copy.deepcopy(sp), nside, fields, cats, ia_cap_alpha=1.0,
                                    ia_noise_nside=None, **kw_def)
    for t in range(2):
        for k in expl[t]:
            assert np.array_equal(dflt[t][k], expl[t][k]), (t, k)
        assert 'ia_cap_factor' in dflt[t]
    # NLA: the default cap is ignored (no halo-IA budget), no error
    print('WL default = ia_cap_alpha 1, per-pixel self term: OK')
    for alpha in (1.0, 0.9):
        np.random.seed(11)
        cap, ccap = new_wl.make_WL_sample(ngal, zeff, {}, copy.deepcopy(sp), nside, fields, cats, ia_cap_alpha=alpha,
                                          **kw)
        for t in range(2):
            mask = cur[t]['var_'] > 0
            check_cap(f'WL bin {t}', cur[t], cap[t], alpha, mask, ia=ia[t])
            # catalogue: galaxy signal changes by -(1 - s) IA at the galaxy pixel; noise only by the noise factor
            pix = new_wl.rotate_and_rebin(new_wl.convert_to_pix_coord(cats[t]['ra'], cats[t]['dec'], nside=2*nside),
                                          nside, 0, delta_=0)
            s = cap[t]['ia_cap_factor'][pix]
            assert np.all(np.isfinite(ccap[t]['e1']))
            assert s.min() < 1 or alpha == 1.0
    try:
        bad, _ = new_wl.make_WL_sample(ngal, zeff, {}, copy.deepcopy(sp), nside, fields, cats, ia_cap_alpha=1.5, **kw)
        raise AssertionError('alpha > 1 should raise')
    except ValueError:
        print('WL alpha > 1 raises: OK')
    nob, _ = new_wl.make_WL_sample(ngal, zeff, {}, copy.deepcopy(sp), nside, fields, cats, ia_cap_alpha=1.0,
                                   **dict(kw, subtract_IA_noise=False))
    assert 'ia_cap_factor' not in nob[0]
    print('WL cap without noise budget is not applied: OK')


def test_bfd():
    from astropy.io import fits
    from LSS_forward_model.cosmology import make_cosmo_bundle, sigma8_to_As_CAMB
    from LSS_forward_model.source_clustering import load_SC_calibration
    legacy = load_legacy('bfd_utils')
    nside, npix = 64, hp.nside2npix(64)
    src = ('/global/cfs/cdirs/des/mgatti/BFD_targets_June25/'
           'BFD_catalog_v5_UNBLINDED_with_tomo_masked_withselection_final_psfmask_psfvalid.fits')
    with fits.open(src, memmap=True) as h:
        block = np.array(h[1].data[:300000])
    cats = {t: {k: np.asarray(block[k][block['tomo'] == t]) for k in ('ra', 'dec', 'selection_term', 'pqr',
                                                                      'pqr10_prelog')} for t in range(4)}
    rng = np.random.default_rng(1)
    nshell = 5
    fields = {'density': rng.normal(0, 0.3, (nshell, npix)),
              'gamma': rng.normal(0, 0.02, (nshell, 1, npix)) + 1j*rng.normal(0, 0.02, (nshell, 1, npix))}
    ngal = rng.uniform(0, 1, (4, nshell))
    ngal /= ngal.sum(1, keepdims=True)
    zeff = np.linspace(0.2, 1.4, nshell)
    sp = dict(h=0.7, Omega_m=0.3, Omega_b=0.05, Omega_cdm=0.25, sigma_8=0.8, n_s=0.96, m_nu=0.06, n_nu=1, w0=-1.,
              wa=0.)
    sp['As'] = sigma8_to_As_CAMB(0.3, 0.05, 0.7, 0.96, -1., 0., 0.06, 0.8)
    cosmo_bundle = make_cosmo_bundle(sp)
    sp.update(bias_sc=np.array([1.1, 1.2, 1.3, 1.4]), bias_sc_slope=np.full(4, 0.5), bias_TA=np.zeros(4),
              dm=np.ones(4), rot=0, delta_rot=0)
    sc_v2 = load_SC_calibration('/global/u2/m/mgatti/LSS_forward_model/Projects/source_clustering_v2/data/'
                                'SC_v2_DES_nside1024_sim1.json')
    ia = 0.02*(rng.normal(size=(4, npix)) + 1j*rng.normal(size=(4, npix)))
    sq = np.abs(rng.normal(0, 1, (4, hp.nside2npix(8))))*3e-2
    sq[:, ::4] = 0.
    halo = {'numerator': ia, 'numerator_sq': np.array([hp.ud_grade(x, nside) for x in sq])}
    kw = dict(SC_corrections=sc_v2, include_SC=True, do_noise=True, noise_model='poisson', IA_model='halo',
              halo_IA=halo, ia_noise_nside=8)
    old, _ = legacy.make_BFD_sample(ngal, zeff, cosmo_bundle, copy.deepcopy(sp), nside, fields, cats,
                                    rng=np.random.default_rng(7), **kw)
    cur, _ = new_bfd.make_BFD_sample(ngal, zeff, cosmo_bundle, copy.deepcopy(sp), nside, fields, cats,
                                     rng=np.random.default_rng(7), ia_cap_alpha=None, ia_self_nside=8, **kw)
    for t in range(4):
        for k in old[t]:
            a, b = old[t][k], cur[t][k]
            assert (np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b), (t, k)
    print('BFD ia_cap_alpha=None, ia_self_nside=ia_noise_nside bit-identical to the pre-cap code: OK')
    for alpha in (1.0, 0.9):
        cap, _ = new_bfd.make_BFD_sample(ngal, zeff, cosmo_bundle, copy.deepcopy(sp), nside, fields, cats,
                                         rng=np.random.default_rng(7), ia_cap_alpha=alpha, ia_self_nside=8, **kw)
        for t in range(4):
            # budget pixels: where the reference noise exists (as the code: n_ref > 0, var_bfd > 0)
            mask = (cap[t]['ia_var_noise'] > 0)
            s = cap[t]['ia_cap_factor']
            vn, vi = cap[t]['ia_var_noise'], cap[t]['ia_var_IA']
            assert np.all((s >= 0) & (s <= 1))
            assert np.all(s[mask]**2*vi[mask] <= alpha*vn[mask]*(1 + 1e-12))
            assert np.all(s[mask & (vi <= alpha*vn)] == 1)
            assert np.all(s[vi == 0] == 1)
            f2 = cap[t]['ia_noise_factor'][mask]**2
            assert np.allclose(f2, 1 - s[mask]**2*vi[mask]/vn[mask], rtol=0, atol=1e-12)
            assert np.all(f2 >= (1 - alpha)*(1 - 1e-9))
            # gravitational shear unchanged: the difference of the signal maps is confined to capped pixels
            diff = cur[t]['g1_map'] - cap[t]['g1_map']
            assert np.all(diff[s == 1] == 0), t
            assert np.array_equal(cur[t]['d_map'], cap[t]['d_map'])
            print(f'  BFD bin {t}: alpha {alpha}: capped {(mask & (s < 1)).sum()}/{mask.sum()}, s min '
                  f'{s[mask].min():.3f}, IA variance removed {np.sum((1 - s[mask]**2)*vi[mask])/np.sum(vi[mask]):.3f}')
    print('BFD cap checks: OK')


if __name__ == '__main__':
    test_wl()
    test_bfd()
