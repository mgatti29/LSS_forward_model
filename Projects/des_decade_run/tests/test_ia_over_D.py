"""Halo IA divided by the source-clustering density (ia_over_D, default since 2026-11-23).

lensing.make_WL_sample and bfd_utils.make_BFD_sample(estimator='count'):
  legacy   ia_over_D=False is bit-identical to the 2026-11-23 backup (previous term I), with and without the cap;
  I/D      IA term of the signal map = s * I / D (D = sum_i n_i (1 + b_i delta_i) floored at 1e-3, per simulation
           pixel; BFD: then projected to the catalogue frame), expected self term var_IA = 0.5 numerator_sq / D^2,
           cap s = min(1, sqrt(var_noise / var_IA)), residual noise factor^2 = 1 - s^2 var_IA / var_noise.
WL: synthetic catalogue, NSIDE 32, source clustering on (inverse_density noise model, unit corrections), D floor hit.
BFD: real N3 catalogue bin 3, NSIDE 1024, synthetic fields (test_bfd_count.fake_inputs), SC v2 Poisson.
Writes tests/out/test_ia_over_D.json.  Usage: python tests/test_ia_over_D.py [wl|bfd|all]
"""
import copy
import json
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
from LSS_forward_model import bfd_utils as new_bfd                 # noqa: E402
from LSS_forward_model import lensing as new_wl                    # noqa: E402
from LSS_forward_model.source_clustering import shell_bias         # noqa: E402

BACKUP = Path('/pscratch/sd/m/mgatti/des_decade_test/code_backup_20261123')
res = {}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_backup(name):
    src = (BACKUP/f'{name}.py').read_text().replace('from .', 'from LSS_forward_model.')
    path = HERE/'out'/f'_backup_{name}_20261123.py'
    path.write_text(src)
    import importlib.util
    spec = importlib.util.spec_from_file_location(f'_backup_{name}_20261123', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def same(a, b):
    return all(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind in 'fc') for k in a
               if isinstance(a[k], np.ndarray))


def test_wl():
    legacy = load_backup('lensing')
    nside, npix = 32, hp.nside2npix(32)
    rng = np.random.default_rng(3)
    nshell = 3
    fields = {'density': np.clip(rng.normal(0, 0.5, (nshell, npix)), -1, None),
              'gamma': rng.normal(0, 0.02, (nshell, 1, npix)) + 1j*rng.normal(0, 0.02, (nshell, 1, npix))}
    ngal = np.array([[0.2, 0.5, 0.3], [0.1, 0.3, 0.6]])
    zeff = np.linspace(0.3, 1.0, nshell)
    n = 60000
    ra, dec = rng.uniform(0, 90, n), np.degrees(np.arcsin(rng.uniform(0, 1, n)))
    cats = {t: dict(ra=ra[t::2], dec=dec[t::2], e1=rng.normal(0, 0.26, n//2), e2=rng.normal(0, 0.26, n//2),
                    w=rng.uniform(0.5, 1.5, n//2)) for t in range(2)}
    num = 0.01*(rng.normal(size=(2, npix)) + 1j*rng.normal(size=(2, npix)))
    nsq = np.abs(num)**2*rng.uniform(1, 30, (2, npix))
    halo = {'numerator': num, 'numerator_sq': nsq}
    zero = {'numerator': np.zeros_like(num), 'numerator_sq': np.zeros_like(nsq)}
    SC = {'corr_variance_fit': [lambda b: 1.0]*2, 'coeff_kurtosis_fit': [lambda b: 0.0]*2,
          'A_corr_fit': [lambda b: 1.0]*2}
    sp = dict(dm=np.array([1.01, 0.98]), rot=0, delta_rot=0, bias_sc=np.array([1.6, 1.2]),
              bias_sc_slope=np.zeros(2))
    kw = dict(include_SC=True, SC_corrections=SC, noise_model='inverse_density', IA_model='halo', do_catalog=True)

    def run(mod, halo_, **extra):
        np.random.seed(11)
        return mod.make_WL_sample(ngal, zeff, {}, copy.deepcopy(sp), nside, fields, cats, halo_IA=halo_,
                                  **kw, **extra)

    r = res['wl'] = {}
    for label, extra in [('cap', {}), ('nocap', dict(ia_cap_alpha=None))]:
        old, cold = run(legacy, halo, **extra)
        cur, ccur = run(new_wl, halo, ia_over_D=False, **extra)
        r[f'legacy_bit_identical_{label}'] = bool(all(same(old[t], cur[t]) and same(cold[t], ccur[t]) for t in (0, 1)))
        log('WL', label, 'ia_over_D=False bit-identical:', r[f'legacy_bit_identical_{label}'])

    # independent D (simulation pixels; the catalogue pixels are where the maps live)
    bias = shell_bias(ngal, zeff, sp['bias_sc'], sp['bias_sc_slope'])
    D = np.array([sum(ngal[t, i]*(1 + bias[t, i]*fields['density'][i]) for i in range(nshell)) for t in (0, 1)])
    S, _ = run(new_wl, zero)
    new, cnew = run(new_wl, halo)
    dflt_eq = None
    for t in (0, 1):
        q = r[f'bin{t}'] = {}
        m = new[t]
        mask = m['var_'] > 0
        Df = np.maximum(D[t], 1e-3)
        q['pixels_D_floored'] = int(np.sum(mask & (D[t] < 1e-3)))
        s = m['ia_cap_factor']
        for k, comp in (('g1_map', num[t].real), ('g2_map', num[t].imag)):
            q[f'{k}_IA_minus_sI_over_D_maxabs'] = float(np.max(np.abs(m[k] - S[t][k] - s*comp/Df)[mask]))
        q['var_IA_minus_half_nsq_over_D2_maxrel'] = float(np.max(np.abs(m['ia_var_IA'] - 0.5*nsq[t]/Df**2)[mask]
                                                               / (0.5*nsq[t]/Df**2)[mask]))
        vn, vi = m['ia_var_noise'], m['ia_var_IA']
        ok = mask & (vn > 0)
        q['cap_budget_ok'] = bool(np.all(s[ok]**2*vi[ok] <= vn[ok]*(1 + 1e-12)))
        q['noise_factor_identity_maxabs'] = float(np.max(np.abs(m['ia_noise_factor'][ok]**2 - (1 - s[ok]**2*vi[ok]/vn[ok]))))
        q['frac_capped'] = float(np.mean(s[mask] < 1))
        # (galaxies with D <= 0 have NaN noise from the existing sqrt(D) noise scale, unrelated to the IA term)
        pix = new_wl.rotate_and_rebin(new_wl.convert_to_pix_coord(cats[t]['ra'], cats[t]['dec'], nside=2*nside),
                                      nside, 0, delta_=0)
        pos = D[t][pix] > 0
        q['catalogue_finite_D_pos'] = bool(np.all(np.isfinite(cnew[t]['e1'][pos])) and np.all(np.isfinite(cnew[t]['e2'][pos])))
        q['var_IA_over_D_vs_I_mean'] = float(np.mean(vi[mask])/np.mean(0.5*nsq[t][mask]))
        log('WL bin', t, json.dumps(q))
    expl, _ = run(new_wl, halo, ia_over_D=True)
    r['default_is_over_D'] = bool(all(same(new[t], expl[t]) for t in (0, 1)))
    # without source clustering D = sum n_i = 1: I/D == I
    kw_nosc = dict(kw, include_SC=False)
    np.random.seed(11)
    a, _ = new_wl.make_WL_sample(ngal, zeff, {}, copy.deepcopy(sp), nside, fields, cats, halo_IA=halo,
                                 ia_over_D=True, **kw_nosc)
    np.random.seed(11)
    b, _ = new_wl.make_WL_sample(ngal, zeff, {}, copy.deepcopy(sp), nside, fields, cats, halo_IA=halo,
                                 ia_over_D=False, **kw_nosc)
    r['no_SC_over_D_equals_I_maxabs'] = float(max(np.max(np.abs(a[t]['e1'] - b[t]['e1'])) for t in (0, 1)))
    log('WL default True:', r['default_is_over_D'], 'no-SC I/D - I:', r['no_SC_over_D_equals_I_maxabs'])


def test_bfd():
    import test_bfd_count as tbc
    legacy = load_backup('bfd_utils')
    stage = tbc.STAGE
    NSIDE, NPIX = tbc.NSIDE, tbc.NPIX
    meta = json.loads((stage/'meta_BFD_N3.json').read_text())
    masks = np.load(stage/'mask_BFD_N3.npz')
    log('loading N3 catalogue')
    t = 3
    cat = np.load(stage/'cat_BFD_N3.npy', allow_pickle=True).item()[t]
    Rbar = np.array(meta['bins'][str(t)]['Rbar_code_sign'])
    rng = np.random.default_rng(200)
    fields, ngal, zeff, halo = tbc.fake_inputs(t, rng, cat)
    zero = {'numerator': [np.zeros(NPIX, complex)], 'numerator_sq': [np.zeros(NPIX)]}
    lam = np.zeros(NPIX)
    lam[masks[f'pix_{t}']] = masks[f'lam_{t}']

    def run(mod, halo_, **extra):
        p = tbc.params()
        maps, _ = mod.make_BFD_sample(ngal, zeff, None, p, NSIDE, fields, {0: cat}, include_SC=True, do_noise=True,
                                      decompose_shear=True, rng=np.random.default_rng(7), lam_maps=[lam],
                                      estimator='count', Rbar=[Rbar], nmin=3, IA_model='halo', halo_IA=halo_,
                                      **extra)
        return maps[0]

    r = res['bfd_bin3'] = {}
    old = run(legacy, halo)
    cur = run(new_bfd, halo, ia_over_D=False)
    r['legacy_bit_identical'] = bool(same(old, cur))
    log('BFD ia_over_D=False bit-identical:', r['legacy_bit_identical'])
    del old, cur
    S = run(new_bfd, zero)
    m = run(new_bfd, halo)
    p = tbc.params()
    bias = shell_bias(ngal, zeff, p['bias_sc'], p['bias_sc_slope'])
    D = sum(ngal[0, i]*(1 + bias[0, i]*fields['density'][i]) for i in range(len(zeff)))
    Df = np.maximum(D, 1e-3)
    pix_cat, pix_sim = m['pix_cat'], m['pix_sim']
    cnt = np.bincount(pix_cat, minlength=NPIX).astype(float)

    def proj(x):                                    # catalogue-pixel mean of the sim-pixel values (rot 1: no mirror)
        out = np.bincount(pix_cat, weights=x[pix_sim], minlength=NPIX)
        return np.divide(out, cnt, out=np.zeros(NPIX), where=cnt > 0)
    M = m['mask']
    s = m['ia_cap_factor']
    num, nsq = halo['numerator'][0], halo['numerator_sq'][0]
    for k, comp in (('g1_map', num.real), ('g2_map', num.imag)):
        r[f'{k}_IA_minus_sI_over_D_maxabs'] = float(np.max(np.abs(m[k] - S[k] - s*proj(comp/Df))[M]))
    exp_vi = 0.5*proj(nsq/Df**2)
    r['var_IA_minus_half_nsq_over_D2_maxrel'] = float(np.max(np.abs(m['ia_var_IA'] - exp_vi)[M]/exp_vi[M]))
    vn, vi = m['ia_var_noise'], m['ia_var_IA']
    ok = M & (vn > 0)
    r['pixels_zero_budget'] = int(np.sum(M & ~(vn > 0)))
    r['cap_budget_ok'] = bool(np.all(s[ok]**2*vi[ok] <= vn[ok]*(1 + 1e-12)))
    r['noise_factor_identity_maxabs'] = float(np.max(np.abs(m['ia_noise_factor'][ok]**2 - (1 - s[ok]**2*vi[ok]/vn[ok]))))
    v = m['var_noise_data']
    tot = m['q_noise_scale_map']**2*v + s**2*vi
    r['white_total_over_budget_pixel_maxdev'] = float(np.max(np.abs(tot[ok]/vn[ok] - 1)))
    r['unit_response_max_abs'] = float(np.max(np.abs(m['e1'] - m['e1_base'] - 1.02*m['g1_map'])[M]))
    r['frac_capped'] = float(np.mean(s[M] < 1))
    r['var_IA_over_D_vs_I_mean'] = float(np.mean(vi[M])/np.mean(0.5*proj(nsq)[M]))
    log('BFD', json.dumps(r))


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('wl', 'all'):
        test_wl()
    if what in ('bfd', 'all'):
        test_bfd()
    out = HERE/'out'/f'test_ia_over_D_{what}.json'
    out.write_text(json.dumps(res, indent=1) + '\n')
    log('wrote', out)
