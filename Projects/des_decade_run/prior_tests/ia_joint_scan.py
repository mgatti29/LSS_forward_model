#!/usr/bin/env python3
"""Joint halo-IA variance-budget scan for DES Y6 mdet, DECADE NGC/SGC and the count-normalised BFD map.

Stage 1 of the joint IA acceptance study (stage 2: ia_joint_eval.py).  For one (simulation, baryon draw b,
rotation r) and a list of samples it stores, per sample,

  geometry  (geom_{sample}_b{b}_r{r}.npz), per tomographic bin, on the observed pixels:
      v_avail     noise variance per component that the mock noise has before the IA subtraction
                  (estimator specific, after source clustering and the map normalisation):
                  - mdet / DECADE: read back exactly from lensing.make_WL_sample (as ia_acceptance_scan.py)
                  - BFD, count-normalised map g_p = -(1/N_p) Rbar^-1 Q_p on the fixed N_gal >= 3 mask:
                    v_p = V_p cR / N_p^2 (V = 1/2 sum |q|^2 of the data, cR = |Rbar^-1|_F^2 / 2), times the
                    Poisson source-clustering modulation s^2 = h(lam D)/h(lam), normalised by
                    cv_new = sum s^2 v / sum v over the mask  (validated in bfd_cnt_validation)
      resp2       squared response applied to the injected IA (BFD: smooth t_64r; 1 for mdet / DECADE)
      W           sparse (observed pixel x NSIDE-64 simulation superpixel) weights of the expected IA
                  self term: indicator for mdet / DECADE (maps in the simulation frame); for BFD the mean over
                  the catalogue rows of the pixel (make_BFD_sample projection, catalogue frame)
  bases (bases_{sample}_b{b}_r{r}_{node}.npz) for each (f_c, f_sat) population node:
      Sc[eta_c, bin, sp], Mc[eta_c, bin]   central self term sum w^2 |e|^2 ((1+z)/1.67)^(2 eta_c) per
                                           superpixel, and its total over the observed pixels (BFD: weighted
                                           by the projection, i.e. sum over the mask of the projected map)
      Ss, Ms                               same for satellites (eta_s)
      a_c, a_s                             NLA-equivalent amplitude coefficients (mdet / DECADE, coarse eta)

For a draw (A_c, eta_c, A_s, eta_s) on a population, numerator_sq = A_c^2 S_c + A_s^2 S_s, and the production
rule (lensing.make_WL_sample) gives per observed pixel
    V_IA(p) = resp2(p) * 0.5 * M * E(p) / sum_q E(q),     E = W S   (expected self term, NSIDE-64 smoothing)
so the invalid-pixel fraction (V_IA > v_avail) is exact for any draw.  The BFD smoothing is done in the
simulation frame, like make_WL_sample (make_BFD_sample currently smooths the projected catalogue-frame map, which
dilutes footprint-edge superpixels; difference reported by the validation).

Validation (rotation 0 only, --validate): (1) the stored IA draw of the rotation is rebuilt with
halo_ia.make_halo_IA_numerator and the invalid fractions are computed directly from the maps and compared with
the emulator on an exact population node at the stored (f_c, f_sat); (2) off-grid populations and re-seeded
populations for the interpolation test of stage 2.

usage: ia_joint_scan.py SIM B R OUTDIR --samples DESY6 BFD [--nproc N] [--validate] [--quick]
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import make_mocks as mm                                                            # noqa: E402
import ia_acceptance_scan as ias                                                   # noqa: E402
from LSS_forward_model.bfd_utils import convert_to_pix_coord                       # noqa: E402
from LSS_forward_model.maps import rotate_and_rebin                                # noqa: E402
from LSS_forward_model.source_clustering import noise_modulation, shell_bias       # noqa: E402

NSIDE = mm.NSIDE
NPIX = hp.nside2npix(NSIDE)
NSIDE_LOW = 64
NSP = hp.nside2npix(NSIDE_LOW)
F_C = [0.2, 0.325, 0.45, 0.575, 0.7]
F_SAT = [0.3, 0.425, 0.55, 0.675, 0.8]
ETA_C = [round(x, 2) for x in np.linspace(-2, 0, 21)]
ETA_S = [-0.5, -0.25, 0.0, 0.25, 0.5]
ETA_C_AEFF = [-2.0, -1.5, -1.0, -0.5, 0.0]
ETA_S_AEFF = [-0.5, 0.0, 0.5]
# off-grid populations (rotation 0) for the interpolation test, and re-seeded grid nodes (population noise)
OFFGRID = [(0.26, 0.74), (0.37, 0.36), (0.49, 0.62), (0.53, 0.47), (0.63, 0.33), (0.66, 0.71)]
RESEED = [(0.45, 0.55), (0.7, 0.3)]
CNT = Path('/pscratch/sd/m/mgatti/des_decade_test/bfd_cnt_validation')
CUT = Path('/pscratch/sd/m/mgatti/des_decade_test/bfd_cut_study')
NMIN_BFD = 3
KEEP_ROWS = False          # set by --validate (rows kept for the direct check)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ------------------------------------------------------------------------------------------------ geometry
def geometry_wl(sample, nuisance, p_b, zeff, ngal, cosmo_bundle, fields, rot):
    """mdet / DECADE: exact var_noise read back from make_WL_sample (+ NLA template for A_eff)."""
    templates = ias.noise_and_template(sample, nuisance, p_b, zeff, ngal, cosmo_bundle, fields, rot)
    out = []
    for tm in templates:
        n = len(tm['obs'])
        W = sparse.csr_matrix((np.ones(n), (np.arange(n), tm['sp'])), shape=(n, NSP))
        obsw = np.zeros(NPIX, np.float32)
        obsw[tm['obs']] = 1.
        out.append(dict(pix=tm['obs'], v_avail=tm['vn'], resp2=np.ones(n), W=W, obsw=obsw, sp_dom=tm['sp'],
                        aTE=tm['aTE'], mask256=tm['mask256'], n_gal=None))
    return out


def geometry_bfd(sample, nuisance, p_b, zeff, ngal, fields, rot):
    """Count-normalised BFD map on the fixed N_gal >= 3 mask (catalogue frame)."""
    bias = shell_bias(ngal, zeff, np.asarray(nuisance['bias_sc']), np.asarray(nuisance['bias_sc_slope']))
    out = []
    for t in range(sample.ntomo):
        c = sample.cats[t]
        pix_cat = convert_to_pix_coord(c['ra'], c['dec'], nside=NSIDE)
        pix_sim = rotate_and_rebin(convert_to_pix_coord(c['ra'], c['dec'], nside=2*NSIDE), NSIDE, rot)
        occ, inv = np.unique(pix_cat, return_inverse=True)
        fld = np.load(CNT/f'bin{t}_fields.npz')
        assert np.array_equal(occ, fld['occ_pix']), 'occupied pixels differ from the validation study'
        n_gal = np.bincount(inv, weights=(np.asarray(c['selection_term']) == 0).astype(float), minlength=len(occ))
        assert np.array_equal(n_gal, fld['n_gal'])
        cnt_rows = np.bincount(inv, minlength=len(occ)).astype(float)
        # projected source density D (make_BFD_sample: mean over all rows of the pixel of the simulation map)
        need, ipix = np.unique(pix_sim, return_inverse=True)
        d_sim = np.zeros(len(need))
        for i in range(min(ngal.shape[1], len(fields['density']))):
            if ngal[t, i] == 0:
                continue
            d_sim += ngal[t, i]*(1. + bias[t, i]*fields['density'][i][need])
        d = np.bincount(inv, weights=d_sim[ipix], minlength=len(occ))/cnt_rows
        lam = sample.lam_maps[t][occ]
        s2 = np.zeros(len(occ))
        gd = np.isfinite(d) & (d > 0)
        s2[gd] = noise_modulation(d[gd], lam[gd], 'poisson')
        m3 = n_gal >= NMIN_BFD
        v = fld['v_pred'].astype(float)
        cv_new = float(np.sum((s2*v)[m3])/np.sum(v[m3]))
        v_mock = s2*v/cv_new
        # projection weights of the M3 pixels: row r of pixel p contributes 1/cnt_rows[p] at its simulation pixel
        keep_rows = m3[inv]
        idx_m3 = -np.ones(len(occ), np.int64)
        idx_m3[m3] = np.arange(m3.sum())
        rp = idx_m3[inv[keep_rows]]
        wr = 1./cnt_rows[inv[keep_rows]]
        sp_rows = ias.parent64(pix_sim[keep_rows])
        W = sparse.csr_matrix((wr, (rp, sp_rows)), shape=(int(m3.sum()), NSP))
        W.sum_duplicates()
        obsw = np.bincount(pix_sim[keep_rows], weights=wr, minlength=NPIX).astype(np.float32)
        sp_dom = np.asarray(W.argmax(axis=1)).ravel()
        out.append(dict(pix=occ[m3], v_avail=v_mock[m3], resp2=fld['t64r'][m3].astype(float)**2, W=W, obsw=obsw,
                        sp_dom=sp_dom, n_gal=n_gal[m3], s2=s2[m3], cv_new=cv_new, d=d[m3],
                        frac_rows_multi_sp=float(np.mean(np.diff(W.indptr) > 1)),
                        rows=(pix_sim[keep_rows], rp, wr) if KEEP_ROWS else None))
        log(f'BFD bin {t}: {m3.sum()} M3 pixels, cv_new {cv_new:.4f}, mean v_avail {v_mock[m3].mean():.3e}, '
            f'pixels spanning >1 superpixel {out[-1]["frac_rows_multi_sp"]:.4f}')
    return out


def save_geometry(path, geo, extra):
    arrs = {}
    for t, g in enumerate(geo):
        arrs[f'pix_{t}'] = g['pix']
        arrs[f'v_avail_{t}'] = g['v_avail']
        arrs[f'resp2_{t}'] = g['resp2']
        arrs[f'W_indptr_{t}'] = g['W'].indptr
        arrs[f'W_indices_{t}'] = g['W'].indices
        arrs[f'W_data_{t}'] = g['W'].data
        arrs[f'sp_dom_{t}'] = g['sp_dom']
        if g.get('n_gal') is not None:
            arrs[f'n_gal_{t}'] = g['n_gal']
            arrs[f's2_{t}'] = g['s2']
    np.savez(path, ntomo=len(geo), meta=json.dumps(extra), **arrs)


# ------------------------------------------------------------------------------------------------ bases
def fine_bases(tab, ntomo, kind, etas):
    sel = tab['central'] if kind == 'c' else ~tab['central']
    key = tab['bin'][sel]*NSP + tab['sp'][sel]
    w2e2, logev, obsw, b = tab['w2e2'][sel], tab['logev'][sel], tab['obsw'][sel], tab['bin'][sel]
    S = np.zeros((len(etas), ntomo, NSP), np.float64)
    M = np.zeros((len(etas), ntomo))
    for i, eta in enumerate(etas):
        v = w2e2*np.exp(2*eta*logev)
        S[i] = np.bincount(key, weights=v, minlength=ntomo*NSP).reshape(ntomo, NSP)
        M[i] = np.bincount(b, weights=v*obsw, minlength=ntomo)
    return S, M


def node_bases(sample, nuisance, shells_info, p_b, cosmo_bundle, ngal, geo, fc, fs, rot, seed_offset=0,
               aeff=True):
    nuis = dict(nuisance)
    if seed_offset:
        nuis['population_seed'] = int((nuisance['population_seed'] + 7919*seed_offset) % (2**32 - 1))
    pop = mm.population_for(sample, nuis, {'f_c': fc, 'f_sat': fs}, shells_info, p_b, cosmo_bundle,
                            ias.HALO_CATALOG, rot)
    tab = ias.source_table(pop, ngal, sample.bin_counts, nuisance['bin_seed'], [g['pix'] for g in geo]
                           if geo[0].get('n_gal') is None else [np.zeros(0, np.int64)]*sample.ntomo)
    del pop
    obsw = np.stack([g['obsw'] for g in geo])
    tab['obsw'] = obsw[tab['bin'], tab['pixel']].astype(float)
    del obsw
    Sc, Mc = fine_bases(tab, sample.ntomo, 'c', ETA_C)
    Ss, Ms = fine_bases(tab, sample.ntomo, 's', ETA_S)
    out = dict(Sc=Sc.astype(np.float32), Mc=Mc, Ss=Ss.astype(np.float32), Ms=Ms, n_sources=len(tab['bin']),
               n_central=int(tab['central'].sum()))
    if aeff and geo[0].get('aTE') is not None:
        templates = [dict(obs=g['pix'], aTE=g['aTE']) for g in geo]
        bc = ias.bases(tab, sample.ntomo, 'c', ETA_C_AEFF, templates)
        bs = ias.bases(tab, sample.ntomo, 's', ETA_S_AEFF, templates)
        out['a_c'] = np.array([x['a'] for x in bc])
        out['a_s'] = np.array([x['a'] for x in bs])
    return out, tab


def fractions_from_bases(geo, Sc, Mc, Ss, Ms, A_c, A_s):
    """invalid fraction, superpixel-budget fraction, excess-variance fraction and mean ratio, per bin."""
    res = []
    for t, g in enumerate(geo):
        S = A_c**2*Sc[t].astype(float) + A_s**2*Ss[t].astype(float)
        M = A_c**2*Mc[t] + A_s**2*Ms[t]
        E = g['W'] @ S
        h = g['resp2']*0.5*M*E/np.sum(E)
        va = g['v_avail']
        res.append(dict(invalid=float(np.mean(h >= va)), mean_ratio=float(np.mean(h/va)),
                        excess=float(np.sum(np.clip(h - va, 0, None))/np.sum(va))))
    return res


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sim', type=Path)
    ap.add_argument('b', type=int)
    ap.add_argument('r', type=int)
    ap.add_argument('out', type=Path)
    ap.add_argument('--samples', nargs='+', required=True)
    ap.add_argument('--nproc', type=int, default=64)
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--quick', action='store_true', help='one node only (timing test)')
    args = ap.parse_args()
    mm.NPROC = args.nproc
    global KEEP_ROWS
    KEEP_ROWS = args.validate
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sim, b, r = args.sim, args.b, args.r
    sims_parameters, cosmo_bundle = mm.read_sims_params(str(sim) + '/')
    shells_info = mm.recover_shell_info(str(sim/'z_values.txt'), max_z=49)
    shells, _, _ = mm.build_shells(shells_info)
    nuisance = np.load(sim/f'nuisance_b{b}_r{r}.npy', allow_pickle=True).item()
    stored = np.load(sim/f'IA_params_halo_b{b}_r{r}.npy', allow_pickle=True).item()
    p_b = copy.deepcopy(sims_parameters)
    density = np.load(sim/f'delta_b_{NSIDE}_{b}.npy')
    need_wl = any(s != 'BFD' for s in args.samples)
    if need_wl:
        fields = mm.compute_lensing_fields(density, shells, cosmo_bundle['pars_camb'], NSIDE,
                                           do_kappa=False, do_shear=True, do_IA=True)
    else:
        fields = {}
    fields['density'] = density
    log(f'setup {time.time() - t0:.0f} s')

    nodes = [(fc, fs, 0) for fc in F_C for fs in F_SAT]
    if args.quick:
        nodes = nodes[:1]
    extra_nodes = []
    if args.validate:
        extra_nodes = ([('val', stored['f_c'], stored['f_sat'], 0)] + [('off', fc, fs, 0) for fc, fs in OFFGRID]
                       + [('reseed', fc, fs, 1) for fc, fs in RESEED])
        if args.quick:
            extra_nodes = extra_nodes[:1]

    for s in args.samples:
        t1 = time.time()
        sample = mm.Sample(s)
        _, _, zeff, ngal = mm.shells_for(sample, nuisance[s], shells_info)
        if sample.is_bfd:
            geo = geometry_bfd(sample, nuisance[s], p_b, zeff, ngal, fields, r)
        else:
            geo = geometry_wl(sample, nuisance[s], p_b, zeff, ngal, cosmo_bundle, fields, r)
        meta = dict(sample=s, b=b, r=r, sim=str(sim), nuisance={k: (np.asarray(v).tolist() if np.ndim(v) else v)
                                                                 for k, v in nuisance[s].items()},
                    n_obs=[len(g['pix']) for g in geo], F_C=F_C, F_SAT=F_SAT, ETA_C=ETA_C, ETA_S=ETA_S,
                    ETA_C_AEFF=ETA_C_AEFF, ETA_S_AEFF=ETA_S_AEFF, stored_ia=stored,
                    cv_new=[g.get('cv_new') for g in geo])
        save_geometry(args.out/f'geom_{s}_b{b}_r{r}.npz', geo, meta)
        log(f'{s}: geometry in {time.time() - t1:.0f} s; observed pixels {meta["n_obs"]}')

        todo = [(f'fc{fc}_fs{fs}', fc, fs, 0) for fc, fs, _ in nodes] + \
               [(f'{tag}_fc{fc:.4f}_fs{fs:.4f}', fc, fs, so) for tag, fc, fs, so in extra_nodes]
        for name, fc, fs, so in todo:
            path = args.out/f'bases_{s}_b{b}_r{r}_{name}.npz'
            if path.exists():
                continue
            t2 = time.time()
            bas, tab = node_bases(sample, nuisance[s], shells_info, p_b, cosmo_bundle, ngal, geo, fc, fs, r,
                                  seed_offset=so, aeff=not name.startswith(('off', 'reseed')))
            if name.startswith('val'):
                # (1) direct check: the stored draw, full numerator from the production function
                bas['validation'] = validate_direct(sample, nuisance[s], shells_info, p_b, cosmo_bundle, ngal, geo,
                                                    stored, r, bas, density)
            del tab
            np.savez(path, f_c=fc, f_sat=fs, seed_offset=so,
                     **{k: (v if not isinstance(v, dict) else json.dumps(v)) for k, v in bas.items()})
            fr = fractions_from_bases(geo, bas['Sc'][10], bas['Mc'][10], bas['Ss'][2], bas['Ms'][2], 1.5, 0.015)
            log(f'{s} {name}: {bas["n_sources"]:,} sources, {time.time() - t2:.0f} s; invalid at A_c 1.5, eta_c -1, '
                f'A_s .015, eta_s 0: {[round(x["invalid"], 4) for x in fr]}')
        del sample, geo
    log(f'done in {time.time() - t0:.0f} s')


def validate_direct(sample, nuisance, shells_info, p_b, cosmo_bundle, ngal, geo, stored, rot, bas, density):
    """Rebuild the stored draw with make_halo_IA_numerator and compare the invalid fractions with the emulator."""
    pop = mm.population_for(sample, nuisance, stored, shells_info, p_b, cosmo_bundle, ias.HALO_CATALOG, rot)
    halo = mm.make_halo_IA_numerator(pop, mm.ia_block(stored), density, ngal, NSIDE, bin_counts=sample.bin_counts,
                                     seed=nuisance['bin_seed'])
    del pop

    def at(etas, arr, x):                       # linear interpolation on the fine eta grid
        k = int(np.clip(np.searchsorted(etas, x) - 1, 0, len(etas) - 2))
        u = (x - etas[k])/(etas[k + 1] - etas[k])
        return (1 - u)*arr[k].astype(float) + u*arr[k + 1].astype(float)
    Sc, Mc = at(ETA_C, bas['Sc'], stored['eta_c']), at(ETA_C, bas['Mc'], stored['eta_c'])
    Ss, Ms = at(ETA_S, bas['Ss'], stored['eta_s']), at(ETA_S, bas['Ms'], stored['eta_s'])
    emu = fractions_from_bases(geo, Sc, Mc, Ss, Ms, stored['A_c'], stored['A_s'])
    out = {'draw': {k: stored[k] for k in ('f_c', 'f_sat', 'A_c', 'eta_c', 'A_s', 'eta_s')},
           'direct': [], 'emulated': emu, 'bfd_catalogue_frame_smoothing': []}
    for t, g in enumerate(geo):
        nsq = np.asarray(halo['numerator_sq'][t], float)
        expected = hp.ud_grade(hp.ud_grade(nsq, NSIDE_LOW), NSIDE)
        if g.get('n_gal') is None:            # simulation frame, as make_WL_sample
            ex = expected[g['pix']]
            ex = ex*np.sum(nsq[g['pix']])/np.sum(ex)
        else:                                 # BFD: mean over the catalogue rows of each pixel, from the rows
            rs, rp, wr = g['rows']
            n = len(g['pix'])
            raw = np.bincount(rp, weights=wr*nsq[rs], minlength=n)
            ex = np.bincount(rp, weights=wr*expected[rs], minlength=n)
            ex = ex*np.sum(raw)/np.sum(ex)
            # current make_BFD_sample: smoothing of the projected catalogue-frame map (edge dilution)
            full = np.zeros(NPIX)
            full[g['pix']] = raw
            exc = hp.ud_grade(hp.ud_grade(full, NSIDE_LOW), NSIDE)[g['pix']]
            exc = exc*np.sum(raw)/np.sum(exc)
            hc = g['resp2']*0.5*exc
            out['bfd_catalogue_frame_smoothing'].append(dict(invalid=float(np.mean(hc >= g['v_avail'])),
                                                             mean_ratio=float(np.mean(hc/g['v_avail']))))
        h = g['resp2']*0.5*ex
        out['direct'].append(dict(invalid=float(np.mean(h >= g['v_avail'])), mean_ratio=float(np.mean(h/g['v_avail'])),
                                  excess=float(np.sum(np.clip(h - g['v_avail'], 0, None))/np.sum(g['v_avail']))))
    log('validation (stored draw)', json.dumps(out))
    return out


if __name__ == '__main__':
    main()
