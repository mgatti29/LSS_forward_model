#!/usr/bin/env python3
"""Acceptance of the halo-IA noise-budget rule over the IA prior, for one (simulation, baryon draw b, rotation r).

The rule of make_mocks.py: a draw is rejected if, in any sample and bin, more than max_fraction of the observed
pixels have var_IA >= var_noise (ia_noise_factor == 0 in lensing.make_WL_sample).  This script evaluates that rule
exactly but cheaply over the whole prior, using two facts:

  * var_noise(p) does not depend on the IA parameters.  It is read back from make_WL_sample run with a tiny
    uniform numerator_sq = C:  ia_noise_factor = sqrt(1 - 0.5 C / var_noise)  ->  var_noise = 0.5 C / (1 - f^2).
  * for a fixed population (f_c, f_sat) the IA self term is quadratic in the amplitudes,
        numerator_sq = A_c^2 S_c(eta_c) + A_s^2 S_s(eta_s),
    and the rule only needs its NSIDE-64 superpixel sums and its total over the observed pixels.  The per-source
    weights and the random bin assignment replicate halo_ia.make_halo_IA_numerator (same seed).

For a direction phi in the (A_c, A_s/S0) plane, the pixel p fails when rho^2 >= r_p = var_noise_p / h_p(phi), so
the critical radius is rho*^2 = the floor(L n)-th smallest r_p (L = max_fraction).  Stored per node, sample, bin,
(eta_c, eta_s), phi and L.

NLA-equivalent amplitude: the halo numerator (linear in A_c, A_s) is regressed on the NLA IA map of the same mock
(make_WL_sample NLA, A_IA = 1, eta_IA = 0, minus the same with no IA) over 100 <= ell <= 600:
    A_eff = A_c a_c(eta_c) + A_s a_s(eta_s),  a = sum_l (2l+1) C_l^{N T} / sum_l (2l+1) C_l^{T T}.

Validation: populations at the stored (f_c, f_sat) of the e2e test, evaluated at the logged attempted marks; compare
with the exact worst fractions of the e2e log.

usage: ia_acceptance_scan.py SIM B R OUT_JSON_PREFIX [--nproc N]
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import make_mocks as mm                                                    # noqa: E402
from LSS_forward_model.halo_ia import PACKAGE_IA_SIGN                       # noqa: E402
from LSS_forward_model.lensing import make_WL_sample                        # noqa: E402

NSIDE = mm.NSIDE
NPIX = hp.nside2npix(NSIDE)
NSIDE_LOW = 64                     # ia_noise_nside of make_WL_sample
NSP = hp.nside2npix(NSIDE_LOW)
NSIDE_CL, LMIN_EFF, LMAX_EFF = 256, 100, 600
HALO_CATALOG = '/pscratch/sd/m/mgatti/source_clustering_v2/halo_ia/halo_catalog_ia_z2_m12_Fiducial_1_big.parquet'
SAMPLES = ['DESY6', 'NGC', 'SGC']
F_C = [0.2, 0.35, 0.5, 0.7]
F_SAT = [0.3, 0.45, 0.6, 0.8]
ETA_C = list(np.round(np.linspace(-2, 0, 9), 4))
ETA_S = [-0.5, 0.0, 0.5]
S0 = 0.015                          # A_s unit of the (A_c, A_s / S0) plane
PHI = list(np.round(np.linspace(0, np.pi/2, 25), 6))
LEVELS = [0.005, 0.01, 0.02, 0.05]
C_PROBE = 1e-12
# logged attempts of the e2e test (tests/run_e2e_gower.sh, logs/e2e_interactive7.log): exact worst fractions
ATTEMPTS = {
    0: [dict(A_c=1.1791, eta_c=-1.0542, A_s=0.015, eta_s=-0.3913, exact=dict(DESY6=0.0005, NGC=0.0187, SGC=0.0217))],
    1: [dict(A_c=-1.465, eta_c=-0.5986, A_s=0.0125, eta_s=0.3782, exact=dict(DESY6=0.0010, NGC=0.0339)),
        dict(A_c=0.9899, eta_c=-0.8903, A_s=0.0141, eta_s=-0.4527, exact=dict(DESY6=0.0, NGC=0.0006, SGC=0.0012))],
    2: [dict(A_c=1.596, eta_c=-0.4894, A_s=0.0266, eta_s=-0.411, exact=dict(DESY6=0.0008, NGC=0.0280)),
        dict(A_c=0.9522, eta_c=-0.56, A_s=0.0001, eta_s=0.2639, exact=dict(DESY6=0.0, NGC=0.0002, SGC=0.0003))],
    3: [dict(A_c=0.7499, eta_c=-1.6401, A_s=0.0061, eta_s=-0.2859, exact=dict(DESY6=0.0001, NGC=0.0004, SGC=0.0007))],
}


def parent64(pixel):
    return hp.nest2ring(NSIDE_LOW, hp.ring2nest(NSIDE, pixel)//(NSIDE//NSIDE_LOW)**2)


def wl_maps(sample, nuisance, p_b, zeff, ngal, cosmo_bundle, fields, rot, model, halo_IA=None, A_IA=0.):
    p = mm.base_parameters(p_b, sample, nuisance, rot)
    p.update(A_IA=A_IA, eta_IA=0.)
    np.random.seed(nuisance['noise_seed'])
    maps, _ = make_WL_sample(ngal, zeff, cosmo_bundle, p, NSIDE, fields, sample.cats, SC_corrections=sample.sc,
                             include_SC=True, compact_savings=False, noise_model='poisson', IA_model=model,
                             halo_IA=halo_IA)
    return maps


def noise_and_template(sample, nuisance, p_b, zeff, ngal, cosmo_bundle, fields, rot):
    """Per bin: observed pixels, var_noise there (exact, from make_WL_sample), NLA template E alm (A=1, eta=0)."""
    nt = sample.ntomo
    probe = {'numerator': np.zeros((nt, NPIX), complex), 'numerator_sq': np.full((nt, NPIX), C_PROBE)}
    m0 = wl_maps(sample, nuisance, p_b, zeff, ngal, cosmo_bundle, fields, rot, 'halo', probe)
    del probe
    m1 = wl_maps(sample, nuisance, p_b, zeff, ngal, cosmo_bundle, fields, rot, 'NLA', A_IA=1.)
    out = []
    for t in range(nt):
        obs = np.flatnonzero(m0[t]['var_'] > 0)
        f = m0[t]['ia_noise_factor'][obs]
        one_minus = 1. - f**2
        vn = np.where(one_minus > 0, 0.5*C_PROBE/np.where(one_minus > 0, one_minus, 1.), 0.)
        vn[f == 0] = 0.
        mask = np.zeros(NPIX)
        mask[obs] = 1.
        T1 = (m1[t]['g1_map'] - m0[t]['g1_map'])*mask
        T2 = (m1[t]['g2_map'] - m0[t]['g2_map'])*mask
        aT = hp.map2alm_spin([hp.ud_grade(T1, NSIDE_CL), hp.ud_grade(T2, NSIDE_CL)], 2, lmax=LMAX_EFF)
        sp = parent64(obs)
        out.append(dict(obs=obs, vn=vn, sp=sp, nobs_sp=np.bincount(sp, minlength=NSP), aTE=aT[0],
                        n_vn_zero=int(np.sum(vn == 0)), mask256=hp.ud_grade(mask, NSIDE_CL)))
    del m0, m1
    return out


def source_table(pop, ngal, bin_counts, seed, obs_list):
    """Per model source, as halo_ia.make_halo_IA_numerator: bin, pixel, weight, unit mark, log evolution, central."""
    ngal = np.asarray(ngal, float)
    ntomo, nshell = ngal.shape
    shell_index = np.asarray(pop['shell_index'])
    is_central = np.asarray(pop['is_central'], bool)
    rng = np.random.default_rng(seed)
    counts_per_shell = (ngal/ngal.sum(1, keepdims=True))*np.asarray(bin_counts, float)[:, None]
    cumulative = np.cumsum(counts_per_shell, axis=0)
    cumulative = cumulative/np.where(cumulative[-1] > 0, cumulative[-1], 1.)
    ordered = np.all(shell_index[1:] >= shell_index[:-1])
    cols = {k: [] for k in ('bin', 'pixel', 'weight', 'mark', 'logev', 'central')}
    for i in np.unique(shell_index):
        if ordered:
            rows = slice(np.searchsorted(shell_index, i, side='left'), np.searchsorted(shell_index, i, side='right'))
        else:
            rows = np.flatnonzero(shell_index == i)
        ra = np.asarray(pop['ra_deg'][rows], float)
        dec = np.asarray(pop['dec_deg'][rows], float)
        logev = np.log((1 + np.asarray(pop['redshift'][rows], float))/1.67)
        mark = PACKAGE_IA_SIGN*(np.asarray(pop['e1'][rows]) + 1j*np.asarray(pop['e2'][rows]))
        pixel = hp.ang2pix(NSIDE, ra, dec, lonlat=True)
        source_bin = np.sum(rng.random(len(pixel))[None, :] > cumulative[:-1, i][:, None], axis=0)
        hosted = pop.get('hosted_fraction')
        hosted_fraction = 1. if hosted is None else float(np.mean(np.asarray(hosted[rows], float)))
        in_fp = pop.get('footprint_fraction')
        if in_fp is not None:
            hosted_fraction *= float(np.mean(np.asarray(in_fp[rows], float)))
        n_model = np.bincount(source_bin, minlength=ntomo)
        weight_bin = np.where((ngal[:, i] > 0) & (n_model > 0), ngal[:, i]*hosted_fraction*NPIX/np.maximum(n_model, 1), 0.)
        weight = weight_bin[source_bin]
        keep = weight > 0
        for k, v in (('bin', source_bin), ('pixel', pixel), ('weight', weight), ('mark', mark), ('logev', logev),
                     ('central', is_central[rows])):
            cols[k].append(v[keep])
    tab = {k: np.concatenate(v) for k, v in cols.items()}
    tab['bin'] = tab['bin'].astype(np.int64)
    tab['sp'] = parent64(tab['pixel'])
    observed = np.zeros((ntomo, NPIX), bool)
    for t, o in enumerate(obs_list):
        observed[t, o] = True
    tab['in_obs'] = observed[tab['bin'], tab['pixel']]
    tab['w2e2'] = tab['weight']**2*np.abs(tab['mark'])**2
    return tab


def bases(tab, ntomo, kind, etas, templates):
    """Self-term superpixel sums S (ntomo, NSP), observed totals M (ntomo,), and A_eff coefficients a (ntomo,)."""
    sel = tab['central'] if kind == 'c' else ~tab['central']
    b, sp, pix = tab['bin'][sel], tab['sp'][sel], tab['pixel'][sel]
    w2e2, logev, inobs = tab['w2e2'][sel], tab['logev'][sel], tab['in_obs'][sel]
    wm = tab['weight'][sel]*tab['mark'][sel]
    out = []
    for eta in etas:
        ev2 = np.exp(2*eta*logev)
        v = w2e2*ev2
        S = np.bincount(b*NSP + sp, weights=v, minlength=ntomo*NSP).reshape(ntomo, NSP)
        M = np.bincount(b, weights=v*inobs, minlength=ntomo)
        num = wm*np.sqrt(ev2)
        idx = b*NPIX + pix
        re = np.bincount(idx, weights=num.real, minlength=ntomo*NPIX).reshape(ntomo, NPIX)
        im = np.bincount(idx, weights=num.imag, minlength=ntomo*NPIX).reshape(ntomo, NPIX)
        a = np.zeros(ntomo)
        cls = []
        for t in range(ntomo):
            tm = templates[t]
            mask = np.zeros(NPIX)
            mask[tm['obs']] = 1.
            aN = hp.map2alm_spin([hp.ud_grade(re[t]*mask, NSIDE_CL), hp.ud_grade(im[t]*mask, NSIDE_CL)], 2,
                                 lmax=LMAX_EFF)
            ell = np.arange(LMAX_EFF + 1)
            sel_l = (ell >= LMIN_EFF) & (ell <= LMAX_EFF)
            nt = hp.alm2cl(aN[0], tm['aTE'])
            tt = hp.alm2cl(tm['aTE'])
            a[t] = np.sum(((2*ell + 1)*nt)[sel_l])/np.sum(((2*ell + 1)*tt)[sel_l])
            ee, bb = hp.alm2cl(aN[0]), hp.alm2cl(aN[1])
            cls.append(dict(EE=float(np.sum(((2*ell+1)*ee)[sel_l])), BB=float(np.sum(((2*ell+1)*bb)[sel_l])),
                            NT=float(np.sum(((2*ell+1)*nt)[sel_l])), TT=float(np.sum(((2*ell+1)*tt)[sel_l]))))
        del re, im
        out.append(dict(S=S, M=M, a=a, cls=cls))
    return out


def fraction(tm, S, M):
    """Exact fraction of observed pixels with var_IA >= var_noise, for numerator_sq superpixel sums S, total M."""
    expected = S/(NSIDE//NSIDE_LOW)**2                    # ud_grade average of the NSIDE-1024 map
    denom = np.sum(expected*tm['nobs_sp'])
    if denom <= 0:
        return 0.
    h = 0.5*(M/denom)*expected[tm['sp']]
    fail = (tm['vn'] <= h) & (h > 0)
    return float(np.mean(fail))


def critical_rho2(tm, Sc, Mc, Ss, Ms):
    """rho*^2 per phi and level: the pixel fails when rho^2 >= vn / h(phi)."""
    out = np.zeros((len(PHI), len(LEVELS)))
    n = len(tm['vn'])
    ks = [int(np.floor(L*n)) for L in LEVELS]
    for j, phi in enumerate(PHI):
        cc, ss = np.cos(phi)**2, (np.sin(phi)*S0)**2
        S = cc*Sc + ss*Ss
        M = cc*Mc + ss*Ms
        expected = S/(NSIDE//NSIDE_LOW)**2
        denom = np.sum(expected*tm['nobs_sp'])
        h = 0.5*(M/denom)*expected[tm['sp']]
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(h > 0, tm['vn']/h, np.inf)
        part = np.partition(r, ks)
        out[j] = [part[k] for k in ks]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sim', type=Path)
    ap.add_argument('b', type=int)
    ap.add_argument('r', type=int)
    ap.add_argument('out', type=Path)
    ap.add_argument('--nproc', type=int)
    ap.add_argument('--quick', action='store_true', help='one node, one sample (timing test)')
    args = ap.parse_args()
    if args.nproc:
        mm.NPROC = args.nproc
    t0 = time.time()
    sim = args.sim
    sims_parameters, cosmo_bundle = mm.read_sims_params(str(sim) + '/')
    shells_info = mm.recover_shell_info(str(sim/'z_values.txt'), max_z=49)
    shells, _, _ = mm.build_shells(shells_info)
    names = ['NGC'] if args.quick else SAMPLES
    samples = {s: mm.Sample(s) for s in names}
    nuisance = np.load(sim/f'nuisance_b{args.b}_r{args.r}.npy', allow_pickle=True).item()
    p_b = copy.deepcopy(sims_parameters)
    density = np.load(sim/f'delta_b_{NSIDE}_{args.b}.npy')
    fields = mm.compute_lensing_fields(density, shells, cosmo_bundle['pars_camb'], NSIDE,
                                       do_kappa=False, do_shear=True, do_IA=True)
    fields['density'] = density
    mm.log(f'setup {time.time() - t0:.0f} s')

    stored = np.load(sim/f'IA_params_halo_b{args.b}_r{args.r}.npy', allow_pickle=True).item()
    nodes = [(fc, fs) for fc in F_C for fs in F_SAT]
    if args.quick:
        nodes = nodes[:1]
    nodes.append(('validation', (stored['f_c'], stored['f_sat'])))
    attempts = ATTEMPTS.get(args.r, []) if args.b == 0 else []

    res = dict(sim=str(sim), b=args.b, r=args.r, F_C=F_C, F_SAT=F_SAT, ETA_C=ETA_C, ETA_S=ETA_S, PHI=PHI,
               LEVELS=LEVELS, S0=S0, samples=names, nodes={}, validation=[], ntomo={})
    rho2 = {}
    for s in names:
        t1 = time.time()
        sample = samples[s]
        _, _, zeff, ngal = mm.shells_for(sample, nuisance[s], shells_info)
        templates = noise_and_template(sample, nuisance[s], p_b, zeff, ngal, cosmo_bundle, fields, args.r)
        res['ntomo'][s] = sample.ntomo
        res.setdefault('n_obs', {})[s] = [len(tm['obs']) for tm in templates]
        res.setdefault('n_vn_zero', {})[s] = [tm['n_vn_zero'] for tm in templates]
        mm.log(f'{s}: noise and NLA template in {time.time() - t1:.0f} s; var_noise == 0 pixels {res["n_vn_zero"][s]}')
        for node in nodes:
            t2 = time.time()
            validation = node[0] == 'validation'
            fc, fs = node[1] if validation else node
            pop = mm.population_for(sample, nuisance[s], {'f_c': fc, 'f_sat': fs}, shells_info, p_b, cosmo_bundle,
                                    HALO_CATALOG, args.r)
            tab = source_table(pop, ngal, sample.bin_counts, nuisance[s]['bin_seed'], [tm['obs'] for tm in templates])
            n_src = len(tab['bin'])
            del pop
            if validation:
                for att in attempts:
                    if s not in att['exact']:
                        continue
                    bc = bases(tab, sample.ntomo, 'c', [att['eta_c']], templates)[0]
                    bs = bases(tab, sample.ntomo, 's', [att['eta_s']], templates)[0]
                    fr = [fraction(templates[t], att['A_c']**2*bc['S'][t] + att['A_s']**2*bs['S'][t],
                                   att['A_c']**2*bc['M'][t] + att['A_s']**2*bs['M'][t]) for t in range(sample.ntomo)]
                    row = dict(sample=s, f_c=fc, f_sat=fs, **{k: att[k] for k in ('A_c', 'eta_c', 'A_s', 'eta_s')},
                               emulated_per_bin=fr, emulated_worst=max(fr), exact_worst=att['exact'][s],
                               A_eff=list(att['A_c']*bc['a'] + att['A_s']*bs['a']))
                    res['validation'].append(row)
                    mm.log('validation', json.dumps(row))
                del tab
                continue
            bc = bases(tab, sample.ntomo, 'c', ETA_C, templates)
            bs = bases(tab, sample.ntomo, 's', ETA_S, templates)
            del tab
            key = f'{fc}_{fs}'
            arr = np.zeros((sample.ntomo, len(ETA_C), len(ETA_S), len(PHI), len(LEVELS)))
            for t in range(sample.ntomo):
                for i, c in enumerate(bc):
                    for j, sb in enumerate(bs):
                        arr[t, i, j] = critical_rho2(templates[t], c['S'][t], c['M'][t], sb['S'][t], sb['M'][t])
            rho2[(s, key)] = arr
            nd = res['nodes'].setdefault(key, {'f_c': fc, 'f_sat': fs})
            nd[s] = dict(n_sources=n_src, a_c=[c['a'].tolist() for c in bc], a_s=[sb['a'].tolist() for sb in bs],
                         cls_c=[c['cls'] for c in bc], cls_s=[sb['cls'] for sb in bs])
            mm.log(f'{s} node f_c {fc} f_sat {fs}: {n_src:,} sources, {time.time() - t2:.0f} s; '
                   f'rho*(phi=0, eta_c=-1, eta_s=0, L=0.02) per bin '
                   f'{np.round(np.sqrt(arr[:, 4, 1, 0, 2]), 3).tolist()}; a_c(eta_c=-1) {np.round(bc[4]["a"], 3).tolist()}')
            np.savez(str(args.out) + '_rho2.npz', **{f'{k[0]}|{k[1]}': v for k, v in rho2.items()})
            Path(str(args.out) + '.json').write_text(json.dumps(res, indent=1, default=float) + '\n')
        del templates
    np.savez(str(args.out) + '_rho2.npz', **{f'{k[0]}|{k[1]}': v for k, v in rho2.items()})
    Path(str(args.out) + '.json').write_text(json.dumps(res, indent=1, default=float) + '\n')
    mm.log(f'done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
