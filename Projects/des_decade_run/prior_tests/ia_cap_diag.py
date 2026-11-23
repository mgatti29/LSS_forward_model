#!/usr/bin/env python3
"""EXPERIMENTAL local IA cap (lensing.make_WL_sample / bfd_utils.make_BFD_sample, ia_cap_alpha): effect at the
existing IA best fits on the production noise-budget geometry of ia_joint_scan.py (Hypatia b0, rotations 0-3).

Per pixel of every sample / rotation / bin (exact production budget, as ia_joint_eval.metrics):
    V_IA(p) = resp2(p) 0.5 M E(p) / sum E,  E = W (A_c^2 S_c + A_s^2 S_s)   (expected IA white variance)
    V(p)    = v_avail (noise variance before the IA subtraction)
    s(p)    = min(1, sqrt(alpha V / V_IA))  (1 where V_IA = 0; 0 where V = 0 < V_IA)
reported: fraction of pixels capped, distribution of s, IA variance removed sum (1 - s^2) V_IA / sum V_IA (and
relative to the noise), residual noise (V - s^2 V_IA) / V >= 1 - alpha, and the dependence on the pixel occupancy
(catalogue objects per pixel; BFD: real objects, selection_term = 0) and on the local density (n(z)-weighted Hypatia
density contrast of the bin at the pixel; BFD: mean over the objects of the catalogue-frame pixel).

usage: ia_cap_diag.py OUTDIR RESULT.json
"""
import json
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import ia_joint_eval as E  # noqa: E402
import make_mocks as mm  # noqa: E402
from LSS_forward_model.maps import convert_to_pix_coord, rotate_and_rebin  # noqa: E402

NSIDE = 1024
ROTS = [0, 1, 2, 3]
ALPHAS = [1.0, 0.9]
FIT = Path('/pscratch/sd/m/mgatti/IA_non_gaussian/tng300/cheap_validation/DES/paper_fit_des_gower_tng')
OCC_EDGES = [1, 2, 3, 4, 6, 9, 13, 21, 41, 10**9]
NDENS = 10


def points():
    pts = {}
    for name, f in (('Gower unconstrained', 'fit_outliers.json'), ('Gower prior box', 'prior_box.json')):
        b = json.loads((FIT/f).read_text())['best']
        pts[name] = {k: float(b[k]) for k in E.PAR}
    pts['TNG halos (IA paper)'] = dict(A_c=2.566, A_s=0.0193, eta_c=-1.0, eta_s=0.0, f_c=0.5, f_sat=0.8)
    return pts


def cap_factor(v, h, alpha):
    s = np.ones_like(h)
    c = (h > 0) & (h > alpha*v)
    s[c] = np.sqrt(alpha*v[c]/h[c])
    return s


def ia_variance(g, Sc, Ss, Mc, Ms, A_c, A_s, unit_resp=False):
    S = A_c**2*Sc + A_s**2*Ss
    M = A_c**2*Mc + A_s**2*Ms
    Ep = S[g['sp']] if g['sp'] is not None else g['W'] @ S
    h = 0.5*M*Ep/np.sum(Ep)
    return h if unit_resp else h*g['resp2']


def summary(v, h, s, occ, dens, alpha):
    capped = s < 1
    removed = (1 - s**2)*h
    out = dict(n_pix=int(len(v)), frac_capped=float(capped.mean()),
               s_quantiles_capped={q: float(np.quantile(s[capped], q)) for q in (0, .05, .25, .5, .75, .95)}
               if capped.any() else None,
               mean_s_all=float(s.mean()), frac_s_below={x: float(np.mean(s < x)) for x in (0.9, 0.7, 0.5, 0.3)},
               var_removed_frac_of_IA=float(removed.sum()/h.sum()),
               var_removed_frac_of_noise=float(removed.sum()/v.sum()),
               amplitude_weighted_removed=float(np.sum((1 - s)*np.sqrt(h))/np.sum(np.sqrt(h))),
               residual_over_noise_min=float(np.min((v - s**2*h)[v > 0]/v[v > 0])),
               n_negative_residual=int(np.sum(v - s**2*h < -1e-12*np.abs(v))),
               invalid_after=float(np.mean(s**2*h > alpha*v*(1 + 1e-9))),
               frac_v_zero=float(np.mean(v == 0)), frac_h_zero=float(np.mean(h == 0)))
    tab = []
    for lo, hi in zip(OCC_EDGES[:-1], OCC_EDGES[1:]):
        m = (occ >= lo) & (occ < hi)
        if m.sum():
            tab.append(dict(n_lo=lo, n_hi=hi - 1, frac_pix=float(m.mean()), frac_capped=float(capped[m].mean()),
                            mean_s=float(s[m].mean()), share_removed=float(removed[m].sum()/max(removed.sum(), 1e-300)),
                            removed_frac_of_IA=float(removed[m].sum()/h[m].sum()) if h[m].sum() > 0 else 0.,
                            median_ratio=float(np.median(h[m]/np.where(v[m] > 0, v[m], np.inf)))))
    out['by_occupancy'] = tab
    if dens is not None:
        edges = np.quantile(dens, np.linspace(0, 1, NDENS + 1))
        idx = np.clip(np.searchsorted(edges, dens, side='right') - 1, 0, NDENS - 1)
        out['by_density'] = [dict(delta_lo=float(edges[k]), delta_hi=float(edges[k + 1]),
                                  frac_capped=float(capped[idx == k].mean()), mean_s=float(s[idx == k].mean()),
                                  share_removed=float(removed[idx == k].sum()/max(removed.sum(), 1e-300)),
                                  mean_occupancy=float(occ[idx == k].mean()))
                             for k in range(NDENS)]
        out['corr_capped_density'] = float(np.corrcoef(capped.astype(float), dens)[0, 1])
        out['corr_capped_occupancy'] = float(np.corrcoef(capped.astype(float), occ)[0, 1])
    return out


def main():
    out, result = Path(sys.argv[1]), Path(sys.argv[2])
    pts = points()
    res = {name: {str(a): {} for a in ALPHAS} for name in pts}
    density = None
    for s in E.SAMPLES:
        t0 = time.time()
        sample = mm.Sample(s)
        for r in ROTS:
            geo, meta = E.load_geometry(out/f'geom_{s}_b0_r{r}.npz')
            bases = E.load_bases(out, s, 0, r, meta['F_C'], meta['F_SAT'])
            sim = Path(meta['sim'])
            shells_info = mm.recover_shell_info(str(sim/'z_values.txt'), max_z=49)
            nuisance = np.load(sim/f'nuisance_b0_r{r}.npy', allow_pickle=True).item()[s]
            _, _, _, ngal = mm.shells_for(sample, nuisance, shells_info)
            if density is None:
                density = np.load(sim/f'delta_b_{NSIDE}_0.npy').astype(np.float32)
            zgeo = np.load(out/f'geom_{s}_b0_r{r}.npz')
            for t, g in enumerate(geo):
                c = sample.cats[t]
                if sample.is_bfd:
                    real = c['selection_term'] == 0
                    ra, dec = c['ra'][real], c['dec'][real]
                    pix_cat = convert_to_pix_coord(ra, dec, nside=NSIDE)
                    pix_sim = rotate_and_rebin(convert_to_pix_coord(ra, dec, nside=2*NSIDE), NSIDE, r)
                else:
                    pix_sim = rotate_and_rebin(convert_to_pix_coord(c['ra'], c['dec'], nside=2*NSIDE), NSIDE, r)
                    pix_cat = pix_sim
                g_pix = zgeo[f'pix_{t}']
                occ = np.bincount(pix_cat, minlength=hp.nside2npix(NSIDE))[g_pix]
                w = ngal[t]/ngal[t].sum()
                need = np.unique(pix_sim)
                dproj = np.zeros(hp.nside2npix(NSIDE), np.float32)
                acc = np.zeros(len(need))
                for i in np.flatnonzero(w):
                    acc += w[i]*density[i][need]
                dproj[need] = acc
                if sample.is_bfd:
                    num = np.bincount(pix_cat, weights=dproj[pix_sim], minlength=hp.nside2npix(NSIDE))
                    den = np.bincount(pix_cat, minlength=hp.nside2npix(NSIDE))
                    dens = (num[g_pix]/np.maximum(den[g_pix], 1)).astype(float)
                else:
                    dens = dproj[g_pix].astype(float)
                g['occ'], g['dens'] = occ.astype(float), dens
            for name, d in pts.items():
                Sc, Ss, Mc, Ms = E.interp_bases(bases, meta, d)
                for t, g in enumerate(geo):
                    h = ia_variance(g, Sc[t], Ss[t], Mc[t], Ms[t], d['A_c'], d['A_s'])
                    v = g['v']
                    for a in ALPHAS:
                        sfac = cap_factor(v, h, a)
                        res[name][str(a)].setdefault(s, {}).setdefault(str(r), {})[str(t)] = summary(
                            v, h, sfac, g['occ'], g['dens'], a)
            E.log(s, 'rotation', r, 'done', f'{time.time() - t0:.0f} s')
            del geo, bases
        del sample
        result.write_text(json.dumps(dict(points=pts, results=res), indent=1) + '\n')
    result.write_text(json.dumps(dict(points=pts, results=res), indent=1) + '\n')
    E.log('saved', result)


if __name__ == '__main__':
    main()
