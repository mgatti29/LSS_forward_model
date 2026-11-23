#!/usr/bin/env python3
"""Trace of the BFD pure-noise map (e1n) vs the noise inside the signal map (e1 - dm g1).

Uses make_bfd_map directly on the BFD catalogue (bins 0 and 3), no source-clustering rescaling:
  A. data (no rotation): pixels with objects, pixels masked by the |g| > 1 cut or a singular R
  B. two independent random-rotation noise maps: masks, overlap of the masked pixels, rms on own / other mask
  C. selection rows: their share of the summed R per pixel; masked pixels vs n_terms, det R
  D. signal injection (decompose_shear): e - g vs the exact noise part (same rotation), R_real vs R_total
output: tests/out/bfd_noise_trace.json
"""
import json, sys
from pathlib import Path
import numpy as np, healpy as hp
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import make_mocks as mm
from LSS_forward_model import bfd_utils as bu

NSIDE = 1024
cfg = mm.SAMPLES_CFG['samples']['BFD']
cats = np.load(mm.large_input(cfg['catalogue']['large_inputs']), allow_pickle=True).item()
rms = lambda a: float(np.sqrt(np.mean(np.asarray(a, float)**2)))
res = {}
for t in (0, 3):
    c = cats[t]
    r = {}
    sel = np.asarray(c['selection_term']) != 0
    r['n_real'], r['n_selection_rows'] = int((~sel).sum()), int(sel.sum())
    # A. data
    e1d, e2d, md, nsd = bu.make_bfd_map(c, nside=NSIDE, random_rotate=False, rotate_selection=False)
    pix_all = bu.convert_to_pix_coord(np.asarray(c['ra']), np.asarray(c['dec']), nside=NSIDE)
    n_terms = np.bincount(pix_all, minlength=hp.nside2npix(NSIDE))
    occupied = n_terms > 0
    r['A_data'] = dict(occupied=int(occupied.sum()), masked_fraction_of_occupied=float(1 - md[occupied].mean()),
                       rms_e1=rms(e1d[md > 0]))
    # B. two noise realisations
    out = []
    for seed in (1, 2):
        e1, e2, m, ns = bu.make_bfd_map(c, nside=NSIDE, random_rotate=True, rotate_selection=True,
                                        rng=np.random.default_rng(seed))
        out.append((e1, e2, m))
    (a1, _, ma), (b1, _, mb) = out
    both = (ma > 0) & (mb > 0)
    r['B_noise'] = dict(masked_fraction_1=float(1 - ma[occupied].mean()), masked_fraction_2=float(1 - mb[occupied].mean()),
                        masked_in_both=float(np.mean((ma[occupied] == 0) & (mb[occupied] == 0))),
                        rms_1_own_mask=rms(a1[ma > 0]), rms_2_own_mask=rms(b1[mb > 0]),
                        rms_1_on_mask_2_with_zeros=rms(a1[mb > 0]), frac_zero_1_on_mask_2=float(np.mean(a1[mb > 0] == 0)),
                        rms_1_on_both=rms(a1[both]), rms_2_on_both=rms(b1[both]),
                        frac_abs_gt_0p5=float(np.mean(np.abs(a1[ma > 0]) > 0.5)))
    # C. R per pixel: real vs selection rows (rotation invariant trace)
    pqr, is_sel = bu.prepare_bfd_pqr_ordered(c, return_selection_mask=True)
    p_, q_, r_ = bu.splitPqr(pqr)
    good = np.isfinite(p_) & np.all(np.isfinite(r_), axis=(1, 2))
    trR = r_[:, 0, 0] + r_[:, 1, 1]
    npix = hp.nside2npix(NSIDE)
    tr_real = np.bincount(pix_all[good & ~is_sel], weights=trR[good & ~is_sel], minlength=npix)
    tr_sel = np.bincount(pix_all[good & is_sel], weights=trR[good & is_sel], minlength=npix)
    occ = occupied
    share = np.divide(tr_sel[occ], tr_real[occ] + tr_sel[occ], out=np.zeros(occ.sum()), where=(tr_real[occ] + tr_sel[occ]) != 0)
    masked = ma[occ] == 0
    r['C_R'] = dict(median_selection_share_trR=float(np.median(share)), mean_selection_share=float(np.mean(share)),
                    sign_tr_real_median=float(np.median(tr_real[occ])), sign_tr_sel_median=float(np.median(tr_sel[occ])),
                    share_in_masked_pixels_median=float(np.median(share[masked])) if masked.any() else None,
                    n_terms_median_all=float(np.median(n_terms[occ])), n_terms_median_masked=float(np.median(n_terms[occ][masked])))
    # D. signal injection, same rotation for signal and noise part
    g1 = np.zeros(npix); g2 = np.zeros(npix)
    rng = np.random.default_rng(5)
    g1[occ] = rng.normal(0, 0.02, occ.sum()); g2[occ] = rng.normal(0, 0.02, occ.sum())
    o = bu.make_bfd_map(c, nside=NSIDE, random_rotate=True, rotate_selection=True, gamma1=g1, gamma2=g2,
                        decompose_shear=True, rng=np.random.default_rng(1))
    e1s, e2s, ms, ns, e1b, e2b, mbase, e1sig, e2sig = o
    use = (ms > 0) & (mbase > 0)
    r['D_signal'] = dict(rms_e_minus_g=rms((e1s - g1)[ms > 0]), rms_base_on_signal_mask=rms(e1b[ms > 0]),
                         rms_base_own_mask=rms(e1b[mbase > 0]), rms_base_where_both=rms(e1b[use]),
                         rms_e_minus_g_where_both=rms((e1s - g1)[use]),
                         response_slope=float(np.sum(e1sig[use]*g1[use])/np.sum(g1[use]**2)),
                         rms_signal_part_minus_g=rms((e1sig - g1)[use]),
                         frac_zero_base_on_signal_mask=float(np.mean(e1b[ms > 0] == 0)),
                         masked_signal_fraction=float(1 - ms[occ].mean()))
    res[f'bin{t}'] = r
    print(t, json.dumps(r, indent=1), flush=True)
(HERE/'out').mkdir(exist_ok=True)
(HERE/'out'/'bfd_noise_trace.json').write_text(json.dumps(res, indent=1) + '\n')
