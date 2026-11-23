#!/usr/bin/env python3
"""Light BFD catalogue for the count-normalised maps: remove the rows of pixels with N_gal < NMIN (default 3).

The count-normalised BFD map (agreed 2026-11-10) is, per tomographic bin and NSIDE-1024 pixel p,

    g_p = -(1/N_p) Rbar^-1 Q_p,   Rbar = sum over the FULL catalogue of R (galaxy + selection rows) / N_total,

on the fixed mask N_p >= NMIN, without |g| cut or weights.  N_p counts the real objects (selection_term == 0)
at their catalogue positions, so the mask depends only on positions: it is identical for every random rotation
of the shapes and every simulation rotation (the maps are built in the catalogue frame).

This script reads the existing light catalogue (BFD_catalog_v5_UNBLINDED_light.npy, never modified), and writes

  OUT/BFD_catalog_v5_UNBLINDED_light_N3.npy   same format ({bin: ra, dec, tomo, selection_term, pqr,
                                               pqr10_prelog}), only the rows (galaxies AND selection rows) of the
                                               retained pixels, original row order
  OUT/BFD_light_N3_mask.npz                    per bin: retained pixels (RING, NSIDE 1024) and N_gal there, removed
                                               pixels and their N_gal, lam (expected counts of the FULL catalogue,
                                               bfd_utils.bfd_expected_counts) on the retained pixels; the shared mask
  OUT/BFD_light_N3_meta.json                   Rbar of the full catalogue (2x2, code and library sign, galaxy and
                                               selection parts), counts, selection Q/R in the removed pixels and the
                                               effective response of the retained pixels, provenance

Rbar must be taken from the metadata (full catalogue).  Recomputing it from the light catalogue drops the
response of the removed rows (reported as 'Rbar_light_over_full').

usage: build_bfd_light_n3.py [--source PATH] [--out-dir DIR] [--nmin 3]
"""
import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from LSS_forward_model import bfd_utils as bu                      # noqa: E402

SOURCE = '/global/cfs/cdirs/m5099/DESY6_bfd/BFD_catalog_v5_UNBLINDED_light.npy'
OUT_DIR = '/global/cfs/cdirs/m5099/DESY6_bfd'
NSIDE = 1024
NPIX = hp.nside2npix(NSIDE)
COLUMNS = ('ra', 'dec', 'tomo', 'selection_term', 'pqr', 'pqr10_prelog')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def sha256(path, chunk=1 << 26):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def rmat(r3):
    return np.array([[r3[0], r3[1]], [r3[1], r3[2]]])


def pixel_sums(cat, pix=None):
    """Per-pixel sums with the production row processing (bfd_utils.prepare_bfd_pqr_ordered, no rotation),
    split into galaxy and selection rows.  Returns dict of full-sky arrays and the number of non-finite rows."""
    if pix is None:
        pix = bu.convert_to_pix_coord(cat['ra'], cat['dec'], nside=NSIDE)
    pqr = bu.prepare_bfd_pqr_ordered(cat, random_rotate=False, rotate_selection=False)
    p, q, r = bu.splitPqr(pqr)
    del pqr
    good = np.isfinite(p) & np.all(np.isfinite(q), 1) & np.all(np.isfinite(r), (1, 2))
    is_sel = np.asarray(cat['selection_term']) != 0
    out = {'n_bad_rows': int((~good).sum())}
    for name, rows in (('gal', good & ~is_sel), ('sel', good & is_sel)):
        pp = pix[rows]
        out['n_' + name] = np.bincount(pp, minlength=NPIX)
        out['p_' + name] = np.bincount(pp, weights=p[rows], minlength=NPIX)
        out['Q_' + name] = np.stack([np.bincount(pp, weights=q[rows, j], minlength=NPIX) for j in range(2)])
        out['R_' + name] = np.stack([np.bincount(pp, weights=r[rows, 0, 0], minlength=NPIX),
                                     np.bincount(pp, weights=r[rows, 0, 1], minlength=NPIX),
                                     np.bincount(pp, weights=r[rows, 1, 1], minlength=NPIX)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default=SOURCE)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--nmin', type=int, default=3)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    tag = f'N{args.nmin}'
    out_cat = out_dir/f'BFD_catalog_v5_UNBLINDED_light_{tag}.npy'
    out_mask = out_dir/f'BFD_light_{tag}_mask.npz'
    out_meta = out_dir/f'BFD_light_{tag}_meta.json'
    for p in (out_cat, out_mask, out_meta):
        if p.exists():
            raise SystemExit(f'{p} exists; remove it first (the script never overwrites)')

    t0 = time.time()
    src = Path(args.source)
    meta = dict(created=datetime.datetime.now().isoformat(timespec='seconds'), script=str(Path(__file__).resolve()),
                source=str(src), source_size=src.stat().st_size, nside=NSIDE, nest=False, nmin=args.nmin,
                estimator='g_p = -(1/N_p) Rbar^-1 Q_p on N_p >= nmin, Rbar = full-catalogue sum R (galaxy + '
                          'selection rows) / N_total; no |g| cut, no weights',
                sign='Rbar_code_sign = sum R of the stored rows (make_bfd_map convention, negative); '
                     'library sign = -code sign (positive, matches mdet)',
                bins={})
    log('sha256 of the source')
    meta['source_sha256'] = sha256(src)
    log(f'loading {src}')
    cats = np.load(src, allow_pickle=True).item()
    log(f'loaded in {time.time() - t0:.0f} s; bins {sorted(cats)}')

    lam_maps = bu.bfd_expected_counts(cats, NSIDE)          # expected counts of the FULL catalogue
    light, masks = {}, {}
    keep_maps = []
    for t in sorted(cats):
        c = cats[t]
        n_rows = len(c['ra'])
        is_sel = np.asarray(c['selection_term']) != 0
        pix = bu.convert_to_pix_coord(c['ra'], c['dec'], nside=NSIDE)
        n_gal = np.bincount(pix[~is_sel], minlength=NPIX)
        occupied = np.bincount(pix, minlength=NPIX) > 0
        keep_pix = n_gal >= args.nmin
        keep_rows = keep_pix[pix]
        keep_maps.append(keep_pix)
        S = pixel_sums(c, pix)
        assert S['n_bad_rows'] == 0, f'bin {t}: {S["n_bad_rows"]} non-finite rows (mask would not be position-only)'
        assert np.array_equal(S['n_gal'], n_gal)
        N_tot = float(n_gal.sum())
        Rg, Rs = S['R_gal'].sum(1), S['R_sel'].sum(1)
        Rbar = rmat(Rg + Rs)/N_tot
        Rinv = np.linalg.inv(Rbar)
        # removed pixels by class
        rem = occupied & ~keep_pix
        classes = {f'N={k}': rem & (n_gal == k) for k in range(args.nmin)}
        sel_total_tr = float(Rs[0] + Rs[2])
        tot_tr = float(Rg[0] + Rg[2] + Rs[0] + Rs[2])
        rem_stats = {}
        for name, m in classes.items():
            rem_stats[name] = dict(
                pixels=int(m.sum()), galaxies=int(n_gal[m].sum()), selection_rows=int(S['n_sel'][m].sum()),
                N_sel_expected=float(-S['p_sel'][m].sum()),
                R_sel_trace_frac_of_total_R=float((S['R_sel'][0][m].sum() + S['R_sel'][2][m].sum())/tot_tr),
                R_sel_trace_frac_of_selection_R=float((S['R_sel'][0][m].sum() + S['R_sel'][2][m].sum())/sel_total_tr),
                R_gal_trace_frac_of_total_R=float((S['R_gal'][0][m].sum() + S['R_gal'][2][m].sum())/tot_tr),
                Q_sel_sum=S['Q_sel'][:, m].sum(1).tolist(), Q_gal_sum=S['Q_gal'][:, m].sum(1).tolist())
        # effective response of the retained map, normalised by the full Rbar
        k = keep_pix
        N_ret = float(n_gal[k].sum())
        R_ret_uncond = rmat(S['R_gal'][:, k].sum(1) + S['R_sel'][:, k].sum(1))/N_ret
        N_sel = -S['p_sel']
        has_sel = k & (N_sel > 0)
        ratio = np.zeros(NPIX)
        ratio[has_sel] = n_gal[has_sel]/N_sel[has_sel]
        R_ret_cond = rmat(S['R_gal'][:, k].sum(1) + (S['R_sel'][:, k]*ratio[k]).sum(1))/N_ret
        R_light_rows = rmat(Rg + Rs - S['R_gal'][:, rem].sum(1) - S['R_sel'][:, rem].sum(1))/N_ret
        g_global = -Rinv @ (S['Q_gal'].sum(1) + S['Q_sel'].sum(1))/N_tot
        g_ret = -Rinv @ (S['Q_gal'][:, k].sum(1) + S['Q_sel'][:, k].sum(1))/N_ret
        b = dict(
            rows_full=n_rows, rows_light=int(keep_rows.sum()), rows_removed=int((~keep_rows).sum()),
            galaxies_full=int(N_tot), galaxies_light=int(N_ret), galaxies_removed=int(N_tot - N_ret),
            galaxies_removed_frac=float(1 - N_ret/N_tot),
            selection_rows_full=int(is_sel.sum()), selection_rows_light=int((is_sel & keep_rows).sum()),
            selection_rows_removed=int((is_sel & ~keep_rows).sum()),
            occupied_pixels=int(occupied.sum()), pixels_with_galaxies=int((n_gal > 0).sum()),
            retained_pixels=int(k.sum()), removed_pixels=int(rem.sum()),
            removed_pixels_frac_of_occupied=float(rem.sum()/occupied.sum()),
            removed_pixels_frac_of_pixels_with_galaxies=float((rem & (n_gal > 0)).sum()/(n_gal > 0).sum()),
            selection_only_pixels=int((occupied & (n_gal == 0)).sum()),
            Rbar_code_sign=Rbar.tolist(), Rbar_library_sign=(-Rbar).tolist(),
            Rbar_galaxy_part_code_sign=(rmat(Rg)/N_tot).tolist(), Rbar_selection_part_code_sign=(rmat(Rs)/N_tot).tolist(),
            selection_share_of_trace=float((Rs[0] + Rs[2])/(Rg[0] + Rg[2] + Rs[0] + Rs[2])),
            abar=float(-0.5*np.trace(Rbar)), N_total=N_tot,
            sum_N_sel_expected=float(N_sel[N_sel > 0].sum()),
            removed=rem_stats,
            effective_response_retained=dict(
                note='mean response of the retained map with the full Rbar: Rbar_full^-1 Rbar_ret (identity = no '
                     'calibration change).  unconditional: sum of R over retained pixels / N_ret (what the global '
                     'BFD sum of the retained rows calibrates); conditional on the detections: selection part '
                     'weighted by N_p / N_sel,p (the response of E[Q_p | N_p])',
                unconditional=(Rinv @ R_ret_uncond).tolist(), conditional=(Rinv @ R_ret_cond).tolist(),
                Rbar_light_over_full=(Rinv @ R_light_rows).tolist(),
                trace_shift_unconditional=float(0.5*np.trace(Rinv @ R_ret_uncond) - 1),
                trace_shift_conditional=float(0.5*np.trace(Rinv @ R_ret_cond) - 1),
                N_over_Nsel_retained=float(n_gal[has_sel].sum()/N_sel[has_sel].sum()),
                N_over_Nsel_removed=float(n_gal[rem & (N_sel > 0)].sum()/max(N_sel[rem & (N_sel > 0)].sum(), 1e-300))),
            global_shear_full=g_global.tolist(), global_shear_retained=g_ret.tolist(),
            lam_full_mean_retained=float(lam_maps[t][k].mean()))
        meta['bins'][str(t)] = b
        log(f'bin {t}: ' + json.dumps({kk: b[kk] for kk in ('rows_removed', 'galaxies_removed', 'galaxies_removed_frac',
                                                              'selection_rows_removed', 'removed_pixels',
                                                              'selection_only_pixels')}))
        log(f'bin {t}: Rbar {Rbar.tolist()}, effective response {b["effective_response_retained"]["trace_shift_unconditional"]:.2e} '
            f'(cond {b["effective_response_retained"]["trace_shift_conditional"]:.2e})')
        light[t] = {name: np.asarray(c[name])[keep_rows] for name in COLUMNS}
        masks[t] = dict(pix=np.flatnonzero(k).astype(np.int64), n_gal=n_gal[k].astype(np.int32),
                        removed_pix=np.flatnonzero(rem).astype(np.int64), removed_n_gal=n_gal[rem].astype(np.int32),
                        lam=lam_maps[t][k].astype(np.float32))
        del S, pix, n_gal, c
        cats[t] = None

    # shared mask (cross-bin statistics): pixels retained in every bin
    shared = np.logical_and.reduce(keep_maps)
    union = np.logical_or.reduce(keep_maps)
    meta['shared_mask'] = dict(
        pixels=int(shared.sum()), union_pixels=int(union.sum()),
        area_deg2=float(shared.sum()*hp.nside2pixarea(NSIDE, degrees=True)),
        extra_loss_vs_bin_mask=[float(1 - shared.sum()/m.sum()) for m in keep_maps],
        galaxies_lost_vs_bin_mask=[float(1 - masks[t]['n_gal'][shared[masks[t]['pix']]].sum()/masks[t]['n_gal'].sum())
                                   for t in sorted(masks)],
        bin_mask_area_deg2=[float(m.sum()*hp.nside2pixarea(NSIDE, degrees=True)) for m in keep_maps])
    log('shared mask', json.dumps(meta['shared_mask']))

    log(f'writing {out_mask}')
    np.savez(out_mask, nside=NSIDE, nmin=args.nmin, shared_pix=np.flatnonzero(shared).astype(np.int64),
             **{f'{k}_{t}': v for t, d in masks.items() for k, v in d.items()})
    log(f'writing {out_cat}')
    np.save(out_cat, light, allow_pickle=True)
    meta['output'] = dict(catalogue=str(out_cat), mask=str(out_mask), catalogue_size=out_cat.stat().st_size,
                          catalogue_sha256=sha256(out_cat))
    out_meta.write_text(json.dumps(meta, indent=1) + '\n')
    log(f'done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
