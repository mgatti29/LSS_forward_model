#!/usr/bin/env python3
"""Exact consistency tests of the N_gal >= NMIN light BFD catalogue (build_bfd_light_n3.py).

Per tomographic bin:
  1. mask: N_gal recomputed from the light catalogue is >= NMIN in every occupied pixel, the occupied pixels equal
     the stored mask, the counts equal the stored counts, no row outside the mask;
  2. data (no rotation): per-pixel sums N, Q, R (galaxy and selection rows) of the light catalogue equal those of
     the full catalogue on the retained pixels (max |difference|, expected exactly 0: same rows, same order);
     the count-normalised map g = -(1/N) Rbar_full^-1 Q is identical; its N-weighted mean equals the global
     estimator of the retained rows;
  3. random rotations (NROT seeds): the per-row angles are drawn exactly as bfd_utils.prepare_bfd_pqr_ordered
     draws them for the full catalogue (checked bit by bit against that function); the light catalogue gets the
     angles of its rows; sums on the retained pixels identical; N_gal and the set of finite rows unchanged by the
     rotation (mask is rotation independent);
  4. production make_bfd_map (random_rotate=False) on both catalogues: se1, se2, mask, n_source identical on the
     retained pixels;
  5. Rbar recomputed from the light catalogue vs the stored full-catalogue Rbar (why it must not be recomputed).

usage: validate_bfd_light_n3.py [--out-dir DIR] [--nmin 3] [--nrot 3]
output: OUT_DIR/BFD_light_N{nmin}_validation.json
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from LSS_forward_model import bfd_utils as bu                      # noqa: E402
from build_bfd_light_n3 import NSIDE, NPIX, pixel_sums, rmat, OUT_DIR, log   # noqa: E402


def process(cat, ang_real=None, ang_sel=None):
    """prepare_bfd_pqr_ordered with explicit per-row angles (None: no rotation)."""
    nobj = len(cat['ra'])
    out = np.zeros((nobj, 6))
    is_real = np.asarray(cat['selection_term']) == 0
    pqr0 = np.asarray(cat['pqr10_prelog'])[is_real]
    if ang_real is not None:
        pqr0 = bu.rotatePqr(ang_real, pqr0)
    out[is_real] = bu.stripMuPqr(bu.logPqr(pqr0))
    pqr1 = copy.deepcopy(np.asarray(cat['pqr'])[~is_real])
    pqr1[:, 0] = np.exp(pqr1[:, 0])
    p_, q_, r_ = bu.splitPqr(pqr1)
    r_ = r_ + q_[..., :, np.newaxis]*q_[..., np.newaxis, :]
    q_ = q_*p_[..., np.newaxis]
    r_ = r_*p_[..., np.newaxis, np.newaxis]
    pqr1 = bu.packPqr(p_, q_, r_, axis=-1)
    if ang_sel is not None:
        pqr1 = bu.rotatePqr(ang_sel, pqr1)
    out[~is_real] = bu.logPqr(pqr1)
    return out


def sums_from_pqr(pqr, pix, is_sel):
    p, q, r = bu.splitPqr(pqr)
    good = np.isfinite(p) & np.all(np.isfinite(q), 1) & np.all(np.isfinite(r), (1, 2))
    out = dict(n_bad=int((~good).sum()), n_gal=np.bincount(pix[good & ~is_sel], minlength=NPIX))
    pp = pix[good]
    out['Q'] = np.stack([np.bincount(pp, weights=q[good, j], minlength=NPIX) for j in range(2)])
    out['R'] = np.stack([np.bincount(pp, weights=r[good, 0, 0], minlength=NPIX),
                         np.bincount(pp, weights=r[good, 0, 1], minlength=NPIX),
                         np.bincount(pp, weights=r[good, 1, 1], minlength=NPIX)])
    return out


def maxdiff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b))) if a.size else 0.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--nmin', type=int, default=3)
    ap.add_argument('--nrot', type=int, default=3)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    tag = f'N{args.nmin}'
    meta = json.loads((out_dir/f'BFD_light_{tag}_meta.json').read_text())
    mk = np.load(out_dir/f'BFD_light_{tag}_mask.npz')
    t0 = time.time()
    full = np.load(meta['source'], allow_pickle=True).item()
    light = np.load(out_dir/f'BFD_catalog_v5_UNBLINDED_light_{tag}.npy', allow_pickle=True).item()
    log(f'loaded both catalogues in {time.time() - t0:.0f} s')
    res = dict(nmin=args.nmin, nrot=args.nrot, bins={})
    for t in sorted(full):
        cf, cl = full[t], light[t]
        b = {}
        pf = bu.convert_to_pix_coord(cf['ra'], cf['dec'], nside=NSIDE)
        pl = bu.convert_to_pix_coord(cl['ra'], cl['dec'], nside=NSIDE)
        sf = np.asarray(cf['selection_term']) != 0
        sl = np.asarray(cl['selection_term']) != 0
        keep = np.zeros(NPIX, bool)
        keep[mk[f'pix_{t}']] = True
        keep_rows = keep[pf]
        # 1. mask
        n_l = np.bincount(pl[~sl], minlength=NPIX)
        occ_l = np.bincount(pl, minlength=NPIX) > 0
        b['mask'] = dict(min_N_in_light=int(n_l[occ_l].min()), occupied_light_equals_mask=bool(np.array_equal(occ_l, keep)),
                         counts_equal_stored=bool(np.array_equal(n_l[keep], mk[f'n_gal_{t}'])),
                         rows_light_equals_kept_rows=bool(int(keep_rows.sum()) == len(pl)),
                         columns_equal=all(np.array_equal(np.asarray(cf[k])[keep_rows], np.asarray(cl[k]))
                                           for k in ('ra', 'dec', 'tomo', 'selection_term', 'pqr', 'pqr10_prelog')))
        # 2. data sums and maps
        Sf, Sl = pixel_sums(cf, pf), pixel_sums(cl, pl)
        k = keep
        b['data_sums_max_abs_diff_retained'] = {name: maxdiff(Sf[name][..., k], Sl[name][..., k])
                                                for name in ('n_gal', 'n_sel', 'p_gal', 'p_sel', 'Q_gal', 'Q_sel',
                                                             'R_gal', 'R_sel')}
        Rbar = np.array(meta['bins'][str(t)]['Rbar_code_sign'])
        Rinv = np.linalg.inv(Rbar)
        Qf, Ql = Sf['Q_gal'] + Sf['Q_sel'], Sl['Q_gal'] + Sl['Q_sel']
        Nf, Nl = Sf['n_gal'], Sl['n_gal']
        gf = -(Rinv @ Qf[:, k])/Nf[k]
        gl = -(Rinv @ Ql[:, k])/Nl[k]
        b['data_map_max_abs_diff'] = maxdiff(gf, gl)
        b['data_map_identical'] = bool(np.array_equal(gf, gl))
        mean_map = (gl*Nl[k]).sum(1)/Nl[k].sum()
        glob = -Rinv @ Ql[:, k].sum(1)/Nl[k].sum()
        b['N_weighted_mean_vs_global_retained'] = dict(map=mean_map.tolist(), global_=glob.tolist(),
                                                       rel_diff=float(np.max(np.abs(mean_map - glob))/np.max(np.abs(glob))))
        # 5. Rbar recomputed from the light rows
        Rl = rmat(Sl['R_gal'].sum(1) + Sl['R_sel'].sum(1))/Nl.sum()
        b['Rbar_recomputed_from_light'] = Rl.tolist()
        b['Rbar_light_over_full_minus_identity'] = (Rinv @ Rl - np.eye(2)).tolist()
        del Sf, Sl, Qf, Ql
        # 3. rotations
        rot = []
        nreal_f, nsel_f = int((~sf).sum()), int(sf.sum())
        for seed in range(args.nrot):
            rng = np.random.default_rng([20261110, t, seed])
            ang_r = rng.uniform(0.0, 2.0*np.pi, nreal_f)                    # as prepare_bfd_pqr_ordered
            ang_s = rng.uniform(0.0, 2.0*np.pi, nsel_f)
            pqr_f = process(cf, ang_r, ang_s)
            entry = {}
            if seed == 0:
                ref = bu.prepare_bfd_pqr_ordered(cf, random_rotate=True, rotate_selection=True,
                                                 rng=np.random.default_rng([20261110, t, seed]))
                entry['explicit_angles_equal_production_bitwise'] = bool(np.array_equal(ref, pqr_f))
                del ref
            pqr_l = process(cl, ang_r[keep_rows[~sf]], ang_s[keep_rows[sf]])
            entry['rows_bitwise_equal'] = bool(np.array_equal(pqr_f[keep_rows], pqr_l))
            Rf_ = sums_from_pqr(pqr_f, pf, sf)
            Rl_ = sums_from_pqr(pqr_l, pl, sl)
            del pqr_f, pqr_l
            entry['n_bad_rows_full'] = Rf_['n_bad']
            entry['N_gal_unchanged_by_rotation'] = bool(np.array_equal(Rf_['n_gal'], Nf))
            entry['sums_max_abs_diff_retained'] = dict(Q=maxdiff(Rf_['Q'][:, k], Rl_['Q'][:, k]),
                                                       R=maxdiff(Rf_['R'][:, k], Rl_['R'][:, k]),
                                                       N=maxdiff(Rf_['n_gal'][k], Rl_['n_gal'][k]))
            g1 = -(Rinv @ Rf_['Q'][:, k])/Rf_['n_gal'][k]
            g2 = -(Rinv @ Rl_['Q'][:, k])/Rl_['n_gal'][k]
            entry['map_identical'] = bool(np.array_equal(g1, g2))
            entry['Rbar_rotated_full_minus_data'] = (rmat(Rf_['R'].sum(1))/Rf_['n_gal'].sum() - Rbar).tolist()
            rot.append(entry)
            del Rf_, Rl_
        b['rotations'] = rot
        # 4. production map maker, no rotation
        a = bu.make_bfd_map(cf, nside=NSIDE, random_rotate=False, rotate_selection=False)
        c = bu.make_bfd_map(cl, nside=NSIDE, random_rotate=False, rotate_selection=False)
        b['make_bfd_map_no_rotation'] = {name: dict(max_abs_diff_retained=maxdiff(x[k], y[k]),
                                                    identical_retained=bool(np.array_equal(x[k], y[k])),
                                                    nonzero_outside_mask_light=int(np.sum(y[~k] != 0)))
                                         for name, x, y in zip(('se1', 'se2', 'mask', 'n_source'), a, c)}
        del a, c
        res['bins'][str(t)] = b
        log(f'bin {t}:', json.dumps(b)[:1500])
        full[t] = light[t] = None
    ok = all(v['mask']['occupied_light_equals_mask'] and v['mask']['counts_equal_stored'] and v['mask']['columns_equal']
             and v['data_map_identical'] and all(e['map_identical'] and e['rows_bitwise_equal'] and
                                                 e['N_gal_unchanged_by_rotation'] for e in v['rotations'])
             and all(x['identical_retained'] for x in v['make_bfd_map_no_rotation'].values())
             for v in res['bins'].values())
    res['all_exact_checks_passed'] = bool(ok)
    (out_dir/f'BFD_light_{tag}_validation.json').write_text(json.dumps(res, indent=1) + '\n')
    log(f'all exact checks passed: {ok}; done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
