#!/usr/bin/env python3
"""Stage 2 of the joint halo-IA variance-budget study: evaluate prior draws on the stage-1 bases (ia_joint_scan.py).

For every draw (A_c, eta_c, A_s, eta_s, f_c, f_sat) of the prior, every rotation r and every sample s (DESY6, NGC,
SGC, BFD [+ BFD with unit response]) and bin, with the self-term bases interpolated bilinearly in (f_c, f_sat) and
linearly in eta (fine grid), it computes exactly, on the observed pixels of that sample:
    invalid   fraction of pixels with V_IA >= V_avail (production rule; the 2% criterion uses the worst bin)
    kappa     critical amplitude multiplier: the draw with (kappa A_c, kappa A_s) has exactly 2% invalid pixels
              (kappa^2 = 2% quantile of V_avail / V_IA); accepted <=> kappa > 1 in every bin
    excess    sum_p max(V_IA - V_avail, 0) / sum_p V_avail  (variance the mock cannot represent if clipped)
    sp_invalid  fraction of observed pixels whose NSIDE-64 superpixel has sum V_IA > sum V_avail (budget per
              superpixel: insensitive to the Poisson count fluctuations of single pixels)
    ratio     mean V_IA / V_avail
Validation: (i) the 'val' node (stored draw rebuilt with the production numerator) - direct vs emulated fractions;
(ii) off-grid populations: exact bases vs the interpolated grid bases on the same marks; (iii) re-seeded
populations (population noise).

usage: ia_joint_eval.py OUTDIR [--ndraw 50000] [--nproc 128] [--rots 0 1 2 3] [--shard i --nshard n] [--no-validation]
       ia_joint_eval.py OUTDIR --merge --nshard n --ndraw 50000
output: OUTDIR/eval_draws.npz (or eval_draws_shard{i}.npz), OUTDIR/eval_validation.json
Sharding: draws i::n of the same seeded set; run one shard per NUMA domain (numactl --cpunodebind=k --membind=k), each
loading its own copy of the bases - a forked pool reading one copy is limited by one domain's memory bandwidth.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse

SAMPLES = ['DESY6', 'NGC', 'SGC', 'BFD']
VARIANTS = ['DESY6', 'NGC', 'SGC', 'BFD', 'BFD_unit']
NOMINAL = dict(A_c=(-2, 2.5), eta_c=(-2, 0), A_s=(0, 0.03), eta_s=(-.5, .5), f_c=(.2, .7), f_sat=(.3, .8))
PAR = ['A_c', 'eta_c', 'A_s', 'eta_s', 'f_c', 'f_sat']
LEVEL = 0.02
G = {}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_geometry(path):
    z = np.load(path)
    meta = json.loads(str(z['meta']))
    geo = []
    for t in range(int(z['ntomo'])):
        n = len(z[f'pix_{t}'])
        W = sparse.csr_matrix((z[f'W_data_{t}'], z[f'W_indices_{t}'], z[f'W_indptr_{t}']), shape=(n, 49152))
        sp = z[f'sp_dom_{t}']
        _, spi = np.unique(sp, return_inverse=True)
        geo.append(dict(v=z[f'v_avail_{t}'].astype(float), resp2=z[f'resp2_{t}'].astype(float), W=W,
                        unit=np.allclose(W.data, 1.) and np.all(np.diff(W.indptr) == 1), sp=W.indices.copy()
                        if np.all(np.diff(W.indptr) == 1) else None, spi=spi, nsp=int(spi.max()) + 1))
    return geo, meta


def load_bases(outdir, s, b, r, F_C, F_SAT):
    Sc = Ss = Mc = Ms = None
    for i, fc in enumerate(F_C):
        for j, fs in enumerate(F_SAT):
            z = np.load(outdir/f'bases_{s}_b{b}_r{r}_fc{fc}_fs{fs}.npz')
            if Sc is None:
                Sc = np.zeros((len(F_C), len(F_SAT)) + z['Sc'].shape, np.float32)
                Ss = np.zeros((len(F_C), len(F_SAT)) + z['Ss'].shape, np.float32)
                Mc = np.zeros((len(F_C), len(F_SAT)) + z['Mc'].shape)
                Ms = np.zeros((len(F_C), len(F_SAT)) + z['Ms'].shape)
            Sc[i, j], Ss[i, j], Mc[i, j], Ms[i, j] = z['Sc'], z['Ss'], z['Mc'], z['Ms']
    return dict(Sc=Sc, Ss=Ss, Mc=Mc, Ms=Ms)


def weights_1d(grid, x):
    k = int(np.clip(np.searchsorted(grid, x) - 1, 0, len(grid) - 2))
    u = (x - grid[k])/(grid[k + 1] - grid[k])
    return k, float(np.clip(u, 0, 1)) if grid[0] <= x <= grid[-1] else u


def interp_bases(B, meta, d):
    """self-term bases of draw d: bilinear in (f_c, f_sat), linear in eta."""
    i, u = weights_1d(meta['F_C'], d['f_c'])
    j, v = weights_1d(meta['F_SAT'], d['f_sat'])
    kc, wc = weights_1d(meta['ETA_C'], d['eta_c'])
    ks, ws = weights_1d(meta['ETA_S'], d['eta_s'])
    wf = [((i, j), (1 - u)*(1 - v)), ((i + 1, j), u*(1 - v)), ((i, j + 1), (1 - u)*v), ((i + 1, j + 1), u*v)]
    Sc = sum(w*((1 - wc)*B['Sc'][a, b2, kc].astype(float) + wc*B['Sc'][a, b2, kc + 1]) for (a, b2), w in wf if w)
    Ss = sum(w*((1 - ws)*B['Ss'][a, b2, ks].astype(float) + ws*B['Ss'][a, b2, ks + 1]) for (a, b2), w in wf if w)
    Mc = sum(w*((1 - wc)*B['Mc'][a, b2, kc] + wc*B['Mc'][a, b2, kc + 1]) for (a, b2), w in wf if w)
    Ms = sum(w*((1 - ws)*B['Ms'][a, b2, ks] + ws*B['Ms'][a, b2, ks + 1]) for (a, b2), w in wf if w)
    return Sc, Ss, Mc, Ms


def node_bases_at(z, meta, d):
    kc, wc = weights_1d(meta['ETA_C'], d['eta_c'])
    ks, ws = weights_1d(meta['ETA_S'], d['eta_s'])
    Sc = (1 - wc)*z['Sc'][kc].astype(float) + wc*z['Sc'][kc + 1]
    Ss = (1 - ws)*z['Ss'][ks].astype(float) + ws*z['Ss'][ks + 1]
    Mc = (1 - wc)*z['Mc'][kc] + wc*z['Mc'][kc + 1]
    Ms = (1 - ws)*z['Ms'][ks] + ws*z['Ms'][ks + 1]
    return Sc, Ss, Mc, Ms


def metrics(geo, Sc, Ss, Mc, Ms, A_c, A_s, unit_resp=False):
    out = np.zeros((len(geo), 5))
    for t, g in enumerate(geo):
        S = A_c**2*Sc[t] + A_s**2*Ss[t]
        M = A_c**2*Mc[t] + A_s**2*Ms[t]
        E = S[g['sp']] if g['sp'] is not None else g['W'] @ S
        h = 0.5*M*E/np.sum(E)
        if not unit_resp:
            h = h*g['resp2']
        v = g['v']
        with np.errstate(divide='ignore'):
            r = np.where(h > 0, v/h, np.inf)
        k = int(np.floor(LEVEL*len(v)))
        kap2 = np.partition(r, k)[k]
        hs = np.bincount(g['spi'], h, g['nsp'])
        vs = np.bincount(g['spi'], v, g['nsp'])
        out[t] = (np.mean(h >= v), np.sqrt(kap2), np.sum(np.clip(h - v, 0, None))/np.sum(v),
                  np.mean((hs > vs)[g['spi']]), np.mean(h/v))
    return out


def work(i):
    d = {k: G['draws'][k][i] for k in PAR}
    res = np.zeros((len(G['rots']), len(VARIANTS), 4, 5), np.float32)
    for ir, r in enumerate(G['rots']):
        for iv, var in enumerate(VARIANTS):
            s = 'BFD' if var.startswith('BFD') else var
            geo, meta = G['geo'][(s, r)]
            Sc, Ss, Mc, Ms = interp_bases(G['bases'][(s, r)], meta, d)
            res[ir, iv] = metrics(geo, Sc, Ss, Mc, Ms, d['A_c'], d['A_s'], unit_resp=var == 'BFD_unit')
    return i, res


def validation(outdir, b, rots):
    """off-grid / re-seeded populations and the direct check of the stored draw."""
    import multiprocessing as mp  # noqa: F401
    rng = np.random.default_rng(5)
    out = {'direct': {}, 'offgrid': {}, 'reseed': {}}
    for s in SAMPLES:
        for r in rots:
            if (s, r) not in G['geo']:
                continue
            geo, meta = G['geo'][(s, r)]
            for f in sorted(outdir.glob(f'bases_{s}_b{b}_r{r}_*_fc*.npz')):
                name = f.stem.split(f'_r{r}_')[1]
                z = np.load(f)
                fc, fs = float(z['f_c']), float(z['f_sat'])
                if name.startswith('val'):
                    out['direct'][f'{s}_r{r}'] = json.loads(str(z['validation']))
                    continue
                marks = [dict(A_c=rng.uniform(-2, 2.5), eta_c=rng.uniform(-2, 0), A_s=rng.uniform(0, .03),
                              eta_s=rng.uniform(-.5, .5), f_c=fc, f_sat=fs) for _ in range(120)]
                rows = []
                for d in marks:
                    if name.startswith('off'):
                        ref = interp_bases(G['bases'][(s, r)], meta, d)
                    else:                                   # re-seeded node: same node of the grid
                        i = meta['F_C'].index(fc)
                        j = meta['F_SAT'].index(fs)
                        B = G['bases'][(s, r)]
                        ref = node_bases_at({k: B[k][i, j] for k in ('Sc', 'Ss', 'Mc', 'Ms')}, meta, d)
                    exact = node_bases_at(z, meta, d)
                    a = metrics(geo, *exact, d['A_c'], d['A_s'])
                    bb = metrics(geo, *ref, d['A_c'], d['A_s'])
                    rows.append((a[:, 0].max(), bb[:, 0].max(), a[:, 1].min(), bb[:, 1].min()))
                rows = np.array(rows)
                key = 'offgrid' if name.startswith('off') else 'reseed'
                acc_a, acc_b = rows[:, 2] > 1, rows[:, 3] > 1
                out[key][f'{s}_r{r}_{name}'] = dict(
                    f_c=fc, f_sat=fs, n=len(rows),
                    worst_invalid_mean_abs_diff=float(np.mean(np.abs(rows[:, 0] - rows[:, 1]))),
                    worst_invalid_max_abs_diff=float(np.max(np.abs(rows[:, 0] - rows[:, 1]))),
                    kappa_rel_diff_rms=float(np.sqrt(np.mean((rows[:, 3]/rows[:, 2] - 1)**2))),
                    kappa_rel_diff_mean=float(np.mean(rows[:, 3]/rows[:, 2] - 1)),
                    accept_agreement=float(np.mean(acc_a == acc_b)), accept_exact=float(acc_a.mean()),
                    accept_interp=float(acc_b.mean()))
                log(key, s, r, name, json.dumps(out[key][f'{s}_r{r}_{name}']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir', type=Path)
    ap.add_argument('--b', type=int, default=0)
    ap.add_argument('--ndraw', type=int, default=50000)
    ap.add_argument('--nproc', type=int, default=128)
    ap.add_argument('--rots', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--no-validation', action='store_true')
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    t0 = time.time()
    if args.merge:
        z = [np.load(args.outdir/f'eval_draws_shard{i}.npz') for i in range(args.nshard)]
        res = np.zeros((args.ndraw,) + z[0]['res'].shape[1:], np.float32)
        for i, zi in enumerate(z):
            assert np.array_equal(zi['idx'], np.arange(i, args.ndraw, args.nshard))
            res[zi['idx']] = zi['res']
        draws = {k: z[0][k] for k in PAR}
        for zi in z[1:]:
            assert all(np.array_equal(zi[k], draws[k]) for k in PAR)
        np.savez(args.outdir/'eval_draws.npz', res=res, rots=z[0]['rots'], variants=z[0]['variants'],
                 metrics=z[0]['metrics'], **draws)
        log(f'merged {args.nshard} shards -> eval_draws.npz')
        return
    G['rots'] = args.rots
    G['geo'], G['bases'] = {}, {}
    for s in SAMPLES:
        for r in args.rots:
            geo, meta = load_geometry(args.outdir/f'geom_{s}_b{args.b}_r{r}.npz')
            G['geo'][(s, r)] = (geo, meta)
            G['bases'][(s, r)] = load_bases(args.outdir, s, args.b, r, meta['F_C'], meta['F_SAT'])
            log(f'loaded {s} r{r}')
    rng = np.random.default_rng(20261110)
    draws = {k: rng.uniform(*NOMINAL[k], args.ndraw) for k in PAR}
    G['draws'] = draws
    idx = np.arange(args.shard, args.ndraw, args.nshard)
    if args.no_validation:
        return run_draws(args, draws, idx, t0)
    val = validation(args.outdir, args.b, args.rots)
    (args.outdir/'eval_validation.json').write_text(json.dumps(val, indent=1) + '\n')
    # special points: TNG300 fits (config/ia_priors.json) and the IA draws stored with the test mocks
    special = {'TNG outlier fit': dict(A_c=2.566, eta_c=-1.0, A_s=0.0193, eta_s=0.0, f_c=0.5, f_sat=0.8),
               'TNG chi2 (DES footprint) fit': dict(A_c=1.71, eta_c=-1.0, A_s=0.025, eta_s=0.0, f_c=0.7, f_sat=0.6)}
    for r in args.rots:
        st = G['geo'][('DESY6', r)][1]['stored_ia']
        special[f'stored draw r{r}'] = {k: st[k] for k in PAR}
    sp_out = {}
    for name, d in special.items():
        G['draws'] = {k: np.array([v]) for k, v in d.items()}
        _, rr = work(0)
        sp_out[name] = dict(params=d, worst_invalid={v: rr[:, iv, :, 0].max(1).tolist() for iv, v in enumerate(VARIANTS)},
                            kappa_min={v: rr[:, iv, :, 1].min(1).tolist() for iv, v in enumerate(VARIANTS)},
                            invalid_per_bin={v: rr[:, iv, :, 0].tolist() for iv, v in enumerate(VARIANTS)})
        log('special', name, json.dumps(sp_out[name]['worst_invalid']))
    val['special_points'] = sp_out
    (args.outdir/'eval_validation.json').write_text(json.dumps(val, indent=1) + '\n')
    G['draws'] = draws
    run_draws(args, draws, idx, t0)


def run_draws(args, draws, idx, t0):
    import multiprocessing as mp
    res = np.zeros((len(idx), len(args.rots), len(VARIANTS), 4, 5), np.float32)
    pos = {int(i): k for k, i in enumerate(idx)}
    with mp.get_context('fork').Pool(args.nproc) as pool:
        for n, (i, r) in enumerate(pool.imap_unordered(work, idx.tolist(), chunksize=4)):
            res[pos[i]] = r
            if n % 1000 == 0:
                log(f'shard {args.shard}: {n}/{len(idx)}')
    name = 'eval_draws.npz' if args.nshard == 1 else f'eval_draws_shard{args.shard}.npz'
    np.savez(args.outdir/name, res=res, idx=idx, rots=np.array(args.rots), variants=np.array(VARIANTS),
             metrics=np.array(['invalid', 'kappa', 'excess', 'sp_invalid', 'ratio']), **draws)
    log(f'shard {args.shard} done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
