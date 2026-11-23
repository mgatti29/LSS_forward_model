#!/usr/bin/env python3
"""Analysis of the joint halo-IA variance-budget evaluation (ia_joint_eval.py): acceptance, overlap, boundary.

Groups: DES = DESY6 mdet; DECADE = NGC and SGC; BFD = count-normalised BFD (smooth response t_64r);
BFD_unit = same with unit response; ALL = DES & DECADE & BFD; OLD_SET = DES & DECADE (the samples of the
previous rule).  A draw is accepted by a sample if its worst-bin invalid fraction is <= 2% (production rule).

usage: ia_joint_analyse.py OUTDIR FIGDIR
output: OUTDIR/ia_joint_summary.json, FIGDIR/ia_joint_fig*.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LEVEL = 0.02
PAR = ['A_c', 'eta_c', 'A_s', 'eta_s', 'f_c', 'f_sat']
NOMINAL = dict(A_c=(-2, 2.5), eta_c=(-2, 0), A_s=(0, 0.03), eta_s=(-.5, .5), f_c=(.2, .7), f_sat=(.3, .8))
COL = {'DES': '#2a6fdb', 'DECADE': '#e8590c', 'BFD': '#2b8a3e', 'BFD_unit': '#82c91e', 'ALL': '#212529',
       'OLD_SET': '#862e9c'}


def binom(p, n):
    return float(np.sqrt(max(p*(1 - p), 0)/n))


def main():
    outdir, figdir = Path(sys.argv[1]), Path(sys.argv[2])
    figdir.mkdir(parents=True, exist_ok=True)
    z = np.load(outdir/'eval_draws.npz')
    res = z['res']                                     # (draw, rot, variant, bin, metric)
    variants = list(z['variants'])
    rots = list(z['rots'])
    d = {k: z[k] for k in PAR}
    n = len(d['A_c'])
    iv = {v: variants.index(v) for v in variants}
    worst = res[..., 0].max(3)                          # (draw, rot, variant)
    kap = res[..., 1].min(3)
    excess = res[..., 2].max(3)
    spinv = res[..., 3].max(3)
    acc_v = worst <= LEVEL
    groups = {'DES': [iv['DESY6']], 'DECADE': [iv['NGC'], iv['SGC']], 'BFD': [iv['BFD']], 'BFD_unit': [iv['BFD_unit']]}
    acc = {g: acc_v[:, :, ix].all(2) for g, ix in groups.items()}          # (draw, rot)
    acc['OLD_SET'] = acc['DES'] & acc['DECADE']
    acc['ALL'] = acc['OLD_SET'] & acc['BFD']
    acc['ALL_unit'] = acc['OLD_SET'] & acc['BFD_unit']
    kg = {g: kap[:, :, ix].min(2) for g, ix in groups.items()}
    kg['OLD_SET'] = np.minimum(kg['DES'], kg['DECADE'])
    kg['ALL'] = np.minimum(kg['OLD_SET'], kg['BFD'])
    out = dict(n_draws=n, rotations=rots, level=LEVEL, prior=NOMINAL)

    # ---------------- acceptance rates
    rates = {}
    for g, a in acc.items():
        per = a.mean(0)
        rates[g] = dict(per_rotation=per.tolist(), binomial_err_per_rotation=[binom(p, n) for p in per],
                        mean=float(per.mean()), rotation_std=float(per.std(ddof=1)),
                        accepted_in_all_rotations=float(a.all(1).mean()))
    out['acceptance_nominal'] = rates
    per_sample = {}
    for v in variants:
        a = acc_v[:, :, iv[v]]
        per_sample[v] = dict(acceptance=float(a.mean()), per_rotation=a.mean(0).tolist(),
                             worst_bin_counts=np.bincount(res[:, :, iv[v], :, 0].argmax(2)[~a], minlength=4).tolist(),
                             invalid_quantiles_50_90_99=np.quantile(worst[:, :, iv[v]], [.5, .9, .99]).tolist())
    out['per_sample'] = per_sample

    # ---------------- overlap of the rejections (per rotation, pooled)
    def frac(m):
        return float(m.mean())
    B, O = acc['BFD'], acc['OLD_SET']
    out['overlap'] = dict(
        pass_BFD_pass_others=frac(B & O), fail_BFD_pass_others=frac(~B & O), pass_BFD_fail_others=frac(B & ~O),
        fail_both=frac(~B & ~O),
        fail_BFD_pass_DES=frac(~B & acc['DES']), fail_BFD_pass_DECADE=frac(~B & acc['DECADE']),
        fail_DES_pass_BFD=frac(~acc['DES'] & B), fail_DECADE_pass_BFD=frac(~acc['DECADE'] & B),
        fail_DES_pass_DECADE=frac(~acc['DES'] & acc['DECADE']), fail_DECADE_pass_DES=frac(~acc['DECADE'] & acc['DES']),
        P_fail_BFD_given_fail_others=float((~B & ~O).sum()/max((~O).sum(), 1)),
        P_fail_others_given_fail_BFD=float((~B & ~O).sum()/max((~B).sum(), 1)),
        unit_response_vs_t64r_disagreement=frac(acc['BFD'] != acc['BFD_unit']))
    pairs = {}
    names = ['DESY6', 'NGC', 'SGC', 'BFD']
    for a in names:
        for b in names:
            pairs[f'{a}|{b}'] = frac(~acc_v[:, :, iv[a]] & acc_v[:, :, iv[b]])
    out['overlap']['fail_row_pass_col'] = pairs

    # ---------------- sensitivity (first-order, from binned conditional means)
    def first_order(y, nb=20):
        S = {}
        for k in PAR:
            e = np.quantile(d[k], np.linspace(0, 1, nb + 1))
            i = np.clip(np.digitize(d[k], e[1:-1]), 0, nb - 1)
            m = np.array([y[i == j].mean() for j in range(nb)])
            S[k] = float(np.var(m)/np.var(y)) if np.var(y) > 0 else 0.
        return S
    sens = {}
    for g in ('DES', 'DECADE', 'BFD', 'ALL'):
        ya = acc[g].mean(1).astype(float)
        with np.errstate(divide='ignore'):
            lac = np.log(np.abs(d['A_c'])[:, None]*kg[g]).mean(1)
        ok = np.isfinite(lac)
        sens[g] = dict(acceptance=first_order(ya), log_critical_Ac={k: v for k, v in first_order_masked(d, lac, ok).items()})
    out['sensitivity_first_order'] = sens

    # critical |A_c| (A_s/A_c ratio of the draw kept) vs f_c, f_sat, eta_c: medians and spread per group
    crit = {}
    fcb = np.linspace(.2, .7, 6)
    for g in ('DES', 'DECADE', 'BFD', 'ALL'):
        ac = (np.abs(d['A_c'])[:, None]*kg[g])               # (draw, rot)
        big = np.abs(d['A_c']) > 0.5                          # A_c-dominated draws (A_s term negligible)
        rows = []
        for a, b in zip(fcb[:-1], fcb[1:]):
            m = big & (d['f_c'] >= a) & (d['f_c'] < b)
            v = ac[m].ravel()
            rows.append(dict(f_c=[float(a), float(b)], q05_50_95=np.quantile(v, [.05, .5, .95]).tolist()))
        crit[g] = rows
    out['critical_Ac_by_fc'] = crit
    # linear fit of log critical |A_c| on (f_c, f_sat, eta_c, eta_s, A_s)
    fits = {}
    big = np.abs(d['A_c']) > 0.5
    X = np.stack([np.ones(n), d['f_c'], d['f_sat'], d['eta_c'], d['eta_s'], d['A_s']/0.03], 1)[big]
    for g in ('DES', 'DECADE', 'BFD', 'ALL'):
        y = np.log(np.abs(d['A_c'])[big, None]*kg[g][big]).mean(1)
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - np.var(y - X @ c)/np.var(y)
        fits[g] = dict(coef=dict(zip(['const', 'f_c', 'f_sat', 'eta_c', 'eta_s', 'A_s/0.03'], c.tolist())), R2=float(r2))
    out['log_critical_Ac_linear_fit'] = fits

    # ---------------- candidate priors (subsets of the uniform draws)
    absA = np.abs(d['A_c'])
    cands = {'nominal': np.ones(n, bool), '|A_c|<1': absA < 1, '|A_c|<1.15': absA < 1.15,
             '|A_c|<1.6-f_c': absA < 1.6 - d['f_c'], 'A_c^2(1-f_sat)<0.4': d['A_c']**2*(1 - d['f_sat']) < 0.4}
    # best linear boundary |A_c| < a - b f_c + c f_sat with >= 99% acceptance (worst rotation) for ALL
    best = {}
    for gname in ('ALL', 'OLD_SET', 'BFD'):
        bst = None
        for a in np.arange(0.8, 3.01, 0.05):
            for b in np.arange(0., 3.01, 0.1):
                for c in (0., 0.25, 0.5, 0.75, 1.0):
                    m = absA < a - b*d['f_c'] + c*d['f_sat']
                    if m.sum() < 1000:
                        continue
                    worst_rot = acc[gname][m].mean(0).min()
                    if worst_rot >= 0.99 and (bst is None or m.mean() > bst[0]):
                        bst = (m.mean(), a, b, c, worst_rot)
        best[gname] = dict(volume=bst[0], a=bst[1], b=bst[2], c=bst[3], acceptance_worst_rotation=bst[4])
        cands[f'best99_{gname}: |A_c|<{bst[1]:.2f}-{bst[2]:.1f}f_c+{bst[3]:.2f}f_sat'] = absA < bst[1] - bst[2]*d['f_c'] + bst[3]*d['f_sat']
    out['best_linear_boundary_99'] = best
    cand_out = {}
    for name, m in cands.items():
        row = dict(prior_volume=float(m.mean()), n=int(m.sum()))
        for g in ('DES', 'DECADE', 'BFD', 'BFD_unit', 'OLD_SET', 'ALL'):
            a = acc[g][m]
            row[g] = dict(mean=float(a.mean()), worst_rotation=float(a.mean(0).min()), err=binom(a.mean(), a.size))
        cand_out[name] = row
    out['candidate_priors'] = cand_out

    # ---------------- is the 2% rule physical?  excess variance and superpixel budget at the boundary
    phys = {}
    for v in ('DESY6', 'NGC', 'SGC', 'BFD'):
        w = worst[:, :, iv[v]].ravel()
        ex = excess[:, :, iv[v]].ravel()
        sp = spinv[:, :, iv[v]].ravel()
        rows = []
        for lo, hi in ((0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.3), (0.3, 1.01)):
            m = (w >= lo) & (w < hi)
            if m.sum() < 10:
                continue
            rows.append(dict(invalid=[lo, hi], n=int(m.sum()), excess_var_q50_95=np.quantile(ex[m], [.5, .95]).tolist(),
                             superpixel_invalid_q50_95=np.quantile(sp[m], [.5, .95]).tolist()))
        phys[v] = rows
    out['excess_and_superpixel_budget_vs_invalid'] = phys
    # acceptance under alternative criteria (pooled rotations)
    alt = {}
    for g, ix in groups.items():
        alt[g] = dict(superpixel_budget_le_2pct=float((spinv[:, :, ix] <= LEVEL).all(2).mean()),
                      superpixel_budget_zero=float((spinv[:, :, ix] <= 0).all(2).mean()),
                      excess_le_0p1pct=float((excess[:, :, ix] <= 1e-3).all(2).mean()),
                      excess_le_1pct=float((excess[:, :, ix] <= 1e-2).all(2).mean()),
                      pixel_rule_2pct=float(acc[g].mean()))
    out['alternative_criteria'] = alt
    # joint acceptance and 99% boundaries under each criterion (a draw passes a sample if the worst bin passes)
    crits = {'pixel_2pct': worst <= LEVEL, 'excess_le_0.1pct': excess <= 1e-3, 'excess_le_0.5pct': excess <= 5e-3,
             'excess_le_1pct': excess <= 1e-2, 'superpixel_none': spinv <= 0, 'superpixel_le_0.1pct': spinv <= 1e-3,
             'superpixel_le_1pct': spinv <= 1e-2}
    joint = {}
    for cname, cv in crits.items():
        ag = {g: cv[:, :, ix].all(2) for g, ix in groups.items()}
        ag['ALL'] = ag['DES'] & ag['DECADE'] & ag['BFD']
        row = {g: dict(mean=float(a.mean()), worst_rotation=float(a.mean(0).min()), all_rotations=float(a.all(1).mean()),
                       err=binom(a.mean(), a.size)) for g, a in ag.items()}
        bst = None
        for a0 in np.arange(0.8, 3.01, 0.05):
            for b0 in np.arange(0., 3.01, 0.1):
                for c0 in (0., 0.25, 0.5, 0.75, 1.0):
                    m = absA < a0 - b0*d['f_c'] + c0*d['f_sat']
                    if m.sum() < 1000:
                        continue
                    wr = ag['ALL'][m].mean(0).min()
                    if wr >= 0.99 and (bst is None or m.mean() > bst[0]):
                        bst = (float(m.mean()), float(a0), float(b0), float(c0), float(wr))
        row['best99_ALL_boundary'] = None if bst is None else dict(volume=bst[0], a=bst[1], b=bst[2], c=bst[3],
                                                                    acceptance_worst_rotation=bst[4])
        row['candidates'] = {name: {g: float(ag[g][m].mean()) for g in ('DES', 'DECADE', 'BFD', 'ALL')}
                             for name, m in cands.items()}
        joint[cname] = row
    out['criteria_joint'] = joint
    (outdir/'ia_joint_summary.json').write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)) + '\n')
    print(json.dumps({k: out[k] for k in ('acceptance_nominal', 'overlap', 'best_linear_boundary_99')}, indent=1, default=float))
    for name, row in cand_out.items():
        print(f'{name:55s} vol {row["prior_volume"]:.3f}  ' + '  '.join(f'{g} {row[g]["mean"]:.3f}/{row[g]["worst_rotation"]:.3f}'
                                                                      for g in ('DES', 'DECADE', 'BFD', 'ALL')))
    for cname, row in joint.items():
        print(f'{cname:22s} ' + '  '.join(f'{g} {row[g]["mean"]:.3f}' for g in ('DES', 'DECADE', 'BFD', 'ALL'))
              + f'  best99 {row["best99_ALL_boundary"]}')
    plots(d, acc, kg, worst, excess, spinv, iv, figdir, out)


def first_order_masked(d, y, ok, nb=20):
    S = {}
    for k in PAR:
        x = d[k][ok]
        yy = y[ok]
        e = np.quantile(x, np.linspace(0, 1, nb + 1))
        i = np.clip(np.digitize(x, e[1:-1]), 0, nb - 1)
        m = np.array([yy[i == j].mean() for j in range(nb)])
        S[k] = float(np.var(m)/np.var(yy))
    return S


def plots(d, acc, kg, worst, excess, spinv, iv, figdir, out):
    gl = ['DES', 'DECADE', 'BFD', 'ALL']
    # fig1: 2D acceptance maps
    planes = [('f_c', 'A_c'), ('f_sat', 'A_c'), ('eta_c', 'A_c'), ('A_s', 'A_c'), ('f_c', 'f_sat')]
    fig, ax = plt.subplots(len(gl), len(planes), figsize=(4*len(planes), 3.6*len(gl)), constrained_layout=True)
    for i, g in enumerate(gl):
        a = acc[g].mean(1)
        for j, (x, y) in enumerate(planes):
            m = np.ones(len(a), bool) if (x, y) != ('f_c', 'f_sat') else np.abs(d['A_c']) > 1.0
            H, xe, ye = np.histogram2d(d[x][m], d[y][m], bins=(16, 18), weights=a[m])
            C, _, _ = np.histogram2d(d[x][m], d[y][m], bins=(xe, ye))
            im = ax[i, j].pcolormesh(xe, ye, (H/np.maximum(C, 1)).T, vmin=0, vmax=1, cmap='viridis', shading='auto')
            ax[i, j].set_xlabel(x)
            ax[i, j].set_ylabel(y)
            ax[i, j].set_title(f'{g}' + (' (|A_c|>1)' if (x, y) == ('f_c', 'f_sat') else ''), fontsize=9)
            if (x, y) == ('f_c', 'A_c'):
                fc = np.linspace(.2, .7, 10)
                ax[i, j].plot(fc, 1.6 - fc, 'w--', lw=1)
                ax[i, j].plot(fc, -(1.6 - fc), 'w--', lw=1)
    fig.colorbar(im, ax=ax, shrink=0.6, label='acceptance (2% rule, mean over rotations)')
    fig.savefig(figdir/'ia_joint_fig1_acceptance_maps.png', dpi=110)
    plt.close(fig)

    # fig2: worst-bin invalid fraction CDFs per sample
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for v, c in (('DESY6', COL['DES']), ('NGC', COL['DECADE']), ('SGC', '#f08c00'), ('BFD', COL['BFD']), ('BFD_unit', COL['BFD_unit'])):
        w = np.sort(worst[:, :, iv[v]].ravel())
        ax[0].semilogx(np.maximum(w, 1e-6), np.arange(1, len(w) + 1)/len(w), color=c, label=v)
        ex = np.sort(excess[:, :, iv[v]].ravel())
        ax[1].semilogx(np.maximum(ex, 1e-8), np.arange(1, len(ex) + 1)/len(ex), color=c, label=v)
    ax[0].axvline(LEVEL, color='k', ls=':')
    ax[0].set_xlabel('worst-bin fraction of pixels with V_IA > V_avail')
    ax[0].set_ylabel('fraction of draws (nominal prior)')
    ax[1].set_xlabel('excess variance sum max(V_IA - V_avail, 0) / sum V_avail (worst bin)')
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(figdir/'ia_joint_fig2_invalid_cdf.png', dpi=110)
    plt.close(fig)

    # fig3: critical |A_c| vs f_c
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    big = np.abs(d['A_c']) > 0.5
    for g in gl:
        ac = (np.abs(d['A_c'])[:, None]*kg[g]).mean(1)
        for k, (x, xl) in enumerate((('f_c', 'f_c'), ('f_sat', 'f_sat'), ('eta_c', 'eta_c'))):
            e = np.linspace(*NOMINAL[x], 11)
            cc = 0.5*(e[1:] + e[:-1])
            q = np.array([np.quantile(ac[big & (d[x] >= a) & (d[x] < b)], [.05, .5, .95]) for a, b in zip(e[:-1], e[1:])])
            ax[k].plot(cc, q[:, 1], '-o' if g != 'ALL' else '--', color=COL[g], ms=3, lw=1 if g == 'ALL' else 1.5,
                       label=g if g != 'ALL' else 'ALL (coincides with DECADE)')
            ax[k].fill_between(cc, q[:, 0], q[:, 2], color=COL[g], alpha=0.12)
            ax[k].set_xlabel(xl)
    ax[0].plot(np.linspace(.2, .7, 5), 1.6 - np.linspace(.2, .7, 5), 'k--', lw=1, label='|A_c| = 1.6 - f_c')
    ax[0].set_ylabel('critical |A_c| (2% rule; 5-95% over the other parameters)')
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir/'ia_joint_fig3_critical_Ac.png', dpi=110)
    plt.close(fig)

    # fig4: conditional distributions of the accepted draws (ALL) vs prior
    fig, ax = plt.subplots(1, 6, figsize=(20, 3.2))
    a = acc['ALL'].mean(1)
    for k, p in enumerate(PAR):
        e = np.linspace(*NOMINAL[p], 21)
        ax[k].hist(d[p], e, density=True, histtype='step', color='0.5', label='prior')
        ax[k].hist(d[p], e, weights=a, density=True, histtype='step', color=COL['ALL'], label='accepted (ALL)')
        ax[k].hist(d[p], e, weights=acc['OLD_SET'].mean(1), density=True, histtype='step', color=COL['OLD_SET'],
                   ls='--', label='accepted (DES+DECADE)')
        ax[k].set_xlabel(p)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir/'ia_joint_fig4_conditional.png', dpi=110)
    plt.close(fig)

    # fig5: physical meaning of the pixel rule
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for v, c in (('DESY6', COL['DES']), ('NGC', COL['DECADE']), ('BFD', COL['BFD'])):
        w = worst[:, :, iv[v]].ravel()
        sel = np.random.default_rng(0).choice(len(w), min(20000, len(w)), replace=False)
        ax[0].loglog(np.maximum(w[sel], 1e-6), np.maximum(excess[:, :, iv[v]].ravel()[sel], 1e-9), '.', ms=1, color=c, label=v)
        ax[1].loglog(np.maximum(w[sel], 1e-6), np.maximum(spinv[:, :, iv[v]].ravel()[sel], 1e-6), '.', ms=1, color=c, label=v)
    for x in ax:
        x.axvline(LEVEL, color='k', ls=':')
        x.set_xlabel('worst-bin pixel invalid fraction')
    ax[0].set_ylabel('excess variance fraction')
    ax[1].set_ylabel('fraction of pixels in superpixels with sum V_IA > sum V_avail')
    ax[0].legend(markerscale=10)
    fig.tight_layout()
    fig.savefig(figdir/'ia_joint_fig5_rule_meaning.png', dpi=110)
    plt.close(fig)

    # fig6: rotation stability
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, g in enumerate(['DES', 'DECADE', 'BFD', 'BFD_unit', 'OLD_SET', 'ALL']):
        r = out['acceptance_nominal'][g]
        ax.errorbar(np.arange(len(r['per_rotation'])) + 0.1*k, r['per_rotation'], r['binomial_err_per_rotation'], fmt='o',
                    color=COL.get(g, 'k'), label=g)
    ax.set_xlabel('rotation')
    ax.set_ylabel('acceptance (nominal prior)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir/'ia_joint_fig6_rotations.png', dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
