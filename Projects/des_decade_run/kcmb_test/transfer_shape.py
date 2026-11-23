"""Shape of the baryon power transfer T(ell) = sqrt(C_b / C_DMO) up to 2 NSIDE_lo, and fit-range choices for the
extension above it.  runsT: kappa over the baryon shells (Born of delta2048 shells 0-40 + glass b0/b1 maps);
FLAMINGO: z < 3 kappa (truth above 2048 available)."""
import sys
from pathlib import Path
import healpy as hp, numpy as np
sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
from LSS_forward_model.coshirex import born_weights
from LSS_forward_model.cosmology import read_sims_params, recover_shell_info
from LSS_forward_model.maps import build_shells
LLO, L = 2048, 4096
ell = np.arange(LLO + 1)
edges = np.array([800, 1000, 1200, 1400, 1600, 1800, 1900, 2000, 2048]); bp = lambda c: np.array([c[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
fits = [(1200, 1800), (1500, 2000), (1700, 2040), (1900, 2040)]
def report(name, T, truth_ext=None, dmo_hi=None):
    print(name, 'T by band', [f'{a}-{b}' for a, b in zip(edges[:-1], edges[1:])], np.round(bp(T), 4).tolist())
    for lo, hi in fits:
        m = (ell >= lo) & (ell <= hi)
        b, a = np.polyfit(np.log(ell[m]), T[m], 1)
        line = f'   fit {lo}-{hi}: T(2048) {a + b*np.log(2048):.4f} (data 2000-2048 {T[2000:2049].mean():.4f})  T(4096) {a + b*np.log(4096):.4f}'
        if truth_ext is not None:
            lq = np.arange(L + 1); t = np.where(lq > LLO, a + b*np.log(np.maximum(lq, 1)), 1.)
            e2 = np.array([2048, 2500, 3000, 3500, 4000])
            ratio = [np.mean((dmo_hi*t**2)[x:y])/np.mean(truth_ext[x:y]) for x, y in zip(e2[:-1], e2[1:])]
            line += '  ext/truth bands 2048-2500..3500-4000: ' + str(np.round(ratio, 4).tolist())
        print(line)

D = Path('/pscratch/sd/m/mgatti/kcmb_test/flamingo')
a = {k: hp.map2alm(np.load(D/f'kappa_{k}.npy'), lmax=L if '2048' in k else LLO, iter=0) for k in ('L1_m9_1024', 'L1_m9_DMO_1024', 'L1_m9_2048', 'L1_m9_DMO_2048')}
report('FLAMINGO', np.sqrt(hp.alm2cl(a['L1_m9_1024'])/hp.alm2cl(a['L1_m9_DMO_1024'])),
       truth_ext=hp.alm2cl(a['L1_m9_2048']), dmo_hi=hp.alm2cl(a['L1_m9_DMO_2048']))

SIM = '/pscratch/sd/m/mgatti/kcmb_test/runsT/run001'; W = Path('/pscratch/sd/m/mgatti/kcmb_test/work_runsT')
_, cb = read_sims_params(SIM); shells, _, _ = build_shells(recover_shell_info(SIM + '/z_values.txt', max_z=49))
idx = list(range(41)); q = born_weights([shells[i] for i in idx], cb['pars_camb'])
k_dmo = sum(w*np.load(W/'delta2048'/f'shell_{i:03d}.npy').astype(np.float64) for i, w in zip(idx, q))
a_dmo = hp.map2alm(k_dmo, lmax=LLO, iter=0)
for r in (0, 1):
    a_b = hp.map2alm(np.load(W/f'kappa_b{r}.npy').astype(np.float64), lmax=LLO, iter=0)
    report(f'runsT draw {r}', np.sqrt(hp.alm2cl(a_dmo + a_b)/hp.alm2cl(a_dmo)))
