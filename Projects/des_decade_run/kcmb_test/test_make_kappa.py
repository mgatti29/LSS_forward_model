"""maps.make_kappa_cmb on runsT run001 (matched Hypatia baryon fields) against the kcmb_test building blocks."""
import json, sys, time
from pathlib import Path
import healpy as hp, numpy as np
sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
from LSS_forward_model.coshirex import born_weights, ladder_kappa
from LSS_forward_model.cosmology import read_sims_params, recover_shell_info
from LSS_forward_model.maps import build_shells, make_kappa_cmb

SIM = '/pscratch/sd/m/mgatti/kcmb_test/runsT/run001'
BARYON = Path('/pscratch/sd/m/mgatti/run001')
WORK = Path('/pscratch/sd/m/mgatti/kcmb_test/work_runsT')
OUT = Path('/pscratch/sd/m/mgatti/kcmb_test/make_kappa'); OUT.mkdir(exist_ok=True)
SEED, L = 20261106, 4096
_, cosmo_bundle = read_sims_params(SIM)
info = recover_shell_info(SIM + '/z_values.txt', max_z=49)
shells, _, _ = build_shells(info)
steps = np.asarray(info['Step'])[::-1]
delta_b = {r: np.load(BARYON/f'delta_b_1024_{r}.npy', mmap_mode='r') for r in (0, 1)}
t0 = time.time()
kappa, meta = make_kappa_cmb(SIM, shells, steps, cosmo_bundle, delta_b=delta_b, seed=SEED)
print(f'make_kappa_cmb: {time.time() - t0:.0f} s', flush=True)
print('meta', json.dumps({k: v for k, v in meta.items() if k != 'native_shells'}), flush=True)
for k, m in kappa.items():
    np.save(OUT/f'kappa_{k}.npy', m)

pars = cosmo_bundle['pars_camb']
hi = ladder_kappa(pars, meta['z_start'], 2048, seed=SEED)
base = meta['native_shells']
ref = sum(w*np.load(WORK/'delta2048'/f'shell_{i:03d}.npy').astype(np.float64)
          for i, w in zip(base, born_weights([shells[i] for i in base], pars)))
native = kappa['dmo'] - hi
print('native part vs Born of delta2048 shells: max|diff|/std', float(np.max(np.abs(native - ref))/ref.std()), flush=True)

edges = np.array([10, 100, 500, 1000, 1500, 1800, 2000, 2048, 2200, 2500, 3000, 3500, 4000])
bp = lambda c: np.array([c[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
print('bands', [f'{a}-{b}' for a, b in zip(edges[:-1], edges[1:])])
a_dmo = hp.map2alm(kappa['dmo'].astype(np.float64), lmax=L, iter=0)
c_dmo = hp.alm2cl(a_dmo)
for r in (0, 1):
    a_r = hp.map2alm(kappa[r].astype(np.float64), lmax=L, iter=0)
    print(f'baryons {r}: C(with)/C(dmo)           ', np.round(bp(hp.alm2cl(a_r))/bp(c_dmo), 4).tolist())
    old = hp.map2alm(np.load(WORK/f'kappa_b{r}.npy').astype(np.float64), lmax=L, iter=0)
    new = a_r - a_dmo
    print(f'baryons {r}: C(new term)/C(glass b{r})   ', np.round(bp(hp.alm2cl(new))/bp(hp.alm2cl(old)), 3).tolist())
    print(f'baryons {r}: r(new term, glass b{r})     ', np.round(bp(hp.alm2cl(new, old))/np.sqrt(bp(hp.alm2cl(new))*bp(hp.alm2cl(old))), 3).tolist(), flush=True)
