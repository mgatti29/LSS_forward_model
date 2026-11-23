"""Timing of every stage of coshirex.ladder_kappa, fast and exact paths (NSIDE 2048, runs3A cosmology)."""
import os, sys, time, multiprocessing
import numpy as np
sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
from LSS_forward_model import coshirex as cx
from LSS_forward_model.cosmology import read_sims_params

NSIDE = 2048
pars = read_sims_params('/pscratch/sd/m/mgatti/kcmb_test/run001')[1]['pars_camb']
def t(label, f):
    t0 = time.time(); out = f(); print(f'{label:62s} {time.time() - t0:8.1f} s', flush=True); return out

lad = cx.ladder_windows(2.0, 1100., 100)
cond = [cx.tophat_window(z, z + 0.04) for z in np.arange(1.6, 1.99, 0.04)]     # ~ the simulated shells within 400 Mpc/h
cores = len(os.sched_getaffinity(0))
print('cores', cores, flush=True)
cls = t('fast: Limber shell spectra (100 shells, lmax 4096)', lambda: cx.limber_shell_cls(pars, lad, 2*NSIDE))
t('fast: one lognormal Gaussian-spectrum solve (lmax 4096)', lambda: cx._gaussian_auto((cls[0], lad[0], NSIDE, 1.)))
for nproc in (8, 16, 32):
    with multiprocessing.get_context('fork').Pool(nproc) as pool:
        gls = t(f'fast: 100 Gaussian-spectrum solves, {nproc} processes', lambda: pool.map(cx._gaussian_auto, [(c, w, NSIDE, 1.) for c, w in zip(cls, lad)]))
        q = cx.born_weights(lad, pars); seeds = np.random.SeedSequence(1).spawn(100)
        tasks = [(gls[k::nproc], q[k::nproc], seeds[k::nproc], [None]*len(q[k::nproc]), 0, NSIDE, 1., max(1, cores//nproc)) for k in range(nproc)]
        t(f'fast: 100 lognormal shells -> kappa, {nproc} x {cores//nproc} threads', lambda: np.sum(pool.map(cx._kappa_part, tasks), axis=0))
t('fast: whole ladder_kappa (default nproc)', lambda: cx.ladder_kappa(pars, 2.0, NSIDE, seed=1))
w = cond + lad
for lmax in (100, 200, 400):
    t(f'exact: CAMB line-of-sight spectra, {len(w)} shells, lmax {lmax}', lambda: cx.shell_cls(pars, w, lmax))
