"""Timing of the kappa_CMB building blocks at NSIDE 2048 / lmax 4096 (one node)."""
import os, time, sys
import numpy as np, healpy as hp, ducc0, camb
sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
import glass.lensing, glass.shells
from cosmology import Cosmology

NSIDE, LMAX = 2048, 4096
W = '/pscratch/sd/m/mgatti/kcmb_test/work/delta2048/'
def t(label, f, n=1):
    t0 = time.time()
    for _ in range(n): out = f()
    print(f'{label:55s} {(time.time() - t0)/n:7.2f} s', flush=True); return out

files = sorted(os.listdir(W))
m = t('np.load one 2048 shell (float32)', lambda: np.load(W + files[30]), 3)
t('np.load 10 shells', lambda: [np.load(W + f) for f in files[40:50]])
p = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.12, ns=0.965, As=2e-9)
conv = glass.lensing.MultiPlaneConvergence(Cosmology.from_camb(p))
ws = [glass.shells.RadialWindow(np.linspace(0.1*i, 0.1*i + 0.1, 50), np.ones(50), 0.1*i + 0.05) for i in range(1, 6)]
md = m.astype(float)
t0 = time.time()
for w in ws: conv.add_window(md, w)
print(f'{"glass add_window (per shell)":55s} {(time.time() - t0)/5:7.2f} s', flush=True)
acc = np.zeros(len(m), np.float32)
t('weighted sum acc += w*delta (float32)', lambda: np.add(acc, np.float32(0.3)*m, out=acc), 5)
alm = t('healpy map2alm lmax 4096', lambda: hp.map2alm(md, lmax=LMAX, iter=0))
t('healpy alm2map lmax 4096', lambda: hp.alm2map(alm, NSIDE, lmax=LMAX))
base = ducc0.healpix.Healpix_Base(NSIDE, 'RING')
geom = base.sht_info()
for nth in (16, 64, 128, 256):
    t(f'ducc0 synthesis lmax 4096 nthreads={nth}',
      lambda: ducc0.sht.synthesis(alm=alm[None], lmax=LMAX, spin=0, nthreads=nth, **geom))
rng = np.random.default_rng(1)
t('rng normal 2 x nalm', lambda: rng.normal(size=(2, len(alm))))
g = hp.alm2map(alm, NSIDE, lmax=LMAX)
t('expm1 on map (float64)', lambda: np.expm1(g))
import numexpr as ne
t('numexpr expm1 (64 threads)', lambda: ne.evaluate('expm1(g)'))
