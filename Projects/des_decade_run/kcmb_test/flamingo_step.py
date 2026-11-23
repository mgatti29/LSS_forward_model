"""Baryon step at ell = 2 NSIDE_lo of the hybrid kappa (DMO 2048 + baryon difference from the 1024 maps), and
extensions above it with the baryon transfer T(ell) = C(hydro, DMO)/C(DMO, DMO) measured on the 1024 maps (FLAMINGO z < 3)."""
import json
from pathlib import Path
import healpy as hp
import numpy as np
D = Path('/pscratch/sd/m/mgatti/kcmb_test/flamingo'); L = 4096; LLO = 2048
alm = {k: hp.map2alm(np.load(D/f'kappa_{k}.npy'), lmax=L if k.endswith('2048') else LLO, iter=0)
       for k in ('L1_m9_2048', 'L1_m9_DMO_2048', 'L1_m9_1024', 'L1_m9_DMO_1024', 'fgas-8sigma_2048')}
ell = np.arange(L + 1)
pw = hp.pixwin(2048, lmax=LLO)/hp.pixwin(1024, lmax=LLO)
T_cross = hp.alm2cl(alm['L1_m9_1024'], alm['L1_m9_DMO_1024'])/hp.alm2cl(alm['L1_m9_DMO_1024'])
T_lo = np.sqrt(hp.alm2cl(alm['L1_m9_1024'])/hp.alm2cl(alm['L1_m9_DMO_1024']))   # power-ratio transfer
fit = (ell[:LLO + 1] >= 1200) & (ell[:LLO + 1] <= 1800)
T_const = T_lo[fit].mean()
slope, icpt = np.polyfit(np.log(ell[:LLO + 1][fit]), T_lo[fit], 1)
def hybrid(ext):
    h = alm['L1_m9_DMO_2048'].copy()
    idx = hp.Alm.getidx(L, *hp.Alm.getlm(LLO))
    h[idx] += hp.almxfl(alm['L1_m9_1024'] - alm['L1_m9_DMO_1024'], pw)
    if ext is not None:
        t = np.ones(L + 1); t[LLO + 1:] = ext[LLO + 1:]
        h = hp.almxfl(h, t)
    return h
truth = hp.alm2cl(alm['L1_m9_2048']); dmo = hp.alm2cl(alm['L1_m9_DMO_2048'])
edges = np.array([1000, 1500, 1800, 2000, 2048, 2200, 2500, 3000, 3500, 4000]); bp = lambda c: np.array([c[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
print('bands', [f'{a}-{b}' for a, b in zip(edges[:-1], edges[1:])])
print('T_cross/T_power', np.round(bp(np.pad(T_cross/T_lo, (0, L - LLO), constant_values=np.nan))[:4], 5).tolist())
print('T measured on 1024 maps (bands <= 2048):', np.round(bp(np.pad(T_lo, (0, L - LLO), constant_values=np.nan))[:4], 4).tolist(), ' T_const', round(T_const, 4), ' slope/ln ell', round(slope, 4))
print(f'{"truth hydro / DMO":32s}', np.round(bp(truth)/bp(dmo), 4).tolist())
for name, ext in (('hard cut', None), ('constant T', np.full(L + 1, T_const)), ('log-linear T', icpt + slope*np.log(np.maximum(ell, 1)))):
    print(f'{"hybrid " + name + " / truth":32s}', np.round(bp(hp.alm2cl(hybrid(ext)))/bp(truth), 4).tolist())
print(f'{"fgas-8sigma / DMO (strong fb)":32s}', np.round(bp(hp.alm2cl(alm["fgas-8sigma_2048"]))/bp(dmo), 4).tolist())
