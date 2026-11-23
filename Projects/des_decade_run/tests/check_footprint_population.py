#!/usr/bin/env python3
"""halo_ia footprint + parallel population vs full-sky parallel population (DES Y6 mean n(z), rotation 0).

Same occupation, different random streams: compare per bin, on the footprint pixels, the mean IA self term
<numerator_sq>, the variance of the numerator, and its E-mode power (A_c = 1, eta_c = -1, A_s = 0.02).
"""
import sys, time
import healpy as hp, numpy as np
sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run')
import make_mocks as mm
from LSS_forward_model.cosmology import read_sims_params, recover_shell_info
from LSS_forward_model.halo_ia import make_halo_IA_numerator

SIM = '/pscratch/sd/m/mgatti/des_decade_test/Fiducial/1_big/'
CAT = '/pscratch/sd/m/mgatti/source_clustering_v2/halo_ia/halo_catalog_ia_z2_m12_Fiducial_1_big.parquet'
sp, cb = read_sims_params(SIM)
info = recover_shell_info(SIM + 'z_values.txt', max_z=49)
s = mm.Sample('DESY6')
nuis = s.draw_nuisance(np.random.default_rng(1))
z, nzs, zeff, ngal = mm.shells_for(s, nuis, info)
pop = {'f_c': 0.5, 'f_sat': 0.8}
fp = s.footprint(0)
mask = np.zeros(hp.nside2npix(1024), bool)
for t, c in s.cats.items():
    mask[mm.rotate_and_rebin(mm.convert_to_pix_coord(c['ra'], c['dec'], nside=2048), 1024, 0)] = True
print('footprint fsky (dilated NSIDE 32)', fp.mean(), 'survey', mask.mean(), flush=True)
ia = mm.ia_block(dict(A_c=1., eta_c=-1., A_s=0.02, eta_s=0.))
res = {}
for name, footprint in (('footprint', fp), ('fullsky', None)):
    t0 = time.time()
    orig = mm.Sample.footprint
    mm.Sample.footprint = (lambda self, rot, f=footprint: f)
    objs = mm.population_for(s, nuis, pop, info, sp, cb, CAT, 0)
    mm.Sample.footprint = orig
    t1 = time.time() - t0
    h = make_halo_IA_numerator(objs, ia, np.zeros((ngal.shape[1], 1)), ngal, 1024, bin_counts=s.bin_counts, seed=3)
    res[name] = h
    print(f'{name}: {len(objs["redshift"]):,} sources, population {t1:.0f} s, numerator {time.time()-t0-t1:.0f} s', flush=True)
    del objs
for t in range(4):
    a, b = res['footprint'], res['fullsky']
    sq = [x['numerator_sq'][t][mask].mean() for x in (a, b)]
    var = [np.mean(np.abs(x['numerator'][t][mask])**2) for x in (a, b)]
    ee = []
    for x in (a, b):
        n = np.where(mask, x['numerator'][t], 0)
        alm = hp.map2alm_spin([n.real, n.imag], 2, lmax=1000)
        cl = hp.alm2cl(alm[0])/mask.mean()
        ee.append([cl[lo:hi].mean() for lo, hi in ((30, 100), (100, 300), (300, 1000))])
    print(f'bin {t+1}: <num_sq> ratio {sq[0]/sq[1]:.4f}  <|num|^2> ratio {var[0]/var[1]:.4f}  EE ratio (30-100,100-300,300-1000)',
          np.round(np.array(ee[0])/np.array(ee[1]), 3).tolist(), flush=True)
