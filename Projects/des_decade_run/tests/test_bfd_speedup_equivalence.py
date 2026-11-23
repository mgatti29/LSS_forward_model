#!/usr/bin/env python3
"""make_BFD_sample with footprint-restricted shell sums vs the previous code (real BFD slice, NSIDE 64):
poisson + NLA, poisson + halo, inverse_density.  usage: test_bfd_speedup_equivalence.py OLD_PKG_DIR"""
import copy, sys
import numpy as np, healpy as hp
from astropy.io import fits
sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
sys.path.insert(0, sys.argv[1])
import lfm_old.bfd_utils as old
from LSS_forward_model import bfd_utils as new
from LSS_forward_model.cosmology import make_cosmo_bundle, sigma8_to_As_CAMB
from LSS_forward_model.source_clustering import load_SC_calibration

NSIDE = 64
SRC = ('/global/cfs/cdirs/des/mgatti/BFD_targets_June25/'
       'BFD_catalog_v5_UNBLINDED_with_tomo_masked_withselection_final_psfmask_psfvalid.fits')
with fits.open(SRC, memmap=True) as h:
    block = np.array(h[1].data[:300000])
cats = {t: {k: np.asarray(block[k][block['tomo'] == t]) for k in ('ra', 'dec', 'selection_term', 'pqr', 'pqr10_prelog')}
        for t in range(4)}
rng = np.random.default_rng(1)
nshell, npix = 5, hp.nside2npix(NSIDE)
fields = {'density': rng.normal(0, 0.3, (nshell, npix)),
          'gamma': rng.normal(0, 0.02, (nshell, 1, npix)) + 1j*rng.normal(0, 0.02, (nshell, 1, npix)),
          'IA_shear': rng.normal(0, 0.5, (nshell, 1, npix)) + 1j*rng.normal(0, 0.5, (nshell, 1, npix))}
ngal = rng.uniform(0, 1, (4, nshell)); ngal[1, 0] = 0; ngal /= ngal.sum(1, keepdims=True)
zeff = np.linspace(0.2, 1.4, nshell)
sp = dict(h=0.7, Omega_m=0.3, Omega_b=0.05, Omega_cdm=0.25, sigma_8=0.8, n_s=0.96, m_nu=0.06, n_nu=1, w0=-1., wa=0.)
sp['As'] = sigma8_to_As_CAMB(0.3, 0.05, 0.7, 0.96, -1., 0., 0.06, 0.8)
cb = make_cosmo_bundle(sp)
sp.update(A_IA=0.7, eta_IA=0.5, A_IA2=0., bias_sc=np.array([1.1, 1.2, 1.3, 1.4]), bias_sc_slope=np.full(4, .5),
          bias_TA=np.zeros(4), dm=np.ones(4), rot=1, delta_rot=0)
sc2 = load_SC_calibration('/global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/data/SC_v2_BFD_nside1024_sim1.json')
sc1 = {key: [(lambda b, k=k: 1.0 + 0.01*k*(b - 1)) for k in range(4)]
       for key in ('corr_variance_fit', 'coeff_kurtosis_fit', 'A_corr_fit')}
halo = {'numerator': 0.02*(rng.normal(size=(4, npix)) + 1j*rng.normal(size=(4, npix))),
        'numerator_sq': np.full((4, npix), 1e-4)}
cases = {'poisson NLA': dict(SC_corrections=sc2, noise_model='poisson', IA_model='NLA'),
         'poisson halo': dict(SC_corrections=sc2, noise_model='poisson', IA_model='halo', halo_IA=halo),
         'inverse_density deltaNLA': dict(SC_corrections=sc1, noise_model='inverse_density')}
for name, kw in cases.items():
    out = [mod.make_BFD_sample(ngal, zeff, cb, copy.deepcopy(sp), NSIDE, fields, cats, include_SC=True, do_noise=True,
                               rng=np.random.default_rng(7), **kw,
                               **(dict(ia_cap_alpha=None, ia_self_nside=64) if mod is new else {}))[0]  # pre-cap defaults
           for mod in (old, new)]
    for t in range(4):
        for k in out[0][t]:
            assert np.array_equal(out[0][t][k], out[1][t][k]), (name, t, k)
    print(f'make_BFD_sample {name}: bit-identical', flush=True)
print('OK')
