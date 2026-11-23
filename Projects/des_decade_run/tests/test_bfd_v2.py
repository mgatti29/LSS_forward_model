#!/usr/bin/env python3
"""make_BFD_sample: legacy path bit-identical to the previous code; Poisson + SC v2 + halo IA runs.

Small real BFD slice (first rows of the unblinded catalogue), synthetic shells, NSIDE 64.
_bfd_utils_legacy.py is the pre-change bfd_utils (only its relative import made absolute).
"""
import copy
import sys
from pathlib import Path

import numpy as np
import healpy as hp
from astropy.io import fits

sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bfd_utils_legacy as legacy
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
print('rows per bin', [len(cats[t]['ra']) for t in range(4)])

rng = np.random.default_rng(1)
nshell, npix = 5, hp.nside2npix(NSIDE)
fields = {'density': rng.normal(0, 0.3, (nshell, npix)),
          'gamma': (rng.normal(0, 0.02, (nshell, 1, npix)) + 1j*rng.normal(0, 0.02, (nshell, 1, npix))),
          'IA_shear': (rng.normal(0, 0.5, (nshell, 1, npix)) + 1j*rng.normal(0, 0.5, (nshell, 1, npix)))}
ngal = rng.uniform(0, 1, (4, nshell)); ngal /= ngal.sum(1, keepdims=True)
zeff = np.linspace(0.2, 1.4, nshell)
sp = dict(h=0.7, Omega_m=0.3, Omega_b=0.05, Omega_cdm=0.25, sigma_8=0.8, n_s=0.96, m_nu=0.06, n_nu=1, w0=-1., wa=0.)
sp['As'] = sigma8_to_As_CAMB(0.3, 0.05, 0.7, 0.96, -1., 0., 0.06, 0.8)
cosmo_bundle = make_cosmo_bundle(sp)
sp.update(A_IA=0.7, eta_IA=0.5, A_IA2=0., bias_sc=np.array([1.1, 1.2, 1.3, 1.4]), bias_TA=np.zeros(4),
          dm=np.ones(4), rot=1, delta_rot=0)
sc_v1 = {key: [(lambda b, k=k: 1.0 + 0.01*k*(b - 1)) for k in range(4)]
         for key in ('corr_variance_fit', 'coeff_kurtosis_fit', 'A_corr_fit')}

kw = dict(SC_corrections=sc_v1, include_SC=True, do_noise=True, do_data=False)
old, _ = legacy.make_BFD_sample(ngal, zeff, cosmo_bundle, copy.deepcopy(sp), NSIDE, fields, cats,
                                rng=np.random.default_rng(7), **kw)
cur, _ = new.make_BFD_sample(ngal, zeff, cosmo_bundle, copy.deepcopy(sp), NSIDE, fields, cats,
                             rng=np.random.default_rng(7), noise_model='inverse_density', **kw)
for t in range(4):
    for key in ('e1', 'e2', 'e1n', 'e2n', 'gamma1_map', 'q_noise_scale_map'):
        assert np.array_equal(old[t][key], cur[t][key]), (t, key)
print('legacy path bit-identical: OK')

try:
    new.make_BFD_sample(ngal, zeff, cosmo_bundle, copy.deepcopy(sp), NSIDE, fields, cats,
                        rng=np.random.default_rng(7), **kw)
    raise AssertionError('v1 SC with poisson should raise')
except ValueError as e:
    print('v1 SC + poisson raises: OK')

sc_v2 = load_SC_calibration('/global/u2/m/mgatti/LSS_forward_model/Projects/source_clustering_v2/data/'
                            'SC_v2_DES_nside1024_sim1.json')        # stand-in until the BFD calibration exists
sp2 = copy.deepcopy(sp); sp2['bias_sc_slope'] = np.full(4, 0.5)
halo = {'numerator': 0.02*(rng.normal(size=(4, npix)) + 1j*rng.normal(size=(4, npix))),
        'numerator_sq': np.full((4, npix), 1e-4)}
maps, _ = new.make_BFD_sample(ngal, zeff, cosmo_bundle, sp2, NSIDE, fields, cats, SC_corrections=sc_v2,
                              include_SC=True, do_noise=True, rng=np.random.default_rng(7),
                              noise_model='poisson', IA_model='halo', halo_IA=halo)
for t in range(4):
    m = maps[t]['mask'] > 0
    print(f"bin {t}: poisson+halo e1 std {maps[t]['e1'][m].std():.4f}, noise std {maps[t]['e1n'][m].std():.4f}, "
          f"ia_noise_factor min {maps[t]['ia_noise_factor'][m].min():.3f} mean {maps[t]['ia_noise_factor'][m].mean():.3f}")
print('poisson + SC v2 + halo IA: OK')
