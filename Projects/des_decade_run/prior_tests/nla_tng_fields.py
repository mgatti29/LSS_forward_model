#!/usr/bin/env python3
"""NLA intrinsic-alignment fields on the TNG300 model frame, for the IA prior comparison.

Same convention as lensing.make_WL_sample (IA_model='NLA'):

    I_t = - sum_i w_ti F_nla(z_i; A=1, eta) IA_shear_i,   F_nla = -A C1 rho_crit Omega_m ((1+z)/1.67)^eta / D(z)

with IA_shear_i the spin-2 tidal field of density shell i (maps.compute_lensing_fields, do_IA) and w_ti the
TNG DES Y6-like n(z) weights per shell (cheap_model_validation target counts).  The predictions are linear
in A, so A = 1 is stored for a grid of eta.

Output: WORK/DES/nla_prior.npz with alm_E (bin, eta, alm) and eta.
"""
import sys

import healpy as hp
import numpy as np
import pyccl as ccl

sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model/Projects/IA_non_gaussian')
sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model/Projects/IA_non_gaussian/cheap_model_validation')
import cheap_ia_validation as v
from LSS_forward_model.cosmology import make_cosmo_bundle, sigma8_to_As_CAMB
from LSS_forward_model.maps import F_nla, compute_lensing_fields

ETA_GRID = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
SURVEY = 'DES'


def main():
    TNG = v.TNG
    sims_parameters = dict(h=TNG['h'], Omega_m=TNG['Omega_m'], Omega_b=TNG['Omega_b'],
                           Omega_cdm=TNG['Omega_m'] - TNG['Omega_b'], sigma_8=TNG['sigma_8'], n_s=TNG['n_s'],
                           m_nu=0.0, n_nu=0, w0=-1.0, wa=0.0)
    sims_parameters['As'] = sigma8_to_As_CAMB(TNG['Omega_m'], TNG['Omega_b'], TNG['h'], TNG['n_s'], -1.0, 0.0, 0.0,
                                              TNG['sigma_8'])
    cosmo_bundle = make_cosmo_bundle(sims_parameters)
    rho_c1 = 5e-14*ccl.physical_constants.RHO_CRITICAL

    zeff = np.load(v.WORK/'shell_zeff.npy')
    density = np.load(v.WORK/'density_model.npy', mmap_mode='r')
    target = np.load(v.WORK/SURVEY/'target.npz')
    weights = target['counts']/target['counts'].sum(1, keepdims=True)        # (bin, shell)
    npix = density.shape[1]
    field = np.zeros((4, len(ETA_GRID), npix), dtype=np.complex128)

    for i in range(v.NSHELL):
        if not np.any(weights[:, i] > 0):
            continue
        # one shell at a time (shells argument unused for the IA field)
        ia = compute_lensing_fields([np.asarray(density[i], dtype=np.float64)], [None], cosmo_bundle['pars_camb'],
                                    v.NSIDE, do_kappa=False, do_shear=False, do_IA=True)['IA_shear'][0, 0]
        for k, eta in enumerate(ETA_GRID):
            f = float(F_nla(z=zeff[i], om0=TNG['Omega_m'], A_ia=1.0, rho_c1=rho_c1, eta=eta, z0=0.67,
                            cosmo=cosmo_bundle['cosmo_pyccl']))
            for t in range(4):
                if weights[t, i] > 0:
                    field[t, k] -= weights[t, i]*f*ia
        print(f'shell {i} (z={zeff[i]:.2f}) done', flush=True)

    alm = np.array([[v.spin_alm_E(field[t, k]) for k in range(len(ETA_GRID))] for t in range(4)])
    np.savez(v.WORK/SURVEY/'nla_prior.npz', alm_E=alm, eta=np.array(ETA_GRID))
    print('saved', v.WORK/SURVEY/'nla_prior.npz', flush=True)


if __name__ == '__main__':
    main()
