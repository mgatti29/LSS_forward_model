import LSS_forward_model
from LSS_forward_model.cosmology import *
from LSS_forward_model.lensing import *
from LSS_forward_model.maps import *
from LSS_forward_model.halos import *
from LSS_forward_model.tsz import *
from LSS_forward_model.theory import *
import os
import pandas as pd
import numpy as np
import healpy as hp
from cosmology import Cosmology
import astropy.io.fits as fits
import copy
import glass
import pyccl as ccl
from mpi4py import MPI
import BaryonForge as bfn
from pathlib import Path



def run(path_simulation, rot, delta_rot ,noise_rel, path_maps_gower):

    print (path_simulation,)
    SC_corrections = np.load('/global/homes/m/mgatti/LSS_forward_model/Data/SC_RR2_fit_nov6.npy',allow_pickle =True).item()
    

    #if True:
    if not os.path.exists(path_maps_gower):        
        # get basic info -------------------------------------------------------------
        sims_parameters, cosmo_bundle = read_sims_params(path_simulation)    
        shells_info = recover_shell_info(path_simulation+'/z_values.txt', max_z=49)
    
        if baryons['enabled']:
            baryons[ "values_to_update"] = draw_params_from_specs(baryon_priors)
            bpar, sys = load_or_save_updated_params(path_simulation,baryons['base_params_path'],baryons['filename_new_params'],baryons['values_to_update'], overwrite = False)
            sims_parameters.update(sys)

        # nuisance parameters --------------------------------------------------------
        dz = np.random.normal(dz_mean,dz_spread)
        dm = np.random.normal(dm_mean,dm_spread)
        A_IA = np.random.uniform(A0_interval[0],A0_interval[1])
        eta_IA = np.random.uniform(eta_interval[0],eta_interval[1])
        bias_sc = [np.random.uniform(bias_SC_interval[0],bias_SC_interval[1]) for tomo in range(len(dz_mean))]
        sims_parameters['dz'] = dz
        sims_parameters['dm'] = dm
        sims_parameters['A_IA'] = A_IA
        sims_parameters['eta_IA'] = eta_IA
        sims_parameters['bias_sc'] = bias_sc

    
        # load n(z) --------------------------------------------------------------------
        nz_RR2 = np.load('/global/cfs/cdirs/m5099/RR2/Reg2_SHE_tombins_unitweights_nz_SOMbin_C2020z_rebinned.npy',allow_pickle=True).item()
        
        nz_shifted, shells, steps, zeff_glass, ngal_glass = apply_nz_shifts_and_build_shells(
            z_rebinned=nz_RR2['z_rebinned'],
            nz_all=nz_RR2['nz_rebinned'],
            dz_values=sims_parameters["dz"],
            shells_info=shells_info,
        )



        # density field ---------------------
        density, label_baryonification = load_and_baryonify_gower_st_shells(
            path_simulation,
            sims_parameters,
            cosmo_bundle,
            baryons,
            nside_maps,
            shells_info,
            shells,
            overwrite_baryonified_shells = False)





        
        # shear field ------------------------------------------------------------------------------------------------
        fields = compute_lensing_fields(density, shells, cosmo_bundle['pars_camb'], nside_maps, do_kappa=True, do_shear=True, do_IA=True)
        fields['density'] = density


        # Test theory and save power spectrum (optional) ------------------------------------------------------------
        theory = LimberTheory(cosmo_bundle['pars_camb'], lmax=4000, nonlinear="mead")  # "euclidemu" | "mead" | "halofit"
        theory.set_Wshear(np.vstack([nz_RR2['z_rebinned'],nz_shifted]).T)
        Cgg = theory.cl_gg(nonlinear=True)
        kappa_tomo = integrate_field(ngal_glass, fields["kappa"])
        Cls = np.array([(hp.anafast(kappa_tomo[tomo,:])) for tomo in range(len(ngal_glass))])
        ratio = [Cls[tomo, :2000]/(Cgg[tomo, tomo, :2000] * (hp.pixwin(nside_maps)[:2000]**2)) for tomo in tomo_bins ]
        np.save(path_simulation+'theory_checks.npy',ratio)

                
        # make RR2 mocks
        path_data_cats = '/global/cfs/cdirs/m5099/RR2/Euclid_cats.npy'
        cats_Euclid  = np.load(path_data_cats,allow_pickle=True).item()
        maps_Gower_WL,_ = make_WL_sample(ngal_glass, zeff_glass, cosmo_bundle, sims_parameters, nside_maps, fields, cats_Euclid, SC_corrections = SC_corrections, do_catalog = False, include_SC = True)


        # save mock
        maps_Gower_WL['sims_parameters'] = sims_parameters
        np.save(path_maps_gower,maps_Gower_WL)



if __name__ == '__main__':


    ########################################################################################################################################
    # covariance run ------------------------------------------------------------------------------------------------------------------------

    do_maps = True
    nside_maps = 1024
    
    tomo_bins = [1,2,3,4,5,6]
    delta_rot_ = [0]

    
    dz_mean = [0,0,0,0,0,0,0]
    dz_spread = [0,0,0,0,0,0,0]
    
    dm_mean = [1,1,1,1,1,1,1]
    dm_spread = [0,0,0,0,0,0,0]
    
    A0_interval  = [0,0]
    eta_interval = [0,0]
    
    bias_SC_interval = [1,1]
    
    baryons = {
            "enabled": False,
            "max_z_halo_catalog": 1.5,
            "mass_cut": 13,
            "do_tSZ": False,
            "base_params_path": "../Data/Baryonification_wl_tsz_flamingo_parameters.npy",
            "filename_new_params": "sys_baryo_0.npy",
            "values_to_update":  None, # or: {'Mc': 10**13,'theta_ej' : 4.} or draw_params_from_specs( {"M_c": (12.5, 15.5, "log10"),   "theta_ej": (3.0, 10.0, "lin"),    "eta": (-2.0, -0.1, "log10")} )
    }

    
    BASE = Path("/global/cfs/cdirs/m5099/GowerSt2/Fiducial/")
    TARGET = "particles_100_4096.parquet"
    have = sorted(p for p in BASE.glob("*_big/") if (p / TARGET).is_file())
    missing = sorted(set(BASE.glob("*_big/")) - set(have))
    
    done = 0 
    runs = []
    for rot in [0]:#,1,2,3]:
        for delta_rot in delta_rot_:
            for noise_rel in [0]:
                for path in have:
                    if baryons['enabled']:
                        label_baryonification = 'baryonified_{0}'.format(noise_rel)
                        path_maps_gower = str(path)+'/maps_Gower_baryonified_{0}_{1}_{2}.npy'.format(rot,delta_rot,noise_rel)
                    else:
                        label_baryonification = 'normal'
                        path_maps_gower = str(path)+'/maps_Gower_{0}_{1}_{2}.npy'.format(rot,delta_rot,noise_rel)
                    if not os.path.exists(path_maps_gower):
                        #if os.path.exists(str(path)+'/density_b_1024.npy'):
                       runs.append([str(path)+'/',rot,delta_rot, noise_rel, path_maps_gower])
                    else:
                        done+=1

    print ('')
    print ('runs done: ',done, 'TODO: ',len(runs))
    print ('')

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Start with run_count = 0, but each process handles tasks based on rank
    run_count = rank 

    while run_count < len(runs):
        path, rot, delta_rot, noise_rel, path_maps_gower = runs[run_count]
        run(path, rot, delta_rot, noise_rel, path_maps_gower)
        run_count += size
    comm.Barrier()


    '''
    ########################################################################################################################################
    # general run ------------------------------------------------------------------------------------------------------------------------
    do_maps = True
    nside_maps = 1024
    tomo_bins = [1,2,3,4,5,6]
    delta_rot = 0.
    
    dz_mean = [0,0,0,0,0,0,0]
    dz_spread = [0.01,0.01,0.01,0.01,0.01,0.01,0.01]
    
    dm_mean = [1,1,1,1,1,1,1]
    dm_spread = [0.05,0.05,0.05,0.05,0.05,0.05,0.05]
    
    A0_interval  = [-2.5,2.5]
    eta_interval = [-2.5,2.5]
    
    bias_SC_interval = [0.5,1.5]
    
 


    baryons = {
            "enabled": True,
            "max_z_halo_catalog": 1,
            "mass_cut": 13.2,
            "do_tSZ": True,
            "base_params_path": "../Data/Baryonification_wl_tsz_flamingo_parameters.npy",
            "filename_new_params": "sys_baryo_0.npy"}

    
    baryons[ "values_to_update"] = draw_params_from_specs(baryon_priors)
    baryon_priors =   {"M_c": (12.5, 15.5, "log10"),   "theta_ej": (3.0, 10.0, "lin"),    "eta": (-2.0, -0.1, "log10")}

    
    

    BASE = Path("/global/cfs/cdirs/m5099/GowerSt2/runsU")
    TARGET = "particles_100_4096.parquet"
    have = sorted(p for p in BASE.glob("run*/") if (p / TARGET).is_file())
    missing = sorted(set(BASE.glob("run*/")) - set(have))

    done = 0 
    runs = []
    for rot in [0,1,2,3]:
        for noise_rel in [0]:
            for path in have:
                if baryonification:
                    label_baryonification = 'baryonified_{0}'.format(noise_rel)
                    path_maps_gower = str(path)+'/maps_Gower_baryonified_{0}_{1}.npy'.format(rot,noise_rel)
                else:
                    label_baryonification = 'normal'
                    path_maps_gower = str(path)+'/maps_Gower_{0}_{1}.npy'.format(rot,noise_rel)
                if not os.path.exists(path_maps_gower):
                    if os.path.exists(str(path)+'/density_b_1024.npy'):
                        runs.append([str(path)+'/',rot,noise_rel])
                else:
                    done+=1


    print (len(runs),done)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Start with run_count = 0, but each process handles tasks based on rank
    run_count = rank 

    while run_count < len(runs):
        run(runs[run_count][0], runs[run_count][1], runs[run_count][2])
        run_count += size
    comm.Barrier()
    '''







#module load python; source activate pyccl_env;  python  generate_mocks.py
#module load python; source activate pyccl_env; srun --nodes=4 --tasks-per-node=4 python  generate_mocks.py
