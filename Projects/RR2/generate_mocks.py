import LSS_forward_model
from LSS_forward_model.cosmology import *
from LSS_forward_model.lensing import *
from LSS_forward_model.maps import *
from LSS_forward_model.halos import *
from LSS_forward_model.tsz import *
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




def run(path_simulation,rot, noise_rel):
    
    SC_corrections = np.load('/global/homes/m/mgatti/LSS_forward_model/Data/SC_RR2_fit.npy',allow_pickle =True).item()
    
    if baryonification:
        label_baryonification = 'baryonified_{0}'.format(noise_rel)
        path_maps_gower = path_simulation+'/maps_Gower_baryonified_{0}_{1}.npy'.format(rot,noise_rel)
    else:
        label_baryonification = 'normal'
        path_maps_gower = path_simulation+'/maps_Gower_{0}_{1}.npy'.format(rot,noise_rel)
    
    #if True:
    if not os.path.exists(path_maps_gower):        
        # get basic info -------------------------------------------------------------
        sims_parameters, cosmo_pyccl, camb_pars, colossus_pars = read_sims_params(path_simulation)    
        shells_info = recover_shell_info(path_simulation+'/z_values.txt', max_z=49)
    
    
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
        sims_parameters['rot'] = rot
        filename_new_baryonic_parameters = f"sys_baryo_{0}.npy".format(noise_rel)
        
    
        # load n(z) --------------------------------------------------------------------
        nz_file = fits.open('/pscratch/sd/m/mgatti/euclid/Reg2_SHE_tombins_unitweights_nz_SOMbin_C2020z.fits')
       
        nz = []
        redshift = np.linspace(0,6,3001)
        n_bins = 50
        bin_factor = len(nz_file[1].data['N_Z'][0]) // n_bins 
        n_trim = bin_factor * n_bins 
        z_rebinned = redshift[:n_trim].reshape(n_bins, bin_factor).mean(axis=1)
        
        ############################################
        nz_ = [nz_file[1].data['N_Z'][0]+nz_file[1].data['N_Z'][1]+nz_file[1].data['N_Z'][2]+nz_file[1].data['N_Z'][3]+nz_file[1].data['N_Z'][4]+nz_file[1].data['N_Z'][5]]
        for i in range(6):
            nz_.append(nz_file[1].data['N_Z'][i])
        
        for ix in range(len(sims_parameters['dz'])):
            nz_rebinned = nz_[ix][:n_trim].reshape(n_bins, bin_factor).sum(axis=1)
        
            # apply a shift in the mean ------
            nz_shifted_on_rebinned = shift_nz(
                        z=z_rebinned,
                        nz=nz_rebinned,
                        z_rebinned=z_rebinned,
                        delta_z=sims_parameters['dz'][ix],
                        renorm="source"  # keep the original integral
                    )
        
            
            norm = np.trapz(nz_shifted_on_rebinned,z_rebinned)
            nz.append(nz_shifted_on_rebinned/norm)
        nz = np.array(nz)
        redshift = copy.deepcopy(z_rebinned)
        
        shells, steps, zeff, ngal_glass = build_shell_windows_and_partitions(
            shells_info=shells_info,
            redshift=redshift,
            nz=nz,
            samples_per_shell=100,
        )



        # Density shells + baryonification ------------------------------------------------------------------------------------------------
        try:
            bpar, sys = load_or_draw_baryon_params(
                path_sim=path_simulation,
                specs=baryon_priors,
                cache_filename=filename_new_baryonic_parameters,
                base_params_path=base_params_path,
                overrides=False,  # set to dict if you need overrides
            )
        except:
            bpar, sys = load_or_draw_baryon_params(
                path_sim=path_simulation,
                specs=baryon_priors,
                cache_filename=filename_new_baryonic_parameters,
                base_params_path=base_params_path,
                overrides=True,  # set to dict if you need overrides
            )
        if baryonification:
            
            label_baryonification = 'baryonified'
            
            # create halo catalog --------------------------------------------
            if not os.path.exists(path_simulation+ 'halo_catalog.parquet'):
                print ('creating halo light cone')
                save_halocatalog(shells_info, sims_parameters, max_redshift = max_redshift_halo_catalog, halo_snapshots_path = path_simulation, catalog_path = path_simulation + 'halo_catalog.parquet')
        
            tsz_path = os.path.join(path_simulation, f"tsz_{nside_maps}.npy")
            dens_path = os.path.join(path_simulation, f"density_b_{nside_maps}.npy")
        
            if (not os.path.exists(dens_path) or (do_tSZ and (not os.path.exists(tsz_path)))):
                halos = load_halo_catalog(path_simulation+ 'halo_catalog.parquet',colossus_pars,sims_parameters,halo_catalog_log10mass_cut)
        
            
        
                make_tsz_and_baryonified_density(path_simulation,sims_parameters,cosmo_pyccl,
                                             halos,bpar,nside_maps,shells_info,dens_path,
                                             tsz_path,do_tSZ,nside_baryonification,
                                             halo_catalog_log10mass_cut)
        
            density = np.load(dens_path,allow_pickle=True)
        
        else:
            label_baryonification = 'normal'
            #save "normal" density shells --------------------------------------
            if not os.path.exists(path_simulation+'/delta_{0}.npy'.format(nside_maps)):
                density = make_density_maps(shells_info,path_simulation,path_simulation+'/delta_{0}.npy'.format(nside_maps),nside_maps)
            else:
                density = np.load(path_simulation+'/delta_{0}.npy'.format(nside_maps),allow_pickle=True)
        
        print ('done')

        
        # compute kappa & shear *********************************************************************************************
        # on the fly
       
        # Note: kappa field ill have the healpy pixel window function applied.
        # however, glass, when it computes gamma, automatically deconvolves it; it will be added later when making mocks and putting simulated galaxies into pixels -- 
        cosmo = Cosmology.from_camb(camb_pars)
        gamma_ = []
        
        convergence = glass.lensing.MultiPlaneConvergence(cosmo)
        for ss in frogress.bar(range(len(density))):
           
            convergence.add_window(density[ss], shells[ss])
            # get convergence field
            kappa = copy.deepcopy(convergence.kappa)
            gamma = glass.lensing.from_convergence(kappa, lmax=nside_maps*3-1, shear=True)
            gamma_.append(gamma)
                 
        gamma_ = np.array(gamma_)


        cosmo = Cosmology.from_camb(camb_pars)
        IA_shear_ = []
        
        convergence = glass.lensing.MultiPlaneConvergence(cosmo)
        for ss in frogress.bar(range(len(density))):
            # get convergence field
            IA_shear = glass.lensing.from_convergence(density[ss]-np.mean(density[ss]), lmax=nside_maps*3-1, shear=True)
            IA_shear_.append(IA_shear)
     
        IA_shear_ = np.array(IA_shear_)
 


        
        # Do maps ****************************************************************************************************************
        corr_variance_array =  [  SC_corrections['corr_variance_fit'][tomo](bias_sc[tomo])       for tomo in range(nz.shape[0])]
        coeff_kurtosis_array = [  SC_corrections['coeff_kurtosis_fit'][tomo](bias_sc[tomo])       for tomo in range(nz.shape[0])]
        A_corr_array = [  SC_corrections['A_corr_fit'][tomo](bias_sc[tomo])       for tomo in range(nz.shape[0])]
        
        
        
        
       # kappa_tot  = np.zeros((nz.shape[0],12*nside_maps**2))
        g1_tot  = np.zeros((nz.shape[0],12*nside_maps**2))
        g2_tot  = np.zeros((nz.shape[0],12*nside_maps**2))
        d_tot  = np.zeros((nz.shape[0], 12*nside_maps**2))
        
        
        # load each lightcone output in turn and add it to the simulation
        # note: I added a -sign to gamma to match data conventions later
        for tomo in range(nz.shape[0]):
            for i in (range(len(gamma_))):       
                C1 = 5e-14
                rho_crit0_h2 = ccl.physical_constants.RHO_CRITICAL
                rho_c1 = C1 * rho_crit0_h2
                IA_f = F_nla(z=zeff_array[i],
                 om0=sims_parameters['Omega_m'],
                 A_ia=A_IA, rho_c1=rho_c1, eta=eta_IA, z0=0.67,
                 cosmo=cosmo_pyccl)
                
               # kappa_tot[tomo] += ngal_glass[tomo,i] * kappa_[i][0].real * (1 + bias_sc[tomo] * density[i])
                g1_tot[tomo] += ngal_glass[tomo,i] * (-gamma_[i][0].real-IA_shear_[i][0].real*IA_f) * (1 + bias_sc[tomo] * density[i])
                g2_tot[tomo] += ngal_glass[tomo,i] * (-gamma_[i][0].imag-IA_shear_[i][0].imag*IA_f) * (1 + bias_sc[tomo] * density[i])
                d_tot[tomo]  += ngal_glass[tomo,i] * (1 + bias_sc[tomo] * density[i] )
           
    
        path_data_cats = '/pscratch/sd/m/mgatti/euclid/Euclid_cats.npy'
        cats_Euclid  = np.load(path_data_cats,allow_pickle=True).item()
        
        cats_Gower = dict()
        maps_Gower = dict()
        for tomo in tomo_bins:
            maps_Gower[tomo] = dict()
        
            pix_ = convert_to_pix_coord(cats_Euclid[tomo]['ra'],cats_Euclid[tomo]['dec'], nside=nside_maps*2)
            pix = rotate_and_rebin(pix_, nside_maps, rot, delta_=delta_rot)
            sims_parameters['delta_rot'] = delta_rot
            # ---------------------------------------------------
            
            # source clustering term ~
            f = 1./np.sqrt(d_tot[tomo])
            f = f[pix]
        
        
            n_map = np.zeros(hp.nside2npix(nside_maps))
            n_map_sc = np.zeros(hp.nside2npix(nside_maps))
        
                            
            unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
        
        
            n_map_sc[unique_pix] += np.bincount(idx_rep, weights=cats_Euclid[tomo]['w']/f**2)
            n_map[unique_pix] += np.bincount(idx_rep, weights=cats_Euclid[tomo]['w'])
        
            g1_ = g1_tot[tomo][pix]
            g2_ = g2_tot[tomo][pix]
        
        
            es1,es2 = apply_random_rotation(cats_Euclid[tomo]['e1']/f, cats_Euclid[tomo]['e2']/f)
            es1_ref,es2_ref = apply_random_rotation(cats_Euclid[tomo]['e1'], cats_Euclid[tomo]['e2'])
            es1a,es2a = apply_random_rotation(cats_Euclid[tomo]['e1']/f, cats_Euclid[tomo]['e2']/f)
        
        
            #x1_sc,x2_sc = addSourceEllipticity({'shear1':g1_,'shear2':g2_},{'e1':es1,'e2':es2},es_colnames=("e1","e2"))
        
        
            e1r_map = np.zeros(hp.nside2npix (nside_maps))
            e2r_map = np.zeros(hp.nside2npix (nside_maps))
            e1r_map0 = np.zeros(hp.nside2npix(nside_maps))
            e2r_map0 = np.zeros(hp.nside2npix(nside_maps))
            e1r_map0_ref = np.zeros(hp.nside2npix(nside_maps))
            e2r_map0_ref = np.zeros(hp.nside2npix(nside_maps))
            g1_map = np.zeros(hp.nside2npix(nside_maps))
            g2_map = np.zeros(hp.nside2npix(nside_maps))
        
            unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
        
        
            e1r_map[unique_pix] += np.bincount(idx_rep, weights=es1*cats_Euclid[tomo]['w'])
            e2r_map[unique_pix] += np.bincount(idx_rep, weights=es2*cats_Euclid[tomo]['w'])
        
            e1r_map0[unique_pix] += np.bincount(idx_rep, weights=es1a*cats_Euclid[tomo]['w'])
            e2r_map0[unique_pix] += np.bincount(idx_rep, weights=es2a*cats_Euclid[tomo]['w'])
        
            e1r_map0_ref[unique_pix] += np.bincount(idx_rep, weights=es1_ref*cats_Euclid[tomo]['w'])
            e2r_map0_ref[unique_pix] += np.bincount(idx_rep, weights=es2_ref*cats_Euclid[tomo]['w'])
        
        
            mask_sims = n_map_sc != 0.
            e1r_map[mask_sims]  = e1r_map[mask_sims]/(n_map_sc[mask_sims])
            e2r_map[mask_sims] =  e2r_map[mask_sims]/(n_map_sc[mask_sims])
            e1r_map0[mask_sims]  = e1r_map0[mask_sims]/(n_map_sc[mask_sims])
            e2r_map0[mask_sims] =  e2r_map0[mask_sims]/(n_map_sc[mask_sims])
            e1r_map0_ref[mask_sims]  = e1r_map0_ref[mask_sims]/(n_map[mask_sims])
            e2r_map0_ref[mask_sims] =  e2r_map0_ref[mask_sims]/(n_map[mask_sims])
        
        
        
            var_ =  e1r_map0_ref**2+e2r_map0_ref**2
        
        
            #'''
            e1r_map   *= 1/(np.sqrt(A_corr_array[tomo]*corr_variance_array[tomo])) * np.sqrt((1+coeff_kurtosis_array[tomo]*var_))
            e2r_map   *= 1/(np.sqrt(A_corr_array[tomo]*corr_variance_array[tomo])) * np.sqrt((1+coeff_kurtosis_array[tomo]*var_))
            e1r_map0  *= 1/(np.sqrt(A_corr_array[tomo]*corr_variance_array[tomo])) * np.sqrt((1+coeff_kurtosis_array[tomo]*var_))
            e2r_map0  *= 1/(np.sqrt(A_corr_array[tomo]*corr_variance_array[tomo])) * np.sqrt((1+coeff_kurtosis_array[tomo]*var_))
        
        
        
            
            
            #'''
            g1_map[unique_pix] += np.bincount(idx_rep, weights= g1_*cats_Euclid[tomo]['w'])
            g2_map[unique_pix] += np.bincount(idx_rep, weights= g2_*cats_Euclid[tomo]['w'])
        
        
        
            g1_map[mask_sims]  = g1_map[mask_sims]/(n_map_sc[mask_sims])
            g2_map[mask_sims] =  g2_map[mask_sims]/(n_map_sc[mask_sims])
        
            e1_ = ((g1_map*dm[tomo]+e1r_map0))[mask_sims]
            e2_ = ((g2_map*dm[tomo]+e2r_map0))[mask_sims]
            e1n_ = ( e1r_map)[mask_sims]
            e2n_ = ( e2r_map)[mask_sims]
            idx_ = np.arange(len(mask_sims))[mask_sims]
    
            maps_Gower[tomo] =     {'e1':e1_,'e2':e2_,'e1n':e1n_,'e2n':e2n_,
                                    'idx':idx_}
    
    
        maps_Gower['sims_parameters'] = sims_parameters
        np.save(path_maps_gower,maps_Gower)
    

        '''
        # make a catalog ---------------------------------------------------------------------------------------------------------------------------------
        SC_per_pixel_correction_noise  = f**2/((np.sqrt(A_corr_array[tomo]*corr_variance_array[tomo])) * np.sqrt((1+coeff_kurtosis_array[tomo]*var_)))[pix]
        
        # the f**2 applied to g1,g2 is the normalisation missing in the g1_tot,g2_tot ---------------------------------------------------
        e1_SC = g1_*f**2+es1a*SC_per_pixel_correction_noise
        e2_SC = g2_*f**2+es2a*SC_per_pixel_correction_noise
        #e1_SC,e2_SC = addSourceEllipticity({'shear1':g1_,'shear2':g2_},{'e1':es1a*SC_per_pixel_correction_noise,'e2':es2a*SC_per_pixel_correction_noise},es_colnames=("e1","e2"))
        cats_Gower[tomo] =  {'ra':cats_Euclid[tomo]['ra'],'dec':cats_Euclid[tomo]['dec'],'e1':e1_SC,'e2':e2_SC,'w':cats_Euclid[tomo]['w']}
        '''



if __name__ == '__main__':


    # nuisance_parameters -----------------------------------------------------
    do_maps = True
    
    dz_mean = [0,0,0,0,0,0,0]
    dz_spread = [0.01,0.01,0.01,0.01,0.01,0.01,0.01]
    
    dm_mean = [1,1,1,1,1,1,1]
    dm_spread = [0.05,0.05,0.05,0.05,0.05,0.05,0.05]
    
    A0_interval  = [-2.5,2.5]
    eta_interval = [-2.5,2.5]
    
    bias_SC_interval = [0.5,1.5]
    
    # Baryonification settings ------------------------------------------------
    
    baryonification = True
    do_tSZ = True
    nside_baryonification = 1024
    max_redshift_halo_catalog = 1.5
    halo_catalog_log10mass_cut = 13.5
    base_params_path = "../../Data/Baryonification_wl_tsz_flamingo_parameters.npy"
    baryon_priors =   {"M_c": (12.5, 15.5, "lin"), 
                      "theta_ej": (3.0, 10.0, "lin"),
                       "eta": (-2.0, -0.1, "log10")} 

    # general maps -------------------------------------------------------------
    nside_maps = 1024
    tomo_bins = [1,2,3,4,5,6]
    delta_rot = 0.

    '''
    dz_mean = [0,0,0,0,0,0,0]
    dz_spread = [0,0,0,0,0,0,0]
    
    dm_mean = [1,1,1,1,1,1,1]
    dm_spread = [0,0,0,0,0,0,0]
    
    A0_interval  = [0,0]
    eta_interval = [0,0]
    
    bias_SC_interval = [1,1]
    path_simulation = '/pscratch/sd/m/mgatti/highres_SBI/Flagship_covariance_big/4_big/'
    rot = 0
    noise_rel = 0
    run(path_simulation, rot, noise_rel)
    '''

  
    from pathlib import Path
    BASE = Path("/pscratch/sd/m/mgatti/highres_SBI/runsU")
    TARGET = "particles_100_4096.parquet"
    have = sorted(p for p in BASE.glob("run*/") if (p / TARGET).is_file())
    missing = sorted(set(BASE.glob("run*/")) - set(have))

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
                    runs.append([str(path)+'/',rot,noise_rel])



    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Start with run_count = 0, but each process handles tasks based on rank
    run_count = rank 

    while run_count < len(runs):
       
        print (runs[run_count][0])
#try:
        run(runs[run_count][0], runs[run_count][1], runs[run_count][2])
       # except:
       #     print ('failed ',runs[run_count][0])
            
        run_count += size
    comm.Barrier()





# do_maps = False 
#module load python; source activate pyccl_env;  python  generate_mocks.py
#module load python; source activate pyccl_env; srun --nodes=4 --tasks-per-node=8 python  generate_mocks.py
# do_maps = True
#module load python; source activate pyccl_env; srun --nodes=4 --tasks-per-node=24 python  generate_mocks.py

#salloc --nodes 4 --qos interactive --time 04:00:00 --constraint cpu  --account=m5099