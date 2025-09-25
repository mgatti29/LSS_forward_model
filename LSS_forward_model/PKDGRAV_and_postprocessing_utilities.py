import camb
from camb import model
import glob
import h5py
import numpy as np
import gc
import frogress
import gc
import pandas as pd
import scipy
from scipy.interpolate import CubicSpline
import healpy as hp
import pyccl as ccl
import os



def As_to_sigma8_CAMB(Omega_m, Omega_b, h, n_s, w0, wa, m_nu, As):
    """
    Compute sigma_8 using CAMB given A_s and cosmological parameters.

    Parameters
    ----------
    Omega_m : float
        Total matter density today (including baryons, CDM, neutrinos).
    Omega_b : float
        Baryon density today.
    h : float
        Dimensionless Hubble parameter.
    n_s : float
        Scalar spectral index.
    w0 : float
        Dark energy equation of state parameter w0.
    wa : float
        Time variation of the dark energy equation of state.
    m_nu : float
        Total neutrino mass in eV.
    As : float
        Primordial scalar amplitude.

    Returns
    -------
    sigma_8 : float
        The resulting sigma_8 from CAMB.
    """
    Omega_nu = m_nu / (93.14 * h**2)
    Omega_cdm = Omega_m - Omega_b - Omega_nu
    ombh2 = Omega_b * h**2
    omch2 = Omega_cdm * h**2

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h*100, ombh2=ombh2, omch2=omch2, mnu=m_nu,
                       tau=0.06, num_massive_neutrinos=3)
    pars.InitPower.set_params(As=As, ns=n_s)
    pars.set_dark_energy(w=w0, wa=wa, dark_energy_model='ppf')
    pars.NonLinear = camb.model.NonLinear_both
    pars.WantCls = False
    pars.DoLensing = False
    pars.WantTransfer = True
    results = camb.get_results(pars)
    return results.get_sigma8()


def sigma8_to_As_CAMB(Omega_m, Omega_b, h, n_s, w0, wa, m_nu, sigma8_target,
                      tol=1e-4, max_iter=20, As_init=2.1e-9):
    """
    Iteratively compute A_s that yields the target sigma_8 in CAMB.

    Parameters
    ----------
    Omega_m, Omega_b, h, n_s, w0, wa, m_nu : float
        Cosmological parameters as in `As_to_sigma8_CAMB`.
    sigma8_target : float
        Desired sigma_8 value.
    tol : float
        Relative tolerance for sigma_8 convergence.
    max_iter : int
        Maximum number of iterations.
    As_init : float
        Initial guess for A_s.

    Returns
    -------
    As : float
        Scalar amplitude that gives the desired sigma_8.
    """
    Omega_nu = m_nu / (93.14 * h**2)
    Omega_cdm = Omega_m - Omega_b - Omega_nu

    ombh2 = Omega_b * h**2
    omch2 = Omega_cdm * h**2
    As = As_init

    for _ in range(max_iter):
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=h*100, ombh2=ombh2, omch2=omch2, mnu=m_nu,
                           tau=0.06, num_massive_neutrinos=3)
        pars.InitPower.set_params(As=As, ns=n_s)
        pars.set_dark_energy(w=w0, wa=wa, dark_energy_model='ppf')
        pars.NonLinear = camb.model.NonLinear_both
        pars.WantCls = False
        pars.DoLensing = False
        pars.WantTransfer = True

        results = camb.get_results(pars)
        sigma8_model = results.get_sigma8()

        # Update As using scaling: sigma8 ∝ sqrt(As)
        As *= (sigma8_target / sigma8_model)**2

        if abs(sigma8_model - sigma8_target) / sigma8_target < tol:
            return As

    raise RuntimeError("Failed to converge on As after {} iterations.".format(max_iter))

def recover_shell_info(path_z_file, max_z=49):
    """
    Reads redshift shell boundary information from a file and returns a dictionary 
    of comoving shell properties, starting from the shell where z_far == max_z.

    Parameters
    ----------
    path_z_file : str
        Path to the file containing shell boundary information. The file is expected 
        to be a CSV-like format with 7 columns: step, z_far, z_near, delta_z, 
        cmd_far, cmd_near, delta_cmd.
    max_z : float, optional
        The maximum redshift to locate in the z_far column, used to truncate earlier shells.
        Defaults to 49.

    Returns
    -------
    resume : dict
        Dictionary containing arrays of shell information starting from the shell
        where z_far == max_z:
            'Step'      : Array of shell indices (renormalized to start from 0)
            'z_far'     : Redshift of the far edge of the shell
            'z_near'    : Redshift of the near edge of the shell
            'delta_z'   : Shell thickness in redshift
            'cmd_far'   : Comoving distance to the far edge (Mpc)
            'cmd_near'  : Comoving distance to the near edge (Mpc)
            'delta_cmd' : Shell thickness in comoving distance (Mpc)
            'z_edges'   : Redshift bin edges (length N+1)
            'mean_z'    : Mean redshift per shell (length N)

    """

    # Initialize the resume dictionary with empty lists
    resume = {
        'Step': [],
        'z_far': [],
        'z_near': [],
        'delta_z': [],
        'cmd_far': [],
        'cmd_near': [],
        'delta_cmd': []
    }

    # Open the file containing z values
    with open(path_z_file) as z_fil:
        z = []
        # Iterate over each line in the file
        for z__, z_ in enumerate(z_fil):
            if z__ > 0:
                # Split the line by commas and convert to float
                mute = np.array(z_.split(',')).astype(float)
                
                # Append each value to the corresponding list in the resume dictionary
                resume['Step'].append(mute[0])
                resume['z_far'].append(mute[1])
                resume['z_near'].append(mute[2])
                resume['delta_z'].append(mute[3])
                resume['cmd_far'].append(mute[4])
                resume['cmd_near'].append(mute[5])
                resume['delta_cmd'].append(mute[6])

    # Find the index of the last occurrence of the value max_z in the 'z_far' list
    init = np.where(np.array(resume['z_far']) == max_z)[0][-1]

    # Adjust the lists in the resume dictionary from the found index
    resume['Step'] = np.array(resume['Step'])[init:] - init
    resume['z_far'] = np.array(resume['z_far'])[init:]
    resume['z_near'] = np.array(resume['z_near'])[init:]
    resume['delta_z'] = np.array(resume['delta_z'])[init:]
    resume['cmd_far'] = np.array(resume['cmd_far'])[init:]
    resume['cmd_near'] = np.array(resume['cmd_near'])[init:]
    resume['delta_cmd'] = np.array(resume['delta_cmd'])[init:]
    resume['z_edges']  = np.hstack([resume['z_far'][0],resume['z_near']])
    resume['mean_z']  = 0.5*(resume['z_edges'][1:]+resume['z_edges'] [:-1])



    # Return the processed resume dictionary
    return resume
    
def read_sims_params(path):

    """
    Extract cosmological parameters from CLASS simulation outputs.

    Parameters
    ----------
    path : str
        Path to the simulation directory containing 'class_processed*' HDF5 file
        and the CLASS 'control.par' parameter file.

    Returns
    -------
    sims_parameters : dict
        Dictionary with the following keys:
            'Omega_b'     : Baryon density parameter
            'Omega_nu'    : Neutrino density parameter
            'Omega_r'     : Photon/radiation density parameter
            'Omega_m'     : Total matter density (baryons + CDM + neutrinos)
            'Omega_cdm'  : Cold dark matter density parameter
            'h'           : Dimensionless Hubble parameter
            'w0'          : Equation of state of dark energy today
            'wa'          : Evolution parameter of the equation of state
            'As'          : Scalar amplitude of primordial power spectrum
            'n_s'         : Scalar spectral index
            'sigma_8'     : RMS matter fluctuation amplitude at 8 Mpc/h
            'm_nu'          : Total neutrino mass in eV
            'dBoxSize Mpc/h' : Box size in Mpc/h

    cosmo_pyccl: pyccl cosmology object
    """

    with h5py.File(glob.glob(path + '/class_processed*')[0], 'r') as f:
        background = f['background']
        
        # Extract density parameters
        Omega_b   = (background['rho_b'][()]      / background['rho_crit'][()])[-1]
        Omega0_cdm = (background['rho_cdm'][()]    / background['rho_crit'][()])[-1]
        try:
            Omega_nu  = ((background['rho_ncdm[0]'][()] +background['rho_ncdm[1]'][()] +background['rho_ncdm[2]'][()] )/ background['rho_crit'][()])[-1]
        except:
            Omega_nu  = (background['rho_ncdm[0]'][()]/ background['rho_crit'][()])[-1]
      
            
        Omega_r   = (background['rho_g'][()]      / background['rho_crit'][()])[-1]
        Omega_m   = Omega_b + Omega0_cdm + Omega_nu

        # Extract Hubble parameter
        h = background['H'][-1] * 977.792 / 100  # Hubble parameter at z=0

        # Equation of state
        if 'w_fld' in background:
            w0 = background['w_fld'][-1] 

            w = background['w_fld'][-100]
            z = np.array(background['z'])[-100]
            fz = z/(1.+z)
            wa = (w-w0)/fz
        else:
            w0 = -1
            wa = 0


    m_nu = Omega_nu*(93.14 * (h)**2)

    # Read scalar amplitude and spectral index from control.par
    file_path = path + '/control.par'
    values = {}
    with open(file_path, 'r') as f:
        for line in f:
            if 'dNormalization' in line or 'dSpectral' in line or "dBoxSize" in line:
                parts = line.split('=')
                if len(parts) > 1:
                    key = parts[0].strip()
                    value = parts[1].split('#')[0].strip()
                    values[key] = float(value)

    As =  values['dNormalization']
    n_s = values['dSpectral']
    dBoxSize = values['dBoxSize']

    # Compute sigma8 from As
    sigma_8 = As_to_sigma8_CAMB(Omega_m, Omega_b, h, n_s, w0, wa, m_nu, As)

    # Return parameters in dictionary form

    sims_parameters = {
        'Omega_b': Omega_b,
        'Omega_nu': Omega_nu,
        'Omega_r': Omega_r,
        'Omega_m': Omega_m,
        'Omega_cdm': Omega0_cdm,
        'h': h,
        'w0': w0,
        'wa': wa,
        'As': As,
        'n_s': n_s,
        'sigma_8': sigma_8[0],
        'm_nu':m_nu,
        'dBoxSize Mpc/h':dBoxSize,
    }

    cosmo_pyccl = ccl.Cosmology(Omega_c = sims_parameters['Omega_cdm'], Omega_b = sims_parameters['Omega_b'], 
                          h =  sims_parameters['h'], sigma8 = sims_parameters['sigma_8'], 
                          n_s = sims_parameters['n_s'], #w0 = cosmological_parameters['w0'], wa = cosmological_parameters['wa'], 
                          m_nu = [sims_parameters['m_nu']/3,sims_parameters['m_nu']/3,sims_parameters['m_nu']/3],mass_split='equal',
                          matter_power_spectrum='linear')



    pars = camb.CAMBparams()
    pars.set_cosmology(H0=sims_parameters['h']*100, ombh2=sims_parameters['Omega_b']*sims_parameters['h']**2, omch2=sims_parameters['Omega_cdm']*sims_parameters['h']**2, mnu=sims_parameters['m_nu'],
                       tau=0.06, num_massive_neutrinos=3)
    pars.InitPower.set_params(As=sims_parameters['As'], ns=sims_parameters['n_s'])
    pars.set_dark_energy(w=sims_parameters['w0'], wa=sims_parameters['wa'], dark_energy_model='ppf')
    pars.NonLinear = camb.model.NonLinear_both
    pars.WantCls = False
    pars.DoLensing = False
    pars.WantTransfer = True


    return sims_parameters, cosmo_pyccl, pars
    


def save_halocatalog(shells_info, sims_parameters, max_redshift = 1.5, halo_snapshots_path = '' , catalog_path = 'halo_catalog.parquet',log10_mass_limit = 0.):

    """
    Generate a lightcone halo catalog by assembling halo snapshots across redshift shells, 
    replicating boxes to fill the shell volume, and saving the final catalog to a FITS file.

    Parameters
    ----------
    shells_info : dict
        Dictionary containing shell definitions with keys like 'cmd_near', 'cmd_far', 'Step', etc.
        Typically output from `recover_shell_info_`.

    sims_parameters : dict
        Dictionary of cosmological parameters and simulation metadata. Must contain:
            - 'Omega_cdm', 'Omega_b', 'h', 'sigma_8', 'n_s', 'm_nu' (in eV), etc.
            - 'dBoxSize Mpc/h' : Box size in comoving Mpc/h units.

    max_redshift : float, optional
        Upper redshift limit for including halos in the lightcone (default is 1.5).

    catalog_path : str, optional
        Path to save the resulting halo catalog in FITS format.

    Notes
    -----
    - Halos are replicated in a cubic grid until they cover the spherical shell between cmd_near and cmd_far.
    - Box replication ensures volume completeness for the shell.
    - Uses `pyccl` to compute comoving distances for redshift-distance interpolation.
    - Outputs a FITS file containing positions, redshifts, masses, and optional angular coordinates.

    Output
    ------
    A binary FITS table saved at `catalog_path`, containing:
        - pix_16384_ring: HEALPix pixel index
        - log_M: Halo mass (in Msun/h, scaled by 1000)
        - R: Halo half-mass radius (in kpc/h, scaled by 1000)
        - redshift: Scaled by 10000
    """  


    def load_snapshot(path_base, c_, Lbox_Mpc, log10_mass_limit = None):
        """
        Loads halo data from a specified path based on the mode.
    
        :param path_base: Base path to the data
        :param c_: Configuration index
        :param mode: Mode of the data ('rockstar' or other)
        :param f_mass: Mass factor
        Lbox_Mpc: Lbox in Mpc
        :return: Dictionary containing halo data
        """
        c__ = f'{int(c_):03}'

        p = f'{path_base}run.00{c__}.fofstats.parquet'
        
        pkd_halo_dtype = np.dtype([("rPot", ("f4", 3)), ("minPot", "f4"), ("rcen", ("f4", 3)),
                                   ("rcom", ("f4", 3)), ("cvom", ("f4", 3)), ("angular", ("f4", 3)),
                                   ("inertia", ("f4", 6)), ("sigma", "f4"), ("rMax", "f4"),
                                   ("fMAss", "f4"), ("fEnvironDensity0", "f4"),
                                   ("fEnvironDensity1", "f4"), ("rHalf", "f4")])

        parquet_ = True
        columns_to_read = ['halo_center', 'rmax', 'log10M']
        halos = pd.read_parquet(p, columns=columns_to_read)
        if log10_mass_limit is not None:
            mask = halos['log10M'] > log10_mass_limit*1000
            halos = halos[mask]
        

        centers = np.array([x for x in np.array(halos['halo_center'])])
        M = np.array([x for x in np.array(halos['log10M'])])
        rmax = np.array([x for x in np.array(halos['rmax'])])


        output = {
            'x': centers[:, 0],
            'y': centers[:, 1],
            'z': centers[:, 2],
            'rhalf': rmax,
            'M': M 
        }

                
        return output
    

    def may_intersect_sphere(x_i, y_i, z_i, Lbox_Mpc, d_min, d_max):
        """
        Determine whether a replicated simulation box intersects with a spherical shell.
    
        Parameters
        ----------
        x_i, y_i, z_i : int
            Replication indices along each axis. The box is translated by (x_i, y_i, z_i) * Lbox_Mpc.
    
        Lbox_Mpc : float
            Side length of the simulation box in comoving Mpc.
    
        d_min : float
            Inner radius of the spherical shell (in Mpc).
    
        d_max : float
            Outer radius of the spherical shell (in Mpc).
    
        Returns
        -------
        intersects : bool
            True if the box may intersect the shell between d_min and d_max.
    
        min_distance : float
            Minimum possible distance from the origin to any point in the box.
    
        max_distance : float
            Maximum possible distance from the origin to any point in the box.
    
        Notes
        -----
        This function assumes the box is axis-aligned and cubic. It uses the box's 
        center and diagonal to conservatively estimate the range of distances covered 
        by the box and checks for overlap with the target spherical shell.
        """
    # Center of the box after translation
        center_x = x_i * Lbox_Mpc + Lbox_Mpc / 2
        center_y = y_i * Lbox_Mpc + Lbox_Mpc / 2
        center_z = z_i * Lbox_Mpc + Lbox_Mpc / 2
        center = np.array([center_x, center_y, center_z])
    
        # Distance from the origin to the center of the box
        center_distance = np.linalg.norm(center)
        
        # Radius of the sphere that contains the entire box (half-diagonal of the box)
        half_diagonal = np.sqrt(3) * (Lbox_Mpc / 2)
    
        # Calculate the minimum and maximum distances any point in the box could be from the origin
        min_distance = max(0, center_distance - half_diagonal)
        max_distance = center_distance + half_diagonal
    
        # Check if there's any overlap between the box and the sphere range
        return (min_distance <= d_max and max_distance >= d_min),min_distance,max_distance


    

    cosmo = ccl.Cosmology(Omega_c = sims_parameters['Omega_cdm'], Omega_b = sims_parameters['Omega_b'], 
                          h =  sims_parameters['h'], sigma8 = sims_parameters['sigma_8'], 
                          n_s = sims_parameters['n_s'], #w0 = cosmological_parameters['w0'], wa = cosmological_parameters['wa'], 
                          m_nu = [sims_parameters['m_nu']/3,sims_parameters['m_nu']/3,sims_parameters['m_nu']/3],mass_split='equal',
                          matter_power_spectrum='linear')
    z_hr = np.linspace(0, 10, 5001)
    d_hr = ccl.comoving_radial_distance(cosmo, 1./(1+z_hr))

    interpolated_distance_to_redshift = CubicSpline(d_hr, z_hr)
    interpolated_redshift_to_distance = CubicSpline(z_hr, d_hr)

    max_step_halocatalog = len(shells_info['z_far'])-int(shells_info['Step'][[shells_info['z_far']<max_redshift][0]][0])+1

    Lbox_Mpc = sims_parameters['dBoxSize Mpc/h']/ sims_parameters['h']
    
    # Initialize the final catalog dictionary
    final_cat = {
        'pix_16384_ring' : [],
        'x': [],
        'y': [],
        'z': [],
        'M': [],
        'redshift': [],
        'R': [],
        'ra': [],
        'dec': [],
        'redshift_hr': [],
        
        
        
    }
    
    count = 0
   
    # Iterate through each step in the halo catalog
    for i_ in frogress.bar(np.arange(0, max_step_halocatalog)):
        i = len(shells_info['Step']) - i_ - 1
        d_min = shells_info['cmd_near'][i]
        d_max = shells_info['cmd_far'][i]
        step = shells_info['Step'][i]

        # Load the snapshot data for the current step
        output_ = load_snapshot(halo_snapshots_path, step, Lbox_Mpc, log10_mass_limit = log10_mass_limit)


        replicas_max = np.ceil(d_max / Lbox_Mpc).astype(int)
        replicas_min = np.ceil(d_min / Lbox_Mpc).astype(int)


        count_i = 0
        add = 0

        f = 1.0
        
        final_cat_x = []
        final_cat_y = []
        final_cat_z = []
        final_cat_M = []
        final_cat_R = []
        final_cat_redshift = []
        # Iterate through replicas
        for x_i in range(-replicas_max, replicas_max ):
            for y_i in range(-replicas_max , replicas_max ):
                for z_i in range(-replicas_max , replicas_max ):
                    may_intersect, close_box,far_box = may_intersect_sphere(x_i, y_i, z_i, Lbox_Mpc ,d_min, d_max)
                        
                    if may_intersect:
                            new_x = output_['x'] + x_i * Lbox_Mpc
                            new_y = output_['y'] + y_i * Lbox_Mpc
                            new_z = output_['z'] + z_i * Lbox_Mpc
                            r = np.sqrt(new_x**2 + new_y**2 + new_z**2)
                            mask = (r >= d_min) & (r < d_max)
                            if np.any(mask):
                                final_cat_x.append(new_x[mask])
                                final_cat_y.append(new_y[mask])
                                final_cat_z.append(new_z[mask])
                                final_cat_M.append(output_['M'][mask])
                                final_cat_R.append(output_['rhalf'][mask])
                                final_cat_redshift.append(interpolated_distance_to_redshift(r[mask]))
                            add += 1
         
                            
                            

        # Append collected data from this step to the final catalog
        if add>0:
            final_cat['pix_16384_ring'].append(hp.pixelfunc.vec2pix(8192 * 2, np.concatenate(final_cat_x), np.concatenate(final_cat_y), np.concatenate(final_cat_z), nest=False))
            final_cat['M'].append(np.concatenate(final_cat_M) )
            final_cat['R'].append(np.concatenate(final_cat_R) )
            final_cat['redshift'].append(np.concatenate(final_cat_redshift) * 10000)

    for key in final_cat:
        if isinstance(final_cat[key], list):
            final_cat[key] = np.concatenate(final_cat[key]) if final_cat[key] else np.array([])

    # Save the final catalog to a FITS file
    if os.path.exists(path_to_save):
        os.remove(path_to_save)

    print ('Done assemblying')
    
    fits_f = dict()
    fits_f['pix_16384_ring'] = (final_cat['pix_16384_ring']).astype('uint32')
    fits_f['log_M'] = (final_cat['M']).astype('uint16')  # this is Msun/h ---
    fits_f['R'] = (final_cat['R']).astype('uint16')
    fits_f['redshift'] = (final_cat['redshift']).astype('uint16')
    
    df = pd.DataFrame(fits_f)

    # Save it as a Parquet file
    df.to_parquet(catalog_path, index=False)

    # Clean up
    del final_cat
    gc.collect()



def reconstruct_inertia_tensor_halo_snapshot(df):
    """
    Reconstructs the original 6-component inertia tensor from
    inertia_auto and inertia_cross columns in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'inertia_auto' and 'inertia_cross' columns.

    Returns
    -------
    inertia : np.ndarray of shape (N, 6)
        Reconstructed inertia tensor per halo, in the order:
        [Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
    """

    # Convert stored lists back to arrays
    inertia_auto_log = np.stack(df['inertia_auto'].apply(np.array).to_numpy())
    inertia_cross_scaled = np.stack(df['inertia_cross'].apply(np.array).to_numpy())

    # Invert transformation for auto components: Ixx, Iyy, Izz
    principal_moments = 10**(inertia_auto_log.astype(float) / 1000) / 1e20
    Ixx, Iyy, Izz = principal_moments[:, 0], principal_moments[:, 1], principal_moments[:, 2]

    # Invert transformation for cross components: Ixy, Ixz, Iyz
    cross = inertia_cross_scaled.astype(float) / 10000 - 1
    Ixy = cross[:, 0] * np.sqrt(Ixx * Iyy)
    Ixz = cross[:, 1] * np.sqrt(Ixx * Izz)
    Iyz = cross[:, 2] * np.sqrt(Iyy * Izz)

    # Stack into (N, 6) array in the original order
    inertia = np.stack([Ixx, Ixy, Ixz, Iyy, Iyz, Izz], axis=1)
    return inertia

def reconstruct_angular_halo_snapshot(df):
    """
    Reconstructs the original angular vectors from the 'angular' column
    in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with an 'angular' column containing transformed int16 triplets.

    Returns
    -------
    angular : np.ndarray of shape (N, 3)
        Reconstructed angular vectors.
    """
    angular_encoded = np.stack(df['angular'].apply(np.array).to_numpy()).astype(float)
    
    # Invert the transformation:
    # x_original = sign * (10**(abs(x_encoded)/1000) - 1) / 1e20
    sign = np.sign(angular_encoded)
    magnitude = (10**(np.abs(angular_encoded) / 1000) - 1) / 1e20
    angular = sign * magnitude
    return angular


def recover_halo_mass(df):
    """
    Recover mass in M_sun/h from df['log10M'].

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a 'log10M' column (uint16, scaled).

    Returns
    -------
    mass : np.ndarray
        Mass in M_sun/h.
    """
    return 10**(df['log10M'].to_numpy() / 1000. + 4)


def recover_halo_redshift(df):
    """
    Recover redshift from scaled df['redshift'] values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a 'redshift' column (uint16, scaled by 10000).

    Returns
    -------
    z : np.ndarray
        Redshift values as float.
    """
    return df['redshift'].to_numpy() / 10000.
    
def recover_halo_centers(df):
    """
    Recover halo centers from df['halo_center'] as a (N, 3) numpy array in Mpc.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a 'halo_center' column (lists of 3 floats).

    Returns
    -------
    centers : np.ndarray of shape (N, 3)
        Halo center coordinates in Mpc.
    """
    return np.array(df['halo_center'].to_list())

def recover_fof_radius(df):
    """
    Recover FOF radius from df['rmax'] column, returning values in Mpc.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'rmax' column (uint16, stored in kpc).

    Returns
    -------
    rmax_mpc : np.ndarray
        FOF radius in Mpc.
    """
    return df['rmax'].to_numpy() / 1000.


def recover_halo_radec(df, nest=False):
    """
    Recover declination and right ascension from HEALPix pixel index.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a 'pix_16384_ring' column (HEALPix pixel index).

    nest : bool, optional
        If True, assumes nested ordering. Default is RING ordering (False).

    Returns
    -------
    dec, ra : np.ndarray
        Declination and Right Ascension in degrees.
    """
    theta, phi = hp.pix2ang(16384, df['pix_16384_ring'].to_numpy(), nest=nest)
    dec = np.degrees(0.5 * np.pi - theta)
    ra = np.degrees(phi)
    return ra, dec




def baryonify_shell(halos, sims_parameters, counts, bpar, min_z, max_z, nside):
    """
    Apply baryonification to a projected lightcone shell using the Baryonification 2D framework.

    This routine constructs a 2D baryonification displacement model using DMO and DMB profiles,
    interpolates the model across a given redshift shell, and applies it to a halo lightcone
    catalog to compute the baryon-modified density field on a HEALPix shell.

    Parameters
    ----------
    halos : dict
        Dictionary containing halo catalog with keys 'ra', 'dec', 'z', and 'M'.
        Halos should be in physical units (M in Msun/h, z is redshift).

    sims_parameters : dict
        Dictionary of cosmological parameters including:
            - 'Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'w0'

    counts : np.ndarray
        2D HEALPix map of halo counts or projected mass on the shell (same `nside`).

    bpar : dict
        Dictionary of baryonification model parameters. Must include at least:
            - profile parameters for DMO and DMB models
            - 'epsilon_max': maximum smoothing scale in Mpc

    min_z, max_z : float
        Redshift bounds of the shell to which baryonification will be applied.

    nside : int
        HEALPix resolution parameter of the shell.

    Returns
    -------
    density_baryonified : np.ndarray
        HEALPix map of the baryonified overdensity field:
            δ = ρ / ⟨ρ⟩ - 1
    """

    DMO = bfn.Profiles.DarkMatterOnly(**bpar)
    DMB = bfn.Profiles.DarkMatterBaryon(**bpar)
    PIX = bfn.HealPixel(nside)
    DMO = bfn.ConvolvedProfile(DMO, PIX)
    DMB = bfn.ConvolvedProfile(DMB, PIX)

    Displacement = bfn.Profiles.Baryonification2D(
        DMO, DMB, cosmo=cosmo, epsilon_max=bpar['epsilon_max']
    )
    Displacement.setup_interpolator(
        z_min=min_z, z_max=max_z, N_samples_z=2, z_linear_sampling=True,
        R_min=1e-4, R_max=300, N_samples_R=2000, verbose=True
    )

    cdict = {
        'Omega_m': sims_parameters['Omega_m'],
        'sigma8': sims_parameters['sigma_8'],
        'h': sims_parameters['h'],
        'n_s': sims_parameters['n_s'],
        'w0': sims_parameters['w0'],
        'Omega_b': sims_parameters['Omega_b'],
    }

    shell = bfn.utils.LightconeShell(map=counts, cosmo=cdict)

    mask_z = (halos['z'] > min_z) & (halos['z'] < max_z)

    halos_ = bfn.utils.HaloLightConeCatalog(
        halos['ra'][mask_z],
        halos['dec'][mask_z],
        halos['M'][mask_z],
        halos['z'][mask_z],
        cosmo=cdict
    )

    Runners = bfn.Runners.BaryonifyShell(
        halos_, shell, epsilon_max=bpar['epsilon_max'], model=Displacement, verbose=True
    )

    baryonified_shell = Runners.process()
    density_baryonified = baryonified_shell / np.mean(baryonified_shell) - 1

    return density_baryonified


def make_density_maps(shells_info,path_simulation,path_output,nside_maps):
    """
    Generate or load downgraded Healpy maps of the density contrast for each simulation shell.

    This function computes the density contrast δ = (n / <n>) - 1 from particle counts 
    stored in parquet files for each shell defined in `shells_info`. It then downgrades 
    the resolution of each map to `nside_maps` using Healpy's `ud_grade` and saves the result 
    to disk. If the maps already exist, they are loaded instead of recomputed.

    Parameters
    ----------
    shells_info : dict
        Dictionary containing metadata for each shell, including a 'Step' key that holds 
        the list of simulation timesteps (should be sortable as integers).
    path_simulation : str
        Path to the directory containing the simulation outputs, where each file is named 
        'particles_<step>_4096.parquet'.
    nside_maps : int
        Desired Nside resolution for the output Healpy maps.

    Returns
    -------
    delta : np.ndarray
        Array of downgraded Healpy maps of the density contrast for each shell. Shape is 
        (number_of_shells, 12 * nside_maps**2).
    """

    delta= []
    for iii in frogress.bar(range(len(shells_info['Step']))):
        try:
            step = shells_info['Step'][::-1][iii]
            path = path_simulation + '/particles_{0}_4096.parquet'.format(int(step))
            counts = np.array(pd.read_parquet(path)).flatten()
            d = counts/np.mean(counts)-1
            delta.append(hp.ud_grade(d,nside_out=1024))
        except:
            pass
    delta = np.array(delta)
    np.save(path_output,delta)

    return delta


def addSourceEllipticity(self,es,es_colnames=("e1","e2"),rs_correction=True,inplace=False):

    """
    :param es: array of intrinsic ellipticities, 
    """

    #Safety check
    assert len(self)==len(es)

    #Compute complex source ellipticity, shear
    es_c = np.array(es[es_colnames[0]]+es[es_colnames[1]]*1j)
    g = np.array(self["shear1"] + self["shear2"]*1j)

    #Shear the intrinsic ellipticity
    e = es_c + g
    if rs_correction:
        e /= (1 + g.conjugate()*es_c)

    #Return
    if inplace:
        self["shear1"] = e.real
        self["shear2"] = e.imag
    else:
        return (e.real,e.imag)

def apply_random_rotation(e1_in, e2_in):
    """
    Applies a random rotation to the input ellipticities.

    Args:
        e1_in (array): Input ellipticities (component 1).
        e2_in (array): Input ellipticities (component 2).

    Returns:
        tuple: Rotated ellipticities.
    """
    np.random.seed()
    rot_angle = np.random.rand(len(e1_in)) * 2 * np.pi
    cos = np.cos(rot_angle)
    sin = np.sin(rot_angle)
    e1_out = e1_in * cos + e2_in * sin
    e2_out = -e1_in * sin + e2_in * cos
    return e1_out, e2_out


def convert_to_pix_coord(ra, dec, nside=1024):
    """
    Converts RA,DEC to hpix coordinates
    """
    theta = (90.0 - dec) * np.pi / 180.
    phi = ra * np.pi / 180.
    pix = hp.ang2pix(nside, theta, phi, nest=False)
    return pix



def g2k_sphere(gamma1, gamma2, mask, nside=1024, lmax=2048):
    """
    Convert shear to convergence on a sphere. In put are all healpix maps.
    """

    gamma1_mask = gamma1 * mask
    gamma2_mask = gamma2 * mask

    KQU_masked_maps = [gamma1_mask, gamma1_mask, gamma2_mask]
    alms = hp.map2alm(KQU_masked_maps, lmax=lmax, pol=True)  # Spin transform!


    ell, emm = hp.Alm.getlm(lmax=lmax)


    almsE = alms[1] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5
    almsB = alms[2] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5

    almsE[ell == 0] = 0.0
    almsB[ell == 0] = 0.0
    almsE[ell == 1] = 0.0
    almsB[ell == 1] = 0.0



    almssm = [alms[0], almsE, almsB]


    kappa_map_alm = hp.alm2map(almssm[0], nside=nside, lmax=lmax, pol=False)
    E_map = hp.alm2map(almssm[1], nside=nside, lmax=lmax, pol=False)
    B_map = hp.alm2map(almssm[2], nside=nside, lmax=lmax, pol=False)

    return E_map, B_map, almsE, alms



def shift_nz(z, nz, z_rebinned, delta_z=0.0, renorm="source"):
    """
    Evaluate the shifted n(z - delta_z) on z_rebinned.
    
    Parameters
    ----------
    z : array-like
        Original redshift grid (must be monotonic).
    nz : array-like
        n(z) sampled on `z`. Can be normalized or not.
    z_rebinned : array-like
        Target redshift grid where you want the shifted n(z) evaluated.
    delta_z : float, optional
        Shift to apply (positive shifts the distribution to higher observed z).
    renorm : {"none", "source", "target"}, optional
        - "none": no renormalization.
        - "source": scale to preserve the original integral ∫ n(z) dz over the original z grid.
        - "target": scale to unit integral over z_rebinned after shifting.
    
    Returns
    -------
    nz_shifted : np.ndarray
        The shifted n(z) evaluated on z_rebinned.
    """
    z = np.asarray(z)
    nz = np.asarray(nz)
    zr = np.asarray(z_rebinned)

    # Query positions: n(z - delta_z) evaluated at zr  -> sample original at (zr - delta_z)
    z_query = zr - delta_z

    # Linear interpolation with zero outside support
    nz_shifted = np.interp(z_query, z, nz, left=0.0, right=0.0)

    if renorm is not None and renorm != "none":
        if renorm == "source":
            area_src = np.trapz(nz, z)
            area_new = np.trapz(nz_shifted, zr)
            # match the original area (if nz was normalized, this keeps it normalized)
            if area_new > 0:
                nz_shifted *= (area_src / area_new)
        elif renorm == "target":
            area_new = np.trapz(nz_shifted, zr)
            if area_new > 0:
                nz_shifted /= area_new
        else:
            raise ValueError("renorm must be one of {'none','source','target'}")

    return nz_shifted


def F_nla(z, om0, A_ia, rho_c1, eta=0.0, z0=0.0, cosmo=None):
    z = np.asarray(z, dtype=float)
    a = 1.0 / (1.0 + z)
    if cosmo is None:
        raise ValueError("Pass a pyccl.Cosmology as `cosmo` to use CCL growth.")
    D = ccl.growth_factor(cosmo, a)          # normalized to 1 at a=1
    return -A_ia * rho_c1 * om0 * ((1+z)/(1+z0))**eta / D

def IndexToDeclRa(index, nside,nest= False):
    theta,phi=hp.pixelfunc.pix2ang(nside ,index,nest=nest)
    return -np.degrees(theta-np.pi/2.),np.degrees(phi)


def rotate_and_rebin(pix_, nside_maps, rot, delta_=0.0):
    # per-rot settings
    angle_by_rot = [0, 180, 90, 270]         # degrees
    flip_by_rot  = [False, False, True, True]

    ang = angle_by_rot[rot] + delta_
    flip = flip_by_rot[rot]

    rotu = hp.rotator.Rotator(rot=[ang, 0, 0], deg=True)

    alpha, delta = hp.pix2ang(nside_maps*2, pix_)          # original angles
    rot_alpha, rot_delta = rotu(alpha, delta)               # rotated

    if flip:
        rot_alpha = np.pi - rot_alpha                       # mirror in alpha

    pix = hp.ang2pix(nside_maps*2, rot_alpha, rot_delta)    # back to pixels

    dec__, ra__ = IndexToDeclRa(pix, nside_maps*2)
    return convert_to_pix_coord(ra__, dec__, nside=nside_maps)