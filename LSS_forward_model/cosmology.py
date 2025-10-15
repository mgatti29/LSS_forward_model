# cosmology.py
import glob
import numpy as np
import camb
from camb import model
import h5py as h5
import pyccl as ccl


def As_to_sigma8_CAMB(Omega_m, Omega_b, h, n_s, w0, wa, m_nu, As):
    """
    Compute sigma8 from CAMB given A_s and cosmological parameters.

    Parameters
    ----------
    Omega_m : float
        Total matter density (baryons + CDM + massive neutrinos), at z=0.
    Omega_b : float
        Baryon density at z=0.
    h : float
        H0 / (100 km s^-1 Mpc^-1).
    n_s : float
        Scalar spectral index.
    w0 : float
        Dark-energy equation-of-state parameter today.
    wa : float
        Time variation of the dark-energy EoS (w(a) = w0 + wa(1-a)).
    m_nu : float
        Total neutrino mass in eV (sum over species).
    As : float
        Primordial scalar amplitude at k=0.05 Mpc^-1 (CAMB default).

    Returns
    -------
    float
        sigma8 (RMS of matter fluctuations in 8 Mpc/h spheres).
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
    pars.NonLinear = model.NonLinear_both
    pars.WantCls = False
    pars.DoLensing = False
    pars.WantTransfer = True

    results = camb.get_results(pars)
    return float(results.get_sigma8())


def sigma8_to_As_CAMB(Omega_m, Omega_b, h, n_s, w0, wa, m_nu, sigma8_target,
                      tol=1e-4, max_iter=20, As_init=2.1e-9):
    """
    Find A_s that yields a target sigma8 in CAMB via fixed-point iteration.

    Parameters
    ----------
    Omega_m, Omega_b, h, n_s, w0, wa, m_nu : float
        Cosmological parameters (see As_to_sigma8_CAMB).
    sigma8_target : float
        Desired sigma8.
    tol : float, optional
        Relative tolerance on |sigma8 - target|/target for convergence.
    max_iter : int, optional
        Maximum iterations.
    As_init : float, optional
        Initial guess for A_s.

    Returns
    -------
    float
        A_s that achieves the target sigma8 (within tolerance).

    Raises
    ------
    RuntimeError
        If convergence is not reached within max_iter iterations.
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
        pars.NonLinear = model.NonLinear_both
        pars.WantCls = False
        pars.DoLensing = False
        pars.WantTransfer = True

        sigma8_model = float(camb.get_results(pars).get_sigma8())

        # sigma8 ∝ sqrt(As)  ⇒  As_new = As_old * (sigma8_target / sigma8_model)^2
        As *= (sigma8_target / sigma8_model)**2

        if abs(sigma8_model - sigma8_target) / sigma8_target < tol:
            return As

    raise RuntimeError(f"Failed to converge on As after {max_iter} iterations.")


def recover_shell_info(path_z_file, max_z=49.0):
    """
    Read shell boundaries from a CSV-like file and return arrays starting
    at the last shell with z_far == max_z. Also returns edges and mean z.

    File format (header + rows):
        step, z_far, z_near, delta_z, cmd_far, cmd_near, delta_cmd

    Parameters
    ----------
    path_z_file : str
        Path to the CSV-like file with 7 columns as above.
    max_z : float, optional
        Start from the last occurrence where z_far == max_z.

    Returns
    -------
    dict
        {
          'Step', 'z_far', 'z_near', 'delta_z',
          'cmd_far', 'cmd_near', 'delta_cmd',
          'z_edges', 'mean_z'
        }
        All values are numpy arrays; 'Step' is renormalized to start at 0.
    """
    res = {k: [] for k in
           ['Step', 'z_far', 'z_near', 'delta_z', 'cmd_far', 'cmd_near', 'delta_cmd']}

    with open(path_z_file) as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue
            vals = np.array(line.split(','), dtype=float)
            res['Step'].append(vals[0])
            res['z_far'].append(vals[1])
            res['z_near'].append(vals[2])
            res['delta_z'].append(vals[3])
            res['cmd_far'].append(vals[4])
            res['cmd_near'].append(vals[5])
            res['delta_cmd'].append(vals[6])

    # convert to arrays
    for k in res:
        res[k] = np.asarray(res[k])

    # find last index with z_far == max_z
    matches = np.where(res['z_far'] == max_z)[0]
    if len(matches) == 0:
        raise ValueError(f"No shell with z_far == {max_z} found in {path_z_file}.")
    init = matches[-1]

    for k in ['Step', 'z_far', 'z_near', 'delta_z', 'cmd_far', 'cmd_near', 'delta_cmd']:
        res[k] = res[k][init:]
    res['Step'] = res['Step'] - res['Step'][0]

    # edges and means
    res['z_edges'] = np.hstack([res['z_far'][0], res['z_near']])
    res['mean_z'] = 0.5 * (res['z_edges'][1:] + res['z_edges'][:-1])

    return res


def read_sims_params(path):
    """
    Extract cosmological parameters from a CLASS run directory and
    build convenience objects for CCL and CAMB.

    Expects:
      - One 'class_processed*' HDF5 in `path` with a 'background' group.
      - A 'control.par' text file with keys:
          dNormalization (=A_s), dSpectral (=n_s), dBoxSize (=Mpc/h).

    Parameters
    ----------
    path : str
        Directory containing CLASS outputs and 'control.par'.

    Returns
    -------
    sims_parameters : dict
        {
          'Omega_b','Omega_nu','Omega_r','Omega_m','Omega_cdm',
          'h','w0','wa','As','n_s','sigma_8','m_nu','dBoxSize Mpc/h'
        }
        Note: 'h' is dimensionless; 'dBoxSize Mpc/h' is in comoving Mpc/h.
    cosmo_pyccl : pyccl.Cosmology
        CCL cosmology (linear P(k), equal mass-split for 3 ν species).
    pars_camb : camb.CAMBparams
        CAMB parameter object matching the above cosmology.
    colossus_params : dict
        Minimal params dict handy for Colossus (if you use it elsewhere).
    """
    # ---- read CLASS background
    h5_path = glob.glob(f"{path}/class_processed*")[0]
    with h5.File(h5_path, "r") as f:
        bg = f['background']
        Omega_b   = (bg['rho_b'][()]   / bg['rho_crit'][()])[-1]
        Omega_cdm = (bg['rho_cdm'][()] / bg['rho_crit'][()])[-1]
        try:
            Omega_nu = ((bg['rho_ncdm[0]'][()] + bg['rho_ncdm[1]'][()] + bg['rho_ncdm[2]'][()]) /
                        bg['rho_crit'][()])[-1]
        except KeyError:
            Omega_nu = (bg['rho_ncdm[0]'][()] / bg['rho_crit'][()])[-1]
        Omega_r   = (bg['rho_g'][()]   / bg['rho_crit'][()])[-1]
        Omega_m   = Omega_b + Omega_cdm + Omega_nu

        # CLASS stores H(a) in 1/Mpc; convert to h = H0/(100 km/s/Mpc)
        # 1/Mpc → km/s/Mpc via c = 299792.458 km/s; factor 977.792 is commonly used.
        h_dimless = float(bg['H'][-1] * 977.792 / 100.0)

        if 'w_fld' in bg:
            w0 = float(bg['w_fld'][-1])
            w  = float(bg['w_fld'][-100])
            z  = float(np.array(bg['z'])[-100])
            fz = z / (1.0 + z) if z > 0 else 1.0
            wa = (w - w0) / fz if fz != 0 else 0.0
        else:
            w0, wa = -1.0, 0.0

    m_nu = float(Omega_nu * (93.14 * h_dimless**2))

    # ---- read control.par (A_s, n_s, box)
    values = {}
    with open(f"{path}/control.par", "r") as f:
        for line in f:
            if any(k in line for k in ('dNormalization', 'dSpectral', 'dBoxSize')):
                key = line.split('=')[0].strip()
                val = line.split('=')[1].split('#')[0].strip()
                values[key] = float(val)

    As       = float(values['dNormalization'])
    n_s      = float(values['dSpectral'])
    dBoxSize = float(values['dBoxSize'])

    # ---- compute sigma8 from CAMB
    sigma_8 = As_to_sigma8_CAMB(Omega_m, Omega_b, h_dimless, n_s, w0, wa, m_nu, As)

    sims_parameters = {
        'Omega_b': Omega_b,
        'Omega_nu': Omega_nu,
        'Omega_r': Omega_r,
        'Omega_m': Omega_m,
        'Omega_cdm': Omega_cdm,
        'h': h_dimless,
        'w0': w0,
        'wa': wa,
        'As': As,
        'n_s': n_s,
        'sigma_8': sigma_8,                 # scalar
        'm_nu': m_nu,                       # eV
        'dBoxSize Mpc/h': dBoxSize,         # comoving
    }

    # ---- CCL cosmology
    cosmo_pyccl = ccl.Cosmology(
        Omega_c=sims_parameters['Omega_cdm'],
        Omega_b=sims_parameters['Omega_b'],
        h=sims_parameters['h'],
        sigma8=sims_parameters['sigma_8'],
        n_s=sims_parameters['n_s'],
        m_nu=[sims_parameters['m_nu']/3.0]*3,  # equal split
        mass_split='equal',
        matter_power_spectrum='linear',
    )

    # ---- CAMB params (matching above)
    pars_camb = camb.CAMBparams()
    pars_camb.set_cosmology(H0=sims_parameters['h']*100,
                            ombh2=sims_parameters['Omega_b']*sims_parameters['h']**2,
                            omch2=sims_parameters['Omega_cdm']*sims_parameters['h']**2,
                            mnu=sims_parameters['m_nu'],
                            tau=0.06, num_massive_neutrinos=3)
    pars_camb.InitPower.set_params(As=sims_parameters['As'], ns=sims_parameters['n_s'])
    pars_camb.set_dark_energy(w=sims_parameters['w0'], wa=sims_parameters['wa'], dark_energy_model='ppf')
    pars_camb.NonLinear = model.NonLinear_both
    pars_camb.WantCls = False
    pars_camb.DoLensing = False
    pars_camb.WantTransfer = True

    # ---- Colossus-friendly dict (no Colossus import here)
    colossus_params = {
        'flat': True,
        'H0': sims_parameters['h']*100.0,
        'Om0': sims_parameters['Omega_m'],
        'Ob0': sims_parameters['Omega_b'],
        'sigma8': sims_parameters['sigma_8'],
        'ns': sims_parameters['n_s'],
        'w0': sims_parameters['w0'],
        'wa': sims_parameters['wa'],
    }

    return sims_parameters, cosmo_pyccl, pars_camb, colossus_params