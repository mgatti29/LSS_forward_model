import numpy as np
from scipy.optimize import root_scalar
from scipy.interpolate import RegularGridInterpolator
import BaryonForge as bfn
import os
import healpy as hp
import frogress
import pyccl as ccl
import pandas as pd
# `colossus` is imported inside infer_overdensity_from_fof to keep it optional


def mu(c):
    """
    Helper for NFW profiles: μ(c) = ln(1+c) − c/(1+c).

    Parameters
    ----------
    c : float or array-like
        Concentration parameter (>0).

    Returns
    -------
    float or ndarray
        μ(c).
    """
    return np.log(1 + c) - c / (1 + c)


def enclosed_overdensity(c, b=0.2, nc=0.652960):
    """
    Effective enclosed overdensity matched to a FoF linking length.

    Implements Eq. (10) of More et al. (2011), giving the mean overdensity
    inside the FoF boundary for an NFW halo of concentration `c`.

    Parameters
    ----------
    c : float or array-like
        NFW concentration.
    b : float, optional
        FoF linking length (in units of mean inter-particle separation). Default 0.2.
    nc : float, optional
        Numerical factor from More+2011 (≈ 0.652960).

    Returns
    -------
    float or ndarray
        Overdensity Δ_FoF/ρ̄ − 1 (i.e., 〈ρ〉/ρ̄ − 1).
    """
    return 3 * nc * b**-3 * mu(c) * (1 + c)**2 / c**2 - 1


def c_zhao2009(M, z):
    """
    Zhao et al. (2009) concentration–mass relation for 200c halos.

    Parameters
    ----------
    M : float or array-like
        Halo mass (Msun/h) defined at 200c.
    z : float or array-like
        Redshift.

    Returns
    -------
    float or ndarray
        Concentration c_200c.
    """
    M_pivot = 1e14  # Msun/h
    A, B, C = 4.67, -0.11, -1.0
    return A * (M / M_pivot)**B * (1 + z)**C


def m_nfw(r, rs, rho_s):
    """
    Enclosed mass for an NFW profile at radius r.

    Parameters
    ----------
    r : float or array-like
        Radius (same distance units as rs).
    rs : float
        NFW scale radius.
    rho_s : float
        NFW scale density.

    Returns
    -------
    float or ndarray
        M(<r) with units of rho_s * rs^3.
    """
    x = r / rs
    return 4 * np.pi * rho_s * rs**3 * (np.log(1 + x) - x / (1 + x))


def c_of_delta(delta, M_fof, z=0):
    """
    Placeholder c–M relation (Duffy et al. 2008-ish) for a given overdensity.

    Parameters
    ----------
    delta : float
        Target overdensity label (unused in this simplified stub).
    M_fof : float
        FoF mass (Msun/h).
    z : float, optional
        Redshift.

    Returns
    -------
    float
        Approximate concentration.

    Notes
    -----
    This is a rough stub (tuned to 200c); for accurate work, use a proper
    concentration model and/or conversion routine.
    """
    A, B, C = 5.71, -0.084, -0.47  # typical for 200c
    return A * (M_fof / 2e12)**B * (1 + z)**C


def infer_overdensity_from_fof(M_fof, M_fof_corr, b=0.2, z=0, tol=1e-3,
                               conc_corr_factor=True, cosmo='planck15'):
    """
    Infer the effective enclosed overdensity for a FoF halo (More+2011 approach).

    Steps:
      1) Get c_200c(M, z) from Colossus (Ishiyama+21 model), optionally scaled.
      2) Convert FoF linking length b to an effective FoF overdensity using
         the NFW mapping (More+2011 Eq. 10).

    Parameters
    ----------
    M_fof : float
        Original FoF mass (Msun/h). (Not used directly; kept for API symmetry.)
    M_fof_corr : float
        Corrected FoF mass used to evaluate the concentration (Msun/h).
    b : float, optional
        FoF linking length (default 0.2).
    z : float, optional
        Redshift.
    tol : float, optional
        Convergence tolerance for the fixed-point iteration on Δ (rarely needs many iters).
    conc_corr_factor : bool, optional
        If True, multiply Colossus c(M) by 0.6*z + 0.8 (empirical tweak).
    cosmo : str or dict, optional
        Colossus cosmology name or definition (default 'planck15').

    Returns
    -------
    Delta_eff : float
        Effective enclosed overdensity (w.r.t. mean density), i.e., 〈ρ〉/ρ̄ − 1.
    c_200c : float
        Concentration used (at 200c).

    Raises
    ------
    RuntimeError
        If the iteration fails to converge.
    """
    # get concentration from Colossus

    from colossus.halo import concentration

    corr = (0.6 * z + 0.8) if conc_corr_factor else 1.0
    c = corr * concentration.concentration(M_fof_corr, '200c', z, model='ishiyama21')

    # fixed-point iteration (usually converges in ≤ a few steps)
    Delta_guess = 200.0
    for _ in range(10):
        Delta_new = enclosed_overdensity(c, b=b)
        if abs(Delta_new - Delta_guess) < tol:
            return Delta_new, c
        Delta_guess = Delta_new

    raise RuntimeError("Overdensity iteration did not converge.")


def fof_to_mdelta(M_fof, Delta_fof, c_fof, Delta_target=200, ref_density='mean',
                  Omega_m=0.3, H0=70.0):
    """
    Convert FoF mass M_FoF to a spherical-overdensity mass M_Δ (NFW assumption).

    Treat the FoF halo as an NFW with concentration c_fof and mean enclosed density
    Δ_fof * ρ_ref_FoF (here taken as Δ_fof * ρ_m). Solve for R_Δ such that
    〈ρ(<R_Δ)〉 = Δ_target * ρ_ref, then return M_Δ = M(<R_Δ).

    Parameters
    ----------
    M_fof : float
        FoF mass (Msun/h).
    Delta_fof : float
        Enclosed overdensity of the FoF halo (w.r.t. mean matter density, ρ_m).
    c_fof : float
        Concentration of the FoF halo (R_FoF / r_s).
    Delta_target : float, optional
        Desired spherical overdensity (e.g., 200).
    ref_density : {"mean","crit"}, optional
        Reference density for the target mass definition (ρ_m or ρ_c).
    Omega_m : float, optional
        Matter density parameter at z relevant for the conversion (default 0.3).
    H0 : float, optional
        Hubble constant [km/s/Mpc] for ρ_c calculation (default 70).

    Returns
    -------
    M_delta : float
        Spherical-overdensity mass M_Δ_target (Msun/h).

    Raises
    ------
    RuntimeError
        If the root finding for R_Δ fails.

    Notes
    -----
    Units: G is taken as 4.30091e-9 Mpc⋅Msun⁻¹⋅(km/s)² so that
           ρ_c = 3 H0² / (8πG) has units Msun/Mpc³ when H0 is in km/s/Mpc.
    """
    # critical and reference densities
    G = 4.30091e-9  # Mpc * Msun^-1 * (km/s)^2
    rho_crit = 3.0 * (H0**2) / (8.0 * np.pi * G)  # Msun / Mpc^3
    rho_m = Omega_m * rho_crit
    rho_ref = rho_m if ref_density == 'mean' else rho_crit

    # FoF radius from mean-density definition (Δ_fof w.r.t. ρ_m)
    R_fof = (3.0 * M_fof / (4.0 * np.pi * Delta_fof * rho_m))**(1.0 / 3.0)

    # NFW params
    rs = R_fof / c_fof
    rho_s = M_fof / (4.0 * np.pi * rs**3 * mu(c_fof))

    # Solve for R_Δ: 〈ρ(<R)〉 = Δ_target * ρ_ref
    def mean_density_minus_target(r):
        return m_nfw(r, rs, rho_s) / ((4.0 / 3.0) * np.pi * r**3) - Delta_target * rho_ref

    # robust bracket (r must be > 0 and < few x R_fof)
    a, b = 1e-6 * rs, 10.0 * R_fof
    sol = root_scalar(mean_density_minus_target, bracket=[a, b], method='brentq')
    if not sol.converged:
        raise RuntimeError("Root finding for R_delta failed.")

    R_delta = sol.root
    M_delta = m_nfw(R_delta, rs, rho_s)
    return M_delta


def build_fof_to_m200c_interpolator(
    sims_parameters: dict,
    cosmo_colossus,
    logM_grid=(12.0, 15.5, 200),
    z_grid=(0.0, 1.5, 200),
):
    """
    Build or load a RegularGridInterpolator mapping (log10 M_fof, z) -> M_200c.
    Use `cache_path` (.npz) to save/reuse the expensive grid.
    """
    from colossus.halo import concentration  # local import to keep it optional

    lmin, lmax, nL = logM_grid
    zmin, zmax, nZ = z_grid
    lg = np.linspace(lmin, lmax, nL)
    zg = np.linspace(zmin, zmax, nZ)


    logM_mesh, z_mesh = np.meshgrid(lg, zg, indexing="ij")
    M_mesh = 10.0**logM_mesh
    M200c_grid = np.empty_like(M_mesh)

    # NOTE: this nested loop is heavy; consider parallelizing or precomputing offline.
    for i in range(logM_mesh.shape[0]):
        for j in range(logM_mesh.shape[1]):
            M_ = M_mesh[i, j]
            z_ = z_mesh[i, j]
            Delta_fof, c_fof = infer_overdensity_from_fof(  # your function from halos.py
                M_, M_, z=z_, conc_corr_factor=True, cosmo=cosmo_colossus
            )
            M200c_grid[i, j] = fof_to_mdelta(               # your function from halos.py
                M_,
                Delta_fof,
                c_fof,
                Delta_target=200,
                ref_density="crit",
                Omega_m=sims_parameters["Omega_m"],
                H0=sims_parameters["h"] * 100.0,
            )

    return RegularGridInterpolator((lg, zg), M200c_grid, bounds_error=False, fill_value=None)


def make_tsz_and_baryonified_density(
    path_simulation: str,
    sims_parameters: dict,
    cosmo_pyccl,
    halos: dict,
    bpar: dict,
    nside_maps: int,
    shells_info: dict,
    dens_path: str,
    tsz_path: str,
    do_tSZ: bool,
    nside_baryonification: int = 1024,
    min_mass: float = 13,                      # Msun/h threshold after FoF->SO
    njobs: int = 16,
):
    """
    Build (if missing) tSZ map and baryonified density shells, saving them to disk.
    Returns the density array for downstream lensing.

    Outputs
    -------
    tsz file:  path_simulation + f"/tsz_{nside_maps}.npy"
    dens file: path_simulation + f"/density_b_{nside_maps}_{noise_rel}.npy"
    """
  
    # ---------- tSZ ----------
    if do_tSZ:
        if not os.path.exists(tsz_path):
            print ('creatng a tSZ map --')
            mask = (halos['M'] > 10**min_mass)
            cdict = {
                "Omega_m": sims_parameters["Omega_m"],
                "sigma8": sims_parameters["sigma_8"],
                "h": sims_parameters["h"],
                "n_s": sims_parameters["n_s"],
                "w0": sims_parameters["w0"],
                "Omega_b": sims_parameters["Omega_b"],
            }
    
            halos_ = bfn.utils.HaloLightConeCatalog(
                halos["ra"][mask], halos["dec"][mask], halos['M'][mask], halos["z"][mask], cosmo=cdict
            )
    
            Gas = bfn.Profiles.Gas(**bpar)
            DMB = bfn.Profiles.DarkMatterBaryon(**bpar, twohalo=0 * bfn.Profiles.TwoHalo(**bpar))
            PRS = bfn.Profiles.Pressure(gas=Gas, darkmatterbaryon=DMB)
            PRS = PRS * (1 - bfn.Profiles.Thermodynamic.NonThermalFrac(**bpar))
            PRS = bfn.Profiles.ThermalSZ(PRS)
            PRS = bfn.Profiles.misc.ComovingToPhysical(PRS, factor=-3)
            Pix = bfn.utils.HealPixel(NSIDE=nside_maps)
            PRS = bfn.utils.ConvolvedProfile(PRS, Pix)
            PRS = bfn.utils.TabulatedProfile(PRS, cosmo_pyccl)
    
            zmin, zmax = float(halos["z"].min()), float(halos["z"].max())
            PRS.setup_interpolator(
                z_min=zmin, z_max=zmax, N_samples_z=10, z_linear_sampling=True,
                R_min=1e-4, R_max=300, N_samples_R=2000, verbose=True
            )
    
            shell = bfn.utils.LightconeShell(np.zeros(hp.nside2npix(nside_maps)), cosmo=cdict)
            runner = bfn.Runners.PaintProfilesShell(halos_, shell, epsilon_max=bpar["epsilon_max"], model=PRS, verbose=True)
            painted_shell = bfn.utils.SplitJoinParallel(runner, njobs=njobs).process()
            np.save(tsz_path, painted_shell)

            
            
    # ---------- Baryonified density shells ----------
    if not os.path.exists(dens_path):
        print ('baryonifying shells --')
        density = []
        steps = shells_info["Step"][::-1]
        z_near = shells_info["z_near"][::-1]
        z_far = shells_info["z_far"][::-1]

        for i in frogress.bar(range(len(steps))):
            try:
                step = steps[i]
                zmin = float(z_near[i]) + (1e-6 if i == 0 else 0.0)
                zmax = float(z_far[i])
    
                # shell thickness for projection cutoff
                chi = ccl.comoving_radial_distance
                shell_thickness = chi(cosmo_pyccl, 1.0 / (1.0 + zmax)) - chi(cosmo_pyccl, 1.0 / (1.0 + zmin))
                bpar["proj_cutoff"] = float(shell_thickness / 2.0)
    
                DMO = bfn.Profiles.DarkMatterOnly(**bpar)
                DMB = bfn.Profiles.DarkMatterBaryon(**bpar)
                Displacement = bfn.Profiles.Baryonification2D(DMO, DMB, cosmo=cosmo_pyccl, epsilon_max=bpar["epsilon_max"])
    
                try:
                    Displacement.setup_interpolator(
                        z_min=zmin, z_max=zmax, N_samples_z=2, z_linear_sampling=True,
                        R_min=1e-4, R_max=300, N_samples_R=2000, verbose=True
                    )
                except Exception:
                    Displacement.setup_interpolator(
                        z_min=zmin, z_max=zmax, N_samples_z=2, z_linear_sampling=True,
                        R_min=1e-9, R_max=2000, N_samples_R=4000, verbose=True
                    )
    
                part_path = os.path.join(path_simulation, f"particles_{int(step)}_4096.parquet")
                counts = np.array(pd.read_parquet(part_path)).astype(np.float32).ravel()
    
                # counts are scalar fields; for resolution change use power=0
                counts = hp.ud_grade(counts, nside_out=nside_baryonification, power=0)
    
                mask_z = (halos["z"] > zmin) & (halos["z"] < zmax)
    
                cdict = {
                    "Omega_m": sims_parameters["Omega_m"],
                    "sigma8": sims_parameters["sigma_8"],
                    "h": sims_parameters["h"],
                    "n_s": sims_parameters["n_s"],
                    "w0": sims_parameters["w0"],
                    "Omega_b": sims_parameters["Omega_b"],
                }
                halos_ = bfn.utils.HaloLightConeCatalog(
                    halos["ra"][mask_z], halos["dec"][mask_z], halos["M"][mask_z], halos["z"][mask_z], cosmo=cdict
                )
    
                shell = bfn.utils.LightconeShell(map=counts, cosmo=cdict)
                runner = bfn.Runners.BaryonifyShell(halos_, shell, epsilon_max=bpar["epsilon_max"], model=Displacement, verbose=True)
                baryonified_shell = runner.process()
    
                if np.mean(baryonified_shell) != 0:
                    density_b = (baryonified_shell / np.mean(baryonified_shell)) - 1.0
                else:
                    density_b = 0.0 * baryonified_shell
    
                density.append(hp.ud_grade(density_b, nside_out=nside_maps, power=0))
            except:
                pass
        density = np.asarray(density, dtype=np.float32)
        np.save(dens_path, density)