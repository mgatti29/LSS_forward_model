from __future__ import annotations
import numpy as np
import glass
from typing import List, Tuple
import pandas as pd
import healpy as hp
import pyccl as ccl
import frogress
import os
from typing import Dict, Mapping, Iterable, Optional, Tuple



ParamSpec = Iterable[float]  # (low, high, scale)

def draw_baryon_params(
    specs: Mapping[str, ParamSpec],
    base: Optional[Mapping[str, float]] = None,
    *,
    overrides: Optional[Mapping[str, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    rng = np.random.default_rng() if rng is None else rng
    bpar: Dict[str, float] = dict(base or {})
    sysdraw: Dict[str, float] = {}
    for k, spec in specs.items():
        try:
            lo, hi, sc = spec
            has_range = True
        except (TypeError, ValueError):
            try:
                value, sc = spec
                has_range = False
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Spec for '{k}' must be (low, high, scale) or (value, scale); got {spec!r}"
                ) from e
        sc = str(sc).lower().strip()
        if sc not in {"lin", "log", "log10"}:
            raise ValueError(f"Unknown scale '{sc}' (use 'lin' or 'log10').")

        try:
            lo, hi = float(lo), float(hi)
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                raise ValueError(f"Invalid bounds for '{k}': {lo}..{hi}")
            u = float(rng.uniform(lo, hi))
            val = 10.0**u if sc in {"log", "log10"} else u
        except:
            val = 10.0**value if sc in {"log", "log10"} else value
        sysdraw[k] = val
        bpar[k] = val
    if overrides:
        bpar.update({k: float(v) for k, v in overrides.items()})
    return bpar, sysdraw

def load_or_draw_baryon_params(
    path_sim: str,
    specs: Mapping[str, ParamSpec],
    cache_filename: str,
    base_params_path: Optional[str] = None,
    overrides: Optional[Mapping[str, float]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    # base params
    base = {}
    if base_params_path:
        base = np.load(base_params_path, allow_pickle=True).item()


    # cache path
    cache_path = cache_filename if os.path.isabs(cache_filename) else os.path.join(path_sim, cache_filename)
    # no specs -> just return base (no cache write)

    # load cached draw if present and no overrides
    if os.path.exists(cache_path) and not overrides:
        _,sysdraw = np.load(cache_path, allow_pickle=True)
        if not isinstance(sysdraw, dict):
            raise ValueError(f"Cache must store a dict: {cache_path}")
        bpar = {**base, **{k: float(v) for k, v in sysdraw.items()}}
        return bpar, sysdraw

    # draw, save drawn-only dict
    if specs is None:
        np.save(cache_path, [base,{}])
        return dict(base), {}
    else:
        bpar, sysdraw = draw_baryon_params(specs, base)
        np.save(cache_path, [bpar,sysdraw])
        return bpar, sysdraw


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
    if np.sum(baryonified_shell)>0:
        density_baryonified = baryonified_shell / np.mean(baryonified_shell) - 1
    else:
        return baryonified_shell
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
            if np.sum(counts) == 0:
                delta.append(hp.ud_grade(counts*1.0,nside_out=1024))
            else:
                d = counts/np.mean(counts)-1
                delta.append(hp.ud_grade(d,nside_out=1024))
        except:
            pass
    delta = np.array(delta)
    np.save(path_output,delta)
    return delta

 


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
    Convert shear (γ1, γ2) to E/B-mode convergence maps on the sphere (HEALPix).

    All inputs are HEALPix maps at `nside`. Masking is applied before the spin-2 transform.

    Returns
    -------
    E_map, B_map : np.ndarray
        E- and B-mode convergence maps.
    almsE : np.ndarray
        E-mode alm coefficients.
    alms : tuple(np.ndarray)
        (T, E, B) alm triple returned by healpy.map2alm with pol=True.
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
    """
    Nonlinear linear-alignment (NLA) amplitude F(z).

    Parameters
    ----------
    z : array-like
        Redshift(s).
    om0 : float
        Present-day matter density parameter Ω_m.
    A_ia : float
        Intrinsic-alignment amplitude.
    rho_c1 : float
        Normalization constant (often 0.0134 for IA in some conventions).
    eta : float, optional
        Redshift evolution exponent (default 0.0).
    z0 : float, optional
        Pivot redshift for evolution (default 0.0).
    cosmo : pyccl.Cosmology
        CCL cosmology; used to compute the growth factor D(a).

    Returns
    -------
    ndarray
        F(z) = -A_ia * rho_c1 * Ω_m * [(1+z)/(1+z0)]^eta / D(z).
    """
    z = np.asarray(z, dtype=float)
    a = 1.0 / (1.0 + z)
    if cosmo is None:
        raise ValueError("Pass a pyccl.Cosmology as `cosmo` to use CCL growth.")
    D = ccl.growth_factor(cosmo, a)  # normalized to 1 at a=1
    return -A_ia * rho_c1 * om0 * ((1 + z) / (1 + z0))**eta / D


def IndexToDeclRa(index, nside, nest=False):
    """
    Convert HEALPix pixel index to (Dec, RA) in degrees.

    Parameters
    ----------
    index : array-like or int
        HEALPix pixel index/indices.
    nside : int
        HEALPix NSIDE.
    nest : bool, optional
        If True, assume NESTED ordering; else RING.

    Returns
    -------
    dec, ra : ndarray
        Declination and Right Ascension in degrees.
    """
    theta, phi = hp.pixelfunc.pix2ang(nside, index, nest=nest)
    return -np.degrees(theta - np.pi/2.0), np.degrees(phi)


def rotate_and_rebin(pix_, nside_maps, rot, delta_=0.0):
    """
    Apply a deterministic rotation/mirror to pixel centers and rebin to `nside_maps`.

    This mimics your map-rotation scheme:
      - rot=0: 0°
      - rot=1: 180°
      - rot=2: 90° + mirror
      - rot=3: 270° + mirror
    You can add a small extra angle `delta_` (degrees).

    Parameters
    ----------
    pix_ : array-like
        HEALPix pixel indices at NSIDE = 2 * nside_maps (RING).
    nside_maps : int
        Target NSIDE for the rebinned output.
    rot : int
        Rotation code in {0, 1, 2, 3}.
    delta_ : float, optional
        Additional rotation angle in degrees.

    Returns
    -------
    pix_rebinned : ndarray
        HEALPix pixel indices at NSIDE = nside_maps (RING) after the transform.
    """
    if rot not in (0, 1, 2, 3):
        raise ValueError("rot must be one of {0, 1, 2, 3}.")

    # per-rot settings
    angle_by_rot = [0, 180, 90, 270]   # degrees
    flip_by_rot  = [False, False, True, True]

    ang = angle_by_rot[rot] + delta_
    flip = flip_by_rot[rot]

    rotu = hp.rotator.Rotator(rot=[ang, 0, 0], deg=True)

    # original directions at NSIDE = 2*nside_maps
    alpha, delta = hp.pix2ang(nside_maps * 2, pix_)
    rot_alpha, rot_delta = rotu(alpha, delta)

    if flip:
        rot_alpha = np.pi - rot_alpha  # mirror in alpha

    # back to pixels (still at 2*nside_maps)
    pix_hi = hp.ang2pix(nside_maps * 2, rot_alpha, rot_delta)

    # convert to (Dec, RA) and then to target NSIDE pixels
    dec__, ra__ = IndexToDeclRa(pix_hi, nside_maps * 2)
    return convert_to_pix_coord(ra__, dec__, nside=nside_maps)

def unrotate_map(rotated_map, nside_maps, rot, delta_=0.0):
    """
    Invert the rotate_and_rebin operation at map level.

    Parameters
    ----------
    rotated_map : array-like, shape (hp.nside2npix(nside_maps),)
        The map AFTER your rotate_and_rebin-style transformation (defined on nside_maps).
    nside_maps : int
        Target NSIDE of the (unrotated) map you want back.
    rot : int
        Same 'rot' used in rotate_and_rebin (0..3).
    delta_ : float, optional
        Same delta_ used in rotate_and_rebin (degrees added to angle_by_rot[rot]).

    Returns
    -------
    unrotated_map : np.ndarray
        Map sampled to undo the rotation/mirroring.
    """
    # Forward settings (same as rotate_and_rebin)
    angle_by_rot = [0, 180, 90, 270]   # degrees
    flip_by_rot  = [False, False, True, True]

    ang  = angle_by_rot[rot] + delta_
    flip = flip_by_rot[rot]

    rotu = hp.rotator.Rotator(rot=[ang, 0, 0], deg=True)

    npix = hp.nside2npix(nside_maps)
    # Pixel centers of the *unrotated* target grid
    pix = np.arange(npix)
    theta, phi = hp.pix2ang(nside_maps, pix)          # (theta, phi)

    # Apply the *forward* transform to find where to sample from in the rotated map
    rot_theta, rot_phi = rotu(theta, phi)
    if flip:
        rot_theta = np.pi - rot_theta                 # same flip as forward op (its own inverse)

    # Pixels in the rotated map corresponding to those directions
    src_pix = hp.ang2pix(nside_maps, rot_theta, rot_phi)

    # Pull sampling (no holes); if you need smoothing, consider bilinear/neighbor averaging
    unrotated_map = np.asarray(rotated_map)[src_pix]
    return unrotated_map





def build_shell_windows_and_partitions(
    shells_info: dict,
    redshift: np.ndarray,
    nz: np.ndarray,
    samples_per_shell: int = 100,
) -> Tuple[List[glass.shells.RadialWindow], np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct GLASS radial windows for each simulation shell and compute
    tomographic partitions (per shell) given n(z) for multiple bins.

    Parameters
    ----------
    shells_info : dict
        Output of `recover_shell_info`. Must contain 'Step', 'z_near', 'z_far'.
    redshift : (Nz,) ndarray
        Redshift grid matching the second axis of `nz`.
    nz : (Ntomo, Nz) ndarray
        n(z) per tomographic bin sampled on `redshift`.
        It need not be normalized; `glass.shells.partition` only uses relative weights.
    samples_per_shell : int, optional
        Number of linearly spaced z-samples to use inside each [zmin, zmax] window.

    Returns
    -------
    shells : list[glass.shells.RadialWindow]
        One RadialWindow per shell (ordered low→high z to match GLASS expectations).
    steps : (Nshell,) ndarray of int
        Integer step index per shell (aligned with `shells`).
    zeff_array : (Nshell,) ndarray of float
        Effective redshift per shell (midpoint).
    ngal_glass : (Ntomo, Nshell) ndarray
        Fractional counts per tomographic bin per shell from `glass.shells.partition`.
        Each row integrates (approximately) to 1 over shells if `nz` is normalized.
    """
    # Reverse once (your inputs were high→low; GLASS prefers low→high)
    steps_rev = shells_info["Step"][::-1]
    z_near_rev = shells_info["z_near"][::-1]
    z_far_rev = shells_info["z_far"][::-1]

    shells: List[glass.shells.RadialWindow] = []
    zeff_list = []
    steps_list = []

    for step, zmin, zmax in zip(steps_rev, z_near_rev, z_far_rev):
        za = np.linspace(float(zmin), float(zmax), samples_per_shell)
        wa = np.ones_like(za)
        zeff = 0.5 * (float(zmin) + float(zmax))
        shells.append(glass.shells.RadialWindow(za, wa, zeff))
        steps_list.append(int(step))
        zeff_list.append(zeff)

    steps = np.asarray(steps_list, dtype=int)
    zeff_array = np.asarray(zeff_list, dtype=float)

    # Partition n(z) into the shell windows: shape (Ntomo, Nshell)
    ngal_glass = np.array([glass.shells.partition(redshift, nz_i, shells)
                           for nz_i in nz], dtype=float)

    return shells, steps, zeff_array, ngal_glass
    