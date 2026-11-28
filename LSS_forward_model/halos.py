import os
import gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import healpy as hp
import pyccl as ccl
import copy
from scipy.interpolate import CubicSpline
from typing import Optional, Dict
import frogress
from colossus.cosmology import cosmology as colossus_cosmology
from scipy.optimize import root_scalar
from scipy.interpolate import RegularGridInterpolator
import BaryonForge as bfn
import astropy.io.fits as fits
from astropy.cosmology import wCDM, Flatw0waCDM
import astropy.units as u
from .halo_inputs import load_halo_snapshot, shell_radial_edges_Mpc
from .halo_ia import (
    HALO_ELLIPTICITY_SCALE,
    build_IA_halo_sources,
    build_IA_halo_sources_from_catalog,
    concentration_200c,
    draw_sources,
    encode_halo_ellipticity,
    expected_counts_in_redshift_bins,
    inverse_profile,
    iter_IA_halo_sources,
    occupation,
    populate_IA_halo_shell,
    project_inertia_tensor,
    projected_nfw_surface_density,
    radius_200c,
    reconstruct_inertia_tensor_halo_snapshot,
    recover_halo_ellipticity,
    sample_radius,
    sky_basis,
    transport_phase,
    unpack_inertia_tensor,
)

def save_halocatalog(shells_info, sims_parameters, max_redshift=1.5, halo_snapshots_path='',
                     catalog_path='halo_catalog.parquet', log10_mass_limit=None,
                     include_ia_shapes=True, distance_convention='auto'):

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

    include_ia_shapes : bool, optional
        Project and store compact halo ellipticities. Disable this for a
        baryonification-only catalogue.

    distance_convention : {'auto', 'file_Mpc', 'from_redshift'}, optional
        How each snapshot's radial edges in comoving Mpc are obtained (see
        ``halo_inputs.shell_radial_edges_Mpc``). ``'auto'`` (default)
        identifies the ``cmd_*`` columns from their values. Package
        ``z_values.txt`` files (χ(z) in Mpc for this cosmology) are used as
        written, so NERSC catalogues are unchanged. Gower St files
        (h·χ_{w=-1}(z) in Mpc/h) get edges χ(z_near), χ(z_far) computed from
        the redshift columns with this run's cosmology. Anything else raises.

    Supported inputs
    ----------------
    Halo snapshots are located per step by ``halo_inputs.find_halo_snapshot``:
    ``run.SSSSS.fofstats.parquet`` (NERSC preprocessing) or the native
    PKDGRAV binary ``fof/run.SSSSS.fofstats.0`` / ``run.SSSSS.fofstats.0``
    (with or without SO fields). Both are normalised to halo positions in
    comoving Mpc in [0, L), with identical encoded masses, radii and shapes.
    SO columns are written only when the snapshots carry SO measurements.

    Notes
    -----
    - Halos are replicated in a cubic grid until they cover each snapshot's spherical shell [χ(z_near), χ(z_far)).
    - Box replication ensures volume completeness for the shell.
    - Uses `pyccl` to compute comoving distances for redshift-distance interpolation.
    - Outputs a FITS file containing positions, redshifts, masses, and optional angular coordinates.

    Output
    ------
    A parquet table saved at `catalog_path`, containing:
        - pix_16384_ring: HEALPix pixel index
        - log_M: log10 halo mass in physical Msun, scaled by 1000
        - R: encoded native halo radius
        - redshift: Scaled by 10000
        - ra_deg, dec_deg: exact replicated lightcone direction
        - shell_index: low-to-high-redshift index aligned with ``build_shells``
        - simulation_step: native PKDGRAV snapshot step
        - host_e1_x10000, host_e2_x10000 and shape_valid when requested
    """  


    halo_snapshots_path = os.path.join(os.fspath(halo_snapshots_path), '')

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
        This function assumes the box is axis-aligned and cubic and computes
        its exact minimum and maximum distance from the observer.
        """
        # Exact radial bounds of this axis-aligned replicated box.  The old
        # circumscribed-sphere test admitted boxes that could not intersect a
        # thin shell, forcing unnecessary full-snapshot scans.
        lower = np.array([x_i, y_i, z_i], dtype=float) * Lbox_Mpc
        upper = lower + Lbox_Mpc
        closest = np.where(
            (lower <= 0.0) & (upper >= 0.0),
            0.0,
            np.minimum(np.abs(lower), np.abs(upper)),
        )
        furthest = np.maximum(np.abs(lower), np.abs(upper))
        min_distance = np.linalg.norm(closest)
        max_distance = np.linalg.norm(furthest)

        return (min_distance <= d_max and max_distance >= d_min),min_distance,max_distance


    

    cosmo = ccl.Cosmology(Omega_c = sims_parameters['Omega_cdm'], Omega_b = sims_parameters['Omega_b'], 
                          h =  sims_parameters['h'], sigma8 = sims_parameters['sigma_8'], 
                          n_s = sims_parameters['n_s'], w0 = sims_parameters['w0'], wa = sims_parameters['wa'],
                          m_nu = [sims_parameters['m_nu']/3,sims_parameters['m_nu']/3,sims_parameters['m_nu']/3],mass_split='equal',
                          matter_power_spectrum='linear',extra_parameters={"camb": {"dark_energy_model": "ppf","halofit_version": "mead2020",}})




    z_hr = np.linspace(0, 10, 5001)
    d_hr = ccl.comoving_radial_distance(cosmo, 1./(1+z_hr))

    interpolated_distance_to_redshift = CubicSpline(d_hr, z_hr)
    maximum_distance = float(
        ccl.comoving_radial_distance(cosmo, 1.0 / (1.0 + max_redshift))
    )
    shell_indices = np.flatnonzero(
        (np.asarray(shells_info['z_near']) < max_redshift)
        & (np.asarray(shells_info['z_far']) > 0.0)
    )
    if not len(shell_indices):
        raise ValueError("No simulation shells intersect the requested redshift range")

    Lbox_Mpc = sims_parameters['dBoxSize Mpc/h']/ sims_parameters['h']

    # Snapshot radial edges in comoving Mpc for this cosmology, whatever the
    # distance convention of the shell table (shells_info is not modified).
    edge_near, edge_far, edge_info = shell_radial_edges_Mpc(
        shells_info, sims_parameters, convention=distance_convention)
    print(f"save_halocatalog: shell distance columns identified as "
          f"{edge_info.get('dialect', 'not checked')}; radial edges from {edge_info['convention']}")
    provenance = {
        'distance_convention': edge_info['convention'],
        'distance_columns_dialect': str(edge_info.get('dialect', 'not checked')),
        'max_redshift': repr(float(max_redshift)),
        'log10_mass_limit': repr(log10_mass_limit),
        'units': 'positions and radial edges in comoving Mpc; redshift = z(chi) for the run cosmology',
    }
    snapshot_formats = set()
    so_columns = None

    partial_path = f"{os.fspath(catalog_path)}.partial"
    if os.path.exists(partial_path):
        raise FileExistsError(
            f"Incomplete output already exists: {partial_path}. "
            "Inspect or remove it before restarting."
        )

    writer = None
    total_rows = 0
    try:
        # Preserve the historical row ordering: shell, replica, source row.
        for i in frogress.bar(shell_indices[::-1]):
            d_min = max(0.0, float(edge_near[i]))
            d_max = min(
                float(edge_far[i]),
                maximum_distance,
            )
            if d_max <= d_min:
                continue
            step = shells_info['Step'][i]

            output_ = load_halo_snapshot(
                halo_snapshots_path,
                step,
                sims_parameters,
                log10_mass_limit=log10_mass_limit,
                include_ia_shapes=include_ia_shapes,
            )
            snapshot_formats.add(output_['format'])
            # One schema for the whole lightcone: SO columns only if the
            # snapshots carry SO data (decided by the first snapshot with halos).
            if so_columns is None and len(output_['M']):
                so_columns = output_['has_so']
            if so_columns is False and output_['has_so']:
                raise ValueError(
                    f"Step {step} carries SO masses but earlier snapshots do not "
                    f"({output_['path']})")
            shell_index = len(shells_info['Step']) - 1 - i
            replicas_max = int(np.ceil(d_max / Lbox_Mpc))

            for x_i in range(-replicas_max, replicas_max):
                for y_i in range(-replicas_max, replicas_max):
                    for z_i in range(-replicas_max, replicas_max):
                        may_intersect, _, _ = may_intersect_sphere(
                            x_i, y_i, z_i, Lbox_Mpc, d_min, d_max
                        )
                        if not may_intersect:
                            continue

                        new_x = output_['x'] + x_i * Lbox_Mpc
                        new_y = output_['y'] + y_i * Lbox_Mpc
                        new_z = output_['z'] + z_i * Lbox_Mpc
                        radius = np.sqrt(new_x**2 + new_y**2 + new_z**2)
                        mask = (radius >= d_min) & (radius < d_max)
                        if not np.any(mask):
                            continue

                        selected_x = new_x[mask]
                        selected_y = new_y[mask]
                        selected_z = new_z[mask]
                        selected_radius = radius[mask]
                        ra = np.degrees(
                            np.arctan2(selected_y, selected_x)
                        ) % 360.0
                        sin_dec = np.divide(
                            selected_z,
                            selected_radius,
                            out=np.zeros_like(selected_z),
                            where=selected_radius > 0,
                        )
                        dec = np.degrees(
                            np.arcsin(np.clip(sin_dec, -1.0, 1.0))
                        )
                        columns = {
                            'pix_16384_ring': hp.vec2pix(
                                16384,
                                selected_x,
                                selected_y,
                                selected_z,
                                nest=False,
                            ).astype(np.uint32),
                            'log_M': np.asarray(output_['M'][mask], dtype=np.uint16),
                            'R': np.asarray(output_['rhalf'][mask], dtype=np.uint16),
                            'redshift': np.asarray(
                                interpolated_distance_to_redshift(selected_radius)
                                * 10000.0,
                                dtype=np.uint16,
                            ),
                            'ra_deg': ra.astype(np.float32),
                            'dec_deg': dec.astype(np.float32),
                            'shell_index': np.full(len(ra), shell_index, dtype=np.uint16),
                            'simulation_step': np.full(len(ra), step, dtype=np.int32),
                            'catalogue_index': np.arange(
                                total_rows, total_rows + len(ra), dtype=np.uint64),
                        }
                        if include_ia_shapes:
                            ellipticity, shape_valid = project_inertia_tensor(
                                ra, dec, output_['inertia_tensor'][mask])
                            encoded_e1, encoded_e2 = encode_halo_ellipticity(
                                ellipticity, shape_valid)
                            columns.update({
                                'host_e1_x10000': encoded_e1,
                                'host_e2_x10000': encoded_e2,
                                'shape_valid': shape_valid.astype(bool),
                            })
                        if so_columns:
                            columns['log_M200c_SO'] = np.asarray(
                                output_['M200c'][mask], dtype=np.uint16
                            ) if 'M200c' in output_ else np.zeros(len(ra), np.uint16)
                            columns['R200c_S0'] = np.asarray(
                                output_['r200c'][mask], dtype=np.uint16
                            ) if 'r200c' in output_ else np.zeros(len(ra), np.uint16)

                        table = pa.table(columns)
                        if writer is None:
                            writer = pq.ParquetWriter(
                                partial_path,
                                table.schema,
                                compression='snappy',
                            )
                        elif table.schema != writer.schema:
                            raise ValueError(
                                "Halo snapshot schemas changed while assembling "
                                "the lightcone"
                            )
                        writer.write_table(table, row_group_size=1_000_000)
                        total_rows += len(table)

            del output_
            gc.collect()
    finally:
        if writer is not None:
            if hasattr(writer, 'add_key_value_metadata'):
                provenance['snapshot_formats'] = ','.join(sorted(snapshot_formats))
                writer.add_key_value_metadata(
                    {f'lss_forward_model.halo_lightcone.{key}': value
                     for key, value in provenance.items()})
            writer.close()

    if total_rows == 0:
        raise ValueError("No halos intersected the requested lightcone")
    if pq.ParquetFile(partial_path).metadata.num_rows != total_rows:
        raise RuntimeError("Incomplete streamed halo catalogue")

    os.replace(partial_path, catalog_path)
    print(f"Done assembling {total_rows} halo occurrences")



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


def recover_halo_mass(df, preferred_mass_col=None):
      if preferred_mass_col is not None and preferred_mass_col in df.columns:
          mass_col = preferred_mass_col
      elif "log10M" in df.columns:
          mass_col = "log10M"
      elif "log_M" in df.columns:
          mass_col = "log_M"
      elif "log_M200c_SO" in df.columns:
          mass_col = "log_M200c_SO"
      elif "log10M200c" in df.columns:
          mass_col = "log10M200c"
      else:
          raise KeyError("No mass column found")

      vals = df[mass_col].to_numpy()

      # New catalogs are already log10(M) * 1000. Do not +4 them.
      if mass_col in ("log_M", "log10M", "log_M200c_SO", "log10M200c"):
          mass = 10.0 ** (vals / 1000.0)
          mass[vals <= 0] = np.nan
          return mass

      good = vals > 0
      if np.any(good) and np.nanmin(vals[good]) / 1000 < 10:
          return 10.0 ** (vals / 1000.0 + 4.0)
      return 10.0 ** (vals / 1000.0)


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
    if "ra_deg" in df.columns and "dec_deg" in df.columns:
        return df["ra_deg"].to_numpy(), df["dec_deg"].to_numpy()
    theta, phi = hp.pix2ang(16384, df['pix_16384_ring'].to_numpy(), nest=nest)
    dec = np.degrees(0.5 * np.pi - theta)
    ra = np.degrees(phi)
    return ra, dec


def load_halo_catalog(
    path_halo_cat: str,
    colossus_pars: [Dict[str, float]],
    sims_parameters: Dict[str, float],
    halo_catalog_log10mass_cut: float,
    no_calib: True,
    shell_index=None,
    verbose=True,
):
    """
    Load a halo catalog, convert FoF masses to M200c (via Colossus-based calibrator),
    apply a log10(M) cut, and return a compact dict.

    Parameters
    ----------
    path_halo_cat : str
        path to the halo catalog
    sims_parameters : dict
        Cosmology dict with at least Omega_m, Omega_b, h, sigma_8, n_s.
    halo_catalog_log10mass_cut : float
        Keep halos with log10(M/Msun h^-1) > cut after FoF->SO conversion.
    colossus_pars : dict, optional
        Parameters to register/select a Colossus cosmology (e.g., {"Om0":..., "Ob0":..., "H0":..., "sigma8":..., "ns":...}).
        If provided, a 'my_cosmo' entry is added/selected.


    Returns
    -------
    halos : dict of np.ndarray
        Keys: 'M' (M200c, Msun/h), 'z', 'ra', 'dec'.
    """
    # Colossus is only needed for the FoF -> M200c calibration.
    if not no_calib:
        colossus_cosmology.addCosmology('my_cosmo', colossus_pars)
        colossus_cosmology.setCosmology('my_cosmo')

    # Read only columns used downstream. A shell filter enables Parquet
    # row-group pruning for catalogues written by ``save_halocatalog``.
    available = set(pq.ParquetFile(path_halo_cat).schema_arrow.names)
    columns = ['log_M', 'redshift']
    columns += ['ra_deg', 'dec_deg'] if {'ra_deg', 'dec_deg'} <= available else ['pix_16384_ring']
    optional = ['log_M200c_SO', 'host_e1_x10000', 'host_e2_x10000',
                'inertia_e1', 'inertia_e2', 'shape_valid', 'shell_index',
                'simulation_step', 'catalogue_index']
    columns += [name for name in optional if name in available]
    filters = None
    if shell_index is not None:
        if 'shell_index' not in available:
            raise KeyError("The halo catalogue has no shell_index column")
        filters = [('shell_index', '==', int(shell_index))]
    df = pd.read_parquet(path_halo_cat, columns=columns, filters=filters, engine="pyarrow")
    has_native_m200c = "log_M200c_SO" in df.columns
    if has_native_m200c:
        M200c_SO = recover_halo_mass(df, preferred_mass_col="log_M200c_SO")




    # Recover base fields
    M_fof = recover_halo_mass(df,preferred_mass_col = 'log_M')          # Msun/h
    z     = recover_halo_redshift(df)
    ra, dec = recover_halo_radec(df)
    has_halo_shape = (
        (
            "host_e1_x10000" in df.columns
            and "host_e2_x10000" in df.columns
        )
        or (
            "inertia_e1" in df.columns
            and "inertia_e2" in df.columns
        )
    ) and "shape_valid" in df.columns
    if has_halo_shape:
        host_ellipticity, shape_valid = recover_halo_ellipticity(df)

    # Convert FoF -> SO (M200c); important for tSZ, neutral for large-scale baryonification
    if no_calib:
        if verbose:
            print('no calib')
        M200c = copy.deepcopy(M_fof)
    else:
        if verbose:
            print('extra calib [fof -> SO]')
        interpolator_calib = build_fof_to_m200c_interpolator(
            sims_parameters, colossus_cosmology)
        coords = np.column_stack([np.log10(M_fof), z])
        M200c  = interpolator_calib(coords)

    # Apply mass cut -----------------------------------
    mask = (np.log10(M200c) > halo_catalog_log10mass_cut)

    if has_native_m200c:
        halos = {
            "M_SO":   M200c_SO[mask],
            "M":   M200c[mask],
            "z":   z[mask],
            "ra":  ra[mask],
            "dec": dec[mask],
        }
    else:
        halos = {
            "M":   M200c[mask],
            "z":   z[mask],
            "ra":  ra[mask],
            "dec": dec[mask],
        }
    if has_halo_shape:
        halos.update({
            "host_e1": host_ellipticity.real[mask],
            "host_e2": host_ellipticity.imag[mask],
            "shape_valid": shape_valid[mask],
        })
    for name in ('shell_index', 'simulation_step', 'catalogue_index'):
        if name in df.columns:
            halos[name] = df[name].to_numpy()[mask]
    return halos


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



def load_Flamingo_halo_catalog(path_catalog,sims_parameters,cosmo_bundle, type_cat = 'fof',halo_catalog_log10mass_cut = 13, max_z_halo_catalog = 1., no_calib = True):

    colossus_cosmology.addCosmology('my_cosmo', cosmo_bundle['colossus_params'])
    colossus_cosmology.setCosmology('my_cosmo')
    
    cat = fits.open(path_catalog)
    if type_cat == 'fof':

        if no_calib:
            M200c = cat[1].data['M']
        else:
            interpolator_calib = build_fof_to_m200c_interpolator(
            sims_parameters, colossus_cosmology)
            coords = np.column_stack([(np.log10(cat[1].data['M'])), cat[1].data['z']])
            M200c = interpolator_calib(coords)
    else:
        M200c = cat[1].data['M']

    z_all   = cat[1].data['z']
    ra_all  = cat[1].data['ra']
    dec_all = cat[1].data['dec']

    mask = (np.log10(M200c) > halo_catalog_log10mass_cut) & (z_all<max_z_halo_catalog)

    halos = {
        "M":   M200c[mask],
        "z":   z_all[mask],
        "ra":  ra_all[mask],
        "dec": dec_all[mask],
    }
    return halos
