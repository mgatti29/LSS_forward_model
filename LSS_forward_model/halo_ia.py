"""Halo-shape projection and portable intrinsic-alignment source modelling."""

from functools import lru_cache

import frogress
import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.special import ndtr


HALO_ELLIPTICITY_SCALE = 10000.0

__all__ = [
    "HALO_ELLIPTICITY_SCALE",
    "build_IA_halo_sources",
    "build_IA_halo_sources_from_catalog",
    "build_halo_ia_maps",
    "concentration_200c",
    "draw_sources",
    "encode_halo_ellipticity",
    "expected_counts_in_redshift_bins",
    "inverse_profile",
    "iter_IA_halo_sources",
    "occupation",
    "pixelize_shapes",
    "populate_IA_halo_shell",
    "project_inertia_tensor",
    "projected_nfw_surface_density",
    "radius_200c",
    "reconstruct_inertia_tensor_halo_snapshot",
    "recover_halo_ellipticity",
    "sample_radius",
    "sky_basis",
    "transport_phase",
    "unpack_inertia_tensor",
]


# Tensor decoding, projection and compact catalogue I/O
# -----------------------------------------------------

def reconstruct_inertia_tensor_halo_snapshot(data):
    """Decode snapshot moments into portable ``(xx,yy,zz,xy,xz,yz)`` order.

    PKDGRAV stores logarithmic diagonal moments and normalized cross terms.
    The Arrow path avoids creating one Python list object per halo.
    """
    if isinstance(data, pa.Table):
        auto = data["inertia_auto"].combine_chunks().values.to_numpy().reshape(-1, 3)
        cross = data["inertia_cross"].combine_chunks().values.to_numpy().reshape(-1, 3)
    else:
        auto = np.stack(data["inertia_auto"].to_numpy())
        cross = np.stack(data["inertia_cross"].to_numpy())

    # Undo the compact snapshot encoding.
    diagonal = 10 ** (auto.astype(float) / 1000) / 1e20
    correlation = cross.astype(float) / 10000 - 1
    off_diagonal = correlation * np.sqrt(
        diagonal[:, [0, 0, 1]] * diagonal[:, [1, 2, 2]]
    )
    return np.column_stack((diagonal, off_diagonal))


def unpack_inertia_tensor(inertia_tensor_packed):
    """Expand portable ``(xx,yy,zz,xy,xz,yz)`` moments to 3x3 matrices."""
    packed = np.asarray(inertia_tensor_packed, dtype=np.float64)
    if packed.ndim != 2 or packed.shape[1] != 6:
        raise ValueError("Expected an (N, 6) packed inertia tensor")
    tensor = np.zeros((len(packed), 3, 3), dtype=np.float64)
    tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 2, 2] = packed[:, :3].T
    tensor[:, 0, 1] = tensor[:, 1, 0] = packed[:, 3]
    tensor[:, 0, 2] = tensor[:, 2, 0] = packed[:, 4]
    tensor[:, 1, 2] = tensor[:, 2, 1] = packed[:, 5]
    return tensor


def project_inertia_tensor(ra_deg, dec_deg, inertia_tensor_packed):
    """Project 3D host moments on the local east/north sky basis.

    The direct contractions below are equivalent to expanding every packed
    tensor to 3x3 and applying ``east.T @ Q @ east`` etc., but use much less
    temporary memory for large lightcones.
    """
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    packed = np.asarray(inertia_tensor_packed, dtype=np.float64)
    if ra.shape != dec.shape:
        raise ValueError("ra_deg and dec_deg must have matching shapes")
    if packed.ndim != 2 or packed.shape[1] != 6:
        raise ValueError("Expected an (N, 6) packed inertia tensor")
    if len(packed) != len(ra):
        raise ValueError("Sky coordinates and inertia tensors must have equal lengths")

    sr, cr = np.sin(ra), np.cos(ra)
    sd, cd = np.sin(dec), np.cos(dec)
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = packed.T

    # Projected second moments in the local tangent plane.
    q11 = Ixx * sr**2 + Iyy * cr**2 - 2 * Ixy * sr * cr
    q22 = (Ixx * sd**2 * cr**2 + Iyy * sd**2 * sr**2 + Izz * cd**2
           + 2 * Ixy * sd**2 * cr * sr - 2 * Ixz * sd * cr * cd
           - 2 * Iyz * sd * sr * cd)
    q12 = sd * sr * cr * (Ixx - Iyy) + Ixy * sd * (sr**2 - cr**2) + cd * (-Ixz * sr + Iyz * cr)
    determinant = q11 * q22 - q12**2
    denominator = q11 + q22 + 2 * np.sqrt(np.maximum(determinant, 0))

    # Epsilon convention: (a-b)/(a+b) exp(2i phi).
    ellipticity = np.full(len(ra), complex(np.nan, np.nan), dtype=np.complex128)
    valid = (np.all(np.isfinite(packed), axis=1) & np.isfinite(q11) & np.isfinite(q22)
             & np.isfinite(q12) & (q11 > 0) & (q22 > 0) & (determinant > 0)
             & np.isfinite(denominator) & (denominator > 0))
    ellipticity[valid] = (q11[valid] - q22[valid] + 2j * q12[valid]) / denominator[valid]
    valid &= np.isfinite(ellipticity) & (np.abs(ellipticity) < 1)
    return ellipticity, valid


def encode_halo_ellipticity(ellipticity, valid):
    """Encode projected ellipticity as two signed int16 columns."""
    ellipticity = np.asarray(ellipticity, dtype=np.complex128)
    valid = np.asarray(valid, dtype=bool)
    if ellipticity.ndim != 1 or valid.shape != ellipticity.shape:
        raise ValueError("ellipticity and valid must be matching 1D arrays")
    if np.any(valid & (~np.isfinite(ellipticity) | (np.abs(ellipticity) >= 1))):
        raise ValueError("valid halo ellipticities must be finite with abs(e) < 1")
    e1 = np.zeros(len(ellipticity), dtype=np.int16)
    e2 = np.zeros(len(ellipticity), dtype=np.int16)
    e1[valid] = np.rint(HALO_ELLIPTICITY_SCALE * ellipticity.real[valid]).astype(np.int16)
    e2[valid] = np.rint(HALO_ELLIPTICITY_SCALE * ellipticity.imag[valid]).astype(np.int16)
    return e1, e2


def recover_halo_ellipticity(data):
    """Decode compact halo ellipticity columns and return values plus validity."""
    valid = data["shape_valid"].to_numpy(dtype=bool)
    compact = "host_e1_x10000" in data.columns and "host_e2_x10000" in data.columns
    if compact:
        e1 = data["host_e1_x10000"].to_numpy(dtype=np.float32)
        e2 = data["host_e2_x10000"].to_numpy(dtype=np.float32)
        ellipticity = (e1 + 1j * e2) / HALO_ELLIPTICITY_SCALE
    elif "inertia_e1" in data.columns and "inertia_e2" in data.columns:
        ellipticity = (data["inertia_e1"].to_numpy(dtype=np.float32)
                       + 1j * data["inertia_e2"].to_numpy(dtype=np.float32))
    else:
        raise KeyError("No compact or legacy halo ellipticity columns found")
    ellipticity = np.asarray(ellipticity, dtype=np.complex64)
    valid &= np.isfinite(ellipticity) & (np.abs(ellipticity) < 1)
    ellipticity[~valid] = complex(np.nan, np.nan)
    return ellipticity, valid


# Halo occupation and target redshift distribution
# ------------------------------------------------

def occupation(mass, target_count, f_sat, sigma=0.2, alpha=1.0, f_c=1.0):
    """Return central probabilities and mean satellite occupations.

    ``target_count`` is enforced in expectation. Centrals follow a softened
    mass threshold, while satellites use a central-gated power law.
    """
    mass = np.asarray(mass, dtype=float)
    if mass.ndim != 1 or np.any(~np.isfinite(mass) | (mass <= 0)):
        raise ValueError("Positive finite one-dimensional halo masses required")
    if not (0 <= f_sat < 1) or sigma <= 0 or not np.isfinite(alpha):
        raise ValueError("Require 0 <= f_sat < 1, sigma > 0 and finite alpha")
    if not np.isfinite(target_count) or target_count < 0:
        raise ValueError("Nonnegative finite target count required")
    if not np.isfinite(f_c) or not 0 < f_c <= 1:
        raise ValueError("Require 0 < f_c <= 1")

    expected_central = (1 - f_sat) * target_count
    expected_satellite = f_sat * target_count
    if expected_central >= f_c * len(mass) and target_count > 0:
        raise ValueError("Central abundance has no finite threshold on eligible halos")
    if target_count == 0:
        metadata = dict(log10_Mmin=None, A_sat=None, expected_central=0.0,
                        expected_satellite=0.0, target_count=0.0, f_sat=f_sat,
                        sigma_logM=sigma, alpha=alpha, f_c=f_c)
        return np.zeros(len(mass)), np.zeros(len(mass)), metadata

    log10_mass = np.log10(mass)

    def central_probability(threshold):
        return f_c * ndtr(np.sqrt(2) * (log10_mass - threshold) / sigma)

    # Solve M_min so that the expected central abundance closes exactly.
    threshold = brentq(lambda value: central_probability(value).sum() - expected_central,
                       log10_mass.min() - 40 * sigma, log10_mass.max() + 40 * sigma,
                       xtol=1e-12)
    central = central_probability(threshold)

    # Normalize the satellite power law independently to the requested fraction.
    satellite_shape = central * 10 ** (alpha * (log10_mass - threshold))
    amplitude = expected_satellite / satellite_shape.sum() if expected_satellite else 0.0
    satellite = amplitude * satellite_shape
    np.testing.assert_allclose([central.sum(), satellite.sum()],
                               [expected_central, expected_satellite], rtol=1e-9, atol=1e-7)
    metadata = dict(log10_Mmin=float(threshold), A_sat=float(amplitude),
                    expected_central=float(central.sum()),
                    expected_satellite=float(satellite.sum()), target_count=float(target_count),
                    f_sat=f_sat, sigma_logM=sigma, alpha=alpha, f_c=f_c)
    return central, satellite, metadata


def _integrate_piecewise_nz(redshift, nz, lower, upper):
    redshift, nz = np.asarray(redshift, float), np.asarray(nz, float)
    if redshift.ndim != 1 or nz.shape != redshift.shape:
        raise ValueError("nz_redshift and nz must be matching 1D arrays")
    if np.any(~np.isfinite(redshift)) or np.any(~np.isfinite(nz)):
        raise ValueError("n(z) inputs must be finite")
    if np.any(np.diff(redshift) <= 0) or np.any(nz < 0):
        raise ValueError("n(z) grid must increase and its values must be nonnegative")
    if upper <= lower:
        return 0.0
    internal = redshift[(redshift > lower) & (redshift < upper)]
    grid = np.concatenate(([lower], internal, [upper]))
    values = np.interp(grid, redshift, nz, left=0, right=0)
    return np.trapz(values, grid)


def expected_counts_in_redshift_bins(redshift_edges, nz_redshift, nz, total_number_counts,
                                      normalization_bounds=None):
    """Convert a relative n(z) into expected counts in each requested bin.

    By default the requested bins receive all ``total_number_counts``. Passing
    ``normalization_bounds`` instead normalizes against that wider interval, so
    bins outside the available halo lightcone are not reassigned to lower z.
    """
    edges = np.asarray(redshift_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("redshift_edges must be strictly increasing")
    if not np.isfinite(total_number_counts) or total_number_counts < 0:
        raise ValueError("total_number_counts must be finite and nonnegative")
    fractions = np.array([_integrate_piecewise_nz(nz_redshift, nz, lo, hi)
                          for lo, hi in zip(edges[:-1], edges[1:])])
    if normalization_bounds is None:
        normalization = fractions.sum()
    else:
        lower, upper = normalization_bounds
        normalization = _integrate_piecewise_nz(nz_redshift, nz, lower, upper)
    if normalization <= 0:
        raise ValueError("n(z) has no support inside the requested bins")
    fractions /= normalization
    return total_number_counts * fractions


# Spherical geometry and projected-NFW sampling
# ---------------------------------------------

def sky_basis(ra_deg, dec_deg):
    """Return radial, east and north unit vectors at sky positions."""
    ra = np.radians(np.asarray(ra_deg, dtype=float))
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    if ra.shape != dec.shape:
        raise ValueError("ra_deg and dec_deg must have identical shapes")
    sr, cr, sd, cd = np.sin(ra), np.cos(ra), np.sin(dec), np.cos(dec)
    radial = np.column_stack((cd * cr, cd * sr, sd))
    east = np.column_stack((-sr, cr, np.zeros(len(ra))))
    north = np.column_stack((-sd * cr, -sd * sr, cd))
    return radial, east, north


def transport_phase(ra_from, dec_from, ra_to, dec_to):
    """Spin-2 phase for parallel transport along a great circle.

    This rotates a shape defined in the tangent basis at the source position
    into the tangent basis at the destination position.
    """
    radial_from, east_from, _ = sky_basis(ra_from, dec_from)
    radial_to, east_to, north_to = sky_basis(ra_to, dec_to)
    denominator = 1 + np.einsum("ij,ij->i", radial_from, radial_to)
    if np.any(denominator <= 1e-12):
        raise ValueError("Antipodal spin transport is undefined")
    projection = np.einsum("ij,ij->i", east_from, radial_to)
    transported = east_from - (projection / denominator)[:, None] * (radial_from + radial_to)
    phase = (np.einsum("ij,ij->i", transported, east_to)
             + 1j * np.einsum("ij,ij->i", transported, north_to))
    return phase**2


def projected_nfw_surface_density(x):
    """Dimensionless projected NFW surface-density profile."""
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("Projected NFW radius must be positive")
    output = np.empty_like(x)
    close = np.abs(x - 1) < 1e-5
    below, above = x < 1 - 1e-5, x > 1 + 1e-5
    if np.any(below):
        value = x[below]
        root = np.sqrt((1 - value) / (1 + value))
        output[below] = (1 - 2 * np.arctanh(root) / np.sqrt(1 - value**2)) / (value**2 - 1)
    if np.any(above):
        value = x[above]
        root = np.sqrt((value - 1) / (1 + value))
        output[above] = (1 - 2 * np.arctan(root) / np.sqrt(value**2 - 1)) / (value**2 - 1)
    output[close] = 1 / 3
    return output


@lru_cache(maxsize=4)
def inverse_profile(number_concentrations=129, number_radii=4097, number_probabilities=4097):
    """Build an interpolator for the projected-NFW inverse CDF."""
    concentrations = np.geomspace(0.5, 50, number_concentrations)
    radii = np.concatenate(([0], np.geomspace(1e-9, 1, number_radii - 1)))
    probabilities = 0.5 * (1 - np.cos(np.linspace(0, np.pi, number_probabilities)))
    inverse_cdf = np.empty((number_concentrations, number_probabilities))

    # Tabulate the radial CDF, including the projected-area Jacobian R.
    for index, concentration in enumerate(concentrations):
        density = np.zeros(number_radii)
        density[1:] = radii[1:] * projected_nfw_surface_density(concentration * radii[1:])
        cdf = cumulative_trapezoid(density, radii, initial=0)
        cdf /= cdf[-1]
        inverse_cdf[index] = np.interp(probabilities, cdf, radii)
    return RegularGridInterpolator((np.log(concentrations), probabilities), inverse_cdf,
                                   bounds_error=True)


def sample_radius(concentration, uniform, refined=False):
    """Draw R/R200 from a truncated projected-NFW distribution."""
    concentration, uniform = np.broadcast_arrays(np.asarray(concentration, float),
                                                  np.asarray(uniform, float))
    if np.any(~np.isfinite(concentration) | (concentration < 0.5) | (concentration > 50)):
        raise ValueError("Concentration must be finite and lie in [0.5, 50]")
    if np.any(~np.isfinite(uniform) | (uniform < 0) | (uniform > 1)):
        raise ValueError("Uniform probabilities must lie in [0,1]")
    interpolator = inverse_profile(257, 8193, 8193) if refined else inverse_profile()
    coordinates = np.column_stack((np.log(concentration).ravel(), uniform.ravel()))
    return interpolator(coordinates).reshape(concentration.shape)


def draw_sources(ra, dec, redshift, central_probability, satellite_mean, r200,
                 concentration, host_e1, host_e2, rng, cosmology, quarter_turns=0,
                 position_ellipticity_scale=1.0, minimum_axis_ratio=0.15,
                 central_host=None, satellite_count=None):
    """Draw explicit central and satellite galaxies from halo occupations.

    Centrals remain at the host centre and inherit its projected ellipticity.
    Satellites follow an elliptical projected-NFW distribution and receive a
    radial spin-2 orientation about the host.
    """
    arrays = [np.asarray(value, dtype=float) for value in
              (ra, dec, redshift, central_probability, satellite_mean, r200,
               concentration, host_e1, host_e2)]
    (ra, dec, redshift, central_probability, satellite_mean, r200, concentration,
     host_e1, host_e2) = arrays
    if any(value.shape != ra.shape for value in arrays):
        raise ValueError("All halo input arrays must have identical shapes")
    if np.any(~np.isfinite(central_probability) | (central_probability < 0)
              | (central_probability > 1)):
        raise ValueError("Invalid central probabilities")
    if np.any(~np.isfinite(satellite_mean) | (satellite_mean < 0)):
        raise ValueError("Invalid satellite means")
    if not 0 <= position_ellipticity_scale <= 1 or not 0 < minimum_axis_ratio <= 1:
        raise ValueError("Invalid position ellipticity scale or minimum axis ratio")

    host_ellipticity = host_e1 + 1j * host_e2
    amplitude = np.abs(host_ellipticity)
    if not np.all(np.isfinite(host_ellipticity) & (amplitude < 1)):
        raise ValueError("Invalid projected host ellipticity in source sampler")
    angle = 0.5 * np.angle(host_ellipticity)
    maximum = (1 - minimum_axis_ratio) / (1 + minimum_axis_ratio)
    axis_ratio = (1 - np.clip(amplitude, 0, maximum)) / (1 + np.clip(amplitude, 0, maximum))

    # Bernoulli centrals and independent Poisson satellite counts. Callers may
    # provide these draws so profile scales are evaluated only for hosts that
    # actually receive satellites.
    if central_host is None:
        central_host = np.flatnonzero(rng.random(len(central_probability)) < central_probability)
    else:
        central_host = np.asarray(central_host, dtype=np.int64)
    if satellite_count is None:
        satellite_count = rng.poisson(satellite_mean)
    else:
        satellite_count = np.asarray(satellite_count, dtype=np.int64)
        if satellite_count.shape != satellite_mean.shape or np.any(satellite_count < 0):
            raise ValueError("Invalid pre-drawn satellite counts")
    satellite_host = np.repeat(np.arange(len(satellite_mean)), satellite_count)
    host_index = np.concatenate((central_host, satellite_host))
    is_central = np.arange(len(host_index)) < len(central_host)
    source_ra, source_dec = ra[host_index].copy(), dec[host_index].copy()
    source_redshift, source_angle = redshift[host_index], angle[host_index]
    source_axis_ratio = axis_ratio[host_index]
    radial, east, north = sky_basis(ra[host_index], dec[host_index])
    position_axis_ratio = 1 - position_ellipticity_scale * (1 - source_axis_ratio)
    central_unit = np.where(is_central, (1 - source_axis_ratio) / (1 + source_axis_ratio)
                            * np.exp(2j * source_angle), 0)
    satellite_unit = np.zeros(len(host_index), dtype=complex)
    elliptical_radius = np.zeros(len(host_index), dtype=float)
    satellite_source = np.flatnonzero(~is_central)

    if len(satellite_source):
        parent = satellite_host

        # Draw elliptical offsets in the host major/minor-axis frame.
        radius_fraction = sample_radius(concentration[parent], rng.random(len(parent)))
        azimuth = 2 * np.pi * rng.random(len(parent))
        major = r200[parent] * radius_fraction * np.cos(azimuth)
        minor = r200[parent] * radius_fraction * position_axis_ratio[satellite_source] * np.sin(azimuth)
        cos_angle = np.cos(source_angle[satellite_source])
        sin_angle = np.sin(source_angle[satellite_source])
        east_offset = major * cos_angle - minor * sin_angle
        north_offset = major * sin_angle + minor * cos_angle
        distance = np.hypot(east_offset, north_offset)
        comoving_distance = np.asarray(cosmology.comoving_distance(redshift[parent]).value)
        if np.any(comoving_distance <= 0):
            raise ValueError("Satellite angular profile requires positive distance")
        separation = distance / comoving_distance

        # Move along the local tangent direction on the sphere.
        tangent = np.divide(east_offset[:, None] * east[satellite_source]
                            + north_offset[:, None] * north[satellite_source], distance[:, None],
                            out=east[satellite_source].copy(), where=distance[:, None] > 0)
        position = (np.cos(separation)[:, None] * radial[satellite_source]
                    + np.sin(separation)[:, None] * tangent)
        satellite_ra = np.degrees(np.arctan2(position[:, 1], position[:, 0])) % 360
        satellite_dec = np.degrees(np.arcsin(np.clip(position[:, 2], -1, 1)))
        source_ra[satellite_source], source_dec[satellite_source] = satellite_ra, satellite_dec

        # Radial spin-2 phase, parallel transported to the satellite position.
        phase = np.divide((east_offset + 1j * north_offset)**2, distance**2,
                          out=np.ones(len(parent), dtype=complex), where=distance > 0)
        satellite_unit[satellite_source] = phase * transport_phase(
            ra[parent], dec[parent], satellite_ra, satellite_dec)
        elliptical_radius[satellite_source] = radius_fraction

    return dict(host_batch_index=host_index, is_central=is_central,
                ra_deg=(source_ra + 90 * quarter_turns) % 360, dec_deg=source_dec,
                redshift=source_redshift, central_unit=central_unit,
                satellite_unit=satellite_unit, elliptical_radius_over_r200=elliptical_radius)


# Effective halo-profile scales
# -----------------------------

def radius_200c(mass, redshift, cosmology):
    """Comoving R200c in Mpc for input mass in physical Msun.

    Passing a FoF mass is the explicitly documented effective-profile choice;
    it does not turn that mass into a measured spherical-overdensity mass.
    """
    mass, redshift = np.broadcast_arrays(np.asarray(mass, float), np.asarray(redshift, float))
    valid = np.isfinite(mass) & (mass > 0) & np.isfinite(redshift) & (redshift >= 0)
    if not np.all(valid):
        raise ValueError("Invalid mass or redshift")
    density = cosmology.critical_density(redshift).to_value("Msun / Mpc3")
    return (3 * mass / (4 * np.pi * 200 * density)) ** (1 / 3) * (1 + redshift)


@lru_cache(maxsize=4)
def _concentration_200c_grid(H0, Om0, Ob0, sigma8, ns):
    """Cache a median Diemer19 c200c grid for one cosmology."""
    from colossus.cosmology import cosmology as colossus_cosmology
    from colossus.halo import concentration

    try:
        previous = colossus_cosmology.getCurrent()
    except Exception:
        previous = None
    colossus_cosmology.setCosmology(
        "ia_diemer19_200c",
        dict(flat=True, H0=H0, Om0=Om0, Ob0=Ob0, sigma8=sigma8, ns=ns),
        persistence="",
    )
    logmass, redshift = np.linspace(11, 17, 241), np.linspace(0, 3, 121)
    values = np.empty((len(logmass), len(redshift)))
    try:
        for index, z in enumerate(redshift):
            value, valid = concentration.concentration(
                10**logmass * (H0 / 100), "200c", float(z), model="diemer19",
                statistic="median", range_return=True)
            if not np.all(valid & np.isfinite(value) & (value > 0)):
                raise ValueError("Invalid Diemer19 concentration grid")
            values[:, index] = np.log(value)
    finally:
        if previous is not None:
            colossus_cosmology.setCurrent(previous)
    return RegularGridInterpolator((logmass, redshift), values, bounds_error=True)


def concentration_200c(mass, redshift, cosmology, *, sigma8, ns):
    """Median Diemer19 c200c for input mass in physical Msun."""
    mass, redshift = np.broadcast_arrays(np.asarray(mass, float), np.asarray(redshift, float))
    valid = np.isfinite(mass) & (mass > 0) & np.isfinite(redshift) & (redshift >= 0)
    if not np.all(valid):
        raise ValueError("Invalid mass or redshift")
    grid = _concentration_200c_grid(float(cosmology.H0.value), float(cosmology.Om0),
                                   float(cosmology.Ob0), float(sigma8), float(ns))
    coordinates = np.column_stack((np.log10(mass).ravel(), redshift.ravel()))
    return np.exp(grid(coordinates)).reshape(mass.shape)


# End-to-end halo population
# --------------------------

def populate_IA_halo_shell(halos, target_count, model_parameters, cosmo_bundle,
                           sims_parameters, rng, shell_index=None, quarter_turns=0,
                           position_ellipticity_scale=1.0):
    """Populate one halo shell without retaining any other shell in memory."""
    required = {"M", "z", "ra", "dec", "host_e1", "host_e2", "shape_valid"}
    missing = required.difference(halos)
    if missing:
        raise KeyError(f"Halo shell is missing required fields: {sorted(missing)}")
    valid = (np.asarray(halos["shape_valid"], bool) & np.isfinite(halos["M"])
             & (np.asarray(halos["M"]) > 0) & np.isfinite(halos["z"])
             & np.isfinite(halos["ra"]) & np.isfinite(halos["dec"])
             & np.isfinite(halos["host_e1"]) & np.isfinite(halos["host_e2"]))
    halos = {name: np.asarray(value)[valid] for name, value in halos.items()}
    if not len(halos["z"]):
        raise ValueError(f"No eligible IA hosts in shell {shell_index}")

    mass = halos["M_SO"] if "M_SO" in halos else halos["M"]
    population = model_parameters["population"]
    central, satellite, _ = occupation(
        mass, target_count, population["f_sat"], population["sigma_logM"],
        population["alpha"], population["f_c"])
    cosmology = cosmo_bundle["cosmo_astropy"]

    # Draw occupations first, then evaluate profiles only for satellite hosts.
    central_host = np.flatnonzero(rng.random(len(central)) < central)
    satellite_count = rng.poisson(satellite)
    satellite_hosts = np.flatnonzero(satellite_count)
    radius = np.ones(len(mass), dtype=float)
    concentration = np.ones(len(mass), dtype=float)
    if len(satellite_hosts):
        radius[satellite_hosts] = radius_200c(
            mass[satellite_hosts], halos["z"][satellite_hosts], cosmology)
        concentration[satellite_hosts] = concentration_200c(
            mass[satellite_hosts], halos["z"][satellite_hosts], cosmology,
            sigma8=sims_parameters["sigma_8"], ns=sims_parameters["n_s"])

    sources = draw_sources(
        halos["ra"], halos["dec"], halos["z"], central, satellite, radius,
        concentration, halos["host_e1"], halos["host_e2"], rng, cosmology,
        quarter_turns=quarter_turns, position_ellipticity_scale=position_ellipticity_scale,
        central_host=central_host, satellite_count=satellite_count)
    source_host = sources.pop("host_batch_index")
    catalogue_index = halos.get("catalogue_index", np.arange(len(mass)))
    ia_unit = sources["central_unit"] + sources["satellite_unit"]
    batch = dict(ra_deg=sources["ra_deg"], dec_deg=sources["dec_deg"],
                 redshift=sources["redshift"], is_central=sources["is_central"],
                 e1=ia_unit.real, e2=ia_unit.imag,
                 host_catalogue_index=catalogue_index[source_host],
                 host_mass=mass[source_host])
    if shell_index is not None:
        batch["shell_index"] = np.full(len(source_host), shell_index, np.uint16)
    return batch


def build_IA_halo_sources(halos, total_number_counts, nz_redshift, nz, shells_info,
                          model_parameters, cosmo_bundle, sims_parameters, seed=913603,
                          quarter_turns=0, position_ellipticity_scale=1.0):
    """Populate a compact halo lightcone with explicit centrals and satellites.

    The target n(z) fixes expected counts per redshift interval, so generated
    objects carry unit statistical weight at this stage.
    """
    required = {"M", "z", "ra", "dec", "host_e1", "host_e2", "shape_valid"}
    missing = required.difference(halos)
    if missing:
        raise KeyError(f"Halo catalogue is missing required fields: {sorted(missing)}")
    valid = (halos["shape_valid"] & np.isfinite(halos["M"]) & (halos["M"] > 0)
             & np.isfinite(halos["z"]) & np.isfinite(halos["ra"])
             & np.isfinite(halos["dec"]) & np.isfinite(halos["host_e1"])
             & np.isfinite(halos["host_e2"]))
    halos = {name: np.asarray(value)[valid] for name, value in halos.items()}
    if not len(halos["z"]):
        raise ValueError("No eligible IA hosts in the halo catalogue")

    # Limit the model to the overlap between the halo lightcone and target n(z).
    nz_redshift, nz = np.asarray(nz_redshift, float), np.asarray(nz, float)
    target_support = np.isfinite(nz_redshift) & np.isfinite(nz) & (nz > 0)
    if not np.any(target_support):
        raise ValueError("The target n(z) has no positive finite support")
    target_max_redshift = np.max(nz_redshift[target_support])
    max_redshift = min(np.max(halos["z"]), target_max_redshift)
    if not np.isfinite(max_redshift) or max_redshift <= 0:
        raise ValueError("The halo catalogue and target n(z) have no positive-redshift overlap")

    # Use the simulation shell boundaries as the HOD calibration intervals.
    try:
        z_near = np.asarray(shells_info["z_near"], float)[::-1]
        z_far = np.asarray(shells_info["z_far"], float)[::-1]
    except (KeyError, TypeError):
        raise ValueError("shells_info must contain z_near and z_far") from None
    if z_near.shape != z_far.shape or np.any(z_far <= z_near):
        raise ValueError("Invalid redshift intervals in shells_info")

    active_shells, target_counts = [], []
    for shell_index, (lower, upper) in enumerate(zip(z_near, z_far)):
        lower, upper = max(0.0, lower), min(upper, max_redshift)
        if upper <= lower:
            continue
        count = expected_counts_in_redshift_bins(
            [lower, upper], nz_redshift, nz, total_number_counts,
            normalization_bounds=(0.0, target_max_redshift))[0]
        active_shells.append(shell_index)
        target_counts.append(count)

    # New catalogues are written in shell order. Searchsorted then gives each
    # shell slice without rescanning the full halo lightcone for every shell.
    stored_shell = halos.get("shell_index")
    ordered_shells = stored_shell is not None and (
        len(stored_shell) < 2 or np.all(stored_shell[1:] >= stored_shell[:-1]))
    catalogue_index = halos.get("catalogue_index", np.arange(len(halos["z"])))
    rng, batches = np.random.default_rng(seed), []

    for shell_index, target_count in zip(active_shells, target_counts):
        if stored_shell is not None and ordered_shells:
            selection = slice(np.searchsorted(stored_shell, shell_index, side="left"),
                              np.searchsorted(stored_shell, shell_index, side="right"))
        elif stored_shell is not None:
            selection = np.flatnonzero(stored_shell == shell_index)
        else:
            lower, upper = z_near[shell_index], min(z_far[shell_index], max_redshift)
            below_upper = halos["z"] < upper
            if upper == max_redshift:
                below_upper |= halos["z"] == max_redshift
            selection = np.flatnonzero((halos["z"] >= lower) & below_upper)

        redshift = halos["z"][selection]
        within_model = redshift <= max_redshift
        if not np.all(within_model):
            if isinstance(selection, slice):
                selection = np.arange(selection.start, selection.stop)[within_model]
            else:
                selection = selection[within_model]
            redshift = halos["z"][selection]
        if not len(redshift):
            if target_count > 0:
                raise ValueError(f"No eligible IA hosts in shell {shell_index}, "
                                 f"expected {target_count:.3f} sources")
            continue
        shell_halos = {name: np.asarray(value)[selection] for name, value in halos.items()}
        shell_halos["catalogue_index"] = np.asarray(catalogue_index[selection])
        batches.append(populate_IA_halo_shell(
            shell_halos, target_count, model_parameters, cosmo_bundle, sims_parameters,
            rng, shell_index=shell_index, quarter_turns=quarter_turns,
            position_ellipticity_scale=position_ellipticity_scale))

    if not batches:
        raise ValueError("No IA sources were generated")
    return {name: np.concatenate([batch[name] for batch in batches]) for name in batches[0]}


# Streaming IA catalogue population
# ---------------------------------

def _halo_catalog_max_redshift(path_halo_cat):
    """Return the maximum encoded catalogue redshift using Parquet statistics."""
    parquet = pq.ParquetFile(path_halo_cat)
    column = parquet.schema_arrow.get_field_index("redshift")
    maxima = []
    if column >= 0:
        for row_group in range(parquet.num_row_groups):
            statistics = parquet.metadata.row_group(row_group).column(column).statistics
            if statistics is not None and statistics.has_min_max:
                maxima.append(statistics.max)
    if maxima:
        return max(maxima) / 10000.0

    maximum = -np.inf
    for batch in parquet.iter_batches(columns=["redshift"], batch_size=1_000_000):
        values = batch.column(0).to_numpy(zero_copy_only=False)
        if len(values):
            maximum = max(maximum, float(np.max(values)) / 10000.0)
    if not np.isfinite(maximum):
        raise ValueError("Halo catalogue contains no redshifts")
    return maximum


def _load_IA_halo_shell(path_halo_cat, shell_index, log10_mass_cut,
                        lower_redshift=None, upper_redshift=None):
    """Read and decode one native-FoF IA shell with Parquet predicate pushdown.

    New catalogues use ``shell_index`` directly. For catalogues made before
    that column was introduced, encoded-redshift filters provide a compatible
    streaming path without requiring a full catalogue rewrite.
    """
    available = set(pq.ParquetFile(path_halo_cat).schema_arrow.names)
    required = {"log_M", "redshift", "ra_deg", "dec_deg",
                "host_e1_x10000", "host_e2_x10000", "shape_valid"}
    missing = required.difference(available)
    if missing:
        raise KeyError(f"Streaming IA catalogue is missing columns: {sorted(missing)}")
    optional = [name for name in ("log_M200c_SO", "catalogue_index", "simulation_step")
                if name in available]
    if "shell_index" in available:
        optional.append("shell_index")
        filters = [("shell_index", "=", int(shell_index))]
    else:
        if lower_redshift is None or upper_redshift is None:
            raise ValueError("Redshift bounds are required when shell_index is unavailable")
        filters = [("redshift", ">=", int(np.floor(10000 * lower_redshift))),
                   ("redshift", "<", int(np.ceil(10000 * upper_redshift)))]
    table = pq.read_table(
        path_halo_cat, columns=sorted(required) + optional, filters=filters)

    def values(name):
        return table[name].combine_chunks().to_numpy(zero_copy_only=False)

    encoded_mass = values("log_M")
    mass = 10.0**(encoded_mass.astype(float) / 1000.0)
    keep = (encoded_mass > 0) & (np.log10(mass) > log10_mass_cut)
    e1 = values("host_e1_x10000").astype(np.float32) / HALO_ELLIPTICITY_SCALE
    e2 = values("host_e2_x10000").astype(np.float32) / HALO_ELLIPTICITY_SCALE
    shape_valid = values("shape_valid").astype(bool)
    shape_valid &= np.isfinite(e1) & np.isfinite(e2) & (np.hypot(e1, e2) < 1)

    halos = {
        "M": mass[keep],
        "z": values("redshift")[keep].astype(float) / 10000.0,
        "ra": values("ra_deg")[keep],
        "dec": values("dec_deg")[keep],
        "host_e1": e1[keep],
        "host_e2": e2[keep],
        "shape_valid": shape_valid[keep],
    }
    if "shell_index" in available:
        halos["shell_index"] = values("shell_index")[keep]
    else:
        halos["shell_index"] = np.full(np.count_nonzero(keep), shell_index, np.uint16)
    if "log_M200c_SO" in available:
        encoded_so = values("log_M200c_SO")
        halos["M_SO"] = 10.0**(encoded_so[keep].astype(float) / 1000.0)
    if "catalogue_index" in available:
        halos["catalogue_index"] = values("catalogue_index")[keep]
    if "simulation_step" in available:
        halos["simulation_step"] = values("simulation_step")[keep]
    return halos


def iter_IA_halo_sources(path_halo_cat, sims_parameters, halo_catalog_log10mass_cut,
                          total_number_counts, nz_redshift, nz, shells_info,
                          model_parameters, cosmo_bundle, seed=913603, quarter_turns=0,
                          position_ellipticity_scale=1.0):
    """Yield populated IA sources one shell at a time from a Parquet catalogue."""
    nz_redshift, nz = np.asarray(nz_redshift, float), np.asarray(nz, float)
    support = np.isfinite(nz_redshift) & np.isfinite(nz) & (nz > 0)
    if not np.any(support):
        raise ValueError("The target n(z) has no positive finite support")
    target_max = np.max(nz_redshift[support])
    model_max = min(_halo_catalog_max_redshift(path_halo_cat), target_max)
    z_near = np.asarray(shells_info["z_near"], float)[::-1]
    z_far = np.asarray(shells_info["z_far"], float)[::-1]
    rng = np.random.default_rng(seed)

    for shell_index, (lower, upper) in enumerate(zip(z_near, z_far)):
        lower, upper = max(0.0, lower), min(upper, model_max)
        if upper <= lower:
            continue
        target_count = expected_counts_in_redshift_bins(
            [lower, upper], nz_redshift, nz, total_number_counts,
            normalization_bounds=(0.0, target_max))[0]
        halos = _load_IA_halo_shell(
            path_halo_cat, shell_index, halo_catalog_log10mass_cut,
            lower_redshift=lower, upper_redshift=upper)
        inside = ((np.asarray(halos["z"]) >= lower)
                  & (np.asarray(halos["z"]) < upper)
                  & (np.asarray(halos["z"]) <= model_max))
        halos = {name: np.asarray(value)[inside] for name, value in halos.items()}
        if not len(halos["z"]):
            if target_count > 0:
                raise ValueError(f"No eligible IA hosts in shell {shell_index}")
            continue
        yield shell_index, populate_IA_halo_shell(
            halos, target_count, model_parameters, cosmo_bundle, sims_parameters,
            rng, shell_index=shell_index, quarter_turns=quarter_turns,
            position_ellipticity_scale=position_ellipticity_scale)


def build_IA_halo_sources_from_catalog(*args, show_progress=True, **kwargs):
    """Concatenate streamed shell batches, optionally showing their progress."""
    shell_batches = iter_IA_halo_sources(*args, **kwargs)
    if show_progress:
        shell_batches = frogress.bar(shell_batches)
    batches = [batch for _, batch in shell_batches]
    if not batches:
        raise ValueError("No IA sources were generated")
    return {name: np.concatenate([batch[name] for batch in batches]) for name in batches[0]}




def _pixelize_shape_marks(pixel, marks, weights, npix, count=None):
    """Accumulate weighted complex marks for already-computed HEALPix pixels."""
    if count is None:
        count = np.bincount(pixel, weights=weights, minlength=npix)
    value = weights*marks
    numerator = (np.bincount(pixel, weights=value.real, minlength=npix)
                 + 1j*np.bincount(pixel, weights=value.imag, minlength=npix))
    mean = np.divide(numerator, count, out=np.zeros_like(numerator), where=count > 0)
    return {'count': count, 'numerator': numerator, 'mean': mean}


def pixelize_shapes(ra, dec, marks, weights, nside):
    """Return weighted counts, sums, and mean complex shape in each pixel.

    Shapes are accumulated in their catalogue basis without applying a
    sub-pixel spin-2 parallel-transport correction.
    """
    ra = np.asarray(ra, float)
    dec = np.asarray(dec, float)
    marks = np.asarray(marks, complex)
    weights = np.asarray(weights, float)
    if not (ra.shape == dec.shape == marks.shape == weights.shape):
        raise ValueError('Object arrays must have identical shapes')
    if np.any(~np.isfinite(marks)) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError('Invalid marks or weights')

    pixel = hp.ang2pix(nside, ra, dec, lonlat=True)
    return _pixelize_shape_marks(pixel, marks, weights, hp.nside2npix(nside))


def build_halo_ia_maps(model_objects, density, ia_parameters, nside):
    """Apply fitted IA amplitudes and pixelize the generated sources."""
    ra = np.asarray(model_objects['ra_deg'], float)
    dec = np.asarray(model_objects['dec_deg'], float)
    redshift = np.asarray(model_objects['redshift'], float)
    is_central = np.asarray(model_objects['is_central'], bool)
    shell_index = np.asarray(model_objects['shell_index'], int)
    unit_shape = np.asarray(model_objects['e1']) + 1j*np.asarray(model_objects['e2'])

    size = len(redshift)
    if not all(len(value) == size for value in (ra, dec, is_central, shell_index, unit_shape)):
        raise ValueError('Inconsistent model-object array lengths')
    if np.any((shell_index < 0) | (shell_index >= len(density))):
        raise ValueError('Model sources refer to unavailable density shells')

    # Interpolate each native density shell only at its own sources.
    delta_source = np.empty(size, dtype=np.float32)
    for shell in frogress.bar(np.unique(shell_index)):
        use = shell_index == shell
        delta_source[use] = hp.get_interp_val(
            density[shell], ra[use], dec[use], lonlat=True)

    evolution = (1 + redshift)/1.67
    central_mark = np.zeros(size, dtype=np.complex64)
    satellite_mark = np.zeros(size, dtype=np.complex64)
    central_mark[is_central] = (
        ia_parameters['ia']['A_c']*evolution[is_central]**ia_parameters['ia']['eta_c']
        *(1 + ia_parameters['ia']['b_c']*delta_source[is_central])*unit_shape[is_central])
    satellite_mark[~is_central] = (
        ia_parameters['ia']['A_s']*evolution[~is_central]**ia_parameters['ia']['eta_s']
        *(1 + ia_parameters['ia']['b_s']*delta_source[~is_central])*unit_shape[~is_central])
    marks = {'central': central_mark, 'satellite': satellite_mark,
             'total': central_mark + satellite_mark}

    # Reuse the pixel geometry and weighted counts for all IA components.
    weights = np.ones(size, dtype=np.float32)
    pixel = hp.ang2pix(nside, ra, dec, lonlat=True)
    npix = hp.nside2npix(nside)
    count = np.bincount(pixel, weights=weights, minlength=npix)
    pixelizations = {
        name: _pixelize_shape_marks(pixel, mark, weights, npix, count=count)
        for name, mark in marks.items()
    }
    maps = {name: value['mean'].astype(np.complex64)
            for name, value in pixelizations.items()}

    # A random rotation makes either ellipticity component carry half of |e|^2.
    # This is the per-component variance of the pixel mean contributed by the
    # modeled IA shapes, in the same convention as data rotation noise maps.
    ia_squared_sum = np.bincount(
        pixel, weights=weights**2*np.abs(marks['total'])**2, minlength=len(count))
    ia_rotation_noise_variance = np.divide(
        0.5*ia_squared_sum, count**2, out=np.zeros_like(count), where=count > 0)
    ia_rotation_noise_std = np.sqrt(ia_rotation_noise_variance)

    return {'maps': maps, 'pixelizations': pixelizations, 'marks': marks,
            'weights': weights, 'delta_source': delta_source,
            'ia_rotation_noise_variance': ia_rotation_noise_variance,
            'ia_rotation_noise_std': ia_rotation_noise_std}
