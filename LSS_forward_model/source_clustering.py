import json
import itertools

import numpy as np
import healpy as hp
from scipy.optimize import minimize
from scipy.special import gammaln

from .maps import convert_to_pix_coord, rotate_and_rebin


# Source clustering in the mock shear maps.
#
# Signal: the shear of each shell is weighted by 1 + b(z) delta, with a source bias that evolves
# inside the tomographic bin, b(z) = b [1 + s (z - zbar)].
# Noise: the variance of the (rotated) data shape noise in pixel p is multiplied by
# h(lambda_p D_p) / h(lambda_p), where h(x) = E[1/N | N > 0] for N ~ Poisson(x), D is the projected
# source-density modulation and lambda_p the expected number of sources in the pixel (depth).
# For lambda -> infinity this is 1/D, the previous ('inverse_density') prescription.


def bin_mean_redshift(ngal_glass, zeff_glass):
    """
    n(z)-weighted mean redshift of each tomographic bin.

    Parameters
    ----------
    ngal_glass : array (ntomo, nshell)
        n(z) weights of each shell.
    zeff_glass : array (nshell,)
        Effective redshift of each shell.

    Returns
    -------
    zbar : np.ndarray (ntomo,)
    """
    ngal_glass = np.asarray(ngal_glass)
    zbar = np.zeros(len(ngal_glass))
    for tomo in range(len(ngal_glass)):
        zbar[tomo] = np.sum(ngal_glass[tomo]*zeff_glass)/np.sum(ngal_glass[tomo])
    return zbar


def shell_bias(ngal_glass, zeff_glass, bias_sc, bias_sc_slope=None):
    """
    Source bias of each tomographic bin and shell, b_t [1 + s_t (z_shell - zbar_t)].

    Parameters
    ----------
    ngal_glass : array (ntomo, nshell)
        n(z) weights of each shell.
    zeff_glass : array (nshell,)
        Effective redshift of each shell.
    bias_sc : array (ntomo,)
        Source bias b_t of each bin.
    bias_sc_slope : array (ntomo,), optional
        Slope s_t of the bias inside the bin. None (or 0) gives a constant bias.

    Returns
    -------
    bias : np.ndarray (ntomo, nshell)
    """
    zbar = bin_mean_redshift(ngal_glass, zeff_glass)
    bias = np.zeros((len(ngal_glass), len(zeff_glass)))
    for tomo in range(len(ngal_glass)):
        slope = 0. if bias_sc_slope is None else bias_sc_slope[tomo]
        bias[tomo] = bias_sc[tomo]*(1 + slope*(np.asarray(zeff_glass) - zbar[tomo]))
    return bias


# expected inverse count of a Poisson variable, tabulated once
_x_table = np.geomspace(1e-3, 300., 400)
_h_table = None


def expected_inverse_count(x):
    """
    h(x) = E[1/N | N > 0] for N ~ Poisson(x) = 1/(e^x - 1) sum_{N>=1} x^N / (N! N).

    Tabulated for x <= 300 (log-log interpolation); above, the large-mean expansion
    (1 + 1/x + 2/x^2)/x.

    Parameters
    ----------
    x : array
        Mean of the Poisson distribution.

    Returns
    -------
    h : np.ndarray
    """
    global _h_table
    if _h_table is None:
        _h_table = np.zeros(len(_x_table))
        for i, xx in enumerate(_x_table):
            n = np.arange(1, int(xx + 20*np.sqrt(xx) + 50))
            # log of x^N / N! for the terms, normalised by P(N > 0) = 1 - e^-x
            log_terms = n*np.log(xx) - gammaln(n + 1) - xx
            _h_table[i] = np.sum(np.exp(log_terms)/n)/(-np.expm1(-xx))
    x = np.asarray(x, dtype=float)
    h = np.exp(np.interp(np.log(np.clip(x, _x_table[0], _x_table[-1])), np.log(_x_table), np.log(_h_table)))
    large = x > _x_table[-1]
    if np.any(large):
        # continuous at the end of the table
        expansion = lambda v: (1 + 1/v + 2/v**2)/v
        h[large] = expansion(x[large])*_h_table[-1]/expansion(_x_table[-1])
    return h


def noise_modulation(d_tot, lam, noise_model='poisson'):
    """
    Modulation of the shape-noise variance by source clustering.

    Parameters
    ----------
    d_tot : array
        Projected source-density modulation D (1 on average).
    lam : array
        Expected number of sources in the pixel without source clustering (same shape as d_tot).
    noise_model : str
        'poisson': h(lam D) / h(lam); 'inverse_density': 1 / D.

    Returns
    -------
    r : np.ndarray
        Variance modulation (0 where D <= 0).
    """
    d_tot = np.asarray(d_tot, dtype=float)
    r = np.zeros(len(d_tot))
    good = d_tot > 0
    if noise_model == 'inverse_density':
        r[good] = 1./d_tot[good]
    elif noise_model == 'poisson':
        lam = np.asarray(lam, dtype=float)
        r[good] = expected_inverse_count(lam[good]*d_tot[good])/expected_inverse_count(lam[good])
    else:
        raise ValueError("noise_model must be 'poisson' or 'inverse_density'")
    return r


def fracdet_map(cats, nside_maps, nside_sub=4096, rot=0, delta_rot=0.):
    """
    Observed fraction of each pixel: share of its NSIDE-nside_sub sub-pixels with at least one source
    (all tomographic bins together). RING ordering, in the rotated frame of the mock.

    Parameters
    ----------
    cats : dict
        Per bin catalogue with 'ra', 'dec'.
    nside_maps : int
        NSIDE of the maps.
    nside_sub : int
        NSIDE of the sub-pixels.
    rot, delta_rot :
        Rotation of the catalogue (as in rotate_and_rebin).

    Returns
    -------
    fracdet : np.ndarray (npix,)
    """
    occupied = np.zeros(hp.nside2npix(nside_sub), dtype=np.float32)
    for tomo in cats.keys():
        pix_ = convert_to_pix_coord(cats[tomo]['ra'], cats[tomo]['dec'], nside=nside_sub*2)
        pix = rotate_and_rebin(pix_, nside_sub, rot, delta_=delta_rot)
        occupied[pix] = 1.
    # average the sub-pixels of each pixel (NESTED ordering keeps them contiguous)
    occupied_nest = hp.reorder(occupied, r2n=True)
    fracdet_nest = occupied_nest.reshape(-1, (nside_sub//nside_maps)**2).mean(axis=1)
    return hp.reorder(fracdet_nest, n2r=True)


def expected_counts_map(pix, nside_maps, fracdet):
    """
    Expected number of sources per pixel without source clustering, lambda_p = nbar * fracdet_p.

    Parameters
    ----------
    pix : array
        Pixel of each galaxy of the bin (rotated frame).
    nside_maps : int
        NSIDE of the maps.
    fracdet : array (npix,)
        Observed fraction of each pixel (fracdet_map).

    Returns
    -------
    lam : np.ndarray (npix,)
        0 in empty pixels.
    """
    counts = np.bincount(pix, minlength=hp.nside2npix(nside_maps))
    nbar = len(pix)/np.sum(fracdet)
    lam = np.zeros(hp.nside2npix(nside_maps))
    good = (fracdet > 0) & (counts > 0)
    lam[good] = np.clip(nbar*fracdet[good], 1e-3, None)
    return lam


# Calibration of the noise rescaling (corr_variance, coeff_kurtosis, A_corr)
# ---------------------------------------------------------------------------
#
# As in the previous model, the rescaled noise maps are matched to the second and fourth moments of
# the randomly rotated data. The three coefficients are calibrated on a grid of (b, s) and fitted per
# bin with low-order polynomials; the fit is stored as JSON:
#   {'format': 'SC_v2_polynomial', 'meta': {...},
#    'corr_variance_fit': [per bin: {'names', 'exponents', 'coeffs', 'rms_residual', 'default'}], ...}
# and each coefficient is  y = sum_k coeffs[k] * prod_j param_j ** exponents[k][j].

SC_KEYS = ['corr_variance', 'coeff_kurtosis', 'A_corr']


def match_noise_moments(x, y):
    """
    Variance and kurtosis rescaling of the mock noise to the data (previous calibration objective).

    Parameters
    ----------
    x : array
        e1^2 + e2^2 of the mock noise maps (per pixel).
    y : array
        e1^2 + e2^2 of the randomly rotated data (per pixel).

    Returns
    -------
    corr_variance, coeff_kurtosis, A_corr : float
    """
    corr_variance = np.mean(x)/np.mean(y)

    def objective(params):
        coeff_kurtosis, A_corr = params
        factor = 1/(A_corr*corr_variance)*(1 + coeff_kurtosis*x)
        second = np.mean(x*factor)/np.mean(y) - 1
        fourth = np.mean((x*factor)**2)/np.mean(y**2) - 1
        return second**2 + fourth**2

    result = minimize(objective, [0.01, 0.995], bounds=[(-1, 1), (0.9, 1.1)])
    return corr_variance, result.x[0], result.x[1]


def calibrate_SC_noise(ngal_glass, zeff_glass, fields, cats, nside_maps, grid, noise_model='poisson',
                       rot=0, delta_rot=0., seed=0):
    """
    Calibrate the noise rescaling for each point of a grid of source-clustering parameters.

    Parameters
    ----------
    ngal_glass, zeff_glass : arrays
        n(z) weights (ntomo, nshell) and effective redshift of each shell.
    fields : dict
        Must contain 'density' (nshell, npix).
    cats : dict
        Per bin data catalogue (ra, dec, e1, e2, w).
    nside_maps : int
        NSIDE of the maps.
    grid : list of dict
        Points {'b': ..., 's': ...}; the same values are used for all bins.
    noise_model : str
        'poisson' or 'inverse_density'.
    rot, delta_rot :
        Rotation of the catalogue.
    seed : int
        Seed of the random rotations.

    Returns
    -------
    calib : list of dict
        {'b', 's', 'corr_variance': [per bin], 'coeff_kurtosis': [...], 'A_corr': [...]}.
    """
    from .lensing import apply_random_rotation

    ntomo = len(ngal_glass)
    npix = hp.nside2npix(nside_maps)
    fracdet = fracdet_map(cats, nside_maps, rot=rot, delta_rot=delta_rot)
    calib = []
    for point in grid:
        bias = shell_bias(ngal_glass, zeff_glass, [point['b']]*ntomo, [point.get('s', 0.)]*ntomo)
        np.random.seed(seed)
        row = {'b': point['b'], 's': point.get('s', 0.), 'corr_variance': [], 'coeff_kurtosis': [], 'A_corr': []}
        for tomo in range(ntomo):
            d_tot = np.zeros(npix)
            for i in range(len(fields['density'])):
                d_tot += ngal_glass[tomo, i]*(1 + bias[tomo, i]*fields['density'][i])

            pix_ = convert_to_pix_coord(cats[tomo]['ra'], cats[tomo]['dec'], nside=nside_maps*2)
            pix = rotate_and_rebin(pix_, nside_maps, rot, delta_=delta_rot)
            lam = expected_counts_map(pix, nside_maps, fracdet)
            d_gal = d_tot[pix]
            noise_scale = d_gal*np.sqrt(noise_modulation(d_gal, lam[pix], noise_model))
            w = cats[tomo]['w']

            n_map = np.bincount(pix, weights=w, minlength=npix)
            n_map_sc = np.bincount(pix, weights=w*d_gal, minlength=npix)
            mask = n_map_sc != 0
            e1, e2 = apply_random_rotation(cats[tomo]['e1']*noise_scale, cats[tomo]['e2']*noise_scale)
            e1_ref, e2_ref = apply_random_rotation(cats[tomo]['e1'], cats[tomo]['e2'])
            e1_map = np.bincount(pix, weights=e1*w, minlength=npix)[mask]/n_map_sc[mask]
            e2_map = np.bincount(pix, weights=e2*w, minlength=npix)[mask]/n_map_sc[mask]
            e1_ref_map = np.bincount(pix, weights=e1_ref*w, minlength=npix)[mask]/n_map[mask]
            e2_ref_map = np.bincount(pix, weights=e2_ref*w, minlength=npix)[mask]/n_map[mask]

            corr_variance, coeff_kurtosis, A_corr = match_noise_moments(e1_map**2 + e2_map**2,
                                                                        e1_ref_map**2 + e2_ref_map**2)
            row['corr_variance'].append(corr_variance)
            row['coeff_kurtosis'].append(coeff_kurtosis)
            row['A_corr'].append(A_corr)
        print(f"b={row['b']:.3f} s={row['s']:.3f}: corr_variance {np.round(row['corr_variance'], 4)}, "
              f"coeff_kurtosis {np.round(row['coeff_kurtosis'], 4)}, A_corr {np.round(row['A_corr'], 4)}")
        calib.append(row)
    return calib


def fit_SC_corrections(calib, names=('b', 's'), degree=2):
    """
    Fit the calibrated noise rescaling per tomographic bin with polynomials in `names`.

    Parameters
    ----------
    calib : list of dict
        calibrate_SC_noise output.
    names : tuple
        Parameters of the polynomial; those that do not vary in the grid are held fixed.
    degree : int
        Total degree of the polynomial.

    Returns
    -------
    SC_corrections : dict
        {'format': 'SC_v2_polynomial', 'meta': {...}, 'corr_variance_fit': [per bin], ...}.
    """
    params = np.array([[row[name] for name in names] for row in calib], dtype=float)
    varying = [j for j in range(len(names)) if np.ptp(params[:, j]) > 0]
    fixed = {names[j]: float(params[0, j]) for j in range(len(names)) if j not in varying}
    used = [names[j] for j in varying]
    exponents = [e for e in itertools.product(range(degree + 1), repeat=len(used)) if sum(e) <= degree]
    exponents = np.array(sorted(exponents, key=lambda e: (sum(e), e)))
    design = np.ones((len(calib), len(exponents)))
    for k, exponent in enumerate(exponents):
        for j, column in enumerate(varying):
            design[:, k] *= params[:, column]**exponent[j]

    SC_corrections = {'format': 'SC_v2_polynomial',
                      'meta': {'names': used, 'degree': degree, 'fixed': fixed, 'npoints': len(calib)}}
    for key in SC_KEYS:
        SC_corrections[key + '_fit'] = []
        for tomo in range(len(calib[0][key])):
            y = np.array([row[key][tomo] for row in calib])
            coeffs = np.linalg.lstsq(design, y, rcond=None)[0]
            residual = y - design @ coeffs
            SC_corrections[key + '_fit'].append({'names': used, 'exponents': exponents.tolist(),
                                                 'coeffs': coeffs.tolist(),
                                                 'rms_residual': float(np.sqrt(np.mean(residual**2))),
                                                 'default': fixed})
    return SC_corrections


def save_SC_calibration(path, SC_corrections, **extra):
    """Write fit_SC_corrections output (plus JSON-serialisable extra entries) to a JSON file."""
    out = dict(SC_corrections)
    out.update(extra)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)


def load_SC_calibration(path):
    """Read a noise-rescaling calibration written by save_SC_calibration."""
    with open(path) as f:
        return json.load(f)


def evaluate_SC_correction(fit, b, s=0.):
    """
    Value of one fitted coefficient at (b, s).

    Parameters
    ----------
    fit : dict or callable
        One entry of SC_corrections['..._fit'] (polynomial dict), or a callable of b (previous format).
    b, s : float
        Source bias and its slope.

    Returns
    -------
    value : float
    """
    if callable(fit):
        return float(fit(b))
    values = dict(fit['default'])
    values.update({'b': b, 's': s})
    value = 0.
    for coeff, exponent in zip(fit['coeffs'], fit['exponents']):
        term = coeff
        for name, power in zip(fit['names'], exponent):
            term *= values.get(name, 0.)**power
        value += term
    return value
