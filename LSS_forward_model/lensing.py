import numpy as np
from typing import Sequence, Tuple
import pandas as pd  # only for typing/self; safe to remove if not needed
import pyccl as ccl  
from .maps import F_nla,F_TT,convert_to_pix_coord,rotate_and_rebin
import healpy as hp
from .source_clustering import shell_bias, noise_modulation, fracdet_map, expected_counts_map, evaluate_SC_correction



def addSourceEllipticity(
    self: "pd.DataFrame",
    es: "pd.DataFrame | np.ndarray",
    es_colnames: Sequence[str] = ("e1", "e2"),
    rs_correction: bool = True,
    inplace: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    Compose intrinsic source ellipticities with shear to obtain observed ellipticities.

    Uses the complex reduced-shear mapping:
        e = (e_s + g) / (1 + g* · e_s)          (exact, reduced-shear form)
    If `rs_correction=False`, uses the linearized approximation:
        e ≈ e_s + g

    Parameters
    ----------
    self : pandas.DataFrame
        Table with per-object shear components in columns "shear1", "shear2".
    es : DataFrame or array-like
        Intrinsic ellipticities for each row in `self`. If a DataFrame, it must
        have columns named by `es_colnames`; if a NumPy structured/record array,
        fields with those names must exist.
    es_colnames : (str, str), optional
        Names of the intrinsic ellipticity columns (default: ("e1", "e2")).
    rs_correction : bool, optional
        If True, apply the exact reduced-shear denominator (default True).
        If False, use the linear approximation e = e_s + g.
    inplace : bool, optional
        If True, overwrite self["shear1"], self["shear2"]; otherwise return (e1, e2).

    Returns
    -------
    (e1, e2) : tuple of ndarray, or None
        Observed ellipticity components. Returns None if `inplace=True`.

    Notes
    -----
    - Conventions: e = e1 + i e2, g = g1 + i g2, and the denominator uses g* (complex conjugate).
    - Assumes `len(self) == len(es)`.
    """
    # Safety check
    assert len(self) == len(es), "Length of `es` must match number of rows in `self`."

    # Build complex intrinsic ellipticity and shear
    if isinstance(es, pd.DataFrame):
        e1s = es[es_colnames[0]].to_numpy()
        e2s = es[es_colnames[1]].to_numpy()
    else:
        # array-like (could be structured array or 2D array)
        try:
            e1s = np.asarray(es[es_colnames[0]])
            e2s = np.asarray(es[es_colnames[1]])
        except Exception:
            es_arr = np.asarray(es)
            if es_arr.ndim == 2 and es_arr.shape[1] >= 2:
                e1s, e2s = es_arr[:, 0], es_arr[:, 1]
            else:
                raise ValueError("`es` must provide two components matching `es_colnames`.")
    es_c = e1s + 1j * e2s

    g = np.asarray(self["shear1"]) + 1j * np.asarray(self["shear2"])

    # Compose
    e = es_c + g
    if rs_correction:
        e = (es_c + g) / (1.0 + np.conjugate(g) * es_c)

    if inplace:
        self["shear1"] = e.real
        self["shear2"] = e.imag
        return None
    else:
        return e.real, e.imag


def apply_random_rotation(e1_in: np.ndarray, e2_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply independent random spin-2 rotations to ellipticities (e1, e2).

    Draws a random phase θ ~ U(0, 2π) per object and rotates:
        [e1']   [ cos θ   sin θ] [e1]
        [e2'] = [-sin θ   cos θ] [e2]
    (θ here already includes the spin-2 factor; equivalently use φ and set θ=2φ.)

    Parameters
    ----------
    e1_in, e2_in : array-like
        Input ellipticity components of equal length.

    Returns
    -------
    (e1_out, e2_out) : tuple of ndarray
        Rotated ellipticity components.

    Notes
    -----
    - For reproducibility, consider controlling the RNG outside this function
      (e.g., pass precomputed angles or set NumPy’s seed before calling).
    """
    e1_in = np.asarray(e1_in)
    e2_in = np.asarray(e2_in)
    if e1_in.shape != e2_in.shape:
        raise ValueError("e1_in and e2_in must have the same shape.")

    # Random spin-2 phase per object
    rot_angle = np.random.random(size=e1_in.shape) * 2.0 * np.pi
    c = np.cos(rot_angle)
    s = np.sin(rot_angle)

    e1_out = e1_in * c + e2_in * s
    e2_out = -e1_in * s + e2_in * c
    return e1_out, e2_out


def make_WL_sample(ngal_glass, zeff_glass, cosmo_bundle, sims_parameters, nside_maps, fields, cats_Euclid, SC_corrections =None,do_catalog = False, include_SC = True, compact_savings = False, *,
                   noise_model='poisson', IA_model='deltaNLA', halo_IA=None, subtract_IA_noise=True, ia_noise_nside=64,
                   ia_numerator=None):
    """
    Make survey signal/noise maps (mock shear maps) of each tomographic bin.

    Signal:
        g_t(p) = sum_i n_ti [w_sc,ti(p) gamma_i(p) + w_ta,ti(p) IA_i(p)] / D_t(p)      (IA_model 'NLA', 'deltaNLA')
        g_t(p) = sum_i n_ti w_sc,ti(p) gamma_i(p) / D_t(p) + I_t(p)                    (IA_model 'halo')
        D_t(p) = sum_i n_ti w_sc,ti(p)
    with the source-clustering weight w_sc = 1 + b_t [1 + s_t (z_i - zbar_t)] delta_i (bias_sc,
    bias_sc_slope) and w_ta = 1 + bias_TA_t delta_i ('deltaNLA') or 1 ('NLA').  The halo IA field I_t
    (sum of the model shapes over the mean number of survey sources, halo_ia.make_halo_IA_numerator) is a
    separate term, not divided by the source-clustering modulation D_t, as in its validation.

    Noise: the data shapes are randomly rotated and, per galaxy, scaled so that the noise variance of
    each pixel is modulated by h(lam D) / h(lam) (noise_model='poisson'), h(x) = E[1/N | N > 0] for a
    Poisson N and lam the expected number of sources of the pixel (mean density x fracdet), or by 1/D
    (noise_model='inverse_density', previous model). The noise maps are then rescaled with the
    calibrated variance/kurtosis corrections (SC_corrections). With IA_model='halo' the white variance
    of the halo IA map, already contained in the data shapes, is removed from the noise.

    Parameters
    ----------
    ngal_glass : array (ntomo, nshell)
        n(z) weights of each shell.
    zeff_glass : array (nshell,)
        Effective redshift of each shell.
    cosmo_bundle : dict
        Cosmology (cosmo_pyccl) for the NLA amplitudes.
    sims_parameters : dict
        bias_sc (per bin), bias_sc_slope (per bin, optional, default 0), dm (1 + m, per bin), rot,
        delta_rot; A_IA, eta_IA, A_IA2, bias_TA for the NLA models.
    nside_maps : int
        NSIDE of the maps.
    fields : dict
        Lensing fields per shell: 'gamma', 'density', 'IA_shear' (and 'IA_shear_TT').
    cats_Euclid : dict
        Per bin data catalogue (ra, dec, e1, e2, w).
    SC_corrections : dict
        Noise rescaling: source_clustering.load_SC_calibration output (for noise_model='poisson'), or the
        previous dict of callables of b (only with noise_model='inverse_density').
    do_catalog : bool
        Also return a mock catalogue.
    include_SC : bool
        If False, no source clustering (b = 0) and no noise rescaling.
    compact_savings : bool
        Store only the observed pixels.
    noise_model : str
        'poisson' (default) or 'inverse_density'.
    IA_model : str
        'deltaNLA' (default), 'NLA' or 'halo'.
    halo_IA : dict
        halo_ia.make_halo_IA_numerator output ('numerator', 'numerator_sq'); only for IA_model='halo'.
    subtract_IA_noise : bool
        Remove the halo IA white variance from the noise (IA_model='halo').
    ia_noise_nside : int or None
        NSIDE of the superpixels on which the expected halo IA self term is computed (None: per pixel).
    ia_numerator : complex array (ntomo, npix), optional
        Previous interface for an external IA numerator: added to the signal numerator and divided by D
        (as before), without noise subtraction.

    Returns
    -------
    maps_sim : dict
        Per bin: g1_map, g2_map, e1, e2 (signal + noise), e1n, e2n (noise only); + e1r_map0_ref,
        e2r_map0_ref, var_ (and ia_noise_factor for the halo model), or idx_ with compact_savings.
    cats_sim : dict or None
        Per bin catalogue (ra, dec, e1, e2, w) if do_catalog.
    """
    ntomo = len(ngal_glass)
    npix = hp.nside2npix(nside_maps)
    sims_parameters.setdefault('rot', 0)
    sims_parameters.setdefault('delta_rot', 0)

    ia_numerator_in_ratio = ia_numerator is not None
    if ia_numerator is not None:
        IA_model = 'halo'
        halo_IA = {'numerator': np.asarray(ia_numerator, dtype=np.complex128)}
        subtract_IA_noise = False
    if IA_model not in ['NLA', 'deltaNLA', 'halo']:
        raise ValueError("IA_model must be 'NLA', 'deltaNLA' or 'halo'")
    if IA_model == 'halo' and halo_IA is None:
        raise ValueError("IA_model='halo' needs halo_IA (halo_ia.make_halo_IA_numerator)")

    # noise rescaling (variance / kurtosis) ---------------------------------------------------------
    if include_SC:
        if SC_corrections is None:
            raise ValueError('include_SC=True needs SC_corrections')
        if noise_model == 'poisson' and SC_corrections.get('format') != 'SC_v2_polynomial':
            raise ValueError("these SC_corrections were calibrated for noise_model='inverse_density'; "
                             "pass a source_clustering.load_SC_calibration file")
        bias_sc_slope = sims_parameters.get('bias_sc_slope', np.zeros(ntomo))
        corr_variance_array = [evaluate_SC_correction(SC_corrections['corr_variance_fit'][tomo], sims_parameters['bias_sc'][tomo], bias_sc_slope[tomo]) for tomo in range(ntomo)]
        coeff_kurtosis_array = [evaluate_SC_correction(SC_corrections['coeff_kurtosis_fit'][tomo], sims_parameters['bias_sc'][tomo], bias_sc_slope[tomo]) for tomo in range(ntomo)]
        A_corr_array = [evaluate_SC_correction(SC_corrections['A_corr_fit'][tomo], sims_parameters['bias_sc'][tomo], bias_sc_slope[tomo]) for tomo in range(ntomo)]
        bias = shell_bias(ngal_glass, zeff_glass, sims_parameters['bias_sc'], bias_sc_slope)
    else:
        corr_variance_array = np.ones(ntomo)
        coeff_kurtosis_array = np.zeros(ntomo)
        A_corr_array = np.ones(ntomo)
        sims_parameters['bias_sc'] = np.zeros(ntomo)
        bias = np.zeros((ntomo, len(zeff_glass)))

    # signal --------------------------------------------------------------------------------------
    # note: I added a -sign to gamma to match data conventions later
    C1 = 5e-14
    rho_crit0_h2 = ccl.physical_constants.RHO_CRITICAL
    rho_c1 = C1 * rho_crit0_h2

    g1_tot = np.zeros((ntomo, npix))
    g2_tot = np.zeros((ntomo, npix))
    d_tot = np.zeros((ntomo, npix))
    for tomo in range(ntomo):
        for i in range(len(fields['gamma'])):
            w_sc = 1 + bias[tomo, i] * fields['density'][i]
            g1_tot[tomo] += ngal_glass[tomo, i] * w_sc * (-fields['gamma'][i][0].real)
            g2_tot[tomo] += ngal_glass[tomo, i] * w_sc * (-fields['gamma'][i][0].imag)
            d_tot[tomo] += ngal_glass[tomo, i] * w_sc

            if IA_model in ['NLA', 'deltaNLA']:
                IA_f = F_nla(z=zeff_glass[i],om0=sims_parameters['Omega_m'],A_ia=sims_parameters['A_IA'], rho_c1=rho_c1, eta=sims_parameters['eta_IA'], z0=0.67,cosmo=cosmo_bundle['cosmo_pyccl'])
                IA_f2 = F_TT(z=zeff_glass[i],om0=sims_parameters['Omega_m'],A2=sims_parameters.get('A_IA2', 0.), rho_c1=rho_c1, eta=sims_parameters['eta_IA'], z0=0.67,cosmo=cosmo_bundle['cosmo_pyccl'])
                w_ta = 1 + sims_parameters['bias_TA'][tomo] * fields['density'][i] if IA_model == 'deltaNLA' else 1.
                g1_IA = -fields['IA_shear'][i][0].real * IA_f
                g2_IA = -fields['IA_shear'][i][0].imag * IA_f
                if 'IA_shear_TT' in fields:
                    g1_IA = g1_IA - fields['IA_shear_TT'][i][0].real * IA_f2
                    g2_IA = g2_IA - fields['IA_shear_TT'][i][0].imag * IA_f2
                g1_tot[tomo] += ngal_glass[tomo, i] * w_ta * g1_IA
                g2_tot[tomo] += ngal_glass[tomo, i] * w_ta * g2_IA

        if ia_numerator_in_ratio:
            # previous interface: external IA numerator inside the source-clustering ratio
            g1_tot[tomo] += halo_IA['numerator'][tomo].real
            g2_tot[tomo] += halo_IA['numerator'][tomo].imag


    # observed fraction of each pixel, for the expected source counts of the Poisson noise model
    fracdet = None
    if include_SC and noise_model == 'poisson':
        fracdet = fracdet_map(cats_Euclid, nside_maps, rot=sims_parameters['rot'], delta_rot=sims_parameters['delta_rot'])

    maps_sim = dict()
    cats_sim = dict() if do_catalog else None
    for tomo in range(ntomo):
        w = cats_Euclid[tomo]['w']
        pix_ = convert_to_pix_coord(cats_Euclid[tomo]['ra'],cats_Euclid[tomo]['dec'], nside=nside_maps*2)
        pix = rotate_and_rebin(pix_, nside_maps, sims_parameters['rot'], delta_=sims_parameters['delta_rot'])
        d_gal = d_tot[tomo][pix]

        # source clustering term ~ per-galaxy scale of the shape noise. The pixel maps are normalised by
        # sum(w D), so a scale D sqrt(r) gives a noise variance modulated by r in the pixel
        if noise_model == 'inverse_density' or not include_SC:
            noise_scale = np.sqrt(d_gal)
        else:
            lam = expected_counts_map(pix, nside_maps, fracdet)
            noise_scale = d_gal * np.sqrt(noise_modulation(d_gal, lam[pix], noise_model))

        # three independent rotations: noise map, noise added to the signal, data noise (reference)
        es1,es2 = apply_random_rotation(cats_Euclid[tomo]['e1']*noise_scale, cats_Euclid[tomo]['e2']*noise_scale)
        es1_ref,es2_ref = apply_random_rotation(cats_Euclid[tomo]['e1'], cats_Euclid[tomo]['e2'])
        es1a,es2a = apply_random_rotation(cats_Euclid[tomo]['e1']*noise_scale, cats_Euclid[tomo]['e2']*noise_scale)

        n_map = np.bincount(pix, weights=w, minlength=npix)
        n_map_sc = np.bincount(pix, weights=w*d_gal, minlength=npix)
        mask_sims = n_map_sc != 0.

        e1r_map = np.zeros(npix)
        e2r_map = np.zeros(npix)
        e1r_map0 = np.zeros(npix)
        e2r_map0 = np.zeros(npix)
        e1r_map0_ref = np.zeros(npix)
        e2r_map0_ref = np.zeros(npix)
        g1_map = np.zeros(npix)
        g2_map = np.zeros(npix)
        e1r_map[mask_sims] = np.bincount(pix, weights=es1*w, minlength=npix)[mask_sims]/n_map_sc[mask_sims]
        e2r_map[mask_sims] = np.bincount(pix, weights=es2*w, minlength=npix)[mask_sims]/n_map_sc[mask_sims]
        e1r_map0[mask_sims] = np.bincount(pix, weights=es1a*w, minlength=npix)[mask_sims]/n_map_sc[mask_sims]
        e2r_map0[mask_sims] = np.bincount(pix, weights=es2a*w, minlength=npix)[mask_sims]/n_map_sc[mask_sims]
        e1r_map0_ref[mask_sims] = np.bincount(pix, weights=es1_ref*w, minlength=npix)[mask_sims]/n_map[mask_sims]
        e2r_map0_ref[mask_sims] = np.bincount(pix, weights=es2_ref*w, minlength=npix)[mask_sims]/n_map[mask_sims]
        g1_map[mask_sims] = np.bincount(pix, weights=g1_tot[tomo][pix]*w, minlength=npix)[mask_sims]/n_map_sc[mask_sims]
        g2_map[mask_sims] = np.bincount(pix, weights=g2_tot[tomo][pix]*w, minlength=npix)[mask_sims]/n_map_sc[mask_sims]

        # halo IA: separate term (the model sources carry their own clustering), not divided by D
        if IA_model == 'halo' and not ia_numerator_in_ratio:
            g1_map[mask_sims] += halo_IA['numerator'][tomo].real[mask_sims]
            g2_map[mask_sims] += halo_IA['numerator'][tomo].imag[mask_sims]

        # variance / kurtosis rescaling of the noise
        var_ = e1r_map0_ref**2 + e2r_map0_ref**2
        noise_factor = 1/(np.sqrt(A_corr_array[tomo]*corr_variance_array[tomo])) * np.sqrt((1+coeff_kurtosis_array[tomo]*var_))

        # halo IA: remove its white variance (already in the data shapes) from the noise
        #   factor(p) = sqrt(1 - var_IA(p) / var_noise(p)),  var_IA = 0.5 <numerator_sq>
        # <numerator_sq> is the expected IA self term: numerator_sq averaged on NSIDE ia_noise_nside
        # superpixels (at low z most pixels hold no model source), normalised to the realised total over
        # the footprint.
        ia_noise_factor = np.ones(npix)
        if IA_model == 'halo' and subtract_IA_noise:
            e_sq = cats_Euclid[tomo]['e1']**2 + cats_Euclid[tomo]['e2']**2
            var_noise = np.zeros(npix)
            var_noise[mask_sims] = 0.5 * noise_factor[mask_sims]**2 * np.bincount(pix, weights=(w*noise_scale)**2*e_sq, minlength=npix)[mask_sims]/n_map_sc[mask_sims]**2
            numerator_sq = halo_IA['numerator_sq'][tomo]
            if ia_noise_nside is not None:
                expected = hp.ud_grade(hp.ud_grade(numerator_sq, ia_noise_nside), nside_maps)
                if np.sum(expected[mask_sims]) > 0:
                    numerator_sq = expected * np.sum(numerator_sq[mask_sims])/np.sum(expected[mask_sims])
            var_IA = 0.5 * numerator_sq
            ratio = np.zeros(npix)
            ratio[mask_sims] = var_IA[mask_sims]/var_noise[mask_sims]
            ia_noise_factor[mask_sims] = np.sqrt(np.clip(1 - ratio[mask_sims], 0., 1.))
            if np.mean(ratio[mask_sims] > 1) > 0.001:
                print(f'bin {tomo}: IA variance > noise variance in {100*np.mean(ratio[mask_sims] > 1):.2f}% of the pixels (noise set to 0 there)')
        noise_factor = noise_factor * ia_noise_factor

        e1r_map *= noise_factor
        e2r_map *= noise_factor
        e1r_map0 *= noise_factor
        e2r_map0 *= noise_factor

        e1_ = g1_map * sims_parameters['dm'][tomo] + e1r_map0
        e2_ = g2_map * sims_parameters['dm'][tomo] + e2r_map0
        if compact_savings:
            maps_sim[tomo] = {'g1_map':g1_map[mask_sims],'g2_map':g2_map[mask_sims],'e1':e1_[mask_sims],'e2':e2_[mask_sims],
                              'e1n':e1r_map[mask_sims],'e2n':e2r_map[mask_sims],'idx_':np.arange(npix)[mask_sims]}
        else:
            maps_sim[tomo] = {'g1_map':g1_map,'g2_map':g2_map,'e1':e1_,'e2':e2_,'e1n':e1r_map,'e2n':e2r_map,
                              'e1r_map0_ref':e1r_map0_ref,'e2r_map0_ref':e2r_map0_ref,'var_':var_}
            if IA_model == 'halo':
                maps_sim[tomo]['ia_noise_factor'] = ia_noise_factor

        if do_catalog:
            # per galaxy: signal g_tot / D (+ halo IA), noise rescaled as the pixel maps
            g1_gal = g1_tot[tomo][pix]/d_gal
            g2_gal = g2_tot[tomo][pix]/d_gal
            if IA_model == 'halo' and not ia_numerator_in_ratio:
                g1_gal = g1_gal + halo_IA['numerator'][tomo].real[pix]
                g2_gal = g2_gal + halo_IA['numerator'][tomo].imag[pix]
            e1_SC = sims_parameters['dm'][tomo] * g1_gal + es1a * noise_factor[pix]/d_gal
            e2_SC = sims_parameters['dm'][tomo] * g2_gal + es2a * noise_factor[pix]/d_gal
            cats_sim[tomo] = {'ra':cats_Euclid[tomo]['ra'],'dec':cats_Euclid[tomo]['dec'],'e1':e1_SC,'e2':e2_SC,'w':w}

    return maps_sim, cats_sim









import h5py

def save_maps_h5(path, d, compression="gzip", compression_opts=4):
    with h5py.File(path, "w") as f:
        for key, val in d.items():
            if isinstance(key, int):
                grp = f.create_group(str(key))
                for subkey, arr in val.items():
                    grp.create_dataset(
                        subkey,
                        data=arr,
                        compression=compression,
                        compression_opts=compression_opts if compression == "gzip" else None,
                    )
            elif key == "sims_parameters":
                meta = f.create_group("sims_parameters")
                for mk, mv in val.items():
                    if isinstance(mv, np.ndarray):
                        meta.create_dataset(
                            mk,
                            data=mv,
                            compression=compression,
                            compression_opts=compression_opts if compression == "gzip" else None,
                        )
                    elif isinstance(mv, (list, tuple)):
                        meta.create_dataset(mk, data=np.asarray(mv))
                    elif isinstance(mv, (str, bytes)):
                        meta.attrs[mk] = mv
                    else:
                        meta.attrs[mk] = mv
            else:
                raise ValueError(f"Unexpected key: {key}")

def load_maps_h5(path):
    d = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            if key.isdigit():
                ikey = int(key)
                d[ikey] = {}
                for subkey in f[key].keys():
                    d[ikey][subkey] = f[key][subkey][()]
            elif key == "sims_parameters":
                d["sims_parameters"] = {}
                for mk in f[key].keys():
                    d["sims_parameters"][mk] = f[key][mk][()]
                for mk, mv in f[key].attrs.items():
                    d["sims_parameters"][mk] = mv
    return d
