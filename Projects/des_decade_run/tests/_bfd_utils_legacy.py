from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import healpy as hp
import pyccl as ccl
from LSS_forward_model.maps import F_nla, F_TT, convert_to_pix_coord, rotate_and_rebin


complexTypeFor = {
    np.float32: np.complex64,
    np.dtype("float32"): np.complex64,
    np.dtype("complex64"): np.complex64,
    np.dtype(">f4"): np.complex64,
    np.float64: np.complex128,
    np.dtype("float64"): np.complex128,
    np.dtype("complex128"): np.complex128,
    np.dtype(">f8"): np.complex128,
}


class Pqr:
    HAS_MU = True
    D0 = 0
    DG1 = 1
    DG2 = 2
    DMU = 3
    DG1DG1 = 4
    DG1DG2 = 5
    DG2DG2 = 6
    DG1DMU = 7
    DG2DMU = 8
    DMUDMU = 9
    NPQR = 10
    QSlice = slice(DG1, DG1DG1)
    RSlice = slice(DG1DG1, NPQR)
    RShape = (3, 3)
    BUILDR = ([DG1DG1, DG1DG2, DG1DMU, DG1DG2, DG2DG2, DG2DMU, DG1DMU, DG2DMU, DMUDMU],)
    GETR = ([0, 0, 1, 0, 1, 2], [0, 1, 1, 2, 2, 2])


class PqrNoMu:
    HAS_MU = False
    D0 = 0
    DG1 = 1
    DG2 = 2
    DG1DG1 = 3
    DG1DG2 = 4
    DG2DG2 = 5
    NPQR = 6
    QSlice = slice(DG1, DG1DG1)
    RSlice = slice(DG1DG1, NPQR)
    RShape = (2, 2)
    BUILDR = ([DG1DG1, DG1DG2, DG1DG2, DG2DG2],)
    GETR = ([0, 0, 1], [0, 1, 1])


def pickPqr(a, axis=-1):
    if a.shape[axis] == Pqr.NPQR:
        return Pqr
    if a.shape[axis] == PqrNoMu.NPQR:
        return PqrNoMu
    raise ValueError(f"PQR axis has invalid length {a.shape[axis]}")


def stripMuPqr(pqr, axis=-1):
    s = [slice(None)] * pqr.ndim
    use = [0] * PqrNoMu.NPQR
    use[PqrNoMu.D0] = Pqr.D0
    use[PqrNoMu.DG1] = Pqr.DG1
    use[PqrNoMu.DG2] = Pqr.DG2
    use[PqrNoMu.DG1DG1] = Pqr.DG1DG1
    use[PqrNoMu.DG1DG2] = Pqr.DG1DG2
    use[PqrNoMu.DG2DG2] = Pqr.DG2DG2
    s[axis] = use
    return np.array(pqr[tuple(s)])


def splitPqr(pqr, axis=-1):
    pp = pickPqr(pqr, axis=axis)
    src = np.moveaxis(pqr, axis, -1)
    p = np.array(src[..., pp.D0])
    q = np.array(src[..., pp.QSlice])
    shape = src.shape[:-1] + pp.RShape
    r = np.array(src[..., pp.BUILDR].reshape(shape))
    return p, q, r


def packPqr(p, q, r, axis=-1):
    if q.shape[-1] == 3:
        pp = Pqr
    elif q.shape[-1] == 2:
        pp = PqrNoMu
    else:
        raise ValueError(f"Bad Q-vector length {q.shape[-1]}")

    pqr = np.zeros(q.shape[:-1] + (pp.NPQR,), dtype=q.dtype)
    pqr[..., pp.D0] = p
    pqr[..., pp.QSlice] = q
    pqr[..., pp.RSlice] = r[(slice(None),) * (q.ndim - 1) + pp.GETR]
    return np.moveaxis(pqr, -1, axis)


def logPqr(pqr, axis=-1):
    p, q, r = splitPqr(pqr, axis=axis)
    good = p > 0.0
    q = q / p[..., np.newaxis]
    r = r / p[..., np.newaxis, np.newaxis]
    r = r - q[..., :, np.newaxis] * q[..., np.newaxis, :]
    p = np.log(p)
    out = packPqr(p, q, r, axis=-1)
    out = np.where(good[..., np.newaxis], out, 0.0)
    return np.moveaxis(out, -1, axis)


def rotatePqr(theta, pqr, axis=-1):
    pp = pickPqr(pqr, axis=axis)
    c2 = np.cos(2 * theta)
    s2 = np.sin(2 * theta)

    out = np.moveaxis(pqr.copy(), axis, -1)
    tmp = out[..., pp.DG1] * c2 - out[..., pp.DG2] * s2
    out[..., pp.DG2] = out[..., pp.DG2] * c2 + out[..., pp.DG1] * s2
    out[..., pp.DG1] = tmp

    if pp.HAS_MU:
        tmp = out[..., pp.DG1DMU] * c2 - out[..., pp.DG2DMU] * s2
        out[..., pp.DG2DMU] = out[..., pp.DG2DMU] * c2 + out[..., pp.DG1DMU] * s2
        out[..., pp.DG1DMU] = tmp

    c4 = c2 * c2 - s2 * s2
    s4 = 2 * c2 * s2
    e0 = 0.5 * (out[..., pp.DG1DG1] + out[..., pp.DG2DG2])
    e1 = 0.5 * (out[..., pp.DG1DG1] - out[..., pp.DG2DG2])
    e2 = out[..., pp.DG1DG2]
    e1, e2 = e1 * c4 - e2 * s4, e1 * s4 + e2 * c4
    out[..., pp.DG1DG1] = e0 + e1
    out[..., pp.DG1DG2] = e2
    out[..., pp.DG2DG2] = e0 - e1

    return np.moveaxis(out, -1, axis)


def convert_to_pix_coord(ra, dec, nside=1024):
    theta = (90.0 - dec) * np.pi / 180.0
    phi = ra * np.pi / 180.0
    return hp.ang2pix(nside, theta, phi, nest=False)


def prepare_bfd_pqr_ordered(
    cat,
    random_rotate=False,
    rotate_selection=None,
    rng=None,
    return_selection_mask=False,
):
    if rng is None:
        rng = np.random.default_rng()

    if rotate_selection is None:
        rotate_selection = random_rotate

    nobj = len(cat["ra"])
    pqr_out = np.zeros((nobj, 6))

    is_real = cat["selection_term"] == 0
    is_selection = ~is_real

    if np.any(is_real):
        pqr0 = cat["pqr10_prelog"][is_real]

        if random_rotate:
            angles = rng.uniform(0.0, 2.0 * np.pi, len(pqr0))
            pqr0 = rotatePqr(angles, pqr0)

        pqr_out[is_real] = stripMuPqr(logPqr(pqr0))

    if np.any(is_selection):
        pqr1 = copy.deepcopy(cat["pqr"][is_selection])

        # logPQR -> raw PQR
        pqr1[:, 0] = np.exp(pqr1[:, 0])
        p_, q_, r_ = splitPqr(pqr1)
        r_ = r_ + q_[..., :, np.newaxis] * q_[..., np.newaxis, :]
        q_ = q_ * p_[..., np.newaxis]
        r_ = r_ * p_[..., np.newaxis, np.newaxis]
        pqr1 = packPqr(p_, q_, r_, axis=-1)

        if rotate_selection:
            angles = rng.uniform(0.0, 2.0 * np.pi, pqr1.shape[0])
            pqr1 = rotatePqr(angles, pqr1)

        pqr_out[is_selection] = logPqr(pqr1)

    if return_selection_mask:
        return pqr_out, is_selection

    return pqr_out


def make_bfd_map(
    cat,
    nside=1024,
    random_rotate=False,
    rotate_selection=None,
    gamma1=None,
    gamma2=None,
    q_noise_scale_map=None,
    decompose_shear=False,
    rng=None,
):
    """
    Build a BFD shear map from a BFD catalog.

    Real rows:
        selection_term == 0
        use pqr10_prelog
        optionally random-rotate
        logPqr
        stripMuPqr
        optionally inject simulated shear via q -> q + R gamma

    Selection rows:
        selection_term != 0
        use pqr
        convert logPqr -> raw PQR
        optionally random-rotate
        logPqr
        never inject simulated shear

    q_noise_scale_map:
        optional per-pixel multiplicative rescaling of q1/q2 only.
        This rescales the stochastic/random-rotated BFD score part.
        It is applied before injecting R gamma.
    """

    if rng is None:
        rng = np.random.default_rng()

    if rotate_selection is None:
        rotate_selection = random_rotate

    ra_all = np.asarray(cat["ra"])
    dec_all = np.asarray(cat["dec"])
    selterm = np.asarray(cat["selection_term"])

    nobj = len(ra_all)
    npix = hp.nside2npix(nside)

    is_real_all = selterm == 0
    is_sel_all = ~is_real_all

    pqr_use = np.zeros((nobj, 6), dtype=float)

    # --------------------------------------------------
    # Real detected-object rows
    # --------------------------------------------------
    if np.any(is_real_all):

        pqr_real = np.asarray(cat["pqr10_prelog"][is_real_all]).copy()

        if random_rotate:
            angles = rng.uniform(0.0, 2.0 * np.pi, pqr_real.shape[0])
            pqr_real = rotatePqr(angles, pqr_real)

        pqr_real = logPqr(pqr_real)
        pqr_real = stripMuPqr(pqr_real)

        pqr_use[is_real_all] = pqr_real

    # --------------------------------------------------
    # Selection rows
    # Stored pqr is already logPqr, so convert back to raw PQR,
    # optionally rotate, then log again.
    # --------------------------------------------------
    if np.any(is_sel_all):

        pqr_sel = np.asarray(cat["pqr"][is_sel_all]).copy()

        # logPqr -> raw PQR
        pqr_sel[:, 0] = np.exp(pqr_sel[:, 0])

        p_sel, q_sel, r_sel = splitPqr(pqr_sel)

        r_sel = r_sel + q_sel[..., :, np.newaxis] * q_sel[..., np.newaxis, :]
        q_sel = q_sel * p_sel[..., np.newaxis]
        r_sel = r_sel * p_sel[..., np.newaxis, np.newaxis]

        pqr_sel = packPqr(p_sel, q_sel, r_sel, axis=-1)

        if rotate_selection:
            angles = rng.uniform(0.0, 2.0 * np.pi, pqr_sel.shape[0])
            pqr_sel = rotatePqr(angles, pqr_sel)

        pqr_sel = logPqr(pqr_sel)

        pqr_use[is_sel_all] = pqr_sel

    # --------------------------------------------------
    # Split PQR and clean rows
    # --------------------------------------------------
    p, q, r = splitPqr(pqr_use)

    good_pqr = (
        np.isfinite(p)
        & np.all(np.isfinite(q), axis=1)
        & np.all(np.isfinite(r), axis=(1, 2))
    )

    ra = ra_all[good_pqr]
    dec = dec_all[good_pqr]
    p = p[good_pqr]
    q = q[good_pqr]
    r = r[good_pqr]

    is_real = is_real_all[good_pqr]
    is_sel = is_sel_all[good_pqr]

    pix = convert_to_pix_coord(ra, dec, nside=nside)

    # --------------------------------------------------
    # SC/noise rescaling.
    # This rescales q only. It does not rescale R.
    # It is applied before injecting deterministic R gamma.
    # --------------------------------------------------
    if q_noise_scale_map is not None:

        q_noise_scale_map = np.asarray(q_noise_scale_map, dtype=float)

        q_scale = q_noise_scale_map[pix]
        q_scale = np.asarray(q_scale, dtype=float)

        q_scale[~np.isfinite(q_scale)] = 0.0

        q[:, 0] *= q_scale
        q[:, 1] *= q_scale

    # --------------------------------------------------
    # Prepare simulated-shear injection.
    # The deterministic R gamma term is linear, so keep it separate
    # when requested and combine m cases outside the catalog pass.
    # --------------------------------------------------
    have_gamma = gamma1 is not None and gamma2 is not None
    if have_gamma:
        gamma1 = np.asarray(gamma1, dtype=float)
        gamma2 = np.asarray(gamma2, dtype=float)

    # --------------------------------------------------
    # Pixel binning.
    # Use sums, not averages.
    # --------------------------------------------------
    unique_pix, idx_rep = np.unique(pix, return_inverse=True)

    n_source = np.zeros(npix, dtype=float)
    n_source[unique_pix] += np.bincount(
        idx_rep,
        weights=is_real.astype(float),
        minlength=len(unique_pix),
    )

    n_terms = np.zeros(npix, dtype=float)
    n_terms[unique_pix] += np.bincount(
        idx_rep,
        weights=np.ones(len(pix)),
        minlength=len(unique_pix),
    )

    p_map = np.zeros(npix, dtype=float)
    q1_map = np.zeros(npix, dtype=float)
    q2_map = np.zeros(npix, dtype=float)
    r11_map = np.zeros(npix, dtype=float)
    r22_map = np.zeros(npix, dtype=float)
    r12_map = np.zeros(npix, dtype=float)

    p_map[unique_pix] += np.bincount(
        idx_rep,
        weights=p,
        minlength=len(unique_pix),
    )

    q1_map[unique_pix] += np.bincount(
        idx_rep,
        weights=q[:, 0],
        minlength=len(unique_pix),
    )

    q2_map[unique_pix] += np.bincount(
        idx_rep,
        weights=q[:, 1],
        minlength=len(unique_pix),
    )

    r11_map[unique_pix] += np.bincount(
        idx_rep,
        weights=r[:, 0, 0],
        minlength=len(unique_pix),
    )

    r22_map[unique_pix] += np.bincount(
        idx_rep,
        weights=r[:, 1, 1],
        minlength=len(unique_pix),
    )

    r12_map[unique_pix] += np.bincount(
        idx_rep,
        weights=r[:, 0, 1],
        minlength=len(unique_pix),
    )

    q1_base_map = q1_map.copy() if decompose_shear and have_gamma else None
    q2_base_map = q2_map.copy() if decompose_shear and have_gamma else None

    if have_gamma:
        g1_obj = gamma1[pix]
        g2_obj = gamma2[pix]
        real = is_real.astype(float)

        q1_inj = np.zeros(npix, dtype=float)
        q2_inj = np.zeros(npix, dtype=float)

        q1_inj[unique_pix] += np.bincount(
            idx_rep,
            weights=real * (r[:, 0, 0] * g1_obj + r[:, 0, 1] * g2_obj),
            minlength=len(unique_pix),
        )
        q2_inj[unique_pix] += np.bincount(
            idx_rep,
            weights=real * (r[:, 1, 0] * g1_obj + r[:, 1, 1] * g2_obj),
            minlength=len(unique_pix),
        )

        q1_map += q1_inj
        q2_map += q2_inj

        del g1_obj, g2_obj, real, q1_inj, q2_inj

    good_pix = n_terms > 0

    def estimate_from_q(q1_in, q2_in):
        # --------------------------------------------------
        # Per-pixel BFD estimator
        # --------------------------------------------------
        pix_good_all = np.where(good_pix)[0]
        n_good = len(pix_good_all)

        r_matrix = np.zeros((n_good, 2, 2), dtype=float)
        r_matrix[:, 0, 0] = r11_map[good_pix]
        r_matrix[:, 0, 1] = r12_map[good_pix]
        r_matrix[:, 1, 0] = r12_map[good_pix]
        r_matrix[:, 1, 1] = r22_map[good_pix]

        q_matrix = np.zeros((n_good, 2), dtype=float)
        q_matrix[:, 0] = q1_in[good_pix]
        q_matrix[:, 1] = q2_in[good_pix]

        det_r = (
            r_matrix[:, 0, 0] * r_matrix[:, 1, 1]
            - r_matrix[:, 0, 1] * r_matrix[:, 1, 0]
        )

        good_matrix = (
            np.isfinite(q_matrix[:, 0])
            & np.isfinite(q_matrix[:, 1])
            & np.isfinite(r_matrix[:, 0, 0])
            & np.isfinite(r_matrix[:, 0, 1])
            & np.isfinite(r_matrix[:, 1, 0])
            & np.isfinite(r_matrix[:, 1, 1])
            & np.isfinite(det_r)
            & (det_r != 0.0)
        )

        pix_good = pix_good_all[good_matrix]
        r_matrix = r_matrix[good_matrix]
        q_matrix = q_matrix[good_matrix]

        cov = np.linalg.inv(-r_matrix)
        g = -np.einsum("ijk,ik->ij", cov, q_matrix)

        se1 = np.zeros(npix, dtype=float)
        se2 = np.zeros(npix, dtype=float)

        se1[pix_good] = g[:, 0]
        se2[pix_good] = g[:, 1]

        bad_shear = (
            ~np.isfinite(se1)
            | ~np.isfinite(se2)
            | (se1**2 > 1.0)
            | (se2**2 > 1.0)
        )

        se1[bad_shear] = 0.0
        se2[bad_shear] = 0.0

        mask = np.zeros(npix, dtype=float)
        mask[pix_good] = 1.0
        mask[bad_shear] = 0.0

        return se1, se2, mask

    se1_map, se2_map, mask_map = estimate_from_q(q1_map, q2_map)

    if decompose_shear and have_gamma:
        se1_base, se2_base, mask_base = estimate_from_q(q1_base_map, q2_base_map)
        se1_signal = se1_map - se1_base
        se2_signal = se2_map - se2_base
        return (
            se1_map, se2_map, mask_map, n_source,
            se1_base, se2_base, mask_base, se1_signal, se2_signal,
        )

    return se1_map, se2_map, mask_map, n_source



def make_BFD_sample(
    ngal_glass,
    zeff_glass,
    cosmo_bundle,
    sims_parameters,
    nside_maps,
    fields,
    cats_BFD,
    SC_corrections=None,
    include_SC=True,
    random_rotate=True,
    rotate_selection=True,
    do_noise=True,
    do_data=False,
    compact_savings=False,
    minimal_output=False,
    decompose_shear=False,
    rng=None,
):
    """
    BFD analogue of make_WL_sample.

    Important sign convention:

        g1_G = gamma_real_sign * Re(gamma)
        g2_G = gamma_imag_sign * Im(gamma)

    To mimic your usual Metacal-like forward model, use:

        gamma_real_sign = -1
        gamma_imag_sign = -1

    To mimic the old anafast test you showed, use:

        gamma_real_sign = +1
        gamma_imag_sign = -1

    The deterministic injected shear enters make_bfd_map through q -> q + R gamma.
    Therefore the physical sim sign should be fixed here, not by flipping the final
    e1/e2 sim map after the fact.
    """

    if rng is None:
        rng = np.random.default_rng()

    if minimal_output and compact_savings:
        raise ValueError("minimal_output and compact_savings are mutually exclusive")

    n_tomo = len(ngal_glass)
    npix = hp.nside2npix(nside_maps)

    sims_parameters.setdefault("rot", 0)
    sims_parameters.setdefault("delta_rot", 0)


    if include_SC:
        bias_sc = np.asarray(sims_parameters["bias_sc"])
    else:
        bias_sc = np.zeros(n_tomo)
        sims_parameters["bias_sc"] = bias_sc

    bias_TA = np.asarray(sims_parameters.get("bias_TA", np.zeros(n_tomo)))
    dm = np.asarray(sims_parameters.get("dm", np.ones(n_tomo)))

    if include_SC and SC_corrections is not None:
        corr_variance_array = [SC_corrections["corr_variance_fit"][tomo](bias_sc[tomo]) for tomo in range(n_tomo)]
        coeff_kurtosis_array = [SC_corrections["coeff_kurtosis_fit"][tomo](bias_sc[tomo]) for tomo in range(n_tomo)]
        A_corr_array = [SC_corrections["A_corr_fit"][tomo](bias_sc[tomo]) for tomo in range(n_tomo)]
    else:
        corr_variance_array = np.ones(n_tomo)
        coeff_kurtosis_array = np.zeros(n_tomo)
        A_corr_array = np.ones(n_tomo)

    g1_tot = np.zeros((n_tomo, npix), dtype=float)
    g2_tot = np.zeros((n_tomo, npix), dtype=float)
    d_tot = np.zeros((n_tomo, npix), dtype=float)

    for tomo in range(n_tomo):
        for i in range(len(fields["gamma"])):

            C1 = 5e-14
            rho_crit0_h2 = ccl.physical_constants.RHO_CRITICAL
            rho_c1 = C1 * rho_crit0_h2

            IA_f = F_nla(
                z=zeff_glass[i],
                om0=sims_parameters["Omega_m"],
                A_ia=sims_parameters["A_IA"],
                rho_c1=rho_c1,
                eta=sims_parameters["eta_IA"],
                z0=0.67,
                cosmo=cosmo_bundle["cosmo_pyccl"],
            )

            IA_f2 = F_TT(
                z=zeff_glass[i],
                om0=sims_parameters["Omega_m"],
                A2=sims_parameters["A_IA2"],
                rho_c1=rho_c1,
                eta=sims_parameters["eta_IA"],
                z0=0.67,
                cosmo=cosmo_bundle["cosmo_pyccl"],
            )

            w_sc = 1.0 + bias_sc[tomo] * fields["density"][i]
            w_ta = 1.0 + bias_TA[tomo] * fields["density"][i]

            g1_G = -1. * fields["gamma"][i][0].real
            g2_G = -1. *  fields["gamma"][i][0].imag

            if "IA_shear" in fields:
                g1_NLA = -1. *  fields["IA_shear"][i][0].real * IA_f
                g2_NLA = -1. *  fields["IA_shear"][i][0].imag * IA_f
            else:
                g1_NLA = 0.0
                g2_NLA = 0.0

            if "IA_shear_TT" in fields:
                g1_TT = -1. *  fields["IA_shear_TT"][i][0].real * IA_f2
                g2_TT = -1. *  fields["IA_shear_TT"][i][0].imag * IA_f2
            else:
                g1_TT = 0.0
                g2_TT = 0.0

            g1_tot[tomo] += ngal_glass[tomo, i] * (w_sc * g1_G + w_ta * (g1_NLA + g1_TT))
            g2_tot[tomo] += ngal_glass[tomo, i] * (w_sc * g2_G + w_ta * (g2_NLA + g2_TT))
            d_tot[tomo] += ngal_glass[tomo, i] * w_sc

    def project_map_to_bfd_footprint(cat, map_in, pix_cat=None, pix_sim=None):
        if pix_cat is None or pix_sim is None:
            pix_hi = convert_to_pix_coord(cat["ra"], cat["dec"], nside=nside_maps * 2)
            pix_sim = rotate_and_rebin(pix_hi, nside_maps, sims_parameters["rot"], delta_=sims_parameters["delta_rot"])
            pix_cat = convert_to_pix_coord(cat["ra"], cat["dec"], nside=nside_maps)
        value_obj = map_in[pix_sim]

        map_out = np.zeros(npix, dtype=float)
        count_out = np.zeros(npix, dtype=float)

        unique_pix, idx_rep = np.unique(pix_cat, return_inverse=True)

        map_out[unique_pix] += np.bincount(idx_rep, weights=value_obj, minlength=len(unique_pix))
        count_out[unique_pix] += np.bincount(idx_rep, weights=np.ones(len(pix_cat)), minlength=len(unique_pix))

        good = count_out > 0
        map_out[good] /= count_out[good]

        return map_out, count_out, pix_cat, pix_sim

    maps_sim = dict()
    cats_sim = None

    for tomo in range(n_tomo):
        print(f"Building BFD tomo {tomo}", flush=True)

        cat_this = cats_BFD[tomo]

        g1_map, n_obj_map, pix_cat, pix_sim = project_map_to_bfd_footprint(cat_this, g1_tot[tomo])
        g2_map, _, _, _ = project_map_to_bfd_footprint(cat_this, g2_tot[tomo], pix_cat=pix_cat, pix_sim=pix_sim)
        d_map, _, _, _ = project_map_to_bfd_footprint(cat_this, d_tot[tomo], pix_cat=pix_cat, pix_sim=pix_sim)

        gamma1_map = dm[tomo] * g1_map
        gamma2_map = dm[tomo] * g2_map

        q_noise_scale_map = np.ones(npix, dtype=float)

        if include_SC:
            q_noise_scale_map[:] = 0.0
            good_d = np.isfinite(d_map) & (d_map > 0.0)
            q_noise_scale_map[good_d] = 1.0 / np.sqrt(d_map[good_d])

        var_bfd = np.zeros(npix, dtype=float)

        if include_SC and SC_corrections is not None:
            rng_ref = np.random.default_rng(rng.integers(0, 2**32 - 1))

            e1_ref, e2_ref, mask_ref, _ = make_bfd_map(
                cat_this,
                nside=nside_maps,
                random_rotate=True,
                rotate_selection=rotate_selection,
                gamma1=None,
                gamma2=None,
                q_noise_scale_map=None,
                rng=rng_ref,
            )

            var_bfd = e1_ref**2 + e2_ref**2
            del e1_ref, e2_ref, mask_ref

            corr_variance = corr_variance_array[tomo]
            coeff_kurtosis = coeff_kurtosis_array[tomo]
            A_corr = A_corr_array[tomo]

            corr_arg = 1.0 + coeff_kurtosis * var_bfd
            corr_arg[~np.isfinite(corr_arg)] = 0.0
            corr_arg[corr_arg < 0.0] = 0.0

            corr_map = 1.0 / np.sqrt(A_corr * corr_variance) * np.sqrt(corr_arg)
            corr_map[~np.isfinite(corr_map)] = 0.0

            q_noise_scale_map *= corr_map
            del corr_map, corr_arg

        rng_sim = np.random.default_rng(rng.integers(0, 2**32 - 1))

        sim_result = make_bfd_map(
            cat_this,
            nside=nside_maps,
            random_rotate=random_rotate,
            rotate_selection=rotate_selection,
            gamma1=gamma1_map,
            gamma2=gamma2_map,
            q_noise_scale_map=q_noise_scale_map,
            decompose_shear=decompose_shear,
            rng=rng_sim,
        )

        if decompose_shear:
            (
                e1_sim, e2_sim, mask_sim, n_source,
                e1_base, e2_base, mask_base, e1_signal, e2_signal,
            ) = sim_result
        else:
            e1_sim, e2_sim, mask_sim, n_source = sim_result

        if minimal_output:
            maps_this = {
                "e1": e1_sim,
                "e2": e2_sim,
                "mask": mask_sim.astype(bool, copy=False),
            }
            if decompose_shear:
                maps_this["e1_base"] = e1_base
                maps_this["e2_base"] = e2_base
                maps_this["mask_base"] = mask_base.astype(bool, copy=False)
                maps_this["e1_signal"] = e1_signal
                maps_this["e2_signal"] = e2_signal
        else:
            maps_this = {
                "g1_map": g1_map,
                "g2_map": g2_map,
                "gamma1_map": gamma1_map,
                "gamma2_map": gamma2_map,
                "d_map": d_map,
                "q_noise_scale_map": q_noise_scale_map,
                "var_bfd": var_bfd,
                "e1": e1_sim,
                "e2": e2_sim,
                "mask": mask_sim,
                "n_source": n_source,
                "n_obj_map": n_obj_map,
                "pix_cat": pix_cat,
                "pix_sim": pix_sim,
            }
            if decompose_shear:
                maps_this["e1_base"] = e1_base
                maps_this["e2_base"] = e2_base
                maps_this["mask_base"] = mask_base
                maps_this["e1_signal"] = e1_signal
                maps_this["e2_signal"] = e2_signal

        if do_noise:
            rng_noise = np.random.default_rng(rng.integers(0, 2**32 - 1))

            e1_noise, e2_noise, mask_noise, _ = make_bfd_map(
                cat_this,
                nside=nside_maps,
                random_rotate=True,
                rotate_selection=rotate_selection,
                gamma1=None,
                gamma2=None,
                q_noise_scale_map=q_noise_scale_map,
                rng=rng_noise,
            )

            maps_this["e1n"] = e1_noise
            maps_this["e2n"] = e2_noise
            maps_this["mask_noise"] = mask_noise

        if do_data:
            rng_data = np.random.default_rng(rng.integers(0, 2**32 - 1))

            e1_data, e2_data, mask_data, n_source_data = make_bfd_map(
                cat_this,
                nside=nside_maps,
                random_rotate=False,
                rotate_selection=False,
                gamma1=None,
                gamma2=None,
                q_noise_scale_map=None,
                rng=rng_data,
            )

            maps_this["e1_data"] = e1_data
            maps_this["e2_data"] = e2_data
            maps_this["mask_data"] = mask_data.astype(bool, copy=False) if minimal_output else mask_data
            if not minimal_output:
                maps_this["n_source_data"] = n_source_data

        if minimal_output:
            del g1_map, g2_map, d_map, gamma1_map, gamma2_map
            del q_noise_scale_map, var_bfd, n_source, n_obj_map, pix_cat, pix_sim
            gc.collect()

        if compact_savings:
            idx_good = np.where(mask_sim > 0)[0]

            compact = {
                "idx_": idx_good,
                "g1_map": g1_map[idx_good],
                "g2_map": g2_map[idx_good],
                "gamma1_map": gamma1_map[idx_good],
                "gamma2_map": gamma2_map[idx_good],
                "d_map": d_map[idx_good],
                "q_noise_scale_map": q_noise_scale_map[idx_good],
                "var_bfd": var_bfd[idx_good],
                "e1": e1_sim[idx_good],
                "e2": e2_sim[idx_good],
                "n_source": n_source[idx_good],
            }

            if do_noise:
                compact["e1n"] = e1_noise[idx_good]
                compact["e2n"] = e2_noise[idx_good]

            if do_data:
                compact["e1_data"] = e1_data[idx_good]
                compact["e2_data"] = e2_data[idx_good]

            maps_sim[tomo] = compact
        else:
            maps_sim[tomo] = maps_this

    return maps_sim, cats_sim



def add_data_noise_BFD(maps_BFD, cats_BFD, nside=1024, rotate_selection=True, seed=12345):
    for t in sorted(maps_BFD.keys()):
        rng_t = np.random.default_rng(seed + 1000 * int(t))

        e1n_data, e2n_data, mask_n_data, _ = make_bfd_map(
            cats_BFD[t],
            nside=nside,
            random_rotate=True,
            rotate_selection=rotate_selection,
            gamma1=None,
            gamma2=None,
            q_noise_scale_map=None,
            rng=rng_t,
        )

        maps_BFD[t]["e1n_data"] = e1n_data
        maps_BFD[t]["e2n_data"] = e2n_data
        maps_BFD[t]["mask_n_data"] = mask_n_data


FD_SIM_E1_SIGN = +1.0
BFD_SIM_E2_SIGN = +1.0

BFD_DATA_E1_SIGN = +1.0
BFD_DATA_E2_SIGN = -1.0


def make_kappaE_map(
    e1,
    e2,
    mask,
    nside=1024,
    lmax=2048,
    e1_sign=1.0,
    e2_sign=1.0,
):
    mask = np.asarray(mask).astype(bool)

    e1_use = e1_sign * np.asarray(e1, dtype=float).copy()
    e2_use = e2_sign * np.asarray(e2, dtype=float).copy()

    e1_use[~mask] = 0.0
    e2_use[~mask] = 0.0

    kE, kB, _, _ = g2k_sphere(
        e1_use,
        e2_use,
        mask=mask,
        nside=nside,
        lmax=lmax,
    )

    kE[~mask] = 0.0
    kB[~mask] = 0.0

    return kE, kB


def make_kappaE_BFD_maps(
    maps_BFD,
    nside=1024,
    lmax=2048,
    use_common_mask=True,
):
    """
    Adds frozen-convention kE/kB maps to maps_BFD.

    Final convention:
      sims: g2k_sphere(e1_sim, e2_sim)
      data: g2k_sphere(e1_data, -e2_data)

    Requires make_BFD_sample to already use:
      g1_G = -Re(gamma)
      g2_G = -Im(gamma)

    Output keys per tomo:
      kE_sim, kB_sim
      kE_noise, kB_noise
      kE_data, kB_data
      kE_data_noise, kB_data_noise  if data-noise exists
      kE_gamma, kB_gamma
      mask_kE
    """

    for t in sorted(maps_BFD.keys()):
        m = maps_BFD[t]

        mask_sim = m["mask"] > 0
        mask_noise = m.get("mask_noise", m["mask"]) > 0
        mask_data = m.get("mask_data", m["mask"]) > 0
        mask_data_noise = m.get("mask_n_data", m.get("mask_data", m["mask"])) > 0

        if use_common_mask:
            mask = mask_sim & mask_noise & mask_data & mask_data_noise
        else:
            mask = mask_data

        m["mask_kE"] = mask.astype(float)

        # ----------------------------
        # Sim signal + BFD noise
        # ----------------------------
        m["kE_sim"], m["kB_sim"] = make_kappaE_map(
            m["e1"],
            m["e2"],
            mask,
            nside=nside,
            lmax=lmax,
            e1_sign=BFD_SIM_E1_SIGN,
            e2_sign=BFD_SIM_E2_SIGN,
        )

        # ----------------------------
        # Sim random-rotation noise
        # ----------------------------
        if "e1n" in m and "e2n" in m:
            m["kE_noise"], m["kB_noise"] = make_kappaE_map(
                m["e1n"],
                m["e2n"],
                mask,
                nside=nside,
                lmax=lmax,
                e1_sign=BFD_SIM_E1_SIGN,
                e2_sign=BFD_SIM_E2_SIGN,
            )

        # ----------------------------
        # Data
        # ----------------------------
        if "e1_data" in m and "e2_data" in m:
            m["kE_data"], m["kB_data"] = make_kappaE_map(
                m["e1_data"],
                m["e2_data"],
                mask,
                nside=nside,
                lmax=lmax,
                e1_sign=BFD_DATA_E1_SIGN,
                e2_sign=BFD_DATA_E2_SIGN,
            )

        # ----------------------------
        # Data random-rotation noise
        # ----------------------------
        if "e1n_data" in m and "e2n_data" in m:
            m["kE_data_noise"], m["kB_data_noise"] = make_kappaE_map(
                m["e1n_data"],
                m["e2n_data"],
                mask,
                nside=nside,
                lmax=lmax,
                e1_sign=BFD_DATA_E1_SIGN,
                e2_sign=BFD_DATA_E2_SIGN,
            )

        # ----------------------------
        # Input gamma diagnostic
        # ----------------------------
        if "gamma1_map" in m and "gamma2_map" in m:
            m["kE_gamma"], m["kB_gamma"] = make_kappaE_map(
                m["gamma1_map"],
                m["gamma2_map"],
                mask,
                nside=nside,
                lmax=lmax,
                e1_sign=BFD_SIM_E1_SIGN,
                e2_sign=BFD_SIM_E2_SIGN,
            )

    return maps_BFD



# ============================================================
# Frozen BFD -> kE convention
# ============================================================

# Sims: use g2k_sphere(e1_sim, e2_sim)
BFD_SIM_E1_SIGN = +1.0
BFD_SIM_E2_SIGN = +1.0

# Data: use g2k_sphere(e1_data, -e2_data)
BFD_DATA_E1_SIGN = +1.0
BFD_DATA_E2_SIGN = -1.0


def bfd_e_to_kE(
    e1,
    e2,
    mask,
    nside=1024,
    lmax=2048,
    e1_sign=1.0,
    e2_sign=1.0,
):
    """
    Convert BFD e1/e2 map to kappa_E.

    Returns full-sky kE map with zero outside mask.
    Does not store B-mode.
    """

    mask = np.asarray(mask).astype(bool)

    e1_use = e1_sign * np.asarray(e1, dtype=float).copy()
    e2_use = e2_sign * np.asarray(e2, dtype=float).copy()

    e1_use[~mask] = 0.0
    e2_use[~mask] = 0.0

    kE, _, _, _ = g2k_sphere(
        e1_use,
        e2_use,
        mask=mask,
        nside=nside,
        lmax=lmax,
    )

    kE[~mask] = 0.0

    return kE


def make_BFD_kE_products(
    ngal_glass,
    zeff_glass,
    cosmo_bundle,
    sims_parameters,
    nside_maps,
    fields,
    cats_BFD,
    SC_corrections=None,
    include_SC=True,
    do_data=False,
    lmax=2048,
    seed=12345,
    save_path=None,
):
    """
    Minimal production function.

    Output per tomo:
      out[tomo]["idx"]      HEALPix pixels kept
      out[tomo]["kE_sim"]   masked/compact sim kE values
      out[tomo]["kE_data"]  masked/compact data kE values, only if do_data=True
      out[tomo]["mask"]     full mask used for compacting

    No noise-only maps.
    No B modes.
    """

    maps_BFD, _ = make_BFD_sample(
        ngal_glass,
        zeff_glass,
        cosmo_bundle,
        sims_parameters,
        nside_maps,
        fields,
        cats_BFD,
        SC_corrections=SC_corrections,
        include_SC=include_SC,
        random_rotate=True,
        rotate_selection=True,
        do_noise=False,
        do_data=do_data,
        compact_savings=False,
        rng=np.random.default_rng(seed),
    )

    out = {}

    for tomo in sorted(maps_BFD.keys()):
        m = maps_BFD[tomo]

        mask_sim = m["mask"] > 0

        if do_data:
            mask_data = m["mask_data"] > 0
            mask = mask_sim & mask_data
        else:
            mask = mask_sim

        kE_sim_full = bfd_e_to_kE(
            m["e1"],
            m["e2"],
            mask,
            nside=nside_maps,
            lmax=lmax,
            e1_sign=BFD_SIM_E1_SIGN,
            e2_sign=BFD_SIM_E2_SIGN,
        )

        idx = np.where(mask)[0]

        out[tomo] = {
            "idx": idx,
            "kE_sim": kE_sim_full[idx],
            "mask": mask.astype(bool),
        }

        if do_data:
            kE_data_full = bfd_e_to_kE(
                m["e1_data"],
                m["e2_data"],
                mask,
                nside=nside_maps,
                lmax=lmax,
                e1_sign=BFD_DATA_E1_SIGN,
                e2_sign=BFD_DATA_E2_SIGN,
            )

            out[tomo]["kE_data"] = kE_data_full[idx]

    meta = {
        "nside": nside_maps,
        "lmax": lmax,
        "sim_convention": "g2k_sphere(e1_sim, e2_sim)",
        "data_convention": "g2k_sphere(e1_data, -e2_data)",
        "gamma_convention": "gamma1=-Re(gamma), gamma2=-Im(gamma)",
    }

    package = {
        "maps": out,
        "meta": meta,
    }

    if save_path is not None:
        np.save(save_path, package, allow_pickle=True)

    return package
