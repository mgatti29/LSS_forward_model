import numpy as np
from typing import Sequence, Tuple
import pandas as pd  # only for typing/self; safe to remove if not needed


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