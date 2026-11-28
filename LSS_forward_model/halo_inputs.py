"""Input normalisation for the halo lightcone (``halos.save_halocatalog``).

The lightcone can be built from two kinds of simulation products. This module
reads both and hands ``save_halocatalog`` one internal convention, so the
lightcone does not depend on where or how the simulation was post-processed.

Supported halo snapshots (one file per PKDGRAV step ``S``), searched in order:

1. ``{sim}/run.{S:05d}.fofstats.parquet``: written by
   ``PKDGRAV_and_postprocessing_utilities.convert_fof_and_cleanup`` (the NERSC
   preprocessing).
2. ``{sim}/fof/run.{S:05d}.fofstats.0`` (Hypatia/Tursa tarball layout) or
   ``{sim}/run.{S:05d}.fofstats.0`` (PKDGRAV output directory): the native FoF
   binary, with or without the two SO fields (r200c, m200c).

A native binary is encoded with exactly the same function that writes the
parquet files (:func:`encode_fof_records`), so both formats give identical
halo properties.

Supported ``z_values.txt`` dialects (columns are read by position by
``cosmology.recover_shell_info``; the header is not trusted):

* ``package``: written by ``build_z_values_file``. ``cmd_*`` are the comoving
  distances χ(z) in Mpc for the run's own (w0, wa). Header ``cmd_far(Mpc)``.
* ``gower``: written by the Gower St team. ``cmd_*`` are h·χ(z) in Mpc/h,
  computed for w = -1 whatever the run's w0. Header ``cmd_far(Mpc/h)``.

In both dialects the redshift columns (``Step``, ``z_far``, ``z_near``) are the
PKDGRAV output redshifts from ``run.log``. They are the authoritative
definition of each snapshot's lightcone shell.

Internal convention used to build the lightcone:

* halo positions: comoving Mpc, in [0, L) with L = dBoxSize[Mpc/h] / h;
* shell radial edges: comoving Mpc, χ(z_near) and χ(z_far) for the run's
  (w0, wa), computed exactly as ``build_z_values_file`` does on NERSC
  (``shell_comoving_distance_Mpc``). ``package`` distances are used as
  written because they already are this; any other dialect is recomputed
  from the redshift columns, so both pathways use the same edges;
* halo redshift: z(χ) with the same CCL cosmology (w0, wa, massive
  neutrinos from ``sims_parameters``).

Nothing here modifies the simulation files or the ``shells_info`` dict.
"""

import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import astropy.units as u
from astropy.cosmology import wCDM, Flatw0waCDM


# ---------------------------------------------------------------------------
# Native FoF binary
# ---------------------------------------------------------------------------

FOF_DTYPE_LEGACY = np.dtype([
    ("rPot", ("f4", 3)), ("minPot", "f4"), ("rcen", ("f4", 3)),
    ("rcom", ("f4", 3)), ("cvom", ("f4", 3)), ("angular", ("f4", 3)),
    ("inertia", ("f4", 6)), ("sigma", "f4"), ("rMax", "f4"),
    ("fMAss", "f4"), ("fEnvironDensity0", "f4"),
    ("fEnvironDensity1", "f4"), ("rHalf", "f4"),
])
"""PKDGRAV FoF record without SO fields (112 bytes)."""

FOF_DTYPE_SO = np.dtype(FOF_DTYPE_LEGACY.descr + [("r200c", "f4"), ("m200c", "f4")])
"""PKDGRAV FoF record with r200c and m200c appended (120 bytes)."""


def _plausible_fof_records(records):
    """True if every record looks like a real halo (box-unit position, mass > 0)."""
    if records.size == 0:
        return True
    pos = records["rPot"].astype(np.float64) + records["rcen"].astype(np.float64)
    mass = records["fMAss"]
    radius = records["rMax"]
    ok = (np.all(np.isfinite(pos)) and np.all(np.abs(pos) <= 0.5 + 1e-2)
          and np.all(np.isfinite(mass)) and np.all(mass > 0)
          and np.all(np.isfinite(radius)) and np.all(radius >= 0))
    if ok and "m200c" in records.dtype.names:
        ok = (np.all(np.isfinite(records["m200c"])) and np.all(records["m200c"] >= 0)
              and np.all(np.isfinite(records["r200c"])) and np.all(records["r200c"] >= 0))
    return bool(ok)


def read_fof_binary(path):
    """Read a native PKDGRAV FoF file and identify its record layout.

    The two layouts differ only in record size (112 or 120 bytes). A file
    whose size is a multiple of both (N a multiple of 15 legacy records, or
    of 14 SO records) is resolved by decoding it both ways and keeping the
    layout whose records are physical.

    Returns
    -------
    records : numpy structured array
    has_so : bool
    """
    size = os.path.getsize(path)
    candidates = [dt for dt in (FOF_DTYPE_SO, FOF_DTYPE_LEGACY) if size % dt.itemsize == 0]
    if not candidates:
        raise ValueError(
            f"Cannot infer FoF record layout of {path}: size {size} is a multiple of neither "
            f"{FOF_DTYPE_LEGACY.itemsize} (legacy) nor {FOF_DTYPE_SO.itemsize} (SO) bytes")
    if size == 0:
        return np.zeros(0, FOF_DTYPE_LEGACY), False
    plausible = []
    for dt in candidates:
        records = np.fromfile(path, dtype=dt)
        if _plausible_fof_records(records):
            plausible.append((records, dt is FOF_DTYPE_SO))
    if len(plausible) != 1:
        raise ValueError(
            f"Cannot infer FoF record layout of {path}: {len(plausible)} of {len(candidates)} "
            "candidate layouts give physical records")
    return plausible[0]


def fof_mass_unit_Msun(sims_parameters):
    """Mass of fMAss = 1, i.e. ρ_crit,0 · L³ in Msun (L in Mpc).

    Identical to the factor used since 2025 by ``convert_fof_and_cleanup``.
    """
    Lbox_Mpc = float(sims_parameters["dBoxSize Mpc/h"]) / float(sims_parameters["h"])
    cosmo = _astropy_cosmology(sims_parameters)
    return ((Lbox_Mpc * u.Mpc) ** 3 * cosmo.critical_density(0).to(u.Msun / u.Mpc ** 3)).value


def encode_fof_records(halos, has_so, sims_parameters):
    """Encode native FoF records into the portable snapshot columns.

    This is the single definition of the snapshot encoding. It is used both
    by ``convert_fof_and_cleanup`` (to write the parquet files) and by
    :func:`load_halo_snapshot` (to read a native binary), so both pathways
    give identical columns.

    Columns (units):
      halo_center   (N,3) float32, comoving Mpc in [0, L): L·(rPot + rcen + 0.5)
      rmax          uint16, kpc (truncated)
      log10M        uint16, 1000·log10(M_FoF / Msun) (truncated)
      r200c         uint16, kpc; zeros without SO fields
      log10M200c    uint16, 1000·log10(M200c / Msun); zeros without SO fields
      inertia_auto  (N,3) uint16, 1000·log10(1e20·I_ii), i = xx, yy, zz
      inertia_cross (N,3) uint16, 10000·(1 + I_ij/sqrt(I_ii I_jj)), ij = xy, xz, yz
      angular       (N,3) int16, signed 1000·log10(1e20·|J| + 1)
    """
    Lbox_Mpc = float(sims_parameters["dBoxSize Mpc/h"]) / float(sims_parameters["h"])
    f_mass = fof_mass_unit_Msun(sims_parameters)
    halo_center = Lbox_Mpc * (halos["rPot"] + halos["rcen"] + 0.5)
    rmax = (Lbox_Mpc * halos["rMax"] * 1000).astype("uint16")
    log10M = (np.log10(halos["fMAss"] * f_mass) * 1000).astype("uint16")
    if has_so:
        r200c = (Lbox_Mpc * halos["r200c"] * 1000).astype("uint16")
        log10M200c = (np.log10(halos["m200c"] * f_mass) * 1000).astype("uint16")
    else:
        r200c = np.zeros(halos.size, dtype="uint16")
        log10M200c = np.zeros(halos.size, dtype="uint16")

    pm = np.empty((halos.size, 3), dtype=np.float32)
    pm[:, 0] = halos["inertia"][:, 0]
    pm[:, 1] = halos["inertia"][:, 3]
    pm[:, 2] = halos["inertia"][:, 5]
    inertia_auto = (np.log10(pm * 1e20) * 1000).astype("uint16")
    cross = np.empty((halos.size, 3), dtype=np.float32)
    cross[:, 0] = halos["inertia"][:, 1] / np.sqrt(pm[:, 0] * pm[:, 1])
    cross[:, 1] = halos["inertia"][:, 2] / np.sqrt(pm[:, 0] * pm[:, 2])
    cross[:, 2] = halos["inertia"][:, 4] / np.sqrt(pm[:, 1] * pm[:, 2])
    inertia_cross = (10000 * (1 + cross)).astype("uint16")
    angular = (np.sign(halos["angular"])
               * np.log10(np.abs(halos["angular"]) * 1e20 + 1) * 1000).astype("int16")
    return {
        "halo_center": halo_center,
        "rmax": rmax,
        "log10M": log10M,
        "r200c": r200c,
        "log10M200c": log10M200c,
        "inertia_auto": inertia_auto,
        "inertia_cross": inertia_cross,
        "angular": angular,
    }


# ---------------------------------------------------------------------------
# Snapshot discovery and loading
# ---------------------------------------------------------------------------

def find_halo_snapshot(path_base, step):
    """Locate the halo snapshot of one PKDGRAV step.

    Returns ``(format, path)`` with format ``'parquet'`` or ``'fof_binary'``.
    The parquet file is preferred when both exist (the NERSC layout).
    """
    step = int(step)
    base = os.fspath(path_base)
    candidates = [
        ("parquet", os.path.join(base, f"run.{step:05d}.fofstats.parquet")),
        ("fof_binary", os.path.join(base, "fof", f"run.{step:05d}.fofstats.0")),
        ("fof_binary", os.path.join(base, f"run.{step:05d}.fofstats.0")),
    ]
    for fmt, path in candidates:
        if os.path.exists(path):
            return fmt, path
    raise FileNotFoundError(
        f"No halo snapshot for step {step} under {base}: looked for "
        + ", ".join(path for _, path in candidates))


def _legacy_log10_offset(encoded_mass):
    """Flamingo_big snapshots store (log10 M - 4)·1000; detect and undo."""
    positive = encoded_mass > 0
    if np.any(positive) and np.min(encoded_mass[positive]) / 1000.0 < 10.0:
        return 4.0
    return 0.0


def load_halo_snapshot(path_base, step, sims_parameters, log10_mass_limit=None,
                       include_ia_shapes=True):
    """Load one halo snapshot in the internal convention.

    Returns a dict with
      x, y, z         float64, comoving Mpc in [0, L)
      M               uint16, 1000·log10(M_FoF / Msun)
      rhalf           uint16, kpc (FoF rMax)
      M200c, r200c    uint16, only if the snapshot carries real SO data
      inertia_tensor  (N,6) float64 (xx,yy,zz,xy,xz,yz), if include_ia_shapes
      format, path    provenance of the snapshot
    """
    from .halo_ia import reconstruct_inertia_tensor_halo_snapshot

    fmt, path = find_halo_snapshot(path_base, step)
    if fmt == "parquet":
        available = set(pq.ParquetFile(path).schema_arrow.names)
        columns = ["halo_center", "rmax", "log10M"]
        columns += [c for c in ("log10M200c", "r200c") if c in available]
        if include_ia_shapes:
            columns += ["inertia_auto", "inertia_cross"]
        table = pq.read_table(path, columns=columns)
        cols = {name: table[name].combine_chunks() for name in table.column_names}
        center = cols["halo_center"].values.to_numpy().reshape(-1, 3)
        data = {name: cols[name].to_numpy() for name in ("rmax", "log10M", "log10M200c", "r200c")
                if name in cols}
        if include_ia_shapes:
            inertia = reconstruct_inertia_tensor_halo_snapshot(
                pa.table({"inertia_auto": cols["inertia_auto"], "inertia_cross": cols["inertia_cross"]}))
    else:
        records, has_so = read_fof_binary(path)
        enc = encode_fof_records(records, has_so, sims_parameters)
        # float32 -> float64 exactly as a parquet round trip of halo_center.tolist()
        center = enc["halo_center"].astype(np.float64)
        data = {name: enc[name] for name in ("rmax", "log10M", "log10M200c", "r200c")}
        if include_ia_shapes:
            inertia = reconstruct_inertia_tensor_halo_snapshot(pa.table({
                name: pa.FixedSizeListArray.from_arrays(pa.array(enc[name].ravel()), 3)
                for name in ("inertia_auto", "inertia_cross")}))

    # SO columns are kept only when they hold measurements: legacy records are
    # encoded with all-zero SO columns, which downstream code would otherwise
    # read as M200c = 1 Msun.
    has_so_data = ("log10M200c" in data and "r200c" in data
                   and np.any(np.asarray(data["log10M200c"]) > 0))

    encoded_mass = np.asarray(data["log10M"])
    offset = _legacy_log10_offset(encoded_mass)
    keep = None
    if log10_mass_limit is not None:
        decoded = encoded_mass.astype(np.float64) / 1000.0 + offset
        keep = (encoded_mass > 0) & (decoded > log10_mass_limit)
    M = encoded_mass.copy()
    if offset:
        M = np.where(M > 0, np.rint(M.astype(np.float64) + 1000.0 * offset), 0).astype(np.uint16)

    out = {"x": center[:, 0], "y": center[:, 1], "z": center[:, 2],
           "rhalf": np.asarray(data["rmax"]), "M": M}
    if has_so_data:
        out["M200c"] = np.asarray(data["log10M200c"])
        out["r200c"] = np.asarray(data["r200c"])
    if include_ia_shapes:
        out["inertia_tensor"] = inertia
    if keep is not None:
        out = {name: np.asarray(value)[keep] for name, value in out.items()}
    out["format"] = fmt
    out["path"] = path
    out["has_so"] = bool(has_so_data)
    return out


# ---------------------------------------------------------------------------
# Shell distances
# ---------------------------------------------------------------------------

DISTANCE_CONVENTIONS = ("auto", "file_Mpc", "from_redshift")


def _astropy_cosmology(sims_parameters, lcdm=False):
    """Background cosmology of ``build_z_values_file`` (flat, Om0 = Omega_m)."""
    Om0 = float(sims_parameters["Omega_m"])
    H0 = float(sims_parameters["h"]) * 100.0
    if lcdm:
        return wCDM(H0=H0, Om0=Om0, w0=-1.0, Ode0=1 - Om0)
    w0 = float(sims_parameters["w0"])
    wa = float(sims_parameters.get("wa", 0.0) or 0.0)
    return (Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa) if abs(wa) > 0
            else wCDM(H0=H0, Om0=Om0, w0=w0, Ode0=1 - Om0))


def shell_comoving_distance_Mpc(z, sims_parameters):
    """χ(z) in Mpc for the run's (w0, wa): the package z_values.txt convention.

    This is the function ``build_z_values_file`` writes to the ``cmd_*``
    columns on NERSC, so edges derived here from the redshift columns equal
    the NERSC files' distances (to their 1e-6 Mpc print precision).
    """
    return _astropy_cosmology(sims_parameters).comoving_distance(np.asarray(z, float)).value


def _ccl_cosmology(sims_parameters, lcdm=False):
    import pyccl as ccl
    m_nu = float(sims_parameters["m_nu"])
    return ccl.Cosmology(
        Omega_c=sims_parameters["Omega_cdm"], Omega_b=sims_parameters["Omega_b"],
        h=sims_parameters["h"], sigma8=sims_parameters["sigma_8"], n_s=sims_parameters["n_s"],
        w0=-1.0 if lcdm else sims_parameters["w0"], wa=0.0 if lcdm else sims_parameters["wa"],
        m_nu=[m_nu / 3] * 3, mass_split="equal", matter_power_spectrum="linear",
        extra_parameters={"camb": {"dark_energy_model": "ppf", "halofit_version": "mead2020"}})


def _chi_Mpc(cosmo, z):
    import pyccl as ccl
    z = np.asarray(z, dtype=float)
    return ccl.comoving_radial_distance(cosmo, 1.0 / (1.0 + z))


def identify_distance_columns(shells_info, sims_parameters, rtol_native=2e-3, rtol_other=1e-2):
    """Identify what the ``cmd_*`` columns of a shell table contain.

    The identification uses the numbers, not the header, because one legacy
    writer labels Mpc as Mpc/h. Rows with 0 < z_far <= 10 are compared with

      native      χ(z) in Mpc for the run's cosmology   (package dialect)
      gower       h·χ_{w=-1}(z), i.e. Mpc/h for w = -1  (Gower St dialect)
      Mpc_h       h·χ(z), Mpc/h for the run's cosmology
      lcdm_Mpc    χ_{w=-1}(z) in Mpc

    Returns a dict with ``dialect`` (one of the above, or ``'unknown'``) and
    the maximum relative deviation from each hypothesis.
    """
    z_far = np.asarray(shells_info["z_far"], dtype=float)
    cmd_far = np.asarray(shells_info["cmd_far"], dtype=float)
    rows = (z_far > 0) & (z_far <= 10)
    if not np.any(rows):
        raise ValueError("No shell with 0 < z_far <= 10 to identify the distance columns")
    h = float(sims_parameters["h"])
    chi = shell_comoving_distance_Mpc(z_far[rows], sims_parameters)
    chi_lcdm = _astropy_cosmology(sims_parameters, lcdm=True).comoving_distance(z_far[rows]).value
    hypotheses = {"native": chi, "gower": h * chi_lcdm, "Mpc_h": h * chi, "lcdm_Mpc": chi_lcdm}
    deviation = {name: float(np.max(np.abs(cmd_far[rows] / ref - 1.0)))
                  for name, ref in hypotheses.items()}
    if deviation["native"] <= rtol_native:
        dialect = "native"
    else:
        others = {k: v for k, v in deviation.items() if k != "native" and v <= rtol_other}
        dialect = min(others, key=others.get) if others else "unknown"
    return {"dialect": dialect, "max_rel_deviation": deviation}


def shell_radial_edges_Mpc(shells_info, sims_parameters, convention="auto"):
    """Radial edges (comoving Mpc) of each snapshot's lightcone shell.

    Parameters
    ----------
    convention : {'auto', 'file_Mpc', 'from_redshift'}
        ``file_Mpc``: use ``cmd_near``/``cmd_far`` as Mpc (package files).
        ``from_redshift``: χ(z_near), χ(z_far) for the run's (w0, wa) with
        ``shell_comoving_distance_Mpc`` (what build_z_values_file writes).
        ``auto``: ``file_Mpc`` if the columns are χ(z) in Mpc for this
        cosmology (within 0.2%); ``from_redshift`` if they match a known
        other convention (Gower St h·χ_{w=-1}, Mpc/h, or w = -1 Mpc);
        otherwise raise.

    Returns
    -------
    near, far : float64 arrays (Mpc), aligned with ``shells_info`` rows
    info : dict with the chosen convention, the identified dialect and the
        deviations used to identify it
    """
    if convention not in DISTANCE_CONVENTIONS:
        raise ValueError(f"convention must be one of {DISTANCE_CONVENTIONS}, got {convention!r}")
    info = {"requested": convention}
    if convention == "auto":
        ident = identify_distance_columns(shells_info, sims_parameters)
        info.update(ident)
        if ident["dialect"] == "native":
            convention = "file_Mpc"
        elif ident["dialect"] == "unknown":
            raise ValueError(
                "The cmd_* columns of the shell table match no known distance convention "
                f"(max relative deviations {ident['max_rel_deviation']}). Check the cosmology in "
                "sims_parameters, or pass distance_convention='from_redshift' to derive the "
                "edges from the redshift columns.")
        else:
            convention = "from_redshift"
    info["convention"] = convention
    if convention == "file_Mpc":
        near = np.asarray(shells_info["cmd_near"], dtype=float)
        far = np.asarray(shells_info["cmd_far"], dtype=float)
    else:
        # rounded like the %.6f columns of build_z_values_file
        near = np.round(shell_comoving_distance_Mpc(shells_info["z_near"], sims_parameters), 6)
        far = np.round(shell_comoving_distance_Mpc(shells_info["z_far"], sims_parameters), 6)
    return near, far, info


def z_values_header_unit(path):
    """Distance unit declared in a z_values.txt header ('Mpc', 'Mpc/h' or None).

    For reporting only: the header is not used to interpret the numbers.
    """
    with open(path) as fh:
        header = fh.readline()
    if "cmd_far(Mpc/h)" in header:
        return "Mpc/h"
    if "cmd_far(Mpc)" in header:
        return "Mpc"
    return None
