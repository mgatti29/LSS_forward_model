"""Regression tests: halo lightcone from native FoF binaries and NERSC parquet.

The same synthetic PKDGRAV run is written in both supported layouts:

* native (Hypatia/Tursa): fof/run.SSSSS.fofstats.0 and a Gower St
  z_values.txt (cmd in Mpc/h, computed for w = -1);
* NERSC: run.SSSSS.fofstats.parquet made by convert_fof_and_cleanup and a
  z_values.txt made by build_z_values_file (cmd in Mpc for the run's w0, wa).

save_halocatalog must give identical catalogues from both, place every halo in
the shell of the snapshot it came from, and reach the requested redshift, for
LCDM, wCDM and w0waCDM cosmologies. Real-data checks run when the NERSC files
exist and are skipped elsewhere.
"""
import math
import os
import shutil

import numpy as np
import pyarrow.parquet as pq
import pytest
from astropy.cosmology import FlatLambdaCDM

from LSS_forward_model.cosmology import recover_shell_info
from LSS_forward_model.halo_inputs import (
    FOF_DTYPE_LEGACY, FOF_DTYPE_SO, encode_fof_records, find_halo_snapshot,
    identify_distance_columns, load_halo_snapshot, read_fof_binary,
    shell_radial_edges_Mpc, _ccl_cosmology, _chi_Mpc)
from LSS_forward_model.halos import save_halocatalog
from LSS_forward_model.PKDGRAV_and_postprocessing_utilities import (
    build_z_values_file, convert_fof_and_cleanup)

COSMOLOGIES = {
    "lcdm": dict(w0=-1.0, wa=0.0, h=0.67),
    "wcdm": dict(w0=-0.75, wa=0.0, h=0.72),   # far from w = -1, like runsV
    "w0wa": dict(w0=-1.054, wa=0.25, h=0.666),
}
MAX_Z = 0.45
BOX = 240.0  # Mpc/h


def sims_parameters(name):
    c = COSMOLOGIES[name]
    Om_c, Om_b, m_nu = 0.26, 0.049, 0.06
    Om_nu = m_nu / (93.14 * c["h"] ** 2)
    return {"Omega_cdm": Om_c, "Omega_b": Om_b, "Omega_nu": Om_nu, "Omega_m": Om_c + Om_b + Om_nu,
            "h": c["h"], "sigma_8": 0.8, "n_s": 0.96, "w0": c["w0"], "wa": c["wa"],
            "m_nu": m_nu, "dBoxSize Mpc/h": BOX}


def redshift_history():
    """PKDGRAV-like output redshifts, 49 -> 0."""
    z = list(np.geomspace(49.0, 3.0, 8)) + list(np.linspace(2.5, 0.0, 26)[:-1]) + [0.0]
    return np.array(sorted(set(np.round(z, 6)), reverse=True))


def write_run_log(directory, z):
    with open(os.path.join(directory, "run.log"), "w") as fh:
        fh.write("# step z ...\n")
        for zi in z:
            fh.write(f"{1/(1+zi):.6e} {zi:.6e} 0 0 0 0 0\n")


def write_gower_z_values(path, z, sp):
    """Gower St dialect: cmd = h·χ_{w=-1}(z) in Mpc/h (header says Mpc/h)."""
    lcdm = FlatLambdaCDM(H0=100.0, Om0=sp["Omega_m"])  # H0 = 100 gives Mpc/h directly
    cmd = lcdm.comoving_distance(z).value
    rows = []
    for s in range(1, len(z)):
        zf, zn, cf, cn = z[s - 1], z[s], cmd[s - 1], cmd[s]
        vol = 4 / 3 * math.pi * (cf ** 3 - cn ** 3)
        rows.append([s, zf, zn, zf - zn, cf, cn, cf - cn, cf / BOX, cn / BOX, (cf - cn) / BOX,
                     vol, vol / BOX ** 3])
    header = ("Step,z_far,z_near,delta_z,cmd_far(Mpc/h),cmd_near(Mpc/h),delta_cmd(Mpc/h),"
              "cmd/box_far,cmd/box_near,delta_cmd/box,cmvolume,cmvolume/boxvolume")
    np.savetxt(path, np.array(rows), fmt=["%i"] + ["%.6f"] * 9 + ["%.2f", "%.6f"],
               delimiter=",", header=header)


def fake_fof_records(rng, n, so=False):
    rec = np.zeros(n, FOF_DTYPE_SO if so else FOF_DTYPE_LEGACY)
    rec["rPot"] = rng.uniform(-0.4999, 0.4999, (n, 3))
    rec["rcen"] = 0.0
    rec["fMAss"] = 10 ** rng.uniform(-6.5, -3.5, n)
    rec["rMax"] = rng.uniform(1e-4, 3e-3, n)
    diag = rng.uniform(1.0, 3.0, (n, 3)) * 1e-14
    corr = rng.uniform(-0.4, 0.4, (n, 3))
    rec["inertia"][:, 0], rec["inertia"][:, 3], rec["inertia"][:, 5] = diag.T
    rec["inertia"][:, 1] = corr[:, 0] * np.sqrt(diag[:, 0] * diag[:, 1])
    rec["inertia"][:, 2] = corr[:, 1] * np.sqrt(diag[:, 0] * diag[:, 2])
    rec["inertia"][:, 4] = corr[:, 2] * np.sqrt(diag[:, 1] * diag[:, 2])
    rec["angular"] = rng.normal(0, 1e-15, (n, 3))
    if so:
        rec["m200c"] = rec["fMAss"] * 0.8
        rec["r200c"] = rec["rMax"] * 0.7
    return rec


def make_simulation(root, name, so=False, n_halos=1500):
    """Write one synthetic run in the native layout and in the NERSC layout."""
    sp = sims_parameters(name)
    z = redshift_history()
    native = os.path.join(root, "native")
    nersc = os.path.join(root, "nersc")
    os.makedirs(os.path.join(native, "fof"))
    os.makedirs(nersc)
    write_run_log(native, z)
    write_gower_z_values(os.path.join(native, "z_values.txt"), z, sp)
    build_z_values_file(native, "run", nersc, sims_parameters=sp)   # NERSC rewrite
    rng = np.random.default_rng(sum(map(ord, name)))
    steps = recover_shell_info(os.path.join(native, "z_values.txt"))["Step"]
    for s in steps:
        fake_fof_records(rng, n_halos, so=so).tofile(
            os.path.join(native, "fof", f"run.{int(s):05d}.fofstats.0"))
    # NERSC preprocessing on links to the native files (it deletes the links only)
    for s in steps:
        os.symlink(os.path.join(native, "fof", f"run.{int(s):05d}.fofstats.0"),
                   os.path.join(nersc, f"run.{int(s):05d}.fofstats.0"))
    resume = recover_shell_info(os.path.join(nersc, "z_values.txt"))
    convert_fof_and_cleanup(nersc, resume, sp, z_max_for_convert=60.0)
    return sp, native, nersc


def build(sim_dir, sp, out, **kw):
    shells = recover_shell_info(os.path.join(sim_dir, "z_values.txt"))
    save_halocatalog(shells, sp, max_redshift=MAX_Z, halo_snapshots_path=sim_dir,
                     catalog_path=out, **kw)
    return pq.read_table(out), shells


# ---------------------------------------------------------------------------
# FoF binary layout and encoding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("so,n", [(False, 15), (False, 30), (True, 14), (True, 28), (False, 7), (True, 9)])
def test_fof_layout_detection_including_ambiguous_sizes(tmp_path, so, n):
    rec = fake_fof_records(np.random.default_rng(n), n, so=so)
    path = tmp_path / "run.00010.fofstats.0"
    rec.tofile(path)
    got, has_so = read_fof_binary(path)
    assert has_so == so
    np.testing.assert_array_equal(got, rec)


def test_convert_and_native_loader_share_one_encoding(tmp_path):
    sp, native, nersc = make_simulation(str(tmp_path), "w0wa", so=True, n_halos=300)
    step = 30
    fmt_n, path_n = find_halo_snapshot(native, step)
    fmt_p, path_p = find_halo_snapshot(nersc, step)
    assert (fmt_n, fmt_p) == ("fof_binary", "parquet")
    # parquet columns are exactly the shared encoding of the native records
    rec, has_so = read_fof_binary(path_n)
    enc = encode_fof_records(rec, has_so, sp)
    table = pq.read_table(path_p)
    np.testing.assert_array_equal(
        table["halo_center"].combine_chunks().values.to_numpy().reshape(-1, 3),
        enc["halo_center"].astype(np.float64))
    for name in ("rmax", "log10M", "r200c", "log10M200c"):
        np.testing.assert_array_equal(table[name].to_numpy(), enc[name])
    # and both loaders return identical snapshots
    a = load_halo_snapshot(native, step, sp, log10_mass_limit=13.5)
    b = load_halo_snapshot(nersc, step, sp, log10_mass_limit=13.5)
    assert 0 < len(a["M"]) < 300
    for name in ("x", "y", "z", "M", "rhalf", "M200c", "r200c", "inertia_tensor"):
        np.testing.assert_array_equal(a[name], b[name])
    L = sp["dBoxSize Mpc/h"] / sp["h"]
    assert a["x"].min() >= 0 and a["x"].max() < L


def test_zero_so_columns_are_not_propagated(tmp_path):
    """Legacy records give all-zero SO columns, which must not reach the lightcone."""
    sp, native, nersc = make_simulation(str(tmp_path), "lcdm", so=False, n_halos=200)
    for d in (native, nersc):
        snap = load_halo_snapshot(d, 30, sp)
        assert not snap["has_so"] and "M200c" not in snap


# ---------------------------------------------------------------------------
# Shell distance conventions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(COSMOLOGIES))
def test_distance_dialects_identified_and_edges_physical(tmp_path, name):
    sp, native, nersc = make_simulation(str(tmp_path), name, n_halos=10)
    cosmo = _ccl_cosmology(sp)
    gower = recover_shell_info(os.path.join(native, "z_values.txt"))
    package = recover_shell_info(os.path.join(nersc, "z_values.txt"))
    # the redshift columns of both dialects agree (both come from run.log)
    np.testing.assert_array_equal(gower["Step"], package["Step"])
    for key in ("z_far", "z_near"):
        np.testing.assert_allclose(gower[key], package[key], atol=1e-5)
    assert identify_distance_columns(package, sp)["dialect"] == "native"
    assert identify_distance_columns(gower, sp)["dialect"] == "gower"
    n_p, f_p, info_p = shell_radial_edges_Mpc(package, sp)
    n_g, f_g, info_g = shell_radial_edges_Mpc(gower, sp)
    assert info_p["convention"] == "file_Mpc" and info_g["convention"] == "from_redshift"
    # package files are used verbatim (backward compatibility)
    np.testing.assert_array_equal(n_p, package["cmd_near"])
    np.testing.assert_array_equal(f_p, package["cmd_far"])
    # both give the physical edges of the run's cosmology
    true_far = _chi_Mpc(cosmo, package["z_far"])
    low = package["z_far"] < 10
    np.testing.assert_allclose(f_p[low], true_far[low], rtol=2e-3)
    np.testing.assert_allclose(f_g[low], true_far[low], rtol=2e-3)
    # edges derived from a Gower file equal the NERSC rewrite of the same table
    np.testing.assert_allclose(f_g, package["cmd_far"], rtol=0, atol=2e-6 * f_g.max())
    np.testing.assert_allclose(n_g, package["cmd_near"], rtol=0, atol=2e-6 * f_g.max())
    # the shell table itself is not modified
    np.testing.assert_array_equal(gower["cmd_far"], recover_shell_info(
        os.path.join(native, "z_values.txt"))["cmd_far"])


def test_unknown_distance_columns_raise(tmp_path):
    sp, native, _ = make_simulation(str(tmp_path), "wcdm", n_halos=10)
    shells = recover_shell_info(os.path.join(native, "z_values.txt"))
    shells = dict(shells, cmd_far=shells["cmd_far"] * 1.7, cmd_near=shells["cmd_near"] * 1.7)
    with pytest.raises(ValueError, match="no known distance convention"):
        shell_radial_edges_Mpc(shells, sp)
    # an explicit convention is still honoured
    near, far, info = shell_radial_edges_Mpc(shells, sp, convention="from_redshift")
    assert info["convention"] == "from_redshift"


# ---------------------------------------------------------------------------
# Full lightcone: native vs NERSC pathway
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,so", [("lcdm", False), ("wcdm", True), ("w0wa", False)])
def test_native_and_nersc_pathways_give_identical_lightcones(tmp_path, name, so):
    sp, native, nersc = make_simulation(str(tmp_path), name, so=so)
    t_native, shells = build(native, sp, str(tmp_path / "native.parquet"))
    t_nersc, _ = build(nersc, sp, str(tmp_path / "nersc.parquet"))

    # identical catalogues: positions, masses, radii, redshifts, steps, shapes
    assert t_native.num_rows > 10000
    assert t_native.column_names == t_nersc.column_names
    assert ("log_M200c_SO" in t_native.column_names) == so
    for column in t_native.column_names:
        assert t_native[column].equals(t_nersc[column]), column

    md_n = pq.ParquetFile(str(tmp_path / "native.parquet")).metadata.metadata
    md_p = pq.ParquetFile(str(tmp_path / "nersc.parquet")).metadata.metadata
    key = b"lss_forward_model.halo_lightcone."
    assert md_n[key + b"snapshot_formats"] == b"fof_binary"
    assert md_p[key + b"snapshot_formats"] == b"parquet"
    assert md_n[key + b"distance_columns_dialect"] == b"gower"
    assert md_p[key + b"distance_columns_dialect"] == b"native"

    # every halo lies in the redshift range of the snapshot it was taken from
    z = t_native["redshift"].to_numpy() / 1e4
    step = t_native["simulation_step"].to_numpy()
    z_near = dict(zip(shells["Step"], shells["z_near"]))
    z_far = dict(zip(shells["Step"], shells["z_far"]))
    lo = np.array([z_near[s] for s in step])
    hi = np.array([z_far[s] for s in step])
    assert np.all((z >= lo - 1e-4) & (z <= hi + 1e-4))

    # radial coverage reaches max_redshift and every selected snapshot is used
    assert z.max() > MAX_Z - 0.01
    used = set(np.unique(step))
    expected = {s for s, zn in zip(shells["Step"], shells["z_near"]) if zn < MAX_Z}
    assert used == expected

    # the stored redshift is z(radius) for the run's cosmology (incl. w0, wa)
    cosmo = _ccl_cosmology(sp)
    ra = np.radians(t_native["ra_deg"].to_numpy().astype(float))
    dec = np.radians(t_native["dec_deg"].to_numpy().astype(float))
    assert np.all(np.isfinite(ra + dec))
    chi = _chi_Mpc(cosmo, z)
    chi_hi = _chi_Mpc(cosmo, z + 1e-4)
    # radius recovered from the box: each halo sits in [edge_near, edge_far) of its step
    near, far, _ = shell_radial_edges_Mpc(shells, sp)
    idx = {s: i for i, s in enumerate(shells["Step"])}
    rows = np.array([idx[s] for s in step])
    assert np.all(chi_hi >= near[rows] - 1e-6) and np.all(chi <= far[rows] + 1e-6)


def test_package_file_behaviour_unchanged(tmp_path):
    """auto on a package z_values.txt equals the explicit file_Mpc convention."""
    sp, _, nersc = make_simulation(str(tmp_path), "w0wa")
    t_auto, _ = build(nersc, sp, str(tmp_path / "auto.parquet"))
    t_file, _ = build(nersc, sp, str(tmp_path / "file.parquet"), distance_convention="file_Mpc")
    assert t_auto.equals(t_file)


def test_hypatia_driver_layout_without_conversion(tmp_path):
    """The Hypatia driver extracts fof/ next to the Gower z_values.txt and nothing else."""
    sp, native, _ = make_simulation(str(tmp_path), "w0wa", n_halos=300)
    assert not any(f.endswith(".parquet") for f in os.listdir(native))
    t, _ = build(native, sp, str(tmp_path / "hyp.parquet"), include_ia_shapes=True)
    assert {"host_e1_x10000", "host_e2_x10000", "shape_valid"} <= set(t.column_names)
    assert t["redshift"].to_numpy().max() / 1e4 > MAX_Z - 0.01


# ---------------------------------------------------------------------------
# Real shell tables on NERSC (skipped elsewhere)
# ---------------------------------------------------------------------------

REAL_SHELL_TABLES = [
    # (simulation directory, expected dialect)
    ("/global/cfs/cdirs/m5099/GowerSt2/runsU/run001", "native"),
    ("/global/cfs/cdirs/m5099/GowerSt2/Fiducial/1_big", "native"),
    ("/global/cfs/cdirs/m5099/GowerSt2/Flamingo_big", "native"),
    ("/global/cfs/cdirs/m5099/GowerSt2/runsV/run001", "gower"),
    ("/pscratch/sd/m/mgatti/run001", "gower"),
]


@pytest.mark.parametrize("sim,dialect", REAL_SHELL_TABLES)
def test_real_shell_tables(sim, dialect):
    path = os.path.join(sim, "z_values.txt")
    if not os.path.exists(path):
        pytest.skip(f"{path} not available")
    from LSS_forward_model.cosmology import read_sims_params
    sp, _ = read_sims_params(sim + "/")
    shells = recover_shell_info(path)
    assert identify_distance_columns(shells, sp)["dialect"] == dialect
