import os, math
import numpy as np
from astropy.cosmology import wCDM, Flatw0waCDM
from pathlib import Path
import healpy as hp
import pyarrow as pa
import glob
import pandas as pd
import pyarrow.parquet as pq
import frogress
import astropy.units as u

def build_z_values_file(directory, out_name, out_dir, sims_parameters=None):
    """
    One-shot: parse z from run.log, cosmology from sims_parameters or log, box from sims_parameters or control.par,
    compute comoving distances/volumes, and write z_values.txt.
    """
    log_file = os.path.join(directory, f"{out_name}.log")
    ctl_file = os.path.join(directory, "control.par")
    out_file = os.path.join(out_dir, "z_values.txt")
    print(f"Writing data to {out_file}...")

    # --- z history from run.log (2nd numeric column), skip comment lines
    z = np.loadtxt(log_file, comments="#", usecols=[1])
    step = np.arange(z.size)


  
    Om0 = float(sims_parameters["Omega_m"])
    w0  = float(sims_parameters["w0"])
    wa  = float(sims_parameters["wa"])
    H0 =  float(sims_parameters["h"]) * 100.0
    Lbox_Mpc = float(sims_parameters["dBoxSize Mpc/h"]) / float(sims_parameters["h"])

    # fallback to control.par for box if still unknown
    if Lbox_Mpc is None:
        with open(ctl_file) as f:
            for line in f:
                if line.split("#")[0].split("=")[0].strip() == "dBoxSize":
                    Lbox_Mpc = float(line.split("#")[0].split("=")[1].strip().strip('"'))
                    break
        if Lbox_Mpc is None:
            raise ValueError("Box size not found (neither sims_parameters nor control.par)")

    # --- cosmology model (flat w0wa if wa != 0)
    cosmo = (Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa)
             if abs(wa) > 0 else
             wCDM(H0=H0, Om0=Om0, w0=w0, Ode0=1-Om0))

    cmd = cosmo.comoving_distance(z).value  # Mpc

    # pair-wise far/near
    s = step[1:]
    z_far, z_near = z[:-1], z[1:]
    dz = z_far - z_near
    cmd_far, cmd_near = cmd[:-1], cmd[1:]
    dcmd = cmd_far - cmd_near

    cmd_box_far, cmd_box_near = cmd_far / Lbox_Mpc, cmd_near / Lbox_Mpc
    dcmd_box = dcmd / Lbox_Mpc

    vol_slice = (4.0/3.0) * math.pi * (cmd_far**3 - cmd_near**3)          # Mpc^3
    vol_over_box = vol_slice / (Lbox_Mpc**3)

    header = ("Step,z_far,z_near,delta_z,"
              "cmd_far(Mpc),cmd_near(Mpc),delta_cmd(Mpc),"
              "cmd/box_far,cmd/box_near,delta_cmd/box,"
              "cmvolume,cmvolume/boxvolume")
    data = np.column_stack([s, z_far, z_near, dz,
                            cmd_far, cmd_near, dcmd,
                            cmd_box_far, cmd_box_near, dcmd_box,
                            vol_slice, vol_over_box])
    fmt = ["%i"] + ["%.6f"]*9 + ["%.2f", "%.6f"]
    np.savetxt(out_file, data, fmt=fmt, delimiter=",", header=header)



def _parquet_path(base_dir, shell, nside):
    return f"{base_dir}/particles_{int(shell)}_{int(nside)}.parquet"

def _hpb_glob(base_dir, shell):
    return glob.glob(f"{base_dir}/run.{int(shell):05d}.hpb*")

def _verify_parquet(path, nside):
    tbl = pq.read_table(path)
    arr = tbl.to_pandas().to_numpy().ravel()
    if arr.size != hp.nside2npix(nside):
        raise ValueError(f"{os.path.basename(path)} size {arr.size} != nside2npix({nside})")
    return True

def _write_parquet_from_hpb(base_dir, shell, nside, dtype=np.uint16):
    # Build HEALPix map, cast, write parquet
    base = f"{base_dir}/run.{int(shell):05d}.hpb"
    mm = one_healpix_map_from_basefilename(base, nside).astype(dtype, copy=False)
    table = pa.Table.from_pandas(pd.DataFrame(mm, columns=["val"]), preserve_index=False)
    out = _parquet_path(base_dir, shell, nside)
    pq.write_table(table, out, compression="zstd")
    return out

def convert_and_cleanup(base_dir, resume, nside=4096, z_max_for_convert=6.0):
    """
    Walk shells in 'resume', create parquet if missing (z_far < z_max_for_convert),
    verify parquet, then remove corresponding .hpb* shards.
    If parquet exists and .hpb* shards are present, verify parquet and remove shards.
    """
    n_pix_expected = hp.nside2npix(nside)

    for s_idx in frogress.bar(range(len(resume["Step"]))):
        shell = int(resume["Step"][s_idx])
        z_far = float(resume["z_far"][s_idx])

        parquet_file = _parquet_path(base_dir, shell, nside)
        hpb_files = _hpb_glob(base_dir, shell)

        try:
            if not os.path.exists(parquet_file):
                # only convert for reasonable z (your original condition)
                if z_far < z_max_for_convert:
                    if not hpb_files:
                        # nothing to convert from; skip
                        continue
                    # write parquet and verify
                    out = _write_parquet_from_hpb(base_dir, shell, nside)
                    _verify_parquet(out, nside)
                    # success: remove hpb shards
                    for p in hpb_files:
                        os.remove(p)
                else:
                    # optional: if z too large, just delete hpb shards (your old else-branch)
                    for p in hpb_files:
                        os.remove(p)
            else:
                # parquet exists; if shard files exist, verify parquet then delete shards
                if hpb_files:
                    _verify_parquet(parquet_file, nside)
                    for p in hpb_files:
                        os.remove(p)
        except Exception as e:
            # leave files in place on failure; report and continue
            print(f"[shell {shell}] skipped due to error: {e}")
            continue


def one_healpix_map_from_basefilename(basefilename, nside):
    """Load basefilename.<int> shards, sum 'grouped'+'ungrouped', trim to NSIDE size; padding must be zero."""
    if not hp.isnsideok(nside):
        raise ValueError(f"Invalid NSIDE: {nside}")

    base = Path(basefilename)
    shards = sorted(
        (p for p in base.parent.glob(f"{base.name}.*") if p.is_file()),
        key=lambda p: int(p.suffix[1:]),
    )
    if not shards:
        raise ValueError(f"No shards found for '{base}'")

    dt = np.dtype([("grouped", "=i4"), ("ungrouped", "=i4"), ("potential", "=f4")])
    parts = []
    for p in shards:
        d = np.fromfile(p, dtype=dt)
        parts.append((d["grouped"] + d["ungrouped"]).astype(np.int64))
    arr = np.concatenate(parts)

    n = hp.nside2npix(nside)
    if arr.size < n:
        raise ValueError(f"Need {n} elems, found {arr.size}")
    if arr.size > n and not np.all(arr[n:] == 0):
        raise ValueError("Non-zero padding past required pixel count")

    return arr[:n]




def convert_fof_and_cleanup(base_dir, resume, sims_parameters, z_max_for_convert=6.0):
    """
    Convert run.00{shell:03}.fofstats.0 -> run.00{shell:03}.fofstats.parquet
    Verify parquet loads, then delete the original .fofstats.0.
    If parquet exists already and .fofstats.0 is present, just verify then delete the .fofstats.0.
    """

    # ---- cosmology & mass unit factors
    Om0 = float(sims_parameters["Omega_m"])
    w0  = float(sims_parameters["w0"])
    wa  = float(sims_parameters["wa"])
    H0  = float(sims_parameters["h"]) * 100.0
    Lbox_Mpc = float(sims_parameters["dBoxSize Mpc/h"]) / float(sims_parameters["h"])

    cosmo = Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa) if abs(wa) > 0 else wCDM(H0=H0, Om0=Om0, w0=w0, Ode0=1-Om0)

    Lbox = Lbox_Mpc * u.Mpc
    f_mass = (Lbox**3 * cosmo.critical_density(0).to(u.Msun/u.Mpc**3)).value  # Msun

    # ---- PKDGRAV halo dtype (as you had it)
    pkd_halo_dtype = np.dtype([
        ("rPot", ("f4", 3)), ("minPot", "f4"), ("rcen", ("f4", 3)),
        ("rcom", ("f4", 3)), ("cvom", ("f4", 3)), ("angular", ("f4", 3)),
        ("inertia", ("f4", 6)), ("sigma", "f4"), ("rMax", "f4"),
        ("fMAss", "f4"), ("fEnvironDensity0", "f4"),
        ("fEnvironDensity1", "f4"), ("rHalf", "f4")
    ])

    def fof_path(shell):    return f"{base_dir}/run.00{int(shell):03d}.fofstats.0"
    def pq_path(shell):     return f"{base_dir}/run.00{int(shell):03d}.fofstats.parquet"

    for s_idx in frogress.bar(range(len(resume["Step"]))):
        shell = int(resume["Step"][s_idx])
        z_far = float(resume["z_far"][s_idx])

        src = fof_path(shell)
        dst = pq_path(shell)

        try:
            if not os.path.exists(dst):
                # convert only when z condition holds and source exists
                if z_far < z_max_for_convert and os.path.exists(src):
                    halos = np.fromfile(src, dtype=pkd_halo_dtype)
                    if halos.size == 0:
                        print(f"[shell {shell:03d}] empty FOF file; skipping")
                        continue

                    # features (your formulas, compacted)
                    halo_center = Lbox_Mpc * (halos["rPot"] + halos["rcen"] + 0.5)    # shape (N,3)
                    rmax        = (Lbox_Mpc * halos["rMax"] * 1000).astype("uint16")  # kpc in uint16
                    log10M      = (np.log10(halos["fMAss"] * f_mass) * 1000).astype("uint16")

                    # inertia autos: take (0,3,5) diagonals, log10 scaled
                    pm = np.empty((halos.size, 3), dtype=np.float32)
                    pm[:, 0] = halos["inertia"][:, 0]
                    pm[:, 1] = halos["inertia"][:, 3]
                    pm[:, 2] = halos["inertia"][:, 5]
                    inertia_auto = (np.log10(pm * 1e20) * 1000).astype("uint16")

                    # inertia crosses normalized: (1,2,4)
                    cross = np.empty((halos.size, 3), dtype=np.float32)
                    cross[:, 0] = halos["inertia"][:, 1] / np.sqrt(pm[:, 0] * pm[:, 1])
                    cross[:, 1] = halos["inertia"][:, 2] / np.sqrt(pm[:, 0] * pm[:, 2])
                    cross[:, 2] = halos["inertia"][:, 4] / np.sqrt(pm[:, 1] * pm[:, 2])
                    inertia_cross = (10000 * (1 + cross)).astype("uint16")

                    # angular: signed log scale
                    angular = (np.sign(halos["angular"]) * np.log10(np.abs(halos["angular"]) * 1e20 + 1) * 1000).astype("int16")

                    df = pd.DataFrame({
                        "halo_center":   halo_center.tolist(),       # list-of-3 float
                        "rmax":          rmax,                        # uint16
                        "log10M":   log10M,                      # uint16
                        "inertia_auto":  inertia_auto.tolist(),       # list-of-3 uint16
                        "inertia_cross": inertia_cross.tolist(),      # list-of-3 uint16
                        "angular":       angular.tolist(),            # list-of-3 int16
                    })
                    df.to_parquet(dst, engine="pyarrow", compression="brotli")

                    # verify parquet loads
                    _ = pq.read_table(dst)  # will raise on corruption
                    os.remove(src)
                else:
                    # parquet missing but condition/source unmet -> do nothing
                    continue
            else:
                # parquet exists; if src present, verify parquet then delete src
                if os.path.exists(src):
                    _ = pq.read_table(dst)
                    os.remove(src)

        except Exception as e:
            print(f"[shell {shell:03d}] error: {e} (left files untouched)")
            continue


def delete_snapshot(path_simulation, resume, size_limit_mb=10, dry_run=False):
    """
    Delete snapshot files run.00000, run.00001, ... in `path_simulation`
    unless they are larger than `size_limit_mb` (default 10 MB).

    Returns a dict with lists: deleted, skipped_large, missing, errors.
    Set dry_run=True to just see what would happen.
    """
    size_limit_bytes = int(size_limit_mb * 1024 * 1024)

    deleted = []
    skipped_large = []
    missing = []
    errors = []

    def _3dsnap_path(base_dir, shell):
        # Original code used glob on a specific filename and took [0].
        # We'll just build the exact expected path, which is safer.
        return Path(base_dir) / f"run.{int(shell):05d}"

    for step in resume['Step']:
        p = _3dsnap_path(path_simulation, step)
        try:
            if not p.exists():
                missing.append(str(p))
                continue
            if p.is_dir():
                errors.append((str(p), "is a directory, not a file"))
                continue

            size = p.stat().st_size
            if size > size_limit_bytes:
                skipped_large.append(str(p))
                continue

            if not dry_run:
                p.unlink()
            deleted.append(str(p))

        except Exception as e:
            errors.append((str(p), repr(e)))

    return {
        "deleted": deleted,
        "skipped_large": skipped_large,
        "missing": missing,
        "errors": errors,
    }
    
