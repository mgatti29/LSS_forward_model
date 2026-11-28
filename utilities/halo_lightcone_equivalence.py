"""Check that the native and NERSC pathways give the same halo lightcone.

Given one simulation in its native layout (fof/run.SSSSS.fofstats.0 and the
z_values.txt it came with) and in the NERSC layout (run.SSSSS.fofstats.parquet
from convert_fof_and_cleanup and z_values.txt from build_z_values_file), this
script

1. compares every snapshot halo by halo (positions, masses, radii, SO fields,
   inertia tensors) as returned by halo_inputs.load_halo_snapshot;
2. builds the halo lightcone from each layout with save_halocatalog;
3. compares the two catalogues column by column, and checks that every halo
   lies in the redshift range of the snapshot it was taken from, that the
   catalogue reaches max_redshift and that every expected snapshot is used.

Usage:
  python halo_lightcone_equivalence.py NATIVE_DIR NERSC_DIR OUT_DIR \
      --max-z 1.0 --mass-cut 13.0 [--shapes] [--snapshots-only] [--skip-snapshots]

Cosmology (sims_parameters) is read from NATIVE_DIR with read_sims_params.
A JSON report is written to OUT_DIR/report_<tag>.json.
"""
import argparse
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq

from LSS_forward_model.cosmology import read_sims_params, recover_shell_info
from LSS_forward_model.halo_inputs import (
    find_halo_snapshot, identify_distance_columns, load_halo_snapshot,
    shell_radial_edges_Mpc, z_values_header_unit)
from LSS_forward_model.halos import save_halocatalog


def compare_snapshots(native, nersc, steps, sp, shapes):
    rows = []
    for step in steps:
        a = load_halo_snapshot(native, step, sp, include_ia_shapes=shapes)
        b = load_halo_snapshot(nersc, step, sp, include_ia_shapes=shapes)
        keys = [k for k in ("x", "y", "z", "M", "rhalf", "M200c", "r200c", "inertia_tensor")
                if k in a or k in b]
        same = {k: bool(k in a and k in b and np.array_equal(a[k], b[k])) for k in keys}
        rows.append({"step": int(step), "n_halos": int(len(a["M"])), "formats": [a["format"], b["format"]],
                     "has_so": [a["has_so"], b["has_so"]], "identical": same,
                     "all_identical": all(same.values()) and len(a["M"]) == len(b["M"])})
        print(f"step {step}: {rows[-1]['n_halos']} halos, identical={rows[-1]['all_identical']}", flush=True)
    return rows


def association(table, shells):
    z = table["redshift"].to_numpy() / 1e4
    step = table["simulation_step"].to_numpy()
    z_near = dict(zip(shells["Step"].astype(int), shells["z_near"]))
    z_far = dict(zip(shells["Step"].astype(int), shells["z_far"]))
    lo = np.vectorize(z_near.get)(step)
    hi = np.vectorize(z_far.get)(step)
    inside = (z >= lo - 1e-4) & (z <= hi + 1e-4)
    return {"n_halos": int(len(z)), "max_redshift": float(z.max()),
            "frac_inside_own_snapshot_z_range": float(inside.mean()),
            "steps_used": sorted(int(s) for s in np.unique(step))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("native")
    ap.add_argument("nersc")
    ap.add_argument("out")
    ap.add_argument("--max-z", type=float, default=1.0)
    ap.add_argument("--mass-cut", type=float, default=13.0)
    ap.add_argument("--shapes", action="store_true")
    ap.add_argument("--snapshots-only", action="store_true")
    ap.add_argument("--skip-snapshots", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or f"z{args.max_z:g}_m{args.mass_cut:g}{'_shapes' if args.shapes else ''}"
    sp, _ = read_sims_params(args.native.rstrip("/") + "/")
    report = {"native": args.native, "nersc": args.nersc, "max_z": args.max_z,
              "mass_cut": args.mass_cut, "shapes": args.shapes,
              "cosmology": {k: float(sp[k]) for k in ("h", "Omega_m", "w0", "wa", "m_nu", "dBoxSize Mpc/h")}}

    shells = {}
    for name, d in (("native", args.native), ("nersc", args.nersc)):
        sh = recover_shell_info(os.path.join(d, "z_values.txt"))
        near, far, info = shell_radial_edges_Mpc(sh, sp)
        shells[name] = sh
        report[f"z_values_{name}"] = {"header_unit": z_values_header_unit(os.path.join(d, "z_values.txt")),
                                     **identify_distance_columns(sh, sp), "convention": info["convention"]}
    a, b = shells["native"], shells["nersc"]
    report["z_columns_max_abs_difference"] = float(max(np.max(np.abs(a["z_far"] - b["z_far"])),
                                                       np.max(np.abs(a["z_near"] - b["z_near"]))))
    near_a, far_a, _ = shell_radial_edges_Mpc(a, sp)
    near_b, far_b, _ = shell_radial_edges_Mpc(b, sp)
    report["radial_edges_max_abs_difference_Mpc"] = float(max(np.max(np.abs(far_a - far_b)),
                                                               np.max(np.abs(near_a - near_b))))
    steps = a["Step"][a["z_near"] < args.max_z].astype(int)

    if not args.skip_snapshots:
        t0 = time.time()
        rows = compare_snapshots(args.native, args.nersc, steps, sp, args.shapes)
        report["snapshots"] = {"n_steps": len(rows), "all_identical": all(r["all_identical"] for r in rows),
                               "n_halos_total": int(sum(r["n_halos"] for r in rows)),
                               "seconds": time.time() - t0, "per_step": rows}
    if not args.snapshots_only:
        tables = {}
        for name, d in (("native", args.native), ("nersc", args.nersc)):
            path = os.path.join(args.out, f"halo_catalog_{name}_{tag}.parquet")
            if os.path.exists(path):
                os.remove(path)
            t0 = time.time()
            save_halocatalog(shells[name], sp, max_redshift=args.max_z, halo_snapshots_path=d,
                             catalog_path=path, log10_mass_limit=args.mass_cut,
                             include_ia_shapes=args.shapes)
            tables[name] = pq.read_table(path)
            md = pq.ParquetFile(path).metadata.metadata
            report[f"lightcone_{name}"] = {
                "seconds": time.time() - t0, **association(tables[name], shells[name]),
                "metadata": {k.decode().split(".")[-1]: v.decode() for k, v in md.items()
                             if k.startswith(b"lss_forward_model")}}
        ta, tb = tables["native"], tables["nersc"]
        report["lightcone_columns_identical"] = {
            c: bool(c in tb.column_names and ta[c].equals(tb[c])) for c in ta.column_names}
        report["lightcone_identical"] = (ta.column_names == tb.column_names
                                         and all(report["lightcone_columns_identical"].values()))
        expected = sorted(int(s) for s in steps)
        report["all_expected_steps_used"] = report["lightcone_native"]["steps_used"] == expected
    with open(os.path.join(args.out, f"report_{tag}.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "snapshots"}, indent=1))
    if "snapshots" in report:
        print("snapshots all identical:", report["snapshots"]["all_identical"])


if __name__ == "__main__":
    main()
