"""Smallest f_c allowed per shell: (1 - f_sat) N_target <= f_c (N_halo + N_virtual), Gower Fiducial/1_big, DES Y6."""
import sys, json, numpy as np, pyarrow.parquet as pq, pyarrow.compute as pc
sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
from LSS_forward_model.cosmology import read_sims_params, recover_shell_info
from LSS_forward_model.halo_ia import virtual_halo_counts, expected_counts_in_redshift_bins, _halo_catalog_max_redshift
from LSS_forward_model.maps import apply_nz_shifts_and_build_shells
from LSS_forward_model.source_clustering import fracdet_map
sim = '/global/cfs/cdirs/m5099/GowerSt2/Fiducial/1_big/'
cat = '/pscratch/sd/m/mgatti/source_clustering_v2/halo_ia/halo_catalog_ia_z2_m12_Fiducial_1_big.parquet'
sp, cb = read_sims_params(sim)
si = recover_shell_info(sim + '/z_values.txt', max_z=49)
nz = np.load('/global/cfs/cdirs/m5099/DESY3/nz_DESY6.npy', allow_pickle=True).item()
nz_shifted, shells, steps, zeff, ngal = apply_nz_shifts_and_build_shells(z_rebinned=nz['z_rebinned'], nz_all=nz['nz_rebinned'], dz_values=[0]*4, shells_info=si)
cats = np.load('/global/cfs/cdirs/m5099/DESY3/DESY6.npy', allow_pickle=True).item()
fsky = fracdet_map(cats, 1024).mean(); counts = np.array([len(cats[t]['e1']) for t in range(4)]); del cats
nz_total = np.sum([counts[t]*nz_shifted[t] for t in range(4)], axis=0)
z = np.asarray(nz['z_rebinned']); support = nz_total > 0; zmax_t = z[support].max()
model_max = min(_halo_catalog_max_redshift(cat), zmax_t)
t = pq.read_table(cat, columns=['shell_index', 'log_M'])
t = t.filter(pc.greater(t['log_M'], 12000))
sh, nh = np.unique(t['shell_index'].to_numpy(), return_counts=True); nhalo = dict(zip(sh, nh))
zn, zf = np.asarray(si['z_near'], float)[::-1], np.asarray(si['z_far'], float)[::-1]
rows = []
for i, (lo, hi) in enumerate(zip(zn, zf)):
    lo, hi = max(0., lo), min(hi, model_max)
    if hi <= lo: continue
    Nt = expected_counts_in_redshift_bins([lo, hi], z, nz_total, counts.sum()/fsky, normalization_bounds=(0., zmax_t))[0]
    if Nt <= 0: continue
    lm, vc = virtual_halo_counts(cb['cosmo_pyccl'], lo, hi, 12.0, mass_def='fof')
    Nh = nhalo.get(i, 0); Nv = vc.sum()
    rows.append(dict(shell=i, z=[round(lo, 3), round(hi, 3)], N_target=float(Nt), N_halo=int(Nh), N_virtual=float(Nv),
                     fc_min={str(fs): float((1-fs)*Nt/(Nh+Nv)) for fs in (0.3, 0.55, 0.8)}))
    print(rows[-1]['shell'], rows[-1]['z'], f"Nt {Nt:.3g} Nh {Nh:.3g} Nv {Nv:.3g}", {k: round(v, 3) for k, v in rows[-1]['fc_min'].items()}, flush=True)
json.dump(rows, open('out/feasibility_gower_des.json', 'w'), indent=1)
