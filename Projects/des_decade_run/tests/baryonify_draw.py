#!/usr/bin/env python3
"""Baryonify one draw of a test simulation exactly as make_mocks.py --make-baryons does (same seed, settings and
sys_baryo file), writing delta_b_1024_{b}.npy atomically (temporary file + rename) so a concurrent make_mocks.py
run only ever sees the complete file.  usage: baryonify_draw.py SIM B"""
import os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_mocks as mm

sim, b = Path(sys.argv[1]).resolve(), int(sys.argv[2])
sim_id = f'{sim.parent.name}/{sim.name}'
final = sim/f'delta_b_{mm.NSIDE}_{b}.npy'
if final.exists():
    sys.exit(f'{final} exists')
sims_parameters, cosmo_bundle = mm.read_sims_params(str(sim) + '/')
shells_info = mm.recover_shell_info(str(sim/'z_values.txt'), max_z=49)
shells, _, _ = mm.build_shells(shells_info)
np.random.seed(mm.stable_seed(sim_id, 'baryons', b))
baryons = dict(mm.BARYONS, filename_new_params=f'sys_baryo_{b}.npy', values_to_update=mm.draw_params_from_specs(mm.BARYON_PRIORS))
mm.load_or_save_updated_params(str(sim) + '/', baryons['base_params_path'], baryons['filename_new_params'], baryons['values_to_update'])
mm.log(f'baryonifying draw {b}:', {k: float(v) for k, v in baryons['values_to_update'].items()})
tmp = sim/f'tmp_delta_b_{mm.NSIDE}_{b}.npy'
mm.load_and_baryonify_gower_st_shells(str(sim) + '/', sims_parameters, cosmo_bundle, baryons, mm.NSIDE, shells_info, shells,
                                      dens_path=str(tmp), tsz_path=str(sim/f'tsz_{mm.NSIDE}_{b}.npy'),
                                      halo_catalog_path=str(sim/'halo_catalog.parquet'))
os.replace(tmp, final)
mm.log('wrote', final)
