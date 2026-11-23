#!/usr/bin/env python3
"""Checks of the make_mocks.py end-to-end test outputs (one simulation directory).

Per sample, baryon draw, rotation and IA model:
  * E/B pseudo-C_ell of the signal maps (g1_map, g2_map; masked full sky / f_sky) against CCL: GG with the n(z)
    realisation (dz) of the mock, x (1 + m)^2 x pixwin^2 (NLA mocks: GG + GI + II of the drawn A_IA, eta_IA);
  * noise: std of the pure-noise map e1n and of e1 - (1 + m) g1 (the noise added to the signal);
  * halo IA: fraction of pixels with var_IA > var_noise, number of mark draws (rejections);
  * NLA vs halo signal on the same density, nuisances and noise seeds.
kappa_CMB: C_ell of the dmo / b0 / b1 maps against CCL (z* = 1100), and the cross-correlation coefficient
with the signal map of the highest bin of each sample (same simulation frame).

usage: validate_e2e.py OUT_DIR   ->  tests/out/e2e_<sim>.json and .png
"""
import json
import os
import sys
from pathlib import Path

import healpy as hp
import numpy as np
import pyccl as ccl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from LSS_forward_model.cosmology import read_sims_params, recover_shell_info   # noqa: E402
from LSS_forward_model.maps import apply_nz_shifts_and_build_shells            # noqa: E402

NSIDE, LMAX = 1024, 2000
EDGES = np.unique(np.rint(np.geomspace(30, LMAX, 11)).astype(int))
CFG = json.loads((HERE.parent/'config'/'samples.json').read_text())
INPUTS = json.loads((HERE.parent/'config'/'large_inputs.json').read_text())
STAGE = Path(os.environ.get('DES_DECADE_INPUTS', '/global/cfs/cdirs/m5099/des_decade_run_inputs'))


def bands(cl):
    return np.array([cl[a:b].mean() for a, b in zip(EDGES[:-1], EDGES[1:])])


def nz_of(sample, params, shells_info):
    nzm = CFG['samples'][sample]['nz_m']
    if nzm['model'] == 'paired_realisations':
        f = np.load(STAGE/INPUTS['files'][nzm['large_inputs']]['file'])
        nz = np.array(f['nzs'][params['idx_rel_nz']], float)
        if 'z_rebinned' in f.files:
            z = f['z_rebinned']
        else:
            z = np.arange(0, 3, 0.01)
            nz[:, -1] = 0.
    else:
        d = np.load(HERE.parent/nzm['nz'], allow_pickle=True).item()
        z, nz = np.asarray(d['z_rebinned']), np.asarray(d['nz_rebinned'])
    nz_shifted = apply_nz_shifts_and_build_shells(z, nz, params['dz'], shells_info)[0]
    return np.asarray(z), np.asarray(nz_shifted)


def theory(cosmo, z, nz, params, model):
    ell = np.arange(LMAX + 1)
    ia = None
    if model == 'nla':
        ia = (z, params['A_IA']*((1 + z)/1.67)**params['eta_IA'])
    pw = hp.pixwin(NSIDE, lmax=LMAX)**2
    out = []
    for t in range(len(nz)):
        good = z > 0
        tr = ccl.WeakLensingTracer(cosmo, dndz=(z[good], nz[t][good]), ia_bias=None if ia is None else (z[good], ia[1][good]))
        out.append(bands(ccl.angular_cl(cosmo, tr, tr, ell)*pw*params['dm'][t]**2))
    return np.array(out)


def full(idx, values):
    m = np.zeros(hp.nside2npix(NSIDE))
    m[idx] = values
    return m


def main():
    out_dir = Path(sys.argv[1])
    sim_dir = out_dir           # simulation files: the output directory, or the simulation of mocks_done.json
    if not (out_dir/'z_values.txt').exists() and (out_dir/'mocks_done.json').exists():
        sim_dir = Path(json.loads((out_dir/'mocks_done.json').read_text())['sim'])
    sims_parameters, cosmo_bundle = read_sims_params(str(sim_dir) + '/')
    cosmo = cosmo_bundle['cosmo_pyccl']
    shells_info = recover_shell_info(str(sim_dir/'z_values.txt'), max_z=49)
    res = {'ell_edges': EDGES.tolist(), 'mocks': {}, 'ia': {}, 'kappa_cmb': {}}
    signal_E = {}
    for path in sorted(out_dir.glob('mock_*_b*_r*_*.npy')):
        model, b, r, sample = path.stem.split('_')[1:]
        mock = np.load(path, allow_pickle=True).item()
        p = mock['sims_parameters']
        p.update({k: v for k, v in mock['IA_params'].items() if k in ('A_IA', 'eta_IA')})
        z, nz = nz_of(sample, p, shells_info)
        th = theory(cosmo, z, nz, p, model)
        row = {'EE/theory': [], 'BB/EE': [], 'std_e1n': [], 'std_noise_in_e': [], 'fsky': []}
        for t in [k for k in mock if isinstance(k, (int, np.integer))]:
            m = mock[t]
            mask = full(m['idx_'], 1.)
            fsky = mask.mean()
            alm = hp.map2alm_spin([full(m['idx_'], m['g1_map']), full(m['idx_'], m['g2_map'])], 2, lmax=LMAX)
            ee, bb = bands(hp.alm2cl(alm[0]))/fsky, bands(hp.alm2cl(alm[1]))/fsky
            row['EE/theory'].append((ee/th[t]).tolist())
            row['BB/EE'].append((bb/ee).tolist())
            row['std_e1n'].append(float(np.std(m['e1n'])))
            row['std_noise_in_e'].append(float(np.std(m['e1'] - p['dm'][t]*m['g1_map'])))
            row['fsky'].append(float(fsky))
            if t == max(k for k in mock if isinstance(k, (int, np.integer))):
                signal_E[(model, b, r, sample)] = alm[0]
        if 'ia_noise_fraction' in mock:
            row['ia_noise_fraction'] = mock['ia_noise_fraction']
        row['IA'] = {k: (float(v) if np.isscalar(v) and not isinstance(v, str) else v) for k, v in mock['IA_params'].items()
                     if k in ('A_c', 'eta_c', 'A_s', 'eta_s', 'f_c', 'f_sat', 'A_IA', 'eta_IA', 'n_attempts')}
        res['mocks'][path.stem] = row
        print(path.stem, 'EE/th per bin (mid bands):', [np.round(np.array(x)[[2, 5, 8]], 3).tolist() for x in row['EE/theory']],
              'noise std', np.round(row['std_e1n'], 4).tolist(), np.round(row['std_noise_in_e'], 4).tolist(),
              row.get('ia_noise_fraction', ''), row['IA'], flush=True)

    # kappa_CMB
    if (out_dir/'kappa_cmb_2048_dmo.npy').exists():
        lk = 4000
        ell = np.arange(lk + 1)
        th = ccl.angular_cl(cosmo, *(2*[ccl.CMBLensingTracer(cosmo, z_source=1100.)]), ell)*hp.pixwin(2048, lmax=lk)**2
        kedges = np.array([10, 30, 100, 300, 1000, 2000, 2048, 2200, 3000, 4000])
        kb = lambda c: np.array([c[a:b].mean() for a, b in zip(kedges[:-1], kedges[1:])])
        res['kappa_cmb']['edges'] = kedges.tolist()
        for name in ('dmo', 'b0', 'b1'):
            f = out_dir/f'kappa_cmb_2048_{name}.npy'
            if not f.exists():
                continue
            k = np.load(f).astype(np.float64)
            alm = hp.map2alm(k, lmax=lk, iter=0)
            res['kappa_cmb'][f'{name}/theory'] = (kb(hp.alm2cl(alm))/kb(th)).tolist()
            print('kappa', name, '/theory', np.round(res['kappa_cmb'][f'{name}/theory'], 3).tolist(), flush=True)
            if name == 'dmo':
                k1024 = hp.map2alm(hp.ud_grade(k, NSIDE), lmax=LMAX)
                ak = k1024
        for key, aE in signal_E.items():
            if key[1] != 'b0':
                continue
            r = bands(hp.alm2cl(ak, aE))/np.sqrt(bands(hp.alm2cl(ak))*bands(hp.alm2cl(aE)))
            res['kappa_cmb'][f'r(kappa_dmo, E {"_".join(key)})'] = r.tolist()
        print('kappa x shear r (b0, last bin):', {k: np.round(v[:6], 2).tolist() for k, v in res['kappa_cmb'].items()
                                                  if k.startswith('r(')}, flush=True)

    tag = out_dir.parent.name + '_' + out_dir.name
    (HERE/'out').mkdir(exist_ok=True)
    (HERE/'out'/f'e2e_{tag}.json').write_text(json.dumps(res, indent=1) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    centre = np.sqrt(EDGES[:-1]*EDGES[1:])
    samples = sorted({k.split('_')[-1] for k in res['mocks']})
    fig, axes = plt.subplots(1, len(samples) + 1, figsize=(4.2*(len(samples) + 1), 3.8), squeeze=False)
    for ax, s in zip(axes[0], samples):
        for key, row in res['mocks'].items():
            if not key.endswith(f'_b0_r0_{s}'):
                continue
            ls = '-' if '_halo_' in key else '--'
            for t, y in enumerate(row['EE/theory']):
                ax.plot(centre, y, ls, color=f'C{t}', label=f"{key.split('_')[1]} bin {t+1}")
        ax.axhline(1, color='k', lw=.5); ax.set_xscale('log'); ax.set_ylim(.6, 1.4)
        ax.set_title(f'{s}: signal EE / CCL GG (b0 r0)', fontsize=9); ax.legend(fontsize=6, ncol=2)
    ax = axes[0][-1]
    if 'edges' in res['kappa_cmb']:
        kc = np.sqrt(np.array(res['kappa_cmb']['edges'][:-1])*np.array(res['kappa_cmb']['edges'][1:]))
        for name in ('dmo', 'b0', 'b1'):
            if f'{name}/theory' in res['kappa_cmb']:
                ax.plot(kc, res['kappa_cmb'][f'{name}/theory'], label=name)
        ax.axhline(1, color='k', lw=.5); ax.set_xscale('log'); ax.set_title('kappa_CMB / CCL', fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE/'out'/f'e2e_{tag}.png', dpi=110)
    print('wrote', HERE/'out'/f'e2e_{tag}.json', HERE/'out'/f'e2e_{tag}.png')


if __name__ == '__main__':
    main()
