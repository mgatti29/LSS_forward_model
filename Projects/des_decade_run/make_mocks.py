#!/usr/bin/env python3
"""Per-simulation driver of the DES Y6 (mdet, BFD) / DECADE (NGC, SGC) mocks with halo IA (+ NLA), SC v2, CMB lensing.

One call processes one simulation directory:

1. IA halo lightcone (only for IA model 'halo'): halo_catalog_ia.parquet (z < 2, M > 1e12 Msun, halo shapes).
   On Hypatia the FoF files are first extracted from the run's FoF tarball (only the steps with z < 2) and
   deleted afterwards (--fof-tar, --cleanup).
2. For every baryon realisation b (delta_b_1024_{b}.npy, NOT recomputed unless --make-baryons):
   lensing fields; for every rotation r:
     * nuisance draws, shared by the IA models (nuisance_b{b}_r{r}.npy): per sample n(z)/m (row of the paired
       realisations for mdet and BFD, Gaussian dz/m for DECADE), SC v2 bias per bin + slope, noise seeds;
     * IA draw per model (IA_params_{model}_b{b}_r{r}.npy), shared by the four samples.  Halo model: population
       (f_c, f_sat) and marks (A_c, eta_c, A_s, eta_s).  The halo IA is capped locally on the noise budget
       (config ia_cap, default alpha = 1); the earlier rejection rule (marks redrawn on the same populations if, for
       any sample and bin, more than max_fraction of the footprint pixels have var_IA > var_noise (the mock
       noise is zero there, ia_noise_factor == 0) is disabled by default (ia_priors.json max_fraction = 1); the
       fraction of capped pixels per bin is stored as ia_noise_fraction.  NLA: A_IA, eta_IA (IA_model 'NLA').
     * mocks mock_{model}_b{b}_r{r}_{sample}.npy (observed pixels only, float32).
   The parameter files are written once and re-used on re-runs, like sys_baryo_{b}.npy for the baryons.
3. kappa_CMB (NSIDE 2048, ell <= 4096, noiseless, full sky, in the simulation frame -- apply the mock rotation
   when cutting a footprint): kappa_cmb_2048_{dmo,b0,b1}.npy (maps.make_kappa_cmb).
4. DONE marker (mocks_done.json) with timings and the package git hash.

Seeds are stable hashes of (simulation id, purpose, b, r, sample), so a re-run reproduces every draw.

usage: make_mocks.py SIM [--out DIR] [--models halo nla] [--samples DESY6 BFD NGC SGC] [--baryons 0 1]
                         [--rots 0 1 2 3] [--halo-catalog PATH] [--fof-tar PATH] [--make-baryons]
                         [--no-kappa-cmb] [--cleanup]
"""
import argparse
import copy
import gc
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import healpy as hp
import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
sys.path.insert(0, str(PACKAGE))
from LSS_forward_model.bfd_utils import bfd_expected_counts, make_BFD_sample           # noqa: E402
from LSS_forward_model.cosmology import read_sims_params, recover_shell_info           # noqa: E402
from LSS_forward_model.halo_ia import build_IA_halo_sources_from_catalog, make_halo_IA_numerator  # noqa: E402
from LSS_forward_model.halos import save_halocatalog                                   # noqa: E402
from LSS_forward_model.lensing import make_WL_sample                                   # noqa: E402
from LSS_forward_model.maps import (apply_nz_shifts_and_build_shells, build_shells,    # noqa: E402
                                    compute_lensing_fields, convert_to_pix_coord, draw_params_from_specs,
                                    rotate_and_rebin,
                                    load_and_baryonify_gower_st_shells, load_or_save_updated_params,
                                    make_kappa_cmb)
from LSS_forward_model.source_clustering import fracdet_map, load_SC_calibration       # noqa: E402

NSIDE = 1024
Z_MAX_IA = 2.0
LOG10_MASS_IA = 12.0
CONFIG = HERE/'config'
SAMPLES_CFG = json.loads((CONFIG/'samples.json').read_text())
IA_CFG = json.loads((CONFIG/'ia_priors.json').read_text())
SC_CFG = json.loads((CONFIG/'sc_priors.json').read_text())
INPUTS_CFG = json.loads((CONFIG/'large_inputs.json').read_text())
STAGE = Path(os.environ.get('DES_DECADE_INPUTS', '/global/cfs/cdirs/m5099/des_decade_run_inputs'))
# baryonification used for the existing Hypatia delta_b_1024_{b}.npy (only with --make-baryons)
BARYONS = {'enabled': True, 'max_z_halo_catalog': 1, 'mass_cut': 13.2, 'do_tSZ': False,
           'base_params_path': str(PACKAGE/'Data'/'Baryonification_default_parameters.npy')}
BARYON_PRIORS = {'M_c': (12.5, 15.5, 'log10'), 'theta_ej': (3.0, 10.0, 'lin'), 'eta': (-2.0, -0.1, 'log10')}
MAX_REJECTIONS = 50
FOOTPRINT_NSIDE, FOOTPRINT_RINGS = 32, 2        # IA hosts populated in the rotated footprint + ~3.7 deg margin
NPROC = len(os.sched_getaffinity(0))           # cores of this task (set by --nproc)


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'big') % (2**32 - 1)


def large_input(key):
    return STAGE/INPUTS_CFG['files'][key]['file']


def load_or_draw(path, draw):
    """Parameters stored once per (model, b, r): re-runs use the same values."""
    if path.exists():
        return np.load(path, allow_pickle=True).item(), True
    return draw(), False


# ------------------------------------------------------------------------------------------------------------
# samples: catalogues, n(z)/m, source clustering
# ------------------------------------------------------------------------------------------------------------

class Sample:
    def __init__(self, name):
        self.name = name
        self.cfg = SAMPLES_CFG['samples'][name]
        self.is_bfd = name == 'BFD'
        self.cats = np.load(large_input(self.cfg['catalogue']['large_inputs']), allow_pickle=True).item()
        self.ntomo = len(self.cats)
        self._footprints = {}
        self.sc = load_SC_calibration(HERE/self.cfg['sc_calibration'])
        self.estimator = self.cfg.get('estimator', {'type': 'ml'})
        if self.is_bfd:
            real = {t: {'ra': c['ra'][c['selection_term'] == 0], 'dec': c['dec'][c['selection_term'] == 0]}
                    for t, c in self.cats.items()}
            self.bin_counts = np.array([len(real[t]['ra']) for t in range(self.ntomo)])
            self.fsky = fracdet_map(real, NSIDE).mean()
            if self.estimator['type'] == 'count':
                # global full-catalogue response (never recomputed from the cut catalogue) and full-catalogue
                # expected counts on the retained pixels
                meta = json.loads(large_input(self.estimator['meta']).read_text())
                self.Rbar = [np.array(meta['bins'][str(t)]['Rbar_code_sign']) for t in range(self.ntomo)]
                masks = np.load(large_input(self.estimator['mask']))
                assert int(masks['nside']) == NSIDE and int(masks['nmin']) == self.estimator['nmin']
                self.lam_maps = []
                for t in range(self.ntomo):
                    lam = np.zeros(hp.nside2npix(NSIDE))
                    lam[masks[f'pix_{t}']] = masks[f'lam_{t}']
                    self.lam_maps.append(lam)
            else:
                self.lam_maps = bfd_expected_counts(self.cats, NSIDE)
            del real
        else:
            self.bin_counts = np.array([len(self.cats[t]['e1']) for t in range(self.ntomo)])
            self.fsky = fracdet_map(self.cats, NSIDE).mean()
            self.lam_maps = None
        nzm = self.cfg['nz_m']
        if nzm['model'] == 'paired_realisations':
            self.nzm = dict(np.load(large_input(nzm['large_inputs'])))
        else:
            mean = np.load(HERE/nzm['nz'], allow_pickle=True).item()
            self.nzm = {'z_rebinned': np.asarray(mean['z_rebinned']), 'nz_rebinned': np.asarray(mean['nz_rebinned'])}
        log(f'{name}: {self.bin_counts.sum():,} objects in {self.ntomo} bins, fsky {self.fsky:.4f}')

    def footprint(self, rot):
        """Boolean NSIDE-32 map (simulation frame) of the rotated footprint, dilated by FOOTPRINT_RINGS rings."""
        if rot not in self._footprints:
            mask = np.zeros(hp.nside2npix(NSIDE), bool)
            for t, c in self.cats.items():
                if self.is_bfd:
                    c = {'ra': c['ra'][c['selection_term'] == 0], 'dec': c['dec'][c['selection_term'] == 0]}
                mask[rotate_and_rebin(convert_to_pix_coord(c['ra'], c['dec'], nside=2*NSIDE), NSIDE, rot)] = True
            low = hp.ud_grade(mask.astype(float), FOOTPRINT_NSIDE) > 0
            for _ in range(FOOTPRINT_RINGS):
                neighbours = hp.get_all_neighbours(FOOTPRINT_NSIDE, np.flatnonzero(low)).ravel()
                low[neighbours[neighbours >= 0]] = True
            self._footprints[rot] = low
        return self._footprints[rot]

    def ia_noise_precheck(self, halo_IA, rot, ia_noise_nside=64):
        """Cheap estimate of the worst-bin fraction of pixels with var_IA > var_noise, before the map makers.

        var_noise = 0.5 sum w^2 e^2 / (sum w)^2 of the data shapes (no source-clustering modulation, no SC noise
        rescaling); var_IA = 0.5 <numerator_sq> on NSIDE-64 superpixels, as make_WL_sample.  BFD: 0 (its q noise
        is far above the IA white term; the exact check after the map maker still applies).
        """
        if self.is_bfd:
            return 0.0
        if getattr(self, '_noise_rot', None) != rot:
            self._noise = []
            npix = hp.nside2npix(NSIDE)
            for t in range(self.ntomo):
                c = self.cats[t]
                pix = rotate_and_rebin(convert_to_pix_coord(c['ra'], c['dec'], nside=2*NSIDE), NSIDE, rot)
                n_map = np.bincount(pix, weights=c['w'], minlength=npix)
                obs = np.flatnonzero(n_map)
                var = 0.5*np.bincount(pix, weights=c['w']**2*(c['e1']**2 + c['e2']**2), minlength=npix)[obs]/n_map[obs]**2
                self._noise.append((obs, var))
            self._noise_rot = rot
        worst = 0.0
        for t, (obs, var) in enumerate(self._noise):
            nsq = halo_IA['numerator_sq'][t]
            expected = hp.ud_grade(hp.ud_grade(nsq, ia_noise_nside), NSIDE)[obs]
            if expected.sum() > 0:
                expected *= nsq[obs].sum()/expected.sum()
            worst = max(worst, float(np.mean(0.5*expected > var)))
        return worst

    def draw_nuisance(self, rng):
        nzm, out = self.cfg['nz_m'], {}
        if nzm['model'] == 'paired_realisations':
            k = int(rng.integers(len(self.nzm['nzs'])))
            out.update(nz_row=k, dz=np.zeros(self.ntomo), dm=1. + self.nzm['ms'][:, k])
        else:
            out.update(dz=rng.normal(nzm['dz_mean'], nzm['dz_sigma']),
                       dm=1. + rng.normal(nzm['m_mean'], nzm['m_sigma']))
        lo, hi = SC_CFG['bias_sc']['range']
        out['bias_sc'] = rng.uniform(lo, hi, self.ntomo)
        lo, hi = SC_CFG['bias_sc_slope']['range']
        out['bias_sc_slope'] = rng.uniform(lo, hi, self.ntomo)
        out['noise_seed'] = int(rng.integers(2**32 - 1))
        out['population_seed'] = int(rng.integers(2**32 - 1))
        out['bin_seed'] = int(rng.integers(2**32 - 1))
        return out

    def nz(self, nuisance):
        """(z grid, n(z) per bin) of this realisation, before the dz shifts."""
        if 'nz_row' in nuisance:
            z = self.nzm['z_rebinned'] if 'z_rebinned' in self.nzm else np.arange(0, 3, 0.01)
            nz = np.array(self.nzm['nzs'][nuisance['nz_row']], dtype=float)
            if 'z_rebinned' not in self.nzm:      # mdet file: grid np.arange(0, 3, 0.01), last node zeroed
                nz[:, -1] = 0.
            return np.asarray(z), nz
        return self.nzm['z_rebinned'], self.nzm['nz_rebinned']


def shells_for(sample, nuisance, shells_info):
    z, nz = sample.nz(nuisance)
    nz_shifted, shells, steps, zeff, ngal = apply_nz_shifts_and_build_shells(
        z_rebinned=z, nz_all=nz, dz_values=nuisance['dz'], shells_info=shells_info)
    return z, np.asarray(nz_shifted), np.asarray(zeff), np.asarray(ngal)


# ------------------------------------------------------------------------------------------------------------
# IA draws
# ------------------------------------------------------------------------------------------------------------

def draw_uniform(rng, parameters):
    return {k: float(rng.uniform(*v['range'])) for k, v in parameters.items()}


def halo_marks(rng, f_c):
    """Mark parameters; the prior is on a_c = f_c A_c (GI amplitude), A_c = a_c / f_c (2026-11-20)."""
    p = IA_CFG['halo']['parameters']
    m = draw_uniform(rng, {k: p[k] for k in ('a_c', 'eta_c', 'A_s', 'eta_s')})
    m['A_c'] = m['a_c'] / f_c
    return m


def halo_draw(rng):
    pop = halo_population(rng)
    return dict(pop, **halo_marks(rng, pop['f_c']))


def halo_population(rng):
    p = IA_CFG['halo']['parameters']
    return draw_uniform(rng, {k: p[k] for k in ('f_c', 'f_sat')})


def ia_block(ia):
    fixed = IA_CFG['halo']['fixed']
    return {'ia': dict(A_c=ia['A_c'], eta_c=ia['eta_c'], A_s=ia['A_s'], eta_s=ia['eta_s'],
                       b_c=fixed['b_c'], b_s=fixed['b_s'])}


# ------------------------------------------------------------------------------------------------------------
# mock makers
# ------------------------------------------------------------------------------------------------------------

def base_parameters(sims_parameters, sample, nuisance, rot):
    p = copy.deepcopy(sims_parameters)
    p.update(rot=rot, delta_rot=0, dz=np.asarray(nuisance['dz']), dm=np.asarray(nuisance['dm']),
             bias_sc=np.asarray(nuisance['bias_sc']), bias_sc_slope=np.asarray(nuisance['bias_sc_slope']),
             A_IA=0., eta_IA=0., A_IA2=0., bias_TA=np.zeros(sample.ntomo))
    if 'nz_row' in nuisance:
        p['idx_rel_nz'] = nuisance['nz_row']
    return p


def make_mock(sample, model, ia, halo_IA, sims_parameters, nuisance, zeff, ngal, cosmo_bundle, fields, rot):
    """Full-sky mock maps of one sample + the fraction of pixels with var_IA > var_noise per bin."""
    p = base_parameters(sims_parameters, sample, nuisance, rot)
    if model == 'nla':
        p.update(A_IA=ia['A_IA'], eta_IA=ia['eta_IA'])
    kwargs = dict(noise_model='poisson', IA_model='halo' if model == 'halo' else 'NLA',
                  halo_IA=halo_IA if model == 'halo' else None)
    if model == 'halo':                                  # local IA cap, per-pixel self term (config ia_cap)
        cap = IA_CFG['halo']['ia_cap']
        kwargs.update(ia_cap_alpha=cap['alpha'])
        kwargs.update({'ia_self_nside' if sample.is_bfd else 'ia_noise_nside': cap['ia_self_nside']})
    if sample.is_bfd:
        if sample.estimator['type'] == 'count':
            kwargs.update(estimator='count', Rbar=sample.Rbar, nmin=sample.estimator['nmin'])
        maps, _ = make_BFD_sample(ngal, zeff, cosmo_bundle, p, NSIDE, fields, sample.cats, SC_corrections=sample.sc,
                                  include_SC=True, do_noise=True, compact_savings=False, minimal_output=False,
                                  rng=np.random.default_rng(nuisance['noise_seed']), lam_maps=sample.lam_maps,
                                  **kwargs)
        mask_key = 'mask'
    else:
        np.random.seed(nuisance['noise_seed'])         # make_WL_sample rotates the shapes with np.random
        maps, _ = make_WL_sample(ngal, zeff, cosmo_bundle, p, NSIDE, fields, sample.cats, SC_corrections=sample.sc,
                                 include_SC=True, compact_savings=False, **kwargs)
        mask_key = None
    out, frac = {}, []
    for t in range(sample.ntomo):
        m = maps[t]
        observed = (m[mask_key] > 0) if mask_key else (m['var_'] > 0)
        idx = np.flatnonzero(observed)
        if model == 'halo':
            frac.append(float(np.mean(m['ia_noise_factor'][idx] == 0)))
        keys = ('g1_map', 'g2_map', 'e1', 'e2', 'e1n', 'e2n')
        out[t] = {k: m[k][idx].astype(np.float32) for k in keys}
        out[t]['idx_'] = idx.astype(np.int32)
        if sample.is_bfd:
            out[t]['n_source'] = m['n_source'][idx].astype(np.float32)
    del maps
    out['sims_parameters'] = p
    return out, frac


def population_for(sample, nuisance, population, shells_info, sims_parameters, cosmo_bundle, halo_catalog, rot):
    z, nz_shifted, _, _ = shells_for(sample, nuisance, shells_info)
    nz_total = np.sum(sample.bin_counts[:, None]*nz_shifted, axis=0)
    fixed = IA_CFG['halo']['fixed']
    pop = dict(f_c=population['f_c'], f_sat=population['f_sat'], sigma_logM=fixed['sigma_logM'], alpha=fixed['alpha'])
    return build_IA_halo_sources_from_catalog(
        path_halo_cat=str(halo_catalog), sims_parameters=sims_parameters, halo_catalog_log10mass_cut=LOG10_MASS_IA,
        total_number_counts=sample.bin_counts.sum()/sample.fsky, nz_redshift=z, nz=nz_total,
        shells_info=shells_info, model_parameters={'population': pop, 'ia': {}}, cosmo_bundle=cosmo_bundle,
        seed=nuisance['population_seed'], virtual_halos=fixed['virtual_halos'], show_progress=False,
        footprint=sample.footprint(rot), nproc=min(32, NPROC))


# ------------------------------------------------------------------------------------------------------------
# per simulation
# ------------------------------------------------------------------------------------------------------------

def git_hash():
    try:
        return subprocess.check_output(['git', '-C', str(PACKAGE), 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


def extract_fof(tar_path, sim, steps):
    """Extract fof/run.NNNNN.fofstats.0 of the requested steps into SIM/fof."""
    wanted = {f'run.{int(s):05d}.fofstats.0' for s in steps}
    (sim/'fof').mkdir(exist_ok=True)
    with tarfile.open(tar_path, 'r:*') as tar:
        for member in tar:
            name = Path(member.name).name
            if member.isfile() and name in wanted:
                member.name = name
                tar.extract(member, sim/'fof')
    return sorted((sim/'fof').glob('run.*.fofstats.0'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sim', type=Path)
    ap.add_argument('--out', type=Path, help='output directory (default: the simulation directory)')
    ap.add_argument('--sim-id', help='identifier for the seeds (default: <parent>/<name> of SIM)')
    ap.add_argument('--models', nargs='+', default=['halo'], choices=['halo', 'nla'])
    ap.add_argument('--samples', nargs='+', default=['DESY6', 'BFD', 'NGC', 'SGC'])
    ap.add_argument('--baryons', nargs='+', type=int, default=[0, 1])
    ap.add_argument('--rots', nargs='+', type=int, default=[0, 1, 2, 3])
    ap.add_argument('--halo-catalog', type=Path, help='IA halo lightcone (default OUT/halo_catalog_ia.parquet)')
    ap.add_argument('--fof-tar', type=Path, help='FoF tarball to extract if SIM/fof is missing (Hypatia)')
    ap.add_argument('--make-baryons', action='store_true', help='baryonify if delta_b_1024_{b}.npy is missing')
    ap.add_argument('--no-kappa-cmb', action='store_true')
    ap.add_argument('--cleanup', action='store_true', help='delete extracted FoF files and the IA halo lightcone')
    ap.add_argument('--nproc', type=int, help='processes for the forked pools (default: cores of this task)')
    args = ap.parse_args()

    global NPROC
    NPROC = args.nproc or NPROC
    t_start = time.time()
    timings = {}
    sim = args.sim.resolve()
    out = (args.out or sim).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sim_id = args.sim_id or f'{sim.parent.name}/{sim.name}'
    log('simulation', sim, 'id', sim_id, 'output', out)

    sims_parameters, cosmo_bundle = read_sims_params(str(sim) + '/')
    shells_info = recover_shell_info(str(sim/'z_values.txt'), max_z=49)
    shells, _, _ = build_shells(shells_info)
    steps = np.asarray(shells_info['Step'])[::-1]

    # 1. IA halo lightcone ---------------------------------------------------------------------------------
    halo_catalog = args.halo_catalog or out/'halo_catalog_ia.parquet'
    extracted = []
    if 'halo' in args.models and not halo_catalog.exists():
        t0 = time.time()
        z_near = np.asarray(shells_info['z_near'])
        ia_steps = np.asarray(shells_info['Step'])[z_near < Z_MAX_IA]
        has_parquet = any((sim/f'run.{int(s):05d}.fofstats.parquet').exists() for s in ia_steps)
        if not has_parquet and not (sim/'fof').exists():
            if args.fof_tar is None:
                raise FileNotFoundError('no FoF catalogues: pass --fof-tar')
            log('extracting FoF from', args.fof_tar)
            extracted = extract_fof(args.fof_tar, sim, ia_steps)
        log('building the IA halo lightcone', halo_catalog)
        save_halocatalog(shells_info, sims_parameters, max_redshift=Z_MAX_IA, halo_snapshots_path=str(sim),
                         catalog_path=str(halo_catalog), log10_mass_limit=LOG10_MASS_IA, include_ia_shapes=True)
        timings['halo_lightcone'] = time.time() - t0

    # samples ------------------------------------------------------------------------------------------------
    t0 = time.time()
    samples = {name: Sample(name) for name in args.samples}
    timings['load_samples'] = time.time() - t0

    # 2. mocks -----------------------------------------------------------------------------------------------
    rejections = {}
    for b in args.baryons:
        todo = [(model, r, s) for model in args.models for r in args.rots for s in args.samples
                if not (out/f'mock_{model}_b{b}_r{r}_{s}.npy').exists()]
        dens_path = sim/f'delta_b_{NSIDE}_{b}.npy'
        if not dens_path.exists():
            dens_path = out/f'delta_b_{NSIDE}_{b}.npy'
        if not todo and dens_path.exists():
            continue
        t0 = time.time()
        p_b = copy.deepcopy(sims_parameters)
        if not dens_path.exists():
            if not args.make_baryons:
                raise FileNotFoundError(f'{dens_path} missing (baryonified density; --make-baryons to create it)')
            if out != sim:      # the package reads sys_baryo_{b}.npy and writes the catalogue next to the particles
                raise ValueError('--make-baryons needs --out = SIM (use a directory of symlinks to the particles)')
            np.random.seed(stable_seed(sim_id, 'baryons', b))       # draw_params_from_specs uses np.random
            baryons = dict(BARYONS, filename_new_params=f'sys_baryo_{b}.npy',
                           values_to_update=draw_params_from_specs(BARYON_PRIORS))
            load_or_save_updated_params(str(out) + '/', baryons['base_params_path'],
                                        baryons['filename_new_params'], baryons['values_to_update'])
            log(f'baryonifying draw {b}:', {k: float(v) for k, v in baryons['values_to_update'].items()})
            load_and_baryonify_gower_st_shells(str(sim) + '/', p_b, cosmo_bundle, baryons, NSIDE, shells_info, shells,
                                               dens_path=str(dens_path), tsz_path=str(out/f'tsz_{NSIDE}_{b}.npy'),
                                               halo_catalog_path=str(out/'halo_catalog.parquet'))
            timings[f'baryonify_b{b}'] = time.time() - t0
        if not todo:
            continue
        for path in (sim/f'sys_baryo_{b}.npy', out/f'sys_baryo_{b}.npy'):
            if path.exists():                     # [bpar, sys]: record the baryon draw in the mock parameters
                p_b.update(np.load(path, allow_pickle=True)[1])
                break
        t0 = time.time()
        density = np.load(dens_path)
        need_ia_shear = 'nla' in args.models
        fields = compute_lensing_fields(density, shells, cosmo_bundle['pars_camb'], NSIDE,
                                        do_kappa=False, do_shear=True, do_IA=need_ia_shear)
        fields['density'] = density
        timings[f'lensing_fields_b{b}'] = time.time() - t0
        log(f'baryon draw {b}: lensing fields in {time.time() - t0:.0f} s')

        for r in args.rots:
            if not any(t[1] == r for t in todo):
                continue
            nuis_path = out/f'nuisance_b{b}_r{r}.npy'

            def draw_nuisance():
                return {s: samples[s].draw_nuisance(np.random.default_rng(stable_seed(sim_id, 'nuisance', b, r, s)))
                        for s in args.samples}
            nuisance, existed = load_or_draw(nuis_path, draw_nuisance)
            for s in args.samples:          # samples added after the first run
                if s not in nuisance:
                    nuisance[s] = samples[s].draw_nuisance(np.random.default_rng(stable_seed(sim_id, 'nuisance', b, r, s)))
                    existed = False
            if not existed:
                np.save(nuis_path, nuisance)
            geometry = {s: shells_for(samples[s], nuisance[s], shells_info) for s in args.samples}

            for model in args.models:
                pending = [s for s in args.samples if not (out/f'mock_{model}_b{b}_r{r}_{s}.npy').exists()]
                if not pending:
                    continue
                t0 = time.time()
                ia_path = out/f'IA_params_{model}_b{b}_r{r}.npy'
                rng = np.random.default_rng(stable_seed(sim_id, 'IA', model, b, r))
                if model == 'nla':
                    ia, existed = load_or_draw(ia_path, lambda: dict(
                        draw_uniform(rng, IA_CFG['nla_comparison']['parameters']), IA_model='NLA',
                        redshift_pivot=IA_CFG['nla_comparison']['fixed']['redshift_pivot']))
                    if not existed:
                        np.save(ia_path, ia)
                    for s in pending:
                        _, _, zeff, ngal = geometry[s]
                        mock, _ = make_mock(samples[s], model, ia, None, p_b, nuisance[s], zeff, ngal,
                                            cosmo_bundle, fields, r)
                        mock['IA_params'] = ia
                        np.save(out/f'mock_{model}_b{b}_r{r}_{s}.npy', mock)
                        log(f'saved mock_{model}_b{b}_r{r}_{s}  A_IA {ia["A_IA"]:.3f} eta_IA {ia["eta_IA"]:.3f}')
                    timings[f'nla_b{b}_r{r}'] = time.time() - t0
                    continue

                # halo model: populations (f_c, f_sat) fixed, marks redrawn until the noise budget is respected
                stored, existed = load_or_draw(ia_path, lambda: halo_draw(rng))
                populations = {}
                for s in (args.samples if not existed else pending):
                    t1 = time.time()
                    populations[s] = population_for(samples[s], nuisance[s], stored, shells_info, p_b, cosmo_bundle,
                                                    halo_catalog, r)
                    log(f'population {s}: {len(populations[s]["redshift"]):,} model sources in {time.time() - t1:.0f} s')
                max_fraction = IA_CFG['halo']['noise_budget_rejection']['max_fraction']
                attempts = stored.get('n_attempts', 1)
                targets = args.samples if not existed else pending
                while True:
                    ia = stored
                    # IA numerators of all samples, then a cheap noise-budget pre-check before the map makers
                    halo, pre = {}, {}
                    for s in targets:
                        _, _, zeff, ngal = geometry[s]
                        halo[s] = make_halo_IA_numerator(populations[s], ia_block(ia), density, ngal, NSIDE,
                                                         bin_counts=samples[s].bin_counts, seed=nuisance[s]['bin_seed'])
                        if not existed and max_fraction < 1:   # rejection disabled when superseded by the cap
                            pre[s] = samples[s].ia_noise_precheck(halo[s], r)
                            if pre[s] > max_fraction:
                                break
                    if not existed and pre and max(pre.values()) > max_fraction:
                        log(f'halo IA b{b} r{r} attempt {attempts} rejected by the pre-check: '
                            + ', '.join(f'{k} {v:.4f}' for k, v in pre.items())
                            + '  ' + json.dumps({k: round(v, 4) for k, v in ia.items() if isinstance(v, float)}))
                    else:
                        mocks, worst = {}, {}
                        for s in targets:
                            _, _, zeff, ngal = geometry[s]
                            mocks[s], frac = make_mock(samples[s], model, ia, halo[s], p_b, nuisance[s], zeff, ngal,
                                                       cosmo_bundle, fields, r)
                            mocks[s]['ia_noise_fraction'] = frac
                            worst[s] = max(frac)
                            halo[s] = None
                            if not existed and worst[s] > max_fraction:
                                break
                        log(f'halo IA b{b} r{r} attempt {attempts}: ' + ', '.join(f'{k} {v:.4f}' for k, v in worst.items())
                            + '  (pre-check ' + ', '.join(f'{k} {v:.4f}' for k, v in pre.items()) + ')  '
                            + json.dumps({k: round(v, 4) for k, v in ia.items() if isinstance(v, float)}))
                        if existed or max(worst.values()) <= max_fraction:
                            break
                    del halo
                    if attempts >= MAX_REJECTIONS:
                        raise RuntimeError(f'halo IA b{b} r{r}: noise budget violated after {attempts} draws')
                    attempts += 1
                    stored = dict(stored, **halo_marks(rng, stored['f_c']))
                    gc.collect()
                rejections[f'b{b}_r{r}'] = attempts - 1
                if not existed:
                    stored.update(n_attempts=attempts, max_fraction=max_fraction, fixed=IA_CFG['halo']['fixed'],
                                  worst_fraction=worst)
                    np.save(ia_path, stored)
                for s, mock in mocks.items():
                    mock['IA_params'] = stored
                    np.save(out/f'mock_{model}_b{b}_r{r}_{s}.npy', mock)
                    log(f'saved mock_{model}_b{b}_r{r}_{s}')
                del populations, mocks
                gc.collect()
                timings[f'halo_b{b}_r{r}'] = time.time() - t0
        del fields, density
        gc.collect()

    # 3. kappa_CMB --------------------------------------------------------------------------------------------
    if not args.no_kappa_cmb and not (out/'kappa_cmb_2048_dmo.npy').exists():
        t0 = time.time()
        delta_b = {}
        for b in (0, 1):
            for d in (sim, out):
                if (d/f'delta_b_{NSIDE}_{b}.npy').exists():
                    delta_b[b] = np.load(d/f'delta_b_{NSIDE}_{b}.npy', mmap_mode='r')
                    break
        kappa, meta = make_kappa_cmb(str(sim), shells, steps, cosmo_bundle, delta_b=delta_b or None,
                                     seed=stable_seed(sim_id, 'kappa_cmb'), nproc=min(16, NPROC))
        for k, m in kappa.items():
            np.save(out/f'kappa_cmb_2048_{"dmo" if k == "dmo" else f"b{k}"}.npy', m)
        meta = {k: (np.asarray(v).tolist() if isinstance(v, (list, np.ndarray)) else v) for k, v in meta.items()}
        (out/'kappa_cmb_2048_meta.json').write_text(json.dumps(meta, indent=1, default=float) + '\n')
        timings['kappa_cmb'] = time.time() - t0
        log(f'kappa_CMB in {time.time() - t0:.0f} s', meta.get('transfer'))

    # 4. cleanup + DONE --------------------------------------------------------------------------------------
    if args.cleanup:
        for f in extracted:
            f.unlink()
        if extracted and not any((sim/'fof').iterdir()):
            (sim/'fof').rmdir()
        if args.halo_catalog is None and halo_catalog.exists():
            halo_catalog.unlink()
    timings['total'] = time.time() - t_start
    done = dict(sim=str(sim), sim_id=sim_id, models=args.models, samples=args.samples, baryons=args.baryons,
                rots=args.rots, rejections=rejections, timings=timings, git=git_hash(),
                finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    (out/'mocks_done.json').write_text(json.dumps(done, indent=1) + '\n')
    log('done', json.dumps(timings))


if __name__ == '__main__':
    main()
