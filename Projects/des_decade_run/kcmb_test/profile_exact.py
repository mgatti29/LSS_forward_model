"""Where the time of the exact (line-of-sight) CAMB shell spectra goes."""
import copy, sys, time
import numpy as np, camb
sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
from LSS_forward_model import coshirex as cx
from LSS_forward_model.cosmology import read_sims_params
pars = read_sims_params('/pscratch/sd/m/mgatti/kcmb_test/run001')[1]['pars_camb']
w = [cx.tophat_window(z, z + 0.04) for z in np.arange(1.6, 1.99, 0.04)] + cx.ladder_windows(2.0, 1100., 100)
def t(label, f):
    t0 = time.time(); out = f(); print(f'{label:60s} {time.time() - t0:7.1f} s', flush=True); return out
def run(lmax, n, lpa=1, kmax=20., total=False):
    p = copy.deepcopy(pars); p.Want_CMB = False; p.WantCls = True; p.WantTransfer = True; p.min_l = 1
    p.set_for_lmax(lmax, lens_potential_accuracy=lpa); p.Transfer.kmax = max(p.Transfer.kmax, kmax)
    p.NonLinear = camb.model.NonLinear_both; p.SourceTerms.limber_windows = False; p.SourceTerms.counts_density = True
    for term in ('counts_redshift', 'counts_lensing', 'counts_velocity', 'counts_radial', 'counts_timedelay', 'counts_ISW', 'counts_potential', 'counts_evolve'):
        setattr(p.SourceTerms, term, False)
    p.SourceWindows = [camb.sources.SplinedSourceWindow(z=x.za, W=x.wa) for x in w[:n]]
    return camb.get_results(p)
t('total-matter ratio: 2 power interpolators (zmax ~1000, kmax 50)', lambda: [camb.get_matter_power_interpolator(copy.deepcopy(pars), var1=v, var2=v, nonlinear=True, hubble_units=True, k_hunit=True, kmax=50., zmax=1000.) for v in ('delta_tot', 'delta_cdm')])
t('get_results lmax 100, 110 shells (lpa 1, kmax 20)', lambda: run(100, 110))
t('get_results lmax 100, 110 shells (lpa 0, kmax 0.9 default)', lambda: run(100, 110, lpa=0, kmax=0.))
t('get_results lmax 100, 30 shells', lambda: run(100, 30))
t('get_results lmax 100, 10 shells', lambda: run(100, 10))
t('get_results lmax 1000, 110 shells', lambda: run(1000, 110))
