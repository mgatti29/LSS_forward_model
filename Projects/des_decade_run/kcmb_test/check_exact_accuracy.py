"""Exact CAMB shell spectra at lmax 100: default accuracy vs lens_potential_accuracy 1 + kmax 20."""
import copy, sys
import numpy as np, camb
sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
from LSS_forward_model import coshirex as cx
from LSS_forward_model.cosmology import read_sims_params
pars = read_sims_params('/pscratch/sd/m/mgatti/kcmb_test/run001')[1]['pars_camb']
w = [cx.tophat_window(z, z + 0.04) for z in np.arange(1.84, 1.99, 0.04)] + cx.ladder_windows(2.0, 1100., 100)[:6] + cx.ladder_windows(2.0, 1100., 100)[60:62]
def run(boost):
    p = copy.deepcopy(pars); p.Want_CMB = False; p.WantCls = True; p.WantTransfer = True; p.min_l = 1
    if boost:
        p.set_for_lmax(100, lens_potential_accuracy=1); p.Transfer.kmax = max(p.Transfer.kmax, 20.)
    else:
        p.set_for_lmax(100)
    p.NonLinear = camb.model.NonLinear_both; p.SourceTerms.limber_windows = False; p.SourceTerms.counts_density = True
    for term in ('counts_redshift', 'counts_lensing', 'counts_velocity', 'counts_radial', 'counts_timedelay', 'counts_ISW', 'counts_potential', 'counts_evolve'):
        setattr(p.SourceTerms, term, False)
    p.SourceWindows = [camb.sources.SplinedSourceWindow(z=x.za, W=x.wa) for x in w]
    return camb.get_results(p).get_source_cls_dict(lmax=100, raw_cl=True)
a, b = run(False), run(True)
for key in ('W1xW1', 'W4xW4', 'W4xW5', 'W5xW6', 'W5xW5', 'W11xW11', 'W11xW12'):
    r = a[key][2:]/b[key][2:]
    print(f'{key:8s} default/boost: max |1-r| {np.max(np.abs(1 - r)):.2e} (ell 2-100)')
