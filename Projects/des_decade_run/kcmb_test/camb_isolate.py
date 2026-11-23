import sys, time, numpy as np
sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
from LSS_forward_model.coshirex import ladder_windows, tophat_window, shell_cls
import kcmb_test as t
_, cb, shells, steps, zeff = t.setup()
pars = cb['pars_camb']
print('dark energy:', type(pars.DarkEnergy).__name__, 'w', getattr(pars.DarkEnergy, 'w', None), 'num_nu_massive', pars.num_nu_massive, flush=True)
lad = ladder_windows(2.0)
case = sys.argv[1]
if case == 'lowz':      w = lad[:5]
elif case == 'highz':   w = lad[-5:]
elif case == 'mid':     w = lad[40:45]
elif case == 'all':     w = lad
lmax = int(sys.argv[2]); limber = sys.argv[3] == 'limber'
t0 = time.time(); cls = shell_cls(pars, w, lmax, limber=limber, total_matter=False)
print(case, lmax, 'limber' if limber else 'exact', 'OK', len(cls), f'{time.time()-t0:.0f}s', 'C_ell[100] auto first', cls[0][100], flush=True)
