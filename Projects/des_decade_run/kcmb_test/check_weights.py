"""born_weights == glass MultiPlaneConvergence on maps, and ladder weights independent of the planes before."""
import sys, numpy as np
sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model')
import camb
from LSS_forward_model.coshirex import born_weights, kappa_cmb, ladder_windows, tophat_window
p = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.12, ns=0.965, As=2e-9)
low = [tophat_window(z, z + 0.1) for z in np.arange(0.0, 2.0, 0.1)]
lad = ladder_windows(2.0, 1100., 100)
rng = np.random.default_rng(1)
maps = rng.normal(size=(len(low) + len(lad), 1000))
q_all = born_weights(low + lad, p)
direct = kappa_cmb(zip(maps, low + lad), p)
print('weights vs glass maps, max rel diff', np.max(np.abs(q_all @ maps - direct))/np.abs(direct).max())
print('ladder weights alone vs joint, max rel diff', np.max(np.abs(born_weights(lad, p) - q_all[len(low):])/q_all[len(low):]))
print('low weights alone vs joint', np.max(np.abs(born_weights(low, p) - q_all[:len(low)])/q_all[:len(low)]))
