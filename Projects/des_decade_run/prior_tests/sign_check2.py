"""E-mode sign of the observed-convention lensing shear (g = -sum w gamma_GLASS) vs density, per bin."""
import sys, numpy as np, healpy as hp
sys.path.insert(0,'/global/u2/m/mgatti/LSS_forward_model/Projects/IA_non_gaussian/cheap_model_validation')
sys.path.insert(0,'/global/u2/m/mgatti/LSS_forward_model/Projects/IA_non_gaussian')
sys.path.insert(0,'/global/u2/m/mgatti/LSS_forward_model')
import cheap_ia_validation as v
t=np.load(v.WORK/'DES'/'target.npz'); w=t['counts']/t['counts'].sum(1,keepdims=True)
d_m,g_m=v.frame_fields('model',w)
bp=lambda a,b: v.bandpower(hp.alm2cl(a,b))
for l in range(4):
    print(f'density bin {l+1} x g_obs bin 4 (lens behind):', bp(d_m[l], g_m[3])[::3])
