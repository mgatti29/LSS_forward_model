import sys, numpy as np, healpy as hp
sys.path.insert(0,'/global/u2/m/mgatti/LSS_forward_model/Projects/IA_non_gaussian/cheap_model_validation')
sys.path.insert(0,'/global/u2/m/mgatti/LSS_forward_model/Projects/IA_non_gaussian')
sys.path.insert(0,'/global/u2/m/mgatti/LSS_forward_model')
import cheap_ia_validation as v
t=np.load(v.WORK/'DES'/'target.npz'); w=t['counts']/t['counts'].sum(1,keepdims=True)
d_m,g_m=v.frame_fields('model',w); d_t,g_t=v.frame_fields('target',w)
nla=np.load(v.WORK/'DES'/'nla_prior.npz'); k=list(nla['eta']).index(0.)
bp=lambda a,b: v.bandpower(hp.alm2cl(a,b))
print('ell', np.sqrt(v.ELL_EDGES[:-1]*v.ELL_EDGES[1:]).round(0))
for b in range(4):
    print('bin',b+1,'NLA A=1 GI/GG', (bp(g_m[b],nla['alm_E'][b,k])/bp(g_m[b],g_m[b]))[::3].round(4),
          ' TNG GI/GG', (bp(g_t[b],t['alm_E'][b])/bp(g_t[b],g_t[b]))[::3].round(4))
