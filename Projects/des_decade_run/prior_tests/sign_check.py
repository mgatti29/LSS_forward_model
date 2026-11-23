"""Sign of GI for the halo model (A_c = 1) vs NLA (A = 1) vs TNG galaxies, same frame and conventions."""
import sys, numpy as np, healpy as hp
sys.path.insert(0,'/global/u2/m/mgatti/LSS_forward_model/Projects/IA_non_gaussian/cheap_model_validation')
sys.path.insert(0,'/global/u2/m/mgatti/LSS_forward_model/Projects/IA_non_gaussian')
sys.path.insert(0,'/global/u2/m/mgatti/LSS_forward_model')
import cheap_ia_validation as v
t=np.load(v.WORK/'DES'/'target.npz'); w=t['counts']/t['counts'].sum(1,keepdims=True)
d_m,g_m=v.frame_fields('model',w); d_t,g_t=v.frame_fields('target',w)
nla=np.load(v.WORK/'DES'/'nla_prior.npz'); kn=list(nla['eta']).index(0.)
m=np.load(v.WORK/'DES'/'model_hod_sim_fc_0.5_fsat_0.8_etaneg.npz'); print('eta_c nodes',m['eta_c'],'eta_s',m['eta_s'])
kc=int(np.argmin(np.abs(m['eta_c']+1)))
bp=lambda a,b: v.bandpower(hp.alm2cl(a,b))
for b in range(4):
    gg=bp(g_m[b],g_m[b])
    print(f'bin {b+1}: halo A_c=1 (eta_c={m["eta_c"][kc]}) GI/GG', (bp(g_m[b],m['alm_central'][b,kc])/gg)[::3].round(4),
          '| NLA A=1', (bp(g_m[b],nla['alm_E'][b,kn])/gg)[::3].round(4),
          '| TNG gal', (bp(g_t[b],t['alm_E'][b])/bp(g_t[b],g_t[b]))[::3].round(4),
          '| dI halo', (bp(d_m[b],m['alm_central'][b,kc])).mean().round(8), 'dI NLA', bp(d_m[b],nla['alm_E'][b,kn]).mean().round(8), 'dI TNG', bp(d_t[b],t['alm_E'][b]).mean().round(8))
