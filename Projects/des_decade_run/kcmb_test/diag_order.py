import sys, numpy as np, healpy as hp
sys.path.insert(0, '/global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/kcmb_test')
import kcmb_test as t
_, _, shells, steps, zeff = t.setup()
db = np.load('/pscratch/sd/m/mgatti/run001/delta_b_1024_0.npy', mmap_mode='r')
for i in (30, 70):
    c = t.counts_of(steps[i]); ns = hp.npix2nside(c.size)
    c1024 = hp.ud_grade(c, 1024, power=-2); d = c1024/c1024.mean() - 1
    best = []
    for j in range(100):
        x = np.asarray(db[j], dtype=float)
        best.append(np.corrcoef(x, d)[0, 1])
    best = np.array(best)
    print(f'native shell {i} (step {steps[i]}, z={zeff[i]:.3f}): best-correlated delta_b index {best.argmax()} (r={best.max():.3f}); r at i={best[i]:.3f}, at 99-i={best[99-i]:.3f}; std native {d.std():.3f}, std delta_b[i] {np.std(db[i]):.3f}', flush=True)
print('control.par / run.log of the delta_b copy vs the tarball:')
