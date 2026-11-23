"""kappa of the baryon term (Delta_b) for runsT/run001: both draws, and the reconstruction floor alone (z > 0.6 shells,
where the two draws are identical: delta_b there is the plain downgrade, so Delta_b is pure reconstruction error)."""
import json, os, numpy as np, healpy as hp
import kcmb_test as t
_, cb, shells, steps, zeff = t.setup()
pars = cb['pars_camb']
sets = json.load(open(t.WORK/'shell_sets.json'))
for name, folder, index in (('b0', 'dbaryon0', sets['baryon_shells'] + sets['floor_shells']),
                            ('b1', 'dbaryon1', sets['baryon_shells'] + sets['floor_shells']),
                            ('floor', 'dbaryon0', sets['floor_shells'])):
    k = t.kappa_cmb(t.shell_stream(sorted(index), shells, folder), pars)
    np.save(t.WORK/f'kappa_{name}.npy', k.astype(np.float32)); print(name, 'done', flush=True)
