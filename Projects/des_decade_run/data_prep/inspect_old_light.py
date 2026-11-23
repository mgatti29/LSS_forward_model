import numpy as np
d = np.load('/global/cfs/cdirs/m5099/DESY3/BFD_catalog_v5_light.npy', allow_pickle=True)
d = d.item() if d.dtype == object and d.shape == () else d
print(type(d), list(d.keys()) if hasattr(d, 'keys') else d.dtype)
for t in d:
    c = d[t]
    if hasattr(c, 'dtype') and c.dtype.names:
        print(t, 'structured', len(c), c.dtype)
    else:
        print(t, type(c), {k: (np.shape(v), np.asarray(v).dtype) for k, v in c.items()})
