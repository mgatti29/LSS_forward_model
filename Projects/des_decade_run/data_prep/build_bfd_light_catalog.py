#!/usr/bin/env python3
"""Light BFD catalogue for the mocks from the unblinded, PSF-masked v5 catalogue.

Keeps, per tomographic bin, what bfd_utils.make_BFD_sample needs, in the format of the previous
BFD_catalog_v5_light.npy: {tomo: {'ra', 'dec', 'tomo', 'selection_term', 'pqr', 'pqr10_prelog'}}.
Rows with tomo outside 0..3 are dropped.  One pass over contiguous row chunks of the FITS table
(column-wise memmap reads of a 79 GB file are far too slow).

usage: build_bfd_light_catalog.py [OUTPUT]
"""
import sys

import numpy as np
from astropy.io import fits

SOURCE = ('/global/cfs/cdirs/des/mgatti/BFD_targets_June25/'
          'BFD_catalog_v5_UNBLINDED_with_tomo_masked_withselection_final_psfmask_psfvalid.fits')
OUTPUT = sys.argv[1] if len(sys.argv) > 1 else '/global/cfs/cdirs/m5099/DESY6_bfd/BFD_catalog_v5_UNBLINDED_light.npy'
COLUMNS = {'ra': np.float64, 'dec': np.float64, 'tomo': np.int64, 'selection_term': np.int64,
           'pqr': np.float64, 'pqr10_prelog': np.float64}
CHUNK = 2_000_000


def main():
    parts = {t: {name: [] for name in COLUMNS} for t in range(4)}
    with fits.open(SOURCE, memmap=True) as hdul:
        hdu = hdul[1]
        nrow = hdu.header['NAXIS2']
        dropped = 0
        for start in range(0, nrow, CHUNK):
            block = np.array(hdu.data[start:min(start + CHUNK, nrow)])     # contiguous read
            tomo = block['tomo']
            dropped += int(np.sum((tomo < 0) | (tomo > 3)))
            for t in range(4):
                use = tomo == t
                for name, dtype in COLUMNS.items():
                    parts[t][name].append(np.asarray(block[name][use], dtype=dtype))
            del block
            print(f'{min(start + CHUNK, nrow)}/{nrow}', flush=True)
    out = {}
    for t in range(4):
        out[t] = {name: np.concatenate(parts[t][name]) for name in COLUMNS}
        parts[t] = None
        sel = out[t]['selection_term']
        print(f'bin {t}: {len(sel)} rows, real {np.sum(sel == 0)}, selection {np.sum(sel != 0)}', flush=True)
    print('dropped (tomo outside 0..3):', dropped, flush=True)
    np.save(OUTPUT, out, allow_pickle=True)
    print('saved', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
