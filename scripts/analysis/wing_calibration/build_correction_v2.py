"""Build C(r) v2: median data/model ratio of the bright half of stars,
full radial resolution, interpolated (no heavy smoothing), clamped:
C = 1 inside r<3 (core untouched), flat beyond r=30 (outer ratio unreliable
in this crowded field -> hold last reliable value).
"""
import sys
import numpy as np

SCRATCH = ('/blue/adamginsburg/adamginsburg/tmp/claude-3663/'
           '-blue-adamginsburg-adamginsburg-repos-jwst-gc-pipeline/'
           'd046d21c-902e-4ed6-b230-4b20ac93f54c/scratchpad/agent_empsf')
DET = sys.argv[1] if len(sys.argv) > 1 else 'nrca1'
FILT = sys.argv[2] if len(sys.argv) > 2 else 'f182m'

d = np.load(f'{SCRATCH}/wingratio_{DET}_{FILT}.npz', allow_pickle=True)
r_mid = d['r_mid']
all_ratio = d['all_ratio']
A = np.array([m[6] for m in d['meta']], float)

bright = A > np.median(A)
med = np.nanmedian(all_ratio[bright], axis=0)

R_IN, R_OUT = 3.0, 30.0
Cr = med.copy()
Cr[r_mid < R_IN] = 1.0
# beyond R_OUT: hold the flux-weighted mean of 20-30 px
hold = np.nanmedian(med[(r_mid >= 20) & (r_mid <= R_OUT)])
Cr[r_mid > R_OUT] = hold
Cr = np.clip(Cr, 0.5, 4.0)

# fine grid, linear interpolation
rgrid = np.arange(0, 61, 0.25)
Cgrid = np.interp(rgrid, np.concatenate([[0], r_mid]),
                  np.concatenate([[1.0], Cr]))

np.savez(f'{SCRATCH}/wing_correction_{DET}_{FILT}.npz',
         r_mid=r_mid, C_train=Cr, C_spline_vals=Cr,
         spline_r=rgrid, spline_C=Cgrid)
print('C(r) v2 (bright-half median ratio):')
for rq in [3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40]:
    print(f'  r={rq:3d}: {np.interp(rq, rgrid, Cgrid):.3f}')
