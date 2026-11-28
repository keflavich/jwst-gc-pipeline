"""The 3x3 box of one pixel of a crf frame, and what the positive spike finder makes of it.

usage: python -I pixel_box.py FRAME.fits X Y --out OUT.txt
Prints SCI and DQ of the 3x3 box around (X, Y), and whether ``find_positive_spikes`` returns the pixel with no neighbour
allowed to be unusable, with one allowed, and with the SATURATED flag of the neighbours cleared in a copy of the DQ.  The
code under test is the checkout that holds this file, or the one that ``PIPE_ROOT`` names.  READ ONLY on the frame."""
import argparse
import os
import sys
import warnings

import numpy as np
from astropy.io import fits

warnings.filterwarnings('ignore')
PIPE_ROOT = os.environ.get('PIPE_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
sys.path.insert(0, PIPE_ROOT)
os.environ.setdefault('GC_ALLOW_DEV', '1')
from jwst_gc_pipeline.photometry import single_exposure_events as S  # noqa: E402

SATURATED = 2
MIN_EXCESS = 300.0              # ``POSITIVE_SPIKE_MASK_MIN_EXCESS`` of the four-tile run

ap = argparse.ArgumentParser()
ap.add_argument('frame')
ap.add_argument('x', type=int)
ap.add_argument('y', type=int)
ap.add_argument('--out', required=True)
args = ap.parse_args()
x, y = args.x, args.y
with fits.open(args.frame, memmap=False) as hdul:
    sci = np.asarray(hdul['SCI'].data, dtype=float)
    dq = np.asarray(hdul['DQ'].data)
box = (slice(y + 1, y - 2, -1), slice(x - 1, x + 2))
lines = [f'frame {os.path.basename(args.frame)}, pixel (x, y) = ({x}, {y})',
         'SCI of the 3x3 box, MJy/sr (top row y+1, bottom row y-1; left column x-1):']
lines += ['  ' + ' '.join(f'{v:8.1f}' for v in row) for row in sci[box]]
lines += ['DQ of the 3x3 box (1 DO_NOT_USE, 2 SATURATED, 4 JUMP_DET):']
lines += ['  ' + ' '.join(f'{int(v):8d}' for v in row) for row in dq[box]]


def found(dq_used, max_unusable):
    spikes = S.find_positive_spikes(sci, dq_used, MIN_EXCESS, max_unusable=max_unusable)
    return [s for s in spikes if (int(s['x']), int(s['y'])) == (x, y)]


for k in (0, 1):
    hit = found(dq, k)
    lines.append(f'find_positive_spikes, max_unusable={k}: {"returns the pixel " + str(hit[0]) if hit else "does not return the pixel"}')
cleared = dq.copy()
cleared[y - 1:y + 2, x - 1:x + 2] &= ~np.uint32(SATURATED)
cleared[y, x] = dq[y, x]
hit = found(cleared, 0)
lines.append('SATURATED flag cleared in the eight neighbours, max_unusable=0: '
             + ('returns the pixel ' + str(hit[0]) if hit else 'does not return the pixel'))
text = '\n'.join(lines) + '\n'
print(text)
with open(args.out, 'w') as fh:
    fh.write(text)
