"""Which per-frame mask already covers the 3x3 box of a pixel: the cross-frame events, the ramp events or the extreme
negative pixels.

usage: python -I which_mask.py TILE BAND FRAME_NAME X Y --root ROOT --out OUT.json
ROOT is the ``gc-treasury`` directory of the tile in the run (``BAND/pipeline/jw10678*_destreak_TILE_crf.fits``).  FRAME_NAME
is the file name of one crf frame of the tile-band and (X, Y) a pixel of it.  The code under test is the checkout that holds
this file, or the one that ``PIPE_ROOT`` names.  READ ONLY on the frames."""
import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
from astropy.io import fits

warnings.filterwarnings('ignore')
PIPE_ROOT = os.environ.get('PIPE_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
sys.path.insert(0, PIPE_ROOT)
from jwst_gc_pipeline.photometry import single_exposure_events as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('tile')
ap.add_argument('band')
ap.add_argument('frame_name')
ap.add_argument('x', type=int)
ap.add_argument('y', type=int)
ap.add_argument('--root', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()
tile, band, frame_name, x, y = args.tile, args.band, args.frame_name, args.x, args.y
NEGATIVE_NSIGMA = 50.0          # the value of the four-tile run, ``NEGATIVE_PIXEL_MASK_NSIGMA``

files = sorted(glob.glob(f'{args.root}/{band}/pipeline/jw10678*_destreak_{tile}_crf.fits'))
assert files, f'no crf frames under {args.root}/{band}/pipeline'
frames = [S.CrfFrame(f) for f in files]
shapes = {fr.name: fr.shape for fr in frames}
target = [fr for fr in frames if os.path.basename(fr.path) == frame_name]
assert len(target) == 1, frame_name
name = target[0].name
res = dict(tile=tile, band=band, frame=frame_name, x=x, y=y)
box = (slice(y - 1, y + 2), slice(x - 1, x + 2))


def near(events):
    """The events of the frame within 8 pixels of (x, y): where they are, how large, and whether the 3x3 box is in one."""
    return [dict(x=float(e.x), y=float(e.y), area=int(e.area), peak=float(e.peak), excess=float(e.excess),
                 n_others=len(e.others),
                 in_box=bool(((e.xs >= x - 1) & (e.xs <= x + 1) & (e.ys >= y - 1) & (e.ys <= y + 1)).any()))
            for e in events if e.frame == name and abs(e.x - x) <= 8 and abs(e.y - y) <= 8]


events = S.find_events(frames)
mask = S.event_masks(events, shapes).get(name)
res['event_px_in_box'] = int(mask[box].sum()) if mask is not None else 0
res['events_near'] = near(events)
if hasattr(S, 'find_ramp_events'):
    ramp_events, n_tested = S.find_ramp_events(frames)
    mask = S.event_masks(ramp_events, shapes).get(name)
    res['ramp_px_in_box'] = int(mask[box].sum()) if mask is not None else 0
    res['ramp_events_near'] = near(ramp_events)
else:
    res['ramp_px_in_box'] = None             # this checkout has no ramp events
with fits.open(target[0].path, memmap=False) as hdul:
    sci = np.asarray(hdul['SCI'].data, dtype=float)
    err = np.asarray(hdul['ERR'].data, dtype=float)
    dq = np.asarray(hdul['DQ'].data).astype(np.int64)
res['negative_px_in_box'] = int(S.negative_pixel_mask(sci, err, dq, NEGATIVE_NSIGMA)[box].sum())
print(json.dumps(res, indent=1))
with open(args.out, 'w') as fh:
    json.dump(res, fh, indent=1)
