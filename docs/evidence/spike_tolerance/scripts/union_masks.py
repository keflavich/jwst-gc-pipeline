"""Pixels that the tolerance adds to the union of the per-frame masks of a render (events, ramp events, extreme negative
pixels and positive spikes).

usage: python -I union_masks.py TILE BAND --root ROOT --out OUT.json
ROOT is the ``gc-treasury`` directory of the tile in the run (``BAND/pipeline/jw10678*_destreak_TILE_crf.fits``).  The
script calls ``single_exposure_events.single_exposure_events`` consumer ``crowdsource_catalogs_long.single_exposure_event_masks``
twice on the crf frames of the tile-band, with ``POSITIVE_SPIKE_MASK_MAX_UNUSABLE`` unset and then 1, and counts per frame
the pixels that only the second call masks.  The other mask switches are the ones of the environment that starts the
script (``POSITIVE_SPIKE_MASK_MIN_EXCESS``, ``SINGLE_EXPOSURE_EVENT_MASK``, ``SINGLE_EXPOSURE_EVENT_RAMP`` and
``NEGATIVE_PIXEL_MASK_NSIGMA``).  The code under test is the checkout that holds this file, or the one that ``PIPE_ROOT``
names.  READ ONLY on the frames."""
import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
PIPE_ROOT = os.environ.get('PIPE_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
sys.path.insert(0, PIPE_ROOT)
from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as C  # noqa: E402
from jwst_gc_pipeline.photometry import single_exposure_events as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('tile')
ap.add_argument('band')
ap.add_argument('--root', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()
tile, band = args.tile, args.band

files = sorted(glob.glob(f'{args.root}/{band}/pipeline/jw10678*_destreak_{tile}_crf.fits'))
assert files, f'no crf frames under {args.root}/{band}/pipeline'
print(len(files), 'frames', flush=True)
os.environ.pop(S.ENV_SPIKE_UNUSABLE, None)
off = C.single_exposure_event_masks(files)
os.environ[S.ENV_SPIKE_UNUSABLE] = '1'
on = C.single_exposure_event_masks(files)
rows = []
for f in files:
    a = off.get(f)
    b = on.get(f)
    n_off = int(a.sum()) if a is not None else 0
    n_on = int(b.sum()) if b is not None else 0
    if n_on != n_off:
        added = b & ~a if a is not None else b
        ys, xs = np.nonzero(added)
        rows.append(dict(frame=os.path.basename(f), masked_off=n_off, masked_on=n_on, added=n_on - n_off,
                         bbox=[int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())]))
        print(os.path.basename(f), n_off, '->', n_on, 'px; box x', xs.min(), xs.max(), 'y', ys.min(), ys.max(), flush=True)
res = dict(tile=tile, band=band, n_frames=len(files),
           masked_off=int(sum(m.sum() for m in off.values())), masked_on=int(sum(m.sum() for m in on.values())),
           frames_changed=rows)
print(res['masked_off'], '->', res['masked_on'], 'px in the union of the frame masks;', len(rows), 'frames changed', flush=True)
with open(args.out, 'w') as fh:
    json.dump(res, fh, indent=1)
