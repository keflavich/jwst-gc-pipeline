"""Leftover-driven check of the SATURATED-neighbour measurement, and the sensitivity of the blocked-spike count to the finder's tests.

For every crf frame of one tile-band the finder is run with the SATURATED bit kept (``own``) and cleared (``clr``) in four variants:
A the shipped isolation (0.15) and 5 sigma test, B without the sigma test, C isolation 0.3, D both relaxations.  The blocked spikes of a
variant are the pixels of ``clr`` that are not in ``own``, carry neither SATURATED nor DO_NOT_USE themselves, and have a SATURATED
neighbour.  Then, for every leftover of the listed tags at or above ``--min-peak`` MJy/sr, in every frame that covers its sky
position: the brightest finite pixel of the 5x5 box at the position, its flags, the SATURATED pixels in the 3x3 and the 7x7 around it,
its excess over the annulus, and whether each variant of the finder returns it.  The leftover is then classified by the frame in which
its brightest pixel has the largest excess.

usage: python -I scan_satnbr_lo.py TILE BAND --root ROOT --score-dir DIR --out OUTDIR [--tags optfin,optnm] [--min-peak 50]
READ ONLY.  The code under test is the checkout that holds this file, or the one that PIPE_ROOT names."""
import argparse
import glob
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy import ndimage

warnings.filterwarnings('ignore')
sys.dont_write_bytecode = True
PIPE_ROOT = os.environ.get('PIPE_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
sys.path.insert(0, PIPE_ROOT)
os.environ.setdefault('GC_ALLOW_DEV', '1')
from jwst_gc_pipeline.photometry import single_exposure_events as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('tile')
ap.add_argument('band')
ap.add_argument('--root', required=True)
ap.add_argument('--score-dir', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--tags', default='optfin,optnm')
ap.add_argument('--min-peak', type=float, default=50.0)
ap.add_argument('--floor', type=float, default=100.0)
ap.add_argument('--max-unusable', type=int, default=1)
ap.add_argument('--match-arcsec', type=float, default=0.1)
args = ap.parse_args()
tile, band = args.tile, args.band
FLOOR, KU = args.floor, args.max_unusable
VARIANTS = {'A': dict(isolation=0.15, min_sigma=5.0), 'B': dict(isolation=0.15, min_sigma=0.0),
            'C': dict(isolation=0.30, min_sigma=5.0), 'D': dict(isolation=0.30, min_sigma=0.0)}
paths = sorted(glob.glob(f'{args.root}/{band}/pipeline/jw10678*_destreak_{tile}_crf.fits'))
assert paths
t0 = time.time()
frames = [S.CrfFrame(p) for p in paths]
RING = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)]
STRUCT8 = np.ones((3, 3), dtype=int)


def unit(ra, dec):
    ra, dec = np.radians(np.atleast_1d(ra)), np.radians(np.atleast_1d(dec))
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]).T


# leftovers of every tag, the union list with a key per (tag, index)
lo = []
for tag in args.tags.split(','):
    with open(f'{args.score_dir}/{tag}_{tile}_{band}_leftovers.json') as fh:
        for r in json.load(fh):
            if r['resid_peak'] >= args.min_peak:
                lo.append(dict(tag=tag, x=r['x'], y=r['y'], ra=r['ra'], dec=r['dec'], resid_peak=r['resid_peak'], cls=r['cls'], frames=[]))
l_ra = np.array([r['ra'] for r in lo])
l_dec = np.array([r['dec'] for r in lo])
print(f'{tile} {band}: {len(frames)} frames, {len(lo)} leftovers >= {args.min_peak:g} ({time.time() - t0:.0f} s)', flush=True)

blocked = {v: [] for v in VARIANTS}
for fr in frames:
    sci, dq = fr.load()
    ny, nx = sci.shape
    dq_nosat = dq & ~np.int64(S.SATURATED)
    satpix = (dq & S.SATURATED) != 0
    lab, nlab = ndimage.label(satpix, structure=STRUCT8)
    area = np.bincount(lab.ravel()) if nlab else np.zeros(1, dtype=int)
    found = {}
    for v, kw in VARIANTS.items():
        own = S.find_positive_spikes(sci, dq, FLOOR, max_unusable=KU, **kw)
        clr = S.find_positive_spikes(sci, dq_nosat, FLOOR, max_unusable=KU, **kw)
        own_xy = {(int(s['x']), int(s['y'])) for s in own}
        found[v] = (own_xy, {(int(s['x']), int(s['y'])): s for s in clr})
        for s in clr:
            x, y = int(s['x']), int(s['y'])
            if (x, y) in own_xy or (dq[y, x] & (S.SATURATED | S.DO_NOT_USE)):
                continue
            if not any(0 <= y + dy < ny and 0 <= x + dx < nx and (dq[y + dy, x + dx] & S.SATURATED) for dx, dy in RING):
                continue
            ra, dec = S._world(fr, x, y)
            blocked[v].append(dict(frame=fr.name, x=x, y=y, ra=float(ra[0]), dec=float(dec[0]), excess=s['excess'], nsigma=s['excess'] / s['sigma'],
                                   nbfrac=s['neighbour'] / s['excess']))
    # leftover-driven
    if lo:
        xs, ys = fr.wcs.world_to_pixel_values(l_ra, l_dec)
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        sel = np.nonzero(np.isfinite(xs) & np.isfinite(ys) & (xs >= 8) & (xs < nx - 9) & (ys >= 8) & (ys < ny - 9))[0]
        for i in sel:
            ix, iy = int(round(xs[i])), int(round(ys[i]))
            box = sci[iy - 2:iy + 3, ix - 2:ix + 3]
            if not np.isfinite(box).any():
                continue
            k = np.nanargmax(np.where(np.isfinite(box), box, -np.inf))
            my, mx = int(iy - 2 + k // 5), int(ix - 2 + k % 5)
            bkg, sig = S._annulus_stats(sci, mx, my, S.SPIKE_BKG_R_IN, S.SPIKE_BKG_R_OUT)
            nbvals = np.array([sci[my + dy, mx + dx] for dx, dy in RING])
            fin = np.isfinite(nbvals)
            nbmax = float(np.nanmax(nbvals)) if fin.any() else float('nan')
            exc = float(sci[my, mx] - bkg) if np.isfinite(bkg) else float('nan')
            sat3 = int(((dq[my - 1:my + 2, mx - 1:mx + 2] & S.SATURATED) != 0).sum())
            sat7 = int(((dq[my - 3:my + 4, mx - 3:mx + 4] & S.SATURATED) != 0).sum())
            comp = 0
            if sat3:
                ids = np.unique(lab[my - 1:my + 2, mx - 1:mx + 2])
                comp = int(max(area[j] for j in ids if j > 0))
            rec = dict(frame=fr.name, x=mx, y=my, sci=float(sci[my, mx]), excess=exc, nsigma=(exc / sig if sig and np.isfinite(sig) and sig > 0 else float('nan')),
                       nbfrac=((nbmax - bkg) / exc if exc and np.isfinite(exc) and exc > 0 else float('nan')),
                       dq_self=int(dq[my, mx]), sat3=sat3, sat7=sat7, sat_comp_area=comp,
                       n_dnu3=int(((dq[my - 1:my + 2, mx - 1:mx + 2] & S.DO_NOT_USE) != 0).sum()),
                       own={v: (mx, my) in found[v][0] for v in VARIANTS}, clr={v: (mx, my) in found[v][1] for v in VARIANTS})
            lo[i]['frames'].append(rec)
    print(f'{fr.name[:36]}: blocked A/B/C/D {[len(blocked[v]) for v in VARIANTS]} ({time.time() - t0:.0f} s)', flush=True)

for r in lo:
    r['n_frames'] = len(r['frames'])
    r['best'] = max(r['frames'], key=lambda f: (f['excess'] if np.isfinite(f['excess']) else -1e30)) if r['frames'] else None

# the blocked spikes of every variant matched to the leftovers
rad = 2.0 * np.sin(np.radians(args.match_arcsec / 3600.0) / 2.0)
from scipy.spatial import cKDTree  # noqa: E402
for v in VARIANTS:
    if blocked[v]:
        tree = cKDTree(unit([b['ra'] for b in blocked[v]], [b['dec'] for b in blocked[v]]))
        for r in lo:
            r.setdefault('blocked_variant', {})[v] = sorted(tree.query_ball_point(unit(r['ra'], r['dec'])[0], rad))
    else:
        for r in lo:
            r.setdefault('blocked_variant', {})[v] = []

os.makedirs(args.out, exist_ok=True)
with open(f'{args.out}/satnbr_lo_{tile}_{band}.json', 'w') as fh:
    json.dump(dict(tile=tile, band=band, floor=FLOOR, max_unusable=KU, variants=VARIANTS, match_arcsec=args.match_arcsec, min_peak=args.min_peak,
                   pipe_root=PIPE_ROOT, blocked=blocked, leftovers=lo), fh)
print(f'DONE {tile} {band}: blocked per variant {[len(blocked[v]) for v in VARIANTS]} ({time.time() - t0:.0f} s)', flush=True)
