"""Spikes that the finder does not judge because a neighbour carries SATURATED, and the leftovers of the residual mosaics at them.

For every crf frame of one tile-band: the isolated one-pixel spike finder as shipped (the tolerance of the pull request 1392 at
``MAX_UNUSABLE``) and the same finder with the SATURATED bit cleared in a copy of the DQ.  A pixel that the second run returns, that
carries neither SATURATED nor DO_NOT_USE itself, and that the first run does not return is a *blocked spike*: it passes the isolation
test with the SATURATED neighbours counted by their SCI values, and the SATURATED bit alone keeps it out of the mask.  For every
blocked spike the script records the 3x3 box (SCI and DQ), the number and the flux of its SATURATED neighbours, the area of the
SATURATED blobs that those neighbours belong to, the distance to the nearest large SATURATED blob, the 3x3 maximum at its sky
position in every other frame that covers it (the fraction of its excess), and the distances to the vetted merged catalog and the
satstar catalog.  Then every leftover of the residual mosaics listed by the scoring step of the run (tags ``prod5``, ``optnm``, ``optfin``)
is matched in sky position to the blocked spikes and to the spikes that the finder as shipped returns at ``SHIP_MIN``.

usage: python -I scan_satnbr.py TILE BAND --root ROOT --score-dir DIR --out OUTDIR [--tags prod5,optnm,optfin]
READ ONLY on the frames, the catalogs and the leftover lists.  The code under test is the checkout that holds this file, or the one that ``PIPE_ROOT`` names."""
import argparse
import glob
import json
import os
import sys
import time
import warnings

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy import ndimage
from scipy.spatial import cKDTree

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
ap.add_argument('--tags', default='prod5,optnm,optfin')
ap.add_argument('--floor', type=float, default=100.0, help='MJy/sr: the lowest excess of a blocked spike that is kept')
ap.add_argument('--ship-min', type=float, default=300.0, help='MJy/sr: the excess of the shipped recipe (POSITIVE_SPIKE_MASK_MIN_EXCESS)')
ap.add_argument('--max-unusable', type=int, default=1)
ap.add_argument('--match-arcsec', type=float, default=0.1)
ap.add_argument('--big-blob-px', type=int, default=25, help='area in pixels from which a SATURATED blob counts as a saturated core')
args = ap.parse_args()
tile, band, outdir = args.tile, args.band, args.out
FLOOR, SHIP_MIN, KU = args.floor, args.ship_min, args.max_unusable
FWHM_ARCSEC = {'F212N': 0.072, 'F480M': 0.164}[band]
JUMP_DET = 4
root = args.root
paths = sorted(glob.glob(f'{root}/{band}/pipeline/jw10678*_destreak_{tile}_crf.fits'))
assert paths, f'no crf frames under {root}/{band}/pipeline'
t0 = time.time()
frames = [S.CrfFrame(p) for p in paths]
print(f'{tile} {band}: {len(frames)} frames, PIPE_ROOT={PIPE_ROOT}, finder limit {S.SPIKE_MAX_UNUSABLE_LIMIT}, floor {FLOOR}, ship {SHIP_MIN}, '
      f'max_unusable {KU} ({time.time() - t0:.0f} s)', flush=True)


def unit(ra, dec):
    ra, dec = np.radians(np.atleast_1d(ra)), np.radians(np.atleast_1d(dec))
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]).T


def chord_of(arcsec):
    return 2.0 * np.sin(np.radians(arcsec / 3600.0) / 2.0)


def arcsec_of(chord):
    return np.degrees(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))) * 3600.0


def tree_of(table, col):
    sc = table[col]
    ra = np.asarray(sc.ra.deg, dtype=float)
    dec = np.asarray(sc.dec.deg, dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    return cKDTree(unit(ra[ok], dec[ok])), ok


vet = Table.read(f'{root}/catalogs/{band.lower()}_merged_{tile}_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits')
sat = Table.read(f'{root}/catalogs/{band.lower()}_{tile}_consolidated_satstar_catalog.fits')
tree_vet, ok_vet = tree_of(vet, 'skycoord')
tree_sat, ok_sat = tree_of(sat, 'skycoord_fit')
nmatch_vet = np.asarray(vet['nmatch'], dtype=float)[ok_vet]
print(f'catalog rows: vetted {int(ok_vet.sum())}, satstar {int(ok_sat.sum())}', flush=True)

RING_DXY = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)]
STRUCT8 = np.ones((3, 3), dtype=int)

blocked = []
shipped = []
per_frame = []
for fr in frames:
    sci, dq = fr.load()
    dq_nosat = dq & ~np.int64(S.SATURATED)
    # the finder as the recipe runs it, at the recipe excess; every spike it returns is already masked
    ship = S.find_positive_spikes(sci, dq, SHIP_MIN, max_unusable=KU)
    # at the floor, with and without the SATURATED bit
    own = S.find_positive_spikes(sci, dq, FLOOR, max_unusable=KU)
    clr = S.find_positive_spikes(sci, dq_nosat, FLOOR, max_unusable=KU)
    own_xy = {(int(s['x']), int(s['y'])) for s in own}
    satpix = (dq & S.SATURATED) != 0
    lab, nlab = ndimage.label(satpix, structure=STRUCT8)
    area = np.bincount(lab.ravel()) if nlab else np.zeros(1, dtype=int)
    big = np.isin(lab, np.nonzero(area >= args.big_blob_px)[0]) & (lab > 0) if nlab else np.zeros(sci.shape, dtype=bool)
    dbig = ndimage.distance_transform_edt(~big) if big.any() else None
    nb_new = 0
    for s in clr:
        x, y = int(s['x']), int(s['y'])
        if (x, y) in own_xy:
            continue
        if dq[y, x] & (S.SATURATED | S.DO_NOT_USE):
            continue
        sat_nb = [(dx, dy) for dx, dy in RING_DXY if 0 <= y + dy < sci.shape[0] and 0 <= x + dx < sci.shape[1]
                  and (dq[y + dy, x + dx] & S.SATURATED)]
        if not sat_nb:
            continue
        ra, dec = S._world(fr, x, y)
        box_sci = sci[y - 1:y + 2, x - 1:x + 2]
        box_dq = dq[y - 1:y + 2, x - 1:x + 2]
        vals = [float(sci[y + dy, x + dx]) for dx, dy in sat_nb]
        comp_area = [int(area[lab[y + dy, x + dx]]) for dx, dy in sat_nb]
        sat_frac = [(v - s['bkg']) / s['excess'] if np.isfinite(v) else float('nan') for v in vals]
        blocked.append(dict(frame=fr.name, x=x, y=y, ra=float(ra[0]), dec=float(dec[0]), peak=s['peak'], excess=s['excess'],
                            sigma=s['sigma'], nsigma=s['excess'] / s['sigma'], bkg=s['bkg'], nbfrac=s['neighbour'] / s['excess'],
                            unusable=int(s.get('unusable', 0)), n_sat_nb=len(sat_nb), sat_nb_vals=vals, sat_nb_frac=sat_frac,
                            sat_nb_comp_area=comp_area, max_comp_area=max(comp_area),
                            n_jump_nb=int(sum(1 for dx, dy in RING_DXY if dq[y + dy, x + dx] & JUMP_DET)),
                            d_bigsat_px=float(dbig[y, x]) if dbig is not None else float('nan'),
                            box_sci=[[float(v) for v in r] for r in box_sci], box_dq=[[int(v) for v in r] for r in box_dq]))
        nb_new += 1
    for s in ship:
        ra, dec = S._world(fr, int(s['x']), int(s['y']))
        shipped.append(dict(frame=fr.name, x=int(s['x']), y=int(s['y']), ra=float(ra[0]), dec=float(dec[0]), excess=s['excess'],
                            unusable=int(s.get('unusable', 0))))
    per_frame.append(dict(name=fr.name, shipped=len(ship), own_floor=len(own), cleared_floor=len(clr), blocked=nb_new,
                          n_sat_blobs=int(nlab), n_big_blobs=int((area[1:] >= args.big_blob_px).sum()) if nlab else 0))
    print(f'{fr.name[:36]}: shipped {len(ship)}, own(floor) {len(own)}, cleared(floor) {len(clr)}, blocked {nb_new} '
          f'({time.time() - t0:.0f} s)', flush=True)

# ---- confirmation in the other frames: 3x3 maximum at the sky position, as a fraction of the excess ----------------------
nb = len(blocked)
others = [[] for _ in range(nb)]
if nb:
    b_ra = np.array([b['ra'] for b in blocked])
    b_dec = np.array([b['dec'] for b in blocked])
    b_bkg = np.array([b['bkg'] for b in blocked])
    b_exc = np.array([b['excess'] for b in blocked])
    b_frame = np.array([b['frame'] for b in blocked])
    for fo in frames:
        xs, ys = fo.wcs.world_to_pixel_values(b_ra, b_dec)
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        ny, nx = fo.shape
        sel = np.nonzero((b_frame != fo.name) & np.isfinite(xs) & np.isfinite(ys) & (xs >= 2) & (xs < nx - 3) & (ys >= 2) & (ys < ny - 3))[0]
        if not sel.size:
            continue
        with fits.open(fo.path, memmap=True) as hdul:
            data = hdul['SCI'].data
            for i in sel:
                ix, iy = int(round(xs[i])), int(round(ys[i]))
                box = np.asarray(data[iy - 1:iy + 2, ix - 1:ix + 2], dtype=float)
                if np.isfinite(box).sum() >= 6:
                    others[i].append((float(np.nanmax(box)) - b_bkg[i]) / b_exc[i])
    d_v, i_v = tree_vet.query(unit(b_ra, b_dec))
    d_s, _ = tree_sat.query(unit(b_ra, b_dec))
    for i, b in enumerate(blocked):
        b['n_other_frames'] = len(others[i])
        b['other_frac_max'] = max(others[i]) if others[i] else float('nan')
        b['other_fracs'] = [round(v, 3) for v in others[i]]
        b['d_vetted_arcsec'] = float(arcsec_of(d_v[i]))
        b['nmatch_nearest'] = float(nmatch_vet[int(i_v[i])])
        b['d_satstar_arcsec'] = float(arcsec_of(d_s[i]))
print(f'blocked spikes: {nb} ({time.time() - t0:.0f} s)', flush=True)

# ---- the leftovers of the residual mosaics against the blocked spikes and the shipped spikes ------------------------------
tree_blk = cKDTree(unit([b['ra'] for b in blocked], [b['dec'] for b in blocked])) if blocked else None
tree_shp = cKDTree(unit([s['ra'] for s in shipped], [s['dec'] for s in shipped])) if shipped else None
rad = chord_of(args.match_arcsec)
leftovers = {}
for tag in args.tags.split(','):
    path = f'{args.score_dir}/{tag}_{tile}_{band}_leftovers.json'
    with open(path) as fh:
        lst = json.load(fh)
    rows = []
    for r in lst:
        u = unit(r['ra'], r['dec'])[0]
        ib = sorted(tree_blk.query_ball_point(u, rad), key=lambda j: -blocked[j]['excess']) if tree_blk is not None else []
        is_ = sorted(tree_shp.query_ball_point(u, rad), key=lambda j: -shipped[j]['excess']) if tree_shp is not None else []
        rows.append(dict(x=r['x'], y=r['y'], ra=r['ra'], dec=r['dec'], resid_peak=r['resid_peak'], data_peak=r['data_peak'], cls=r['cls'],
                         blocked=ib, shipped=is_))
    leftovers[tag] = rows
    n100 = sum(1 for r in rows if r['resid_peak'] >= 100)
    nb100 = sum(1 for r in rows if r['resid_peak'] >= 100 and r['blocked'])
    ns100 = sum(1 for r in rows if r['resid_peak'] >= 100 and r['shipped'])
    print(f'{tag}: leftovers {len(rows)}, >=100: {n100}, with a blocked spike {nb100}, with a shipped spike {ns100}', flush=True)

res = dict(tile=tile, band=band, n_frames=len(frames), fwhm_arcsec=FWHM_ARCSEC, floor=FLOOR, ship_min=SHIP_MIN, max_unusable=KU,
           match_arcsec=args.match_arcsec, big_blob_px=args.big_blob_px, pipe_root=PIPE_ROOT, per_frame=per_frame, blocked=blocked,
           shipped=shipped, leftovers=leftovers)
os.makedirs(outdir, exist_ok=True)
with open(f'{outdir}/satnbr_{tile}_{band}.json', 'w') as fh:
    json.dump(res, fh)
print(f'DONE {tile} {band}: {nb} blocked spikes, {len(shipped)} shipped spikes ({time.time() - t0:.0f} s)', flush=True)
