"""Safety scan of the spike finder tolerance on one tile-band of a run.  Reads the frames and the catalogs, writes one JSON.

For every crf frame of the tile-band: the finder as shipped (``max_unusable=0``) and the tolerant one with the tolerance
raised to six (the scan lifts ``SPIKE_MAX_UNUSABLE_LIMIT`` in its own process, so the table covers more than the shipped
limit; every spike it returns carries ``unusable``).  Every spike that the tolerant finder adds is listed with its excess,
its sigma, the fraction of the excess in its brightest usable neighbour, the DQ of its missing neighbours, its distance to
the nearest SATURATED pixel, the distance to the nearest row of the vetted merged catalog and of the satstar catalog of the
run, and the 3x3 maximum at its sky position in the other frames that cover it.  A second table covers the complete-box
local maxima that are not spikes (the cores of stars): how many of them would pass the isolation test if their k brightest
neighbours were missing, k = 0 to 7, and for the cores that pass at k = 4 or less the 3x3 maximum at their sky position in
the other frames.

usage: python -I scan_spikes.py TILE BAND --root ROOT --out OUTDIR
ROOT is the ``gc-treasury`` directory of the tile in the run: it holds ``BAND/pipeline/jw10678*_destreak_TILE_crf.fits`` and
``catalogs/<band>_merged_TILE_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits`` and
``catalogs/<band>_TILE_consolidated_satstar_catalog.fits``.  The code under test is the checkout that holds this file, or
the one that ``PIPE_ROOT`` names.  READ ONLY on the frames and the catalogs."""
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
ap.add_argument('--out', required=True)
args = ap.parse_args()
tile, band, outdir = args.tile, args.band, args.out
MIN_EXC = 300.0
KMAX = 6
S.SPIKE_MAX_UNUSABLE_LIMIT = KMAX       # the scan goes beyond the shipped limit
FWHM_ARCSEC = {'F212N': 0.072, 'F480M': 0.164}[band]
root = args.root
paths = sorted(glob.glob(f'{root}/{band}/pipeline/jw10678*_destreak_{tile}_crf.fits'))
assert paths, f'no crf frames under {root}/{band}/pipeline'
t0 = time.time()
frames = [S.CrfFrame(p) for p in paths]
print(f'{tile} {band}: {len(frames)} frames ({time.time() - t0:.0f} s)', flush=True)


def unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]).T


def tree_of(table, col):
    sc = table[col]
    ra = np.asarray(sc.ra.deg, dtype=float)
    dec = np.asarray(sc.dec.deg, dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    return cKDTree(unit(ra[ok], dec[ok])), int(ok.sum())


vet = Table.read(f'{root}/catalogs/{band.lower()}_merged_{tile}_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits')
sat = Table.read(f'{root}/catalogs/{band.lower()}_{tile}_consolidated_satstar_catalog.fits')
tree_vet, n_vet = tree_of(vet, 'skycoord')
tree_sat, n_sat = tree_of(sat, 'skycoord_fit')
nmatch_vet = np.asarray(vet['nmatch'], dtype=float)
ok_vet = np.isfinite(np.asarray(vet['skycoord'].ra.deg)) & np.isfinite(np.asarray(vet['skycoord'].dec.deg))
nmatch_vet = nmatch_vet[ok_vet]            # same order as the rows of tree_vet
multi = nmatch_vet >= 2
vsub = vet[ok_vet][multi]
tree_multi, n_multi = tree_of(vsub, 'skycoord')
print(f'catalog rows: vetted {n_vet} (nmatch >= 2: {n_multi}), satstar {n_sat}', flush=True)


def sep_arcsec(tree, ra, dec, with_index=False):
    chord, idx = tree.query(unit(np.atleast_1d(ra), np.atleast_1d(dec)))
    sep = float(np.degrees(2.0 * np.arcsin(np.clip(chord[0] / 2.0, 0.0, 1.0))) * 3600.0)
    return (sep, int(idx[0])) if with_index else sep


def other_frames_max(fr, ra, dec, bkg, excess):
    """3x3 maximum at the sky position in each other frame that covers it, as a fraction of the excess."""
    out = []
    for fo in frames:
        if fo.name == fr.name:
            continue
        xo, yo = fo.wcs.world_to_pixel_values(ra, dec)
        xo, yo = float(np.ravel(xo)[0]), float(np.ravel(yo)[0])
        ny, nx = fo.shape
        if not (2 <= xo < nx - 3 and 2 <= yo < ny - 3):
            continue
        ix, iy = int(round(xo)), int(round(yo))
        with fits.open(fo.path, memmap=True) as hdul:
            box = np.asarray(hdul['SCI'].data[iy - 1:iy + 2, ix - 1:ix + 2], dtype=float)
        if np.isfinite(box).sum() >= 6:
            out.append((float(np.nanmax(box)) - bkg) / excess)
    return out


RING = np.ones((3, 3), dtype=bool)
RING[1, 1] = False
ring_dxy = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)]

per_frame = []
added = []
shipped_list = []
core = {'n_candidates': 0, 'n_starlike': 0, 'pass': [0] * 8, 'pass_starlike': [0] * 8, 'closest': [None] * 8,
        'closest_starlike': [None] * 8, 'vulnerable': [], 'n_vulnerable_by_kmin': [0] * 8}
STARLIKE = 0.3
KDETAIL = 4       # the cores that pass with up to this many of their brightest neighbours missing get the sky check in the other frames (the counts cover k up to 7)
for fr in frames:
    sci, dq = fr.load()
    ship = S.find_positive_spikes(sci, dq, MIN_EXC)
    tol = S.find_positive_spikes(sci, dq, MIN_EXC, max_unusable=KMAX)
    ship_xy = {(int(s['x']), int(s['y'])) for s in ship}
    new = [s for s in tol if (int(s['x']), int(s['y'])) not in ship_xy]
    by_k = {k: sum(1 for s in tol if s['unusable'] <= k) for k in range(1, KMAX + 1)}
    per_frame.append(dict(name=fr.name, shipped=len(ship), tolerant_by_k=by_k, added=len(new)))
    for s in ship:
        shipped_list.append(dict(frame=fr.name, x=int(s['x']), y=int(s['y']), excess=s['excess']))
    if new:
        satpix = (dq & S.SATURATED) != 0
        dsat = ndimage.distance_transform_edt(~satpix) if satpix.any() else None
        dnu = ~np.isfinite(sci) | ((dq & S.DO_NOT_USE) != 0)
    for s in new:
        x, y = int(s['x']), int(s['y'])
        ra, dec = S._world(fr, x, y)
        missing = []
        for dx, dy in ring_dxy:
            yy, xx = y + dy, x + dx
            if not (0 <= yy < sci.shape[0] and 0 <= xx < sci.shape[1]):
                missing.append(dict(dx=dx, dy=dy, dq=-1, finite=False))   # outside the array
            elif (not np.isfinite(sci[yy, xx])) or (dq[yy, xx] & (S.DO_NOT_USE | S.SATURATED)):
                missing.append(dict(dx=dx, dy=dy, dq=int(dq[yy, xx]), finite=bool(np.isfinite(sci[yy, xx]))))
        d_v, i_v = sep_arcsec(tree_vet, ra, dec, with_index=True)
        d_m = sep_arcsec(tree_multi, ra, dec)
        d_s = sep_arcsec(tree_sat, ra, dec)
        # nearest other missing pixel outside the 3x3 box of the spike
        box_mask = np.zeros(sci.shape, bool)
        box_mask[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = True
        outside_dnu = dnu & ~box_mask
        d_dnu = float(ndimage.distance_transform_edt(~outside_dnu)[y, x]) if outside_dnu.any() else float('nan')
        others = other_frames_max(fr, ra, dec, s['bkg'], s['excess'])
        added.append(dict(frame=fr.name, x=x, y=y, peak=s['peak'], excess=s['excess'], sigma=s['sigma'],
                          nsigma=s['excess'] / s['sigma'], nbfrac=s['neighbour'] / s['excess'],
                          unusable=s['unusable'], missing=missing,
                          d_sat_px=float(dsat[y, x]) if dsat is not None else float('nan'),
                          d_dnu_outside_px=d_dnu,
                          d_vetted_arcsec=d_v, d_vetted_fwhm=d_v / FWHM_ARCSEC, nmatch_nearest=float(nmatch_vet[i_v]),
                          d_multiframe_fwhm=d_m / FWHM_ARCSEC,
                          d_satstar_arcsec=d_s, d_satstar_fwhm=d_s / FWHM_ARCSEC,
                          n_other_frames=len(others), other_frac_max=(max(others) if others else float('nan')),
                          other_fracs=[round(o, 3) for o in others]))
    # star-core table: complete-box local maxima with excess >= MIN_EXC and >= 5 sigma
    usable = np.isfinite(sci) & ((dq & (S.DO_NOT_USE | S.SATURATED)) == 0)
    complete = ndimage.minimum_filter(usable.astype(np.uint8), size=3, mode='constant', cval=0) > 0
    filled = np.where(usable, sci, -np.inf)
    lm = (filled == ndimage.maximum_filter(filled, size=3, mode='constant', cval=-np.inf)) & complete & (sci >= 100.0)
    for y, x in zip(*np.nonzero(lm)):
        bkg, sig = S._annulus_stats(sci, x, y, S.SPIKE_BKG_R_IN, S.SPIKE_BKG_R_OUT)
        if not np.isfinite(bkg):
            continue
        exc = float(sci[y, x] - bkg)
        if exc < MIN_EXC or exc < S.SPIKE_MIN_SIGMA * sig:
            continue
        nb = np.array([sci[y + dy, x + dx] for dx, dy in ring_dxy]) - bkg
        r = np.sort(nb / exc)[::-1]
        core['n_candidates'] += 1
        starlike = bool(r[0] >= STARLIKE)
        core['n_starlike'] += int(starlike)
        for k in range(8):
            if r[k] <= S.SPIKE_ISOLATION:
                core['pass'][k] += 1
                core['pass_starlike'][k] += int(starlike)
            if r[0] > S.SPIKE_ISOLATION:       # a non-spike: how close is its (k+1)-th brightest neighbour to the limit
                for key, ok in (('closest', True), ('closest_starlike', starlike)):
                    c = core[key][k]
                    if ok and (c is None or r[k] < c['ratio']):
                        core[key][k] = dict(ratio=float(r[k]), frame=fr.name, x=int(x), y=int(y), excess=exc)
        if r[0] > S.SPIKE_ISOLATION:
            kmin = next((k for k in range(8) if r[k] <= S.SPIKE_ISOLATION), None)
            if kmin is not None:
                core['n_vulnerable_by_kmin'][kmin] += 1
                if kmin <= KDETAIL:
                    ra, dec = S._world(fr, int(x), int(y))
                    others = other_frames_max(fr, ra, dec, bkg, exc)
                    d_m = sep_arcsec(tree_multi, ra, dec)
                    core['vulnerable'].append(dict(frame=fr.name, x=int(x), y=int(y), excess=exc, kmin=kmin,
                                                   r=[round(float(v), 3) for v in r], n_other_frames=len(others),
                                                   other_frac_max=(max(others) if others else float('nan')),
                                                   d_multiframe_fwhm=d_m / FWHM_ARCSEC, starlike=starlike))
    print(f'{fr.name[:36]}: shipped {len(ship)}, tolerant(<= {KMAX}) {len(tol)}, added {len(new)} ({time.time() - t0:.0f} s)', flush=True)

tot = dict(shipped=sum(f['shipped'] for f in per_frame),
           tolerant_by_k={k: sum(f['tolerant_by_k'][k] for f in per_frame) for k in range(1, KMAX + 1)})
res = dict(tile=tile, band=band, n_frames=len(frames), fwhm_arcsec=FWHM_ARCSEC, min_excess=MIN_EXC, max_unusable_scanned=KMAX,
           n_vetted=n_vet, n_satstar=n_sat, per_frame=per_frame, totals=tot, shipped=shipped_list, added=added, starcore=core)
os.makedirs(outdir, exist_ok=True)
with open(f'{outdir}/spikes_{tile}_{band}.json', 'w') as fh:
    json.dump(res, fh, indent=1)
print(f'TOTAL {tile} {band}: shipped {tot["shipped"]}, by k {tot["tolerant_by_k"]}, added {len(added)}; '
      f'cores {core["n_candidates"]} ({core["n_starlike"]} star-like), pass by k {core["pass"]}, star-like {core["pass_starlike"]}', flush=True)
