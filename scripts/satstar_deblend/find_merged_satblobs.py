#!/usr/bin/env python
"""Locate MERGED saturated-star blobs in gc2211 frames.

Problem: the satstar finder labels connected saturated-DQ pixels and fits ONE
PSF per connected component, placing it at the component centroid (bbox centre).
In gc2211 the field is so crowded that many bright stars' saturated cores TOUCH,
merging into one connected component -> one centroid that sits BETWEEN two stars
-> the satstar is fit at the wrong place and the pair is badly subtracted.

This diagnostic builds the saturated mask EXACTLY as the production finder does
(reusing ``find_saturated_stars``), labels components, and for each component
counts how many independent reference stars fall inside it.  Reference stars come
from two sources we cross-check:
  * GALACTICNUCLEUS 2021 (ground JHK; knows the bright GC stars that saturate)
  * the JWST merged daophot catalog rows flagged ``is_saturated`` (the pipeline's
    own saturated-star positions, at JWST resolution)
A component containing >=2 reference stars is a MERGED double (the problem case).

Outputs a table of the worst blobs (most stars, with pixel + sky coords) so we
can pick a few to zoom in on, and saves diagnostic cutout PNGs.

Run:
  python find_merged_satblobs.py [FILTER] [N_EXAMPLES]
"""
import os, sys, glob
os.environ.setdefault('STPSF_PATH', '/orange/adamginsburg/jwst/stpsf-data/')
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.ndimage import label, find_objects, sum_labels

sys.path.insert(0, '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-satdeblend')
from jwst_gc_pipeline.reduction.saturated_star_finding import find_saturated_stars

GC = '/orange/adamginsburg/jwst/gc2211'
FILTER = sys.argv[1].upper() if len(sys.argv) > 1 else 'F200W'
N_EX = int(sys.argv[2]) if len(sys.argv) > 2 else 8
OUTDIR = os.path.join(os.path.dirname(__file__), 'out')
os.makedirs(OUTDIR, exist_ok=True)

# ---- pick one deep GC obs frame (o028) ----
cands = sorted(glob.glob(f'{GC}/{FILTER}/jw02211028*nrca1_cal.fits'))
if not cands:
    cands = sorted(glob.glob(f'{GC}/{FILTER}/jw02211*nrca1_cal.fits'))
calfn = cands[0]
print(f'frame = {calfn}', flush=True)

fitsdata = fits.open(calfn)
hdr = fitsdata['SCI'].header
ww = WCS(hdr)
data = fitsdata['SCI'].data

saturated, sources, coms = find_saturated_stars(fitsdata)
nsrc = int(sources.max())
sizes = sum_labels(saturated, sources, np.arange(nsrc) + 1)
slices = find_objects(sources)
print(f'{nsrc} saturated components', flush=True)

# ---- reference catalogs ----
gns = Table.read(f'{GC}/catalogs/GALACTICNUCLEUS_2021_gc2211.fits')
gns_sc = SkyCoord(gns['RAJ2000'], gns['DEJ2000'], unit=(u.deg, u.deg))
gx, gy = ww.world_to_pixel(gns_sc)

dao = Table.read(f'{GC}/catalogs/{FILTER.lower()}_merged_indivexp_merged_dao_basic.fits')
dsc = SkyCoord(dao['skycoord'])
issat = np.asarray(dao['is_saturated'], dtype=bool)
dsat_sc_all = dsc[issat]
# Project onto THIS frame and keep only in-frame rows (huge reduction).
_dsx_all, _dsy_all = ww.world_to_pixel(dsat_sc_all)
_inframe = (np.isfinite(_dsx_all) & np.isfinite(_dsy_all) &
            (_dsx_all > -5) & (_dsx_all < data.shape[1] + 5) &
            (_dsy_all > -5) & (_dsy_all < data.shape[0] + 5))
_px = np.column_stack([_dsx_all[_inframe], _dsy_all[_inframe]])
# The merged daophot catalog is PER-EXPOSURE: the same physical star appears in
# every overlapping exposure (~7-28 rows at the same position).  Friends-of-
# friends link at 2 px in pixel space collapses those to DISTINCT stars so each
# blob's count is the real number of independent saturated stars (located by
# their unsaturated-wing daophot detections).
from scipy.spatial import cKDTree
def fof_dedupe(px, link_px=2.0):
    if len(px) == 0:
        return px
    tree = cKDTree(px)
    pairs = tree.query_pairs(link_px)
    parent = list(range(len(px)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in pairs:
        parent[find(a)] = find(b)
    groups = {}
    for i in range(len(px)):
        groups.setdefault(find(i), []).append(i)
    return np.array([px[idx].mean(axis=0) for idx in groups.values()])

_ded = fof_dedupe(_px, link_px=2.0)
dsx, dsy = _ded[:, 0], _ded[:, 1]
print(f'DAO is_saturated: {int(issat.sum())} rows total, {int(_inframe.sum())} '
      f'in-frame -> {len(dsx)} distinct stars', flush=True)

ny, nx = data.shape

def count_in_bbox(px, py, sl, margin=3):
    y0, y1 = sl[0].start - margin, sl[0].stop + margin
    x0, x1 = sl[1].start - margin, sl[1].stop + margin
    m = (px >= x0) & (px < x1) & (py >= y0) & (py < y1)
    return m

rows = []
for i in range(nsrc):
    sl = slices[i]
    if sl is None:
        continue
    area = int(sizes[i])
    if area < 4:            # tiny: not a bright merged core
        continue
    gm = count_in_bbox(gx, gy, sl)
    dm = count_in_bbox(dsx, dsy, sl)
    n_gns = int(gm.sum())
    n_dao = int(dm.sum())
    cy, cx = coms[i]
    h = sl[0].stop - sl[0].start
    w = sl[1].stop - sl[1].start
    rows.append((i + 1, area, w, h, n_gns, n_dao, float(cx), float(cy)))

T = Table(rows=rows, names=['label', 'sat_area', 'bbox_w', 'bbox_h',
                            'n_gns', 'n_dao_sat', 'cx', 'cy'])
# merged double = >=2 reference stars from EITHER source, or a fat bbox
T['n_ref'] = np.maximum(T['n_gns'], T['n_dao_sat'])
sky = ww.pixel_to_world(T['cx'], T['cy'])
T['ra'] = sky.ra.deg
T['dec'] = sky.dec.deg
T.sort('n_ref')
T.reverse()

T.write(os.path.join(OUTDIR, f'merged_satblobs_{FILTER.lower()}.ecsv'),
        overwrite=True)

multi = T[T['n_ref'] >= 2]
print(f'\n{len(multi)} / {len(T)} sizeable saturated blobs contain >=2 reference '
      f'stars (merged doubles+)', flush=True)
print('\nTop blobs (label area wxh n_gns n_dao cx cy ra dec):', flush=True)
for r in T[:25]:
    print(f"  L{r['label']:5d} area={r['sat_area']:4d} {r['bbox_w']:3d}x{r['bbox_h']:3d} "
          f"gns={r['n_gns']} dao={r['n_dao_sat']} ref={r['n_ref']} "
          f"px=({r['cx']:.0f},{r['cy']:.0f}) sky=({r['ra']:.6f},{r['dec']:.6f})",
          flush=True)

# ---- save zoom PNGs for the first few merged doubles ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ex = multi[:N_EX]
for k, r in enumerate(ex):
    lab = int(r['label'])
    sl = slices[lab - 1]
    pad = 25
    y0 = max(0, sl[0].start - pad); y1 = min(ny, sl[0].stop + pad)
    x0 = max(0, sl[1].start - pad); x1 = min(nx, sl[1].stop + pad)
    sub = data[y0:y1, x0:x1]
    satsub = saturated[y0:y1, x0:x1]
    vmax = np.nanpercentile(sub[np.isfinite(sub)], 99.5) if np.isfinite(sub).any() else 1
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.3))
    for a in ax:
        a.imshow(sub, origin='lower', vmin=0, vmax=vmax, cmap='gray')
    ax[1].contour(satsub, levels=[0.5], colors='red', linewidths=0.6)
    # overlay reference stars
    gmask = (gx >= x0) & (gx < x1) & (gy >= y0) & (gy < y1)
    ax[0].scatter(gx[gmask] - x0, gy[gmask] - y0, marker='+', c='cyan',
                  s=80, label=f'GNS ({gmask.sum()})')
    dmask = (dsx >= x0) & (dsx < x1) & (dsy >= y0) & (dsy < y1)
    ax[0].scatter(dsx[dmask] - x0, dsy[dmask] - y0, marker='x', c='lime',
                  s=60, label=f'DAO sat ({dmask.sum()})')
    ax[0].scatter(r['cx'] - x0, r['cy'] - y0, marker='o', facecolors='none',
                  edgecolors='orange', s=120, label='blob centroid (current seed)')
    ax[0].legend(fontsize=6, loc='upper right')
    ax[0].set_title(f"L{lab} ref={int(r['n_ref'])} gns={int(r['n_gns'])} "
                    f"dao={int(r['n_dao_sat'])}", fontsize=8)
    ax[1].set_title('saturated mask (red)', fontsize=8)
    fig.suptitle(f"sky=({r['ra']:.6f},{r['dec']:.6f}) px=({r['cx']:.0f},{r['cy']:.0f})",
                 fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUTDIR, f'blob_{FILTER.lower()}_{k:02d}_L{lab}.png')
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f'  wrote {out}', flush=True)

print('\nDONE', flush=True)
