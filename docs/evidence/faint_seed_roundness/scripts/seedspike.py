"""Full-frame test of the loose-roundness spike guard (#1020).

Brick NRCB F182M: m6 residual i2d minus the m6 smoothed background, with the
m6 vetted catalog as the previous seed (as the m7/m6 seed step does).  Runs
_build_i2d_augmented_seed with (A) the tight cut only, (B) loose 0.8 without
the spike guard, (C) loose 0.8 with the spike guard (radius 3").  Then
histograms the position angle (mod 60 deg, pixel frame) of each new seed
about its nearest top-0.5% star within 2", for each population.

usage: python seedspike.py <worktree> <field> <filt> <module>
"""
import json
import os
import shutil
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import wcs
from scipy.spatial import cKDTree

WT, FIELD, FILT, MOD = sys.argv[1:5]
sys.path.insert(0, WT)
from jwst_gc_pipeline.photometry import cataloging as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = '/orange/adamginsburg/jwst'
CFG = {
    'brick': dict(pipe=f'{R}/brick/{FILT.upper()}/pipeline',
                  stem=f'jw02221-o001_t001_nircam_clear-{FILT}-{MOD}_resbgsub_m6_daophot_basic_mergedcat_residual',
                  cat=f'{R}/brick/catalogs/{FILT}_{MOD}_o001_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits'),
    'sgrb2': dict(pipe=f'{R}/sgrb2/NB/{FILT.upper()}/pipeline',
                  stem=f'jw05365-o001_t001_nircam_clear-{FILT}-{MOD}_resbgsub_m6_daophot_basic_mergedcat_residual',
                  cat=f'{R}/sgrb2/catalogs/{FILT}_{MOD}_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits'),
}[FIELD]
det = f"{CFG['pipe']}/{CFG['stem']}_i2d.fits"
bg = f"{CFG['pipe']}/{CFG['stem']}_smoothed_bg_i2d.fits"
for p in (det, bg, CFG['cat']):
    assert os.path.exists(p), p


def main():
    RUNS = {'A_tight': dict(round_loose_max=0.0),
            'B_loose': dict(round_loose_max=0.8, round_loose_spike_radius_as=0.0),
            'C_guard': dict(round_loose_max=0.8, round_loose_spike_radius_as=3.0)}
    out = {}
    tabs = {}
    for name, kw in RUNS.items():
        d = f'{HERE}/run_{FIELD}_{FILT}_{MOD}/{name}'
        os.makedirs(d, exist_ok=True)
        prev = f'{d}/{os.path.basename(CFG["cat"])}'
        if not os.path.exists(prev):
            shutil.copy(CFG['cat'], prev)
        path = C._build_i2d_augmented_seed(det, prev, FILT.upper(), local_snr_min=5.0,
                                           roundlo=-0.5, roundhi=0.5, bg_subtract_path=bg,
                                           round_loose_prom_min=5.0, label=name, **kw)
        tabs[name] = Table.read(path)
        print(name, len(tabs[name]), flush=True)

    with fits.open(det) as h:
        ww = wcs.WCS(h['SCI'].header)
    prevt = Table.read(CFG['cat'])
    pf = np.asarray(prevt['flux'], float)
    g = np.isfinite(pf) & (pf > 0)
    thr = np.percentile(pf[g], 99.5)
    bsky = C._L._resolve_seed_skycoords(prevt[g & (pf >= thr)])['skycoord']
    bx, by = ww.world_to_pixel(bsky)
    tree = cKDTree(np.c_[bx, by])
    pixscale = float(np.sqrt(abs(np.linalg.det(ww.pixel_scale_matrix))) * 3600)
    r2 = 2.0 / pixscale

    def pa_hist(t, sel):
        x, y = ww.world_to_pixel(t['skycoord'][sel])
        d, j = tree.query(np.c_[x, y])
        near = np.isfinite(d) & (d <= r2) & (d > 2)
        pa = np.degrees(np.arctan2(y[near] - by[j[near]], x[near] - bx[j[near]])) % 60
        h, _ = np.histogram(pa, bins=12, range=(0, 60))
        return dict(n=int(sel.sum()), n_near=int(near.sum()), hist=h.tolist(),
                    max_over_median=float(h.max() / max(np.median(h), 1)))

    A, B, Cg = tabs['A_tight'], tabs['B_loose'], tabs['C_guard']
    newA = np.asarray(A['seed_origin']).astype(str) == 'i2d'
    looseB = np.asarray(B['seed_round_loose'], bool)
    looseC = np.asarray(Cg['seed_round_loose'], bool)
    out['counts'] = dict(A_total=len(A), A_new=int(newA.sum()), B_total=len(B), B_loose=int(looseB.sum()),
                         C_total=len(Cg), C_loose=int(looseC.sum()))
    out['tight_new'] = pa_hist(A, newA)
    out['loose_no_guard'] = pa_hist(B, looseB)
    out['loose_guard'] = pa_hist(Cg, looseC)
    # dropped by the guard: loose in B, absent from C within 30 mas
    from astropy.coordinates import SkyCoord  # noqa: E402
    bl = B['skycoord'][looseB]
    cl = Cg['skycoord'][looseC]
    _, sep, _ = bl.match_to_catalog_sky(cl)
    dropped = np.zeros(len(B), bool)
    dropped[np.where(looseB)[0][sep.arcsec > 0.03]] = True
    out['dropped_by_guard'] = pa_hist(B, dropped)
    # chance-flag rate: the guard applied to the tight new seeds (never applied in production)
    xa, ya = ww.world_to_pixel(A['skycoord'][newA])
    with fits.open(det) as h:
        img = np.asarray(h['SCI'].data, float)
    with fits.open(bg) as h:
        img = img - np.nan_to_num(np.asarray(h['SCI'].data, float))
    flag = C._radially_elongated(img, xa, ya, bx, by, radius_pix=3.0 / pixscale, half=3)
    d3, _ = tree.query(np.c_[xa, ya])
    w3 = d3 <= 3.0 / pixscale
    out['tight_new_flag_rate'] = dict(n_within_3as=int(w3.sum()), n_flagged=int(flag.sum()),
                                      rate_within_3as=float(flag.sum() / max(w3.sum(), 1)))
    xb, yb = ww.world_to_pixel(B['skycoord'][looseB])
    d3b, _ = tree.query(np.c_[xb, yb])
    out['loose_within_3as'] = int((d3b <= 3.0 / pixscale).sum())
    print(json.dumps(out, indent=1))
    json.dump(out, open(f'{HERE}/seedspike_{FIELD}_{FILT}_{MOD}.json', 'w'), indent=1)

    # emission: rank of the m6 smoothed background at each seed among all finite
    # bg pixels (filaments = high rank)
    with fits.open(bg) as h:
        bgim = np.asarray(h['SCI'].data, float)
    fin = np.sort(bgim[np.isfinite(bgim)].ravel())

    def bg_rank(t, sel):
        x, y = ww.world_to_pixel(t['skycoord'][sel])
        ix = np.clip(np.round(x).astype(int), 0, bgim.shape[1] - 1)
        iy = np.clip(np.round(y).astype(int), 0, bgim.shape[0] - 1)
        v = bgim[iy, ix]
        v = v[np.isfinite(v)]
        r = np.searchsorted(fin, v) / len(fin)
        return dict(n=int(len(r)), median_rank=float(np.median(r)) if len(r) else None,
                    frac_top10=float(np.mean(r >= 0.9)) if len(r) else None,
                    frac_top25=float(np.mean(r >= 0.75)) if len(r) else None)

    out['bg_rank'] = dict(base=bg_rank(A, ~newA), tight_new=bg_rank(A, newA),
                          loose_no_guard=bg_rank(B, looseB), loose_guard=bg_rank(Cg, looseC),
                          dropped_by_guard=bg_rank(B, dropped))
    print(json.dumps(out['bg_rank'], indent=1))
    json.dump(out, open(f'{HERE}/seedspike_{FIELD}_{FILT}_{MOD}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
