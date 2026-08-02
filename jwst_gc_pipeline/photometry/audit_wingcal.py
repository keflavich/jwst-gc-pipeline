"""Audit the wing-selfcal correction on REAL data (no injection).

Cross-frame consistency: a real saturated star seen in multiple exposures has a
constant TRUE flux, so its wing-selfcal-CORRECTED flux should not depend on its
per-frame mask radius (rmask). If the corrected flux trends DOWN with rmask, the
correction OVER-does deep stars (recovered too faint = the CMD "hook"); flat =
the correction is well-behaved.

For each frame: run get_saturated_stars (current code, wing-selfcal ON) -> catalog
with flux_fit (corrected), flux_fit_raw, wingcal_ratio, wingcal_rmask. Cross-match
stars across frames; per star normalise flux to its cross-frame median; pool
(rmask, delta-mag) and bin. Compares RAW (the bias the wingcal targets) vs
CORRECTED (what's left after the wingcal) vs rmask.
"""
import os
import sys
import glob

os.environ.setdefault('CRDS_PATH', '/orange/adamginsburg/jwst/crds')
os.environ.setdefault('STPSF_PATH', '/orange/adamginsburg/jwst/stpsf-data/')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u

D = '/orange/adamginsburg/jwst/brick/F200W/pipeline'
PSFS = '/orange/adamginsburg/jwst/brick/psfs/'
FRAMES = [f'{D}/jw01182004001_04101_{i:05d}_nrca1_destreak_o004_crf.fits'
          for i in range(1, 9)]
OUT = os.environ.get('AUDIT_OUT', '/blue/adamginsburg/adamginsburg/tmp/wingcal_audit')


def gen_catalogs():
    from jwst_gc_pipeline.reduction import saturated_star_finding as ssf
    os.makedirs(OUT, exist_ok=True)
    cats = []
    for k, f in enumerate(FRAMES):
        if not os.path.exists(f):
            print(f'[audit] missing {f}', flush=True); continue
        cp = f'{OUT}/cat_{k}.fits'
        if os.path.exists(cp):
            cats.append(Table.read(cp)); continue
        cat = ssf.get_saturated_stars(fits.open(f), path_prefix=PSFS, plot=False,
                                      use_merged_psf_for_merged=False)
        if cat is None or not len(cat):
            print(f'[audit] no satstars in frame {k}', flush=True); continue
        cat['frame'] = k
        cat.write(cp, overwrite=True)
        cats.append(cat)
        print(f'[audit] frame {k}: {len(cat)} satstars, '
              f'wingcal cols={"wingcal_rmask" in cat.colnames}', flush=True)
    return cats


def link_stars(allcat, tol_arcsec=0.15):
    sc = SkyCoord(allcat['skycoord_fit'])
    fin = np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg)
    idx_all = np.where(fin)[0]
    scf = sc[fin]
    from scipy.spatial import cKDTree
    xyz = np.array(scf.cartesian.xyz).T
    tree = cKDTree(xyz)
    r = (tol_arcsec * u.arcsec).to(u.rad).value
    pairs = tree.query_pairs(r * 2)      # chord ~ angle for small r
    par = list(range(len(scf)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for a, b in pairs:
        if scf[a].separation(scf[b]).arcsec < tol_arcsec:
            par[find(a)] = find(b)
    groups = {}
    for i in range(len(scf)):
        groups.setdefault(find(i), []).append(idx_all[i])
    return groups


def main():
    cats = gen_catalogs()
    cats = [c for c in cats if c is not None and 'wingcal_rmask' in c.colnames]
    if len(cats) < 2:
        raise SystemExit('[audit] <2 catalogs with wingcal columns')
    allcat = vstack(cats)
    for col in ('flux_fit', 'flux_fit_raw', 'wingcal_rmask', 'wingcal_ratio'):
        if col not in allcat.colnames:
            raise SystemExit(f'[audit] missing column {col}')
    groups = link_stars(allcat)
    rmask, dmag_corr, dmag_raw, ratio = [], [], [], []
    n_multi = 0
    for members in groups.values():
        if len(members) < 3:
            continue
        m = np.array(members)
        fc = np.asarray(allcat['flux_fit'][m], float)
        fr = np.asarray(allcat['flux_fit_raw'][m], float)
        rm = np.asarray(allcat['wingcal_rmask'][m], float)
        rt = np.asarray(allcat['wingcal_ratio'][m], float)
        ok = np.isfinite(fc) & np.isfinite(fr) & np.isfinite(rm) & (fc > 0) & (fr > 0)
        if ok.sum() < 3:
            continue
        n_multi += 1
        medc, medr = np.median(fc[ok]), np.median(fr[ok])
        rmask += list(rm[ok]); ratio += list(rt[ok])
        dmag_corr += list(-2.5 * np.log10(fc[ok] / medc) * 1000)   # mmag vs star's own median
        dmag_raw += list(-2.5 * np.log10(fr[ok] / medr) * 1000)
    rmask = np.array(rmask); dmag_corr = np.array(dmag_corr)
    dmag_raw = np.array(dmag_raw); ratio = np.array(ratio)
    print(f'\n[audit] {n_multi} stars seen in >=3 frames; {len(rmask)} detections')

    print('\n=== corrected & raw flux (mmag vs each star\'s cross-frame median) by rmask ===')
    print(f'  {"rmask":>10} {"N":>5} {"raw":>9} {"corrected":>10}   (mmag; corrected should be FLAT if wingcal ok)')
    edges = [0, 5, 8, 12, 16, 20, 25, 40]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (rmask >= lo) & (rmask < hi)
        if s.sum() >= 5:
            print(f'  {lo:4d}-{hi:<4d}  {int(s.sum()):5d} {np.median(dmag_raw[s]):+9.0f} '
                  f'{np.median(dmag_corr[s]):+10.0f}')
    # slope of corrected vs rmask (the mis-calibration signature)
    good = np.isfinite(dmag_corr) & np.isfinite(rmask)
    if good.sum() > 20:
        sl_c = np.polyfit(rmask[good], dmag_corr[good], 1)[0]
        sl_r = np.polyfit(rmask[good], dmag_raw[good], 1)[0]
        print(f'\n  slope corrected vs rmask: {sl_c:+.1f} mmag/px  (0=good; <0=over-correct deep=hook)')
        print(f'  slope raw       vs rmask: {sl_r:+.1f} mmag/px  (the bias the wingcal targets)')

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    a.scatter(rmask, dmag_raw, s=6, alpha=0.25, color='0.6', label='raw (pre-wingcal)')
    a.scatter(rmask, dmag_corr, s=6, alpha=0.35, color='tab:red', label='corrected')
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (rmask >= lo) & (rmask < hi)
        if s.sum() >= 5:
            a.plot(0.5 * (lo + hi), np.median(dmag_corr[s]), 'o', color='darkred', ms=9, mec='k')
            a.plot(0.5 * (lo + hi), np.median(dmag_raw[s]), 's', color='k', ms=8)
    a.axhline(0, color='0.5'); a.set_ylim(-600, 600)
    a.set_xlabel('per-frame wingcal_rmask (px)')
    a.set_ylabel('flux vs star cross-frame median (mmag)')
    a.set_title('(a) cross-frame consistency vs mask radius\n(corrected should be FLAT)')
    a.legend(fontsize=8)
    a = ax[1]
    a.scatter(rmask, ratio, s=6, alpha=0.3, color='tab:blue')
    a.set_xlabel('wingcal_rmask (px)'); a.set_ylabel('applied wingcal_ratio (÷)')
    a.set_title('(b) applied correction vs mask radius')
    fig.suptitle('Wing-selfcal audit on REAL brick F200W (cross-frame, no injection)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = f'{OUT}/wingcal_audit.png'
    fig.savefig(p, dpi=120)
    print(f'\n[audit] wrote {p}')


if __name__ == '__main__':
    main()
