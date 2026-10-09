"""Which side carries the area-correlated term?  Regress JWST - ground residuals on the pixel-area prediction.

Ground reference: Ascenso et al. 2007 VLT/ISAAC H, Ks (thirdref/thirdref_matched*.fits; A = ours main2, B = dolphot,
mutual matches within 0.3 arcsec).  The ground photometry has no NIRCam pixel-area pattern, so for a band X:
    r_A = A_X - ground - colour term  vs  pred   slope 1 if ours lacks only the area correction,
    r_B = B_X - ground - colour term  vs  pred   slope 0 if dolphot carries the correct area correction.
pred = 2.5 log10(mean over exposures of PIXAR_SR * AREA(x, y) / proj_plane_pixel_area), as in areapred.py.
Isolation and blend cuts follow thirdref.py (blend fraction < 5 % / 20 %, eH < 0.1 / 0.2 for F150W).
Colour term a + b * (H - Ks) fitted separately for A and B (3-sigma clipped).  The magnitude trend of the residual
is removed with medians in 1-mag bins of the ground magnitude, separately for JWST-saturated and unsaturated stars,
before the Theil-Sen fit against pred.
usage: python groundarea.py [default|loose] > groundarea_<sel>.txt"""
import sys
import glob
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import stats

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sel = sys.argv[1] if len(sys.argv) > 1 else 'default'
t = Table.read(f'{Q}/thirdref/thirdref_matched{"" if sel == "default" else "_" + sel}.fits')
sky = SkyCoord(t['RA'], t['DEC'], unit='deg')
n = len(t)


def areapred(band):
    fsum, nexp = np.zeros(n), np.zeros(n)
    for fn in sorted(glob.glob(f'{Q}/tree_main2/F{band}/pipeline/jw03523005001_*_align_o005_crf.fits')):
        with fits.open(fn, memmap=True) as fh:
            h = fh['SCI'].header
            area = np.asarray(fh['AREA'].data, float)
        w = WCS(h)
        ny, nx = area.shape
        x, y = w.world_to_pixel(sky)
        inn = (x > 0) & (x < nx - 1) & (y > 0) & (y < ny - 1)
        a = np.full(n, np.nan)
        a[inn] = area[np.round(y[inn]).astype(int), np.round(x[inn]).astype(int)] * h['PIXAR_SR'] / w.proj_plane_pixel_area().to(u.sr).value
        good = np.isfinite(a) & (a > 0)
        fsum[good] += a[good]
        nexp[good] += 1
    pred = np.full(n, np.nan)
    pred[nexp > 0] = 2.5 * np.log10(fsum[nexp > 0] / nexp[nexp > 0])
    return pred


def colourfit(res, col, ok):
    m = ok.copy()
    for _ in range(5):
        b, a = np.polyfit(col[m], res[m], 1)
        r = res - (a + b * col)
        s = 1.4826 * np.median(np.abs(r[m] - np.median(r[m])))
        m = ok & (np.abs(r) < 3 * s)
    return r


def detrend(r, g, sat, ok):
    out = np.full(n, np.nan)
    for cls in (sat, ~sat):
        for lo in np.arange(np.floor(np.nanmin(g[ok])), np.nanmax(g[ok]) + 1):
            s = ok & cls & (g >= lo) & (g < lo + 1)
            if s.sum() >= 5:
                out[s] = r[s] - np.median(r[s])
    return out


def ts(y, x, m):
    m = m & np.isfinite(y) & np.isfinite(x)
    if m.sum() < 20:
        return f'N {m.sum()}'
    s = stats.theilslopes(y[m], x[m])
    ols = stats.linregress(x[m], y[m])
    return f'Theil-Sen {s[0]:+.2f} [{s[2]:+.2f}, {s[3]:+.2f}], OLS {ols.slope:+.2f} +- {ols.stderr:.2f} (N {m.sum()})'


col = np.asarray(t['H'] - t['Ks'], float)
for band, gname, err in (('150W', 'H', 'eH'), ('200W', 'Ks', None)):
    pred = areapred(band)
    g = np.asarray(t[gname], float)
    A, B = np.asarray(t[f'A{band[:3]}'], float), np.asarray(t[f'B{band[:3]}'], float)
    sat = np.asarray(t[f'sat{band[:3]}'], bool)
    ok = np.isfinite(A) & np.isfinite(B) & np.isfinite(g) & np.isfinite(col) & np.isfinite(pred) & np.asarray(t['ground_isolated'], bool)
    ok &= np.asarray(t[f'nfrac{band[:3]}'], float) < (0.05 if sel == 'default' else 0.2)
    if err:
        ok &= np.asarray(t[err], float) < (0.1 if sel == 'default' else 0.2)
    rA = detrend(colourfit(A - g, col, ok & ~sat), g, sat, ok)
    rB = detrend(colourfit(B - g, col, ok & ~sat), g, sat, ok)
    dAB = detrend(A - B, g, sat, ok)
    print(f'\n### F{band} vs {gname} ({sel} selection; rms(pred) {np.nanstd(pred[ok]):.4f}, N sat {int((ok & sat).sum())}, unsat {int((ok & ~sat).sum())})')
    print('| sample | ours - ground vs pred | dolphot - ground vs pred | ours - dolphot vs pred |')
    print('|---|---|---|---|')
    for lab, m in (('all', ok), ('unsaturated', ok & ~sat), ('saturated', ok & sat)):
        print(f'| {lab} | {ts(rA, pred, m)} | {ts(rB, pred, m)} | {ts(dAB, pred, m)} |')
