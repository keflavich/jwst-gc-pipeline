"""Cap analysis: per matched star, flux_fit vs flux_fit_precap in each arm, vs dolphot mag (uses kf_step_stars_<band>_withref.fits)."""
import sys
import numpy as np
from astropy.table import Table
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
band = sys.argv[1]
st = Table.read(f'{Q}/kf_lwmad/kf_step_stars_{band}_withref.fits')
fr = np.array([q.decode() if isinstance(q, bytes) else str(q) for q in st['frame']])
cols = {}
for arm in ('main2', 'main2kf'):
    out = {k: np.full(len(st), np.nan) for k in ('f', 'pre', 'pk', 'psf', 'ipk', 'opk')}
    for f in np.unique(fr):
        t = Table.read(f'{Q}/tree_{arm}/{band}/pipeline/{f}_align_o005_crf_resbgsub_m7_satstar_catalog.fits')
        x = np.ma.filled(t['xcentroid'], np.nan).astype(float); y = np.ma.filled(t['ycentroid'], np.nan).astype(float)
        for j in np.where(fr == f)[0]:
            # use off-arm coordinates for both arms (match within 0.5 px)
            d = (x - st['x'][j]) ** 2 + (y - st['y'][j]) ** 2
            k = int(np.argmin(d))
            if d[k] > 0.25:
                continue
            out['f'][j] = t['flux_fit'][k]; out['pre'][j] = t['flux_fit_precap'][k]
            out['psf'][j] = t['cap_psf_frac'][k]; out['ipk'][j] = t['satstar_implied_peak'][k]; out['opk'][j] = t['satstar_observed_peak'][k]
    cols[arm] = out
m = st['dmag_ref']
print(f'## {band}: capped fraction (flux_fit < 0.999 flux_fit_precap) and median flux/precap, per arm')
print('| dolphot mag | N | off capped | off f/precap | on capped | on f/precap | on/off precap | on/off f | med obs-peak on/off | med implied-peak on/off |')
print('|---|---|---|---|---|---|---|---|---|---|')
for lo, hi in [(12, 13), (13, 14), (14, 15), (15, 15.5), (15.5, 16), (16, 17)]:
    s = (m >= lo) & (m < hi) & np.isfinite(cols['main2']['f']) & np.isfinite(cols['main2kf']['f'])
    if not s.any():
        continue
    a, b = cols['main2'], cols['main2kf']
    print(f"| {lo}-{hi} | {s.sum()} | {np.mean(a['f'][s] < 0.999 * a['pre'][s]):.2f} | {np.nanmedian(a['f'][s] / a['pre'][s]):.3f} | "
          f"{np.mean(b['f'][s] < 0.999 * b['pre'][s]):.2f} | {np.nanmedian(b['f'][s] / b['pre'][s]):.3f} | "
          f"{np.nanmedian(b['pre'][s] / a['pre'][s]):.3f} | {np.nanmedian(b['f'][s] / a['f'][s]):.3f} | "
          f"{np.nanmedian(b['opk'][s] / a['opk'][s]):.3f} | {np.nanmedian(b['ipk'][s] / a['ipk'][s]):.3f} |")
