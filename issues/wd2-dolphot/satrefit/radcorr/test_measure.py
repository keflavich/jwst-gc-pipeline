"""In-frame Delta(r) (satstar_psf_radial.measure_radial_correction) vs the offline Delta_unsat used by satrefit v7b."""
import sys, time, warnings
import numpy as np
sys.path.insert(0, '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-satwing')
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
from astropy.io import fits
import jwst_gc_pipeline.reduction.satstar_psf_radial as R
import jwst_gc_pipeline.reduction.saturated_star_finding as S
warnings.filterwarnings('ignore', category=RuntimeWarning)
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
band, det, exp = sys.argv[1], sys.argv[2], int(sys.argv[3])
lw = band in ('250M', '300M')
vg = {'150W': '10101', '200W': '12101', '250M': '04101', '300M': '12101'}[band]
stem = f'jw03523005001_{vg}_{exp:05d}_{det}_align_o005_crf'
fn = f'{Q}/tree_main2/F{band}/pipeline/{stem}.fits'
fh = fits.open(fn)
hdr = fh[0].header
data = np.array(fh['SCI'].data, float)
data[np.isnan(fh['VAR_POISSON'].data)] = 0
err = np.array(fh['ERR'].data, float)
dq = fh['DQ'].data
SAT, DNU = 2, 1
sat = (dq & SAT) > 0
bad = sat | ((dq & DNU) > 0) | ~np.isfinite(data) | (data == 0) | ~np.isfinite(err) | (err <= 0)
pix = float(np.sqrt(fh['SCI'].header['PIXAR_A2']))
fwhm, fwhm_pix = S.get_fwhm(hdr, instrument_replacement='NIRCam')
import satrefit_core as C
grid, gf = C.load_grid(Q + '/tree_main2/psfs', hdr, lw)
rmax = 0.5 if lw else 0.35
t0 = time.time()
out = R.measure_radial_correction(data, err, sat, bad, grid, fwhm_pix=float(fwhm_pix), pixscale=pix,
                                  rmax_arcsec=rmax, iso_arcsec=1.0 if lw else 0.6)
print(f'{band} {det} exp{exp}: fwhm_pix={float(fwhm_pix):.2f} pix={pix:.4f} n_cal={out["n_cal"]} {time.time()-t0:.0f}s', flush=True)
# offline reference
if band in ('150W', '250M'):
    import re
    txt = open(Q + '/radprof/radprof.md').read().split('\n')
    key = f'### F{band} main2 per-bin Delta(r) (neighbours removed), iso, satstars precap'
    i = [k for k, l in enumerate(txt) if l.startswith(key)][0]
    rr, dd = [], []
    for l in txt[i + 4:]:
        if not l.startswith('|'):
            break
        cells = [c.strip() for c in l.strip('|').split('|')]
        m = re.match(r'([+-]?\d+\.\d+)', cells[-1])
        if m:
            rr.append(float(cells[0])); dd.append(float(m.group(1)))
    ref = (np.array(rr), np.array(dd))
else:
    a = np.loadtxt(f'{Q}/satrefit/out/delta_unsat_F{band}.txt')
    ref = (a[:, 0], a[:, 1])
print(' r_px  r_as   inframe  n    offline')
for r, d, n in zip(out['r_pix'], out['delta'], out['n']):
    ras = r * pix
    j = np.argmin(np.abs(ref[0] - ras))
    off = ref[1][j] if abs(ref[0][j] - ras) < 0.3 * max(ras, 0.01) + 0.004 else np.nan
    print(f'{r:5.2f} {ras:5.3f} {d:+8.3f} {n:4d} {off:+8.3f}')
print('correction', out['correction'])
