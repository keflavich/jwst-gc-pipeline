"""Stage 4 (per frame, all LW satstar rows mapped to dolphot stars): group-0 peak near the fit position.  usage: stage4.py BAND PIXFILE"""
import sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import satrefit_core as C
import run_frames5 as R5
from satrefit_core import S, fits, ndimage, Table
band, pix = sys.argv[1], sys.argv[2]
H = C.Q + '/satrefit/bright13'
fn = f'{R5.TREE}/F{band}/pipeline/{pix}.fits'
rows = Table.read(f'{H}/allrows_{band}.fits'); rows = rows[rows['pixfile'] == pix]
cat = Table.read(fn.replace('.fits', '') + '_resbgsub_m7_satstar_catalog.fits')
FL = R5.load_frame(fn)
g0, sat = FL['g0'], FL['sat']
fin = np.isfinite(g0)
gs = g0[sat & fin & (g0 > 0)]
ceiling = 0.9 * np.nanpercentile(gs, 99.0)
g0sat = S._find_group0_saturation_for(fn, do_not_use=True)
g0sat = np.asarray(g0sat, bool) if g0sat is not None else np.zeros(g0.shape, bool)
ny, nx = g0.shape
out = []
for r in rows:
    c = cat[int(r['row'])]
    x0g = int(round(float(c['xcentroid']) - float(c['x_fit']))); y0g = int(round(float(c['ycentroid']) - float(c['y_fit'])))
    xf, yf = float(c['x_fit']) + x0g, float(c['y_fit']) + y0g
    yy, xx = np.mgrid[max(0, int(yf) - 12):min(ny, int(yf) + 13), max(0, int(xf) - 12):min(nx, int(xf) + 13)]
    rr = np.hypot(xx - xf, yy - yf)
    gg = g0[yy, xx]; ss = sat[yy, xx]; g0s = g0sat[yy, xx]
    m3 = rr <= 3; m8 = (rr <= 8) & ss
    out.append(dict(istar=int(r['istar']), pixfile=pix, row=int(r['row']),
                    g0pk3=float(np.nanmax(gg[m3])) if np.isfinite(gg[m3]).any() else np.nan,
                    g0pk3_sat=bool(g0s[m3].any()), nsat8=int(m8.sum()), nsat8_g0sat=int((m8 & g0s).sum()),
                    g0_p90_sat8=float(np.nanpercentile(gg[m8], 90)) if m8.sum() > 3 else np.nan,
                    ceiling=float(ceiling)))
Table(rows=out).write(f'{H}/s4_{band}_{pix}.fits', overwrite=True)
print('done', len(out))
