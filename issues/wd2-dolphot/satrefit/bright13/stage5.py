"""Stage 5 (per frame; all mapped LW satstar rows): core-ring data/model ratios (uncapped model) and group-0 peak.  usage: stage5.py BAND PIXFILE"""
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
hdr = fits.getheader(fn)
grid, gf = C.load_grid(R5.TREE + '/psfs', hdr, True)
P = C.prep_frame(fn); FL = R5.load_frame(fn)
g0, sat = FL['g0'], FL['sat']
fin = np.isfinite(g0)
ceiling = 0.9 * np.nanpercentile(g0[sat & fin & (g0 > 0)], 99.0)
g0sat = S._find_group0_saturation_for(fn, do_not_use=True)
g0sat = np.asarray(g0sat, bool) if g0sat is not None else np.zeros(g0.shape, bool)
shape = g0.shape
out = []
edges = [0, 2, 3, 4, 5]
for r in rows:
    c = cat[int(r['row'])]
    win = C.window_of(c, shape)
    y0, y1, x0, x1 = win
    sl = (slice(y0, y1), slice(x0, x1))
    aH = float(r['a_H'])
    if not np.isfinite(aH):
        continue
    psf, m = C.render(grid, win, float(c['x_fit']), float(c['y_fit']), aH)
    d = P['data'][sl]
    yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0]
    rr = np.hypot(xx - float(c['x_fit']), yy - float(c['y_fit']))
    gg = g0[sl]; gs = g0sat[sl]
    o = dict(istar=int(r['istar']), pixfile=pix, row=int(r['row']), ceiling=float(ceiling), aH=aH,
             g0pk3=float(np.nanmax(gg[rr <= 3])) if np.isfinite(gg[rr <= 3]).any() else np.nan,
             g0pk_valid3=float(np.nanmax(gg[(rr <= 3) & ~gs & ~sat[sl]])) if ((rr <= 3) & ~gs & ~sat[sl]).any() else np.nan,
             nsat8=int(((rr <= 8) & sat[sl]).sum()), n_g0sat8=int(((rr <= 8) & gs).sum()),
             nrim3=int(((rr <= 3) & P['rim'][sl]).sum()), nvalid3=int(((rr <= 3) & (d > 0)).sum()))
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (rr >= lo) & (rr < hi) & (d > 0) & (m > 0)
        o[f'q{lo}_{hi}'] = float(np.median(d[s] / m[s])) if s.sum() >= 3 else np.nan
    # ratio at the pixel of max recovered data
    s = (rr <= 4) & (d > 0)
    if s.any():
        k = np.argmax(np.where(s, d, -np.inf)); iy, ix = np.unravel_index(k, d.shape)
        o['dmax'] = float(d[iy, ix]); o['q_dmax'] = float(d[iy, ix] / m[iy, ix]) if m[iy, ix] > 0 else np.nan
        o['g0_at_dmax'] = float(gg[iy, ix]); o['r_dmax'] = float(rr[iy, ix])
    out.append(o)
Table(rows=out).write(f'{H}/s5_{band}_{pix}.fits', overwrite=True)
print('done', len(out))
