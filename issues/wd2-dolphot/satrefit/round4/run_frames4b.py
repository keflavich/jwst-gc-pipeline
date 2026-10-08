"""Round 4 addendum.  usage: python run_frames4b.py BAND DET EXP [MAXROWS]
(writes out4/<band>_<stem>_satrefit4b.fits and out4/pix_<band>_<stem>.npz)

1. norim: the amplitude fit excludes every ZEROFRAME-rewritten (rim) pixel (mask = deep core dilation + rim); base and bgfree.
2. Per-pixel records for the data/model ratio: for each satstar row, pixels within 30 px of the fitted position with psf_unit >= 1e-3 peak:
   u = (neighbour-subtracted data - annulus bkg) / psf_unit  [ratio to a model F*PSF is u/F], g0 (ramp SCI[0,0]), r, category
   0 = rim pixel in the fit (value after rewrite), 1 = never-rewritten crf pixel in the fit, 2 = rim pixel masked from the fit; peak flag = model peak pixel.
"""
import copy
import os
import sys
import numpy as np

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import satrefit_core as C
from satrefit_core import S, fits, Table, SkyCoord

band, det, e = sys.argv[1], sys.argv[2], int(sys.argv[3])
max_rows = int(sys.argv[4]) if len(sys.argv) > 4 else None
vgroup = {'150W': '10101', '200W': '12101'}[band]
tree = C.Q + '/tree_main2'
stem = f'jw03523005001_{vgroup}_{e:05d}_{det}_align_o005_crf'
fn = f'{tree}/F{band}/pipeline/{stem}.fits'
tag = '_test' if max_rows else ''
outfn = f'{C.Q}/satrefit/out4/{band}_{stem}_satrefit4b{tag}.fits'
pixfn = f'{C.Q}/satrefit/out4/pix_{band}_{stem}{tag}.npz'
hdr = fits.getheader(fn)
grid, gf = C.load_grid(tree + '/psfs', hdr, False)
base = fn.replace('.fits', '')
cat = Table.read(base + '_resbgsub_m7_satstar_catalog.fits')
P = C.prep_frame(fn)
with fits.open(S._find_ramp_for(fn), memmap=True) as r:
    g0 = np.array(r['SCI'].data[0, 0], float)
shape = P['data'].shape
n = len(cat) if not max_rows else min(len(cat), max_rows)
ww = P['ww']
labels = np.full(n, -1, int)
for i in range(n):
    r = cat[i]
    if int(r['sat_area']) <= 0 or not np.isfinite(float(r['sat_com_ra'])):
        continue
    cx, cy = ww.world_to_pixel(SkyCoord(float(r['sat_com_ra']), float(r['sat_com_dec']), unit='deg'))
    d = np.hypot(P['comcen'][:, 0] - float(cx), P['comcen'][:, 1] - float(cy))
    j = int(np.nanargmin(d))
    if d[j] < 1.0:
        labels[i] = j + 1
wins, models = [], []
for i in range(n):
    r = cat[i]
    win = C.window_of(r, shape)
    wins.append(win)
    models.append(C.render(grid, win, float(r['x_fit']), float(r['y_fit']), float(r['flux_fit_raw']))[1])
seq = P['data'].copy()
rows = []
pix = {k: [] for k in ('row', 'u', 'g0', 'r', 'cat', 'peak')}
for i in range(n):
    r = cat[i]
    y0, y1, x0, x1 = wins[i]
    xfit, yfit = float(r['x_fit']), float(r['y_fit'])
    out = dict(idx=i, label=labels[i], a_cat=float(r['flux_fit_precap']), a_raw=float(r['flux_fit_raw']), flux_fit=float(r['flux_fit']))
    if labels[i] > 0:
        cut_seq = seq[y0:y1, x0:x1].copy()
        st0 = C.make_setup(P, wins[i], r, cut_seq, labels[i])
        psf = S.psf_in_cutout_coords(grid, x0, y0)
        yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0]
        psf_unit = np.maximum(psf.evaluate(xx, yy, 1.0, xfit, yfit), 0.0)
        reg = C.fit_region(st0, psf_unit.shape)
        rimcut = P['rim'][y0:y1, x0:x1]
        fitpx = reg & ~st0.mask & np.isfinite(st0.cut)
        stn = copy.copy(st0)
        stn.mask = st0.mask | rimcut
        out['nfit'] = int(fitpx.sum())
        out['nfit_norim'] = int((reg & ~stn.mask & np.isfinite(stn.cut)).sum())
        out['nrim_fit'] = int((fitpx & rimcut).sum())
        out['sat_area'] = int(r['sat_area'])
        out['a_base'] = C.solve(st0, psf_unit, xfit, yfit, {}, kpois=P['kpois'])
        out['a_bgfree'] = C.solve(st0, psf_unit, xfit, yfit, {'bg': 'free'}, kpois=P['kpois'])
        out['a_norim'] = C.solve(stn, psf_unit, xfit, yfit, {}, kpois=P['kpois'])
        out['a_norim+bgfree'] = C.solve(stn, psf_unit, xfit, yfit, {'bg': 'free'}, kpois=P['kpois'])
        rr = np.hypot(xx - xfit, yy - yfit)
        pkv = psf_unit.max()
        ipk = np.unravel_index(np.argmax(psf_unit), psf_unit.shape)
        sel = reg & (rr <= 30) & (psf_unit >= 1e-3 * pkv) & np.isfinite(st0.cut) & (st0.cut != 0)
        cat_ = np.where(rimcut & ~st0.mask, 0, np.where(~rimcut & ~st0.mask, 1, np.where(rimcut, 2, -1)))
        sel &= cat_ >= 0
        pk_flag = np.zeros(psf_unit.shape, bool)
        pk_flag[ipk] = True
        w = sel
        pix['row'].append(np.full(int(w.sum()), i, np.int32))
        pix['u'].append(((st0.cut - st0.bkg)[w] / psf_unit[w]).astype(np.float32))
        pix['g0'].append(g0[y0:y1, x0:x1][w].astype(np.float32))
        pix['r'].append(rr[w].astype(np.float32))
        pix['cat'].append(cat_[w].astype(np.int8))
        pix['peak'].append(pk_flag[w])
    rows.append(out)
    seq[y0:y1, x0:x1] -= models[i]
    if i % 100 == 0:
        C.log('row', i, n)
tab = Table(rows=rows)
sk = cat['skycoord_fit'][:n]
sk = sk if isinstance(sk, SkyCoord) else SkyCoord(sk)
tab['ra'] = sk.ra.deg
tab['dec'] = sk.dec.deg
tab.write(outfn, overwrite=True)
np.savez_compressed(pixfn, **{k: np.concatenate(v) for k, v in pix.items()})
C.log('wrote', outfn, pixfn)
