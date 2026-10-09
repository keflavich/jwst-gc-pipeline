"""Stage 2 (per frame): own-star saturation/g0/rim stats + cutouts + model.  usage: stage2.py BAND PIXFILE"""
import sys, os, pickle
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import satrefit_core as C
import run_frames5 as R5
from satrefit_core import S, fits, ndimage, Table, SkyCoord
band, pix = sys.argv[1], sys.argv[2]
H = C.Q + '/satrefit/bright13'
fn = f'{R5.TREE}/F{band}/pipeline/{pix}.fits'
rows = Table.read(f'{H}/rows_{band}.fits'); stars = Table.read(f'{H}/star_{band}.fits')
rows = rows[rows['pixfile'] == pix]
cat = Table.read(fn.replace('.fits', '') + '_resbgsub_m7_satstar_catalog.fits')
hdr = fits.getheader(fn)
grid, gf = C.load_grid(R5.TREE + '/psfs', hdr, True)
P = C.prep_frame(fn); FL = R5.load_frame(fn)
cur0 = R5.pipeline_curve_N(fn, FL, 0)
ceiling = cur0['ceiling']; c0 = cur0['curve']
g0 = FL['g0']; sat = FL['sat']
g0sat = S._find_group0_saturation_for(fn, do_not_use=True)
g0sat = np.asarray(g0sat, bool) if g0sat is not None else np.zeros(g0.shape, bool)
rim = P['rim']; zfd = P['zf_deep'] if P['zf_deep'] is not None else P['saturated']
ww = P['ww']; shape = g0.shape
out = []; cut = {}
for r in rows:
    k = int(r['row']); c = cat[k]; lab = int(r['label'])
    y0, y1, x0, x1 = C.window_of(c, shape)
    sl = (slice(y0, y1), slice(x0, x1))
    own = (P['sources'][sl] == lab)
    ownsat = own & sat[sl]
    nsat = int(ownsat.sum())
    gs = g0sat[sl]
    st = stars[stars['istar'] == r['istar']][0]
    ex, ey = ww.world_to_pixel(SkyCoord(st['ra'], st['dec'], unit='deg'))
    g0c = g0[sl]
    # ring of rim pixels near this star (within 12 px of own sat region)
    edt_own = ndimage.distance_transform_edt(~ownsat) if nsat else np.full(own.shape, 99.)
    near = edt_own <= 12
    rimn = rim[sl] & near
    amp = min(float(r['a_H']), float(r['cap_H']) if np.isfinite(r['cap_H']) else np.inf)
    psf, model = C.render(grid, (y0, y1, x0, x1), float(c['x_fit']), float(c['y_fit']), amp)
    d = dict(istar=int(r['istar']), pixfile=pix, row=k, sat_area=int(c['sat_area']), nsat=nsat,
             n_g0sat=int((ownsat & gs).sum()), n_zfdeep=int((own & zfd[sl]).sum()),
             n_rim_near=int(rimn.sum()),
             n_rim_above_ceil=int((rimn & (g0c > ceiling)).sum()),
             n_rim_above_curve=int((rimn & (g0c > c0[0][-1])).sum()),
             n_core_above_ceil=int((ownsat & (g0c > ceiling)).sum()),
             g0max_core=float(np.nanmax(g0c[ownsat])) if nsat else np.nan,
             g0max_rim=float(np.nanmax(g0c[rimn])) if rimn.any() else np.nan,
             ceiling=float(ceiling), curve_top=float(c0[0][-1]), Rhdr=float(FL['Rhdr']),
             amp=amp, a_cat=float(r['a_cat']), a_raw=float(r['a_raw']), a_H=float(r['a_H']), cap_H=float(r['cap_H']), cap_base=float(r['cap_base']),
             nfit=int(r['nfit']), nrim_fit=int(r['nrim_fit']), nrw_h=int(r['nrw_h']),
             x_fit=float(c['x_fit']) , y_fit=float(c['y_fit']), x0=x0, y0=y0, ex=float(ex) - x0, ey=float(ey) - y0,
             qfit=float(c['qfit']), flags=int(c['flags']), model_peak=float(model.max()),
             data_rimmax=float(np.nanmax(P['data'][sl][rimn])) if rimn.any() else np.nan)
    # fitted position in cutout coords
    d['fx'] = d['x_fit']; d['fy'] = d['y_fit']
    out.append(d)
    cut[f'{r["istar"]}_{k}'] = dict(data=P['data'][sl].astype(np.float32), cal=FL['cal'][sl].astype(np.float32), sat=ownsat, sat_all=sat[sl], rim=rimn, model=model.astype(np.float32), g0=g0c.astype(np.float32), g0sat=gs)
Table(rows=out).write(f'{H}/s2_{band}_{pix}.fits', overwrite=True)
pickle.dump(cut, open(f'{H}/cut_{band}_{pix}.pkl', 'wb'))
print('done', band, pix, len(out))
