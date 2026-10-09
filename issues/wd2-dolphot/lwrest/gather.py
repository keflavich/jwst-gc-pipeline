"""Per star-frame diagnostics for the main2kfpk LW no-row stars (read-only on pipeline).
usage: python gather.py BAND  (250M|277W|300M) -> sf_BAND.ecsv, star_BAND.ecsv"""
import glob, re, sys, warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from scipy import ndimage
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, Q)
import analyze as an
REPO = '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wd2main2kfpk'
sys.path.insert(0, REPO)
from jwst_gc_pipeline.photometry import cataloging as C
assert C.__file__.startswith(REPO), C.__file__
b = sys.argv[1].upper().lstrip('F')
FWHM = {'277W': 1.48, '250M': 1.33, '300M': 1.58}[b]
RAD = max(1.0, 0.5 * FWHM)
T = f'{Q}/tree_main2kfpk'
P = f'{T}/F{b}/pipeline'
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2kfpk')
rs = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)
ref = A.ref[b]
noro = np.asarray(Table.read(f'{Q}/f277w_gap/trace_mainfcbg_{b}.ecsv')['dolphot_idx'])
# merged m7 catalog status, same ZP recipe as score_lw.py
t = Table.read(f'{T}/catalogs/f{b.lower()}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
sk = t['skycoord']; fl = np.asarray(t['flux'], float)
ok = np.isfinite(sk.ra.deg) & (fl > 0)
t, sk, fl = t[ok], sk[ok], fl[ok]
mi = -2.5 * np.log10(fl)
rep = np.asarray(t['replaced_saturated'], bool)
mid = np.where(np.isfinite(ref) & (ref >= 18.6) & (ref < 21))[0]
j, d, _ = rs[mid].match_to_catalog_sky(sk)
sel = (d.arcsec < 0.05) & ~rep[j]
zp = np.median(ref[mid][sel] - mi[j][sel])
m = mi + zp
jj, dd, _ = rs[noro].match_to_catalog_sky(sk)
hit = dd.arcsec < 0.08
dm = np.where(hit, m[jj] - ref[noro], np.nan)
star = Table({'dolphot_idx': noro, 'ref_mag': ref[noro], 'sep_arcsec': dd.arcsec, 'hit': hit, 'dm': dm,
              'rep': np.where(hit, rep[jj], False),
              'nmatch': np.where(hit, np.asarray(t['nmatch'])[jj], 0),
              'flags': np.where(hit, np.asarray(t['flags'])[jj], 0)})
star.write(f'star_{b}.ecsv', overwrite=True)
frames = sorted(glob.glob(f'{P}/jw03523005001_*_nrc?long_align_o005_crf.fits'))
rows = []
for f in frames:
    mm = re.search(r'_(\d{5})_(nrc[ab]long)_', f)
    exp, det = int(mm.group(1)), mm.group(2)
    pf = glob.glob(f'{T}/F{b}/f{b.lower()}_{det}_visit001_vgroup*_exp{exp:05d}_resbgsub_m7_daophot_basic.fits')
    pft = Table.read(pf[0])
    px = np.column_stack([np.asarray(pft['x_fit'], float), np.asarray(pft['y_fit'], float)])
    pfl = np.asarray(pft['flux_fit'], float)
    good = np.isfinite(px).all(1)
    ptree = cKDTree(px[good]); pfl = pfl[good]; pxg = px[good]
    sci = fits.getdata(f, 'SCI').astype(float)
    dq = fits.getdata(f, 'DQ')
    err = fits.getdata(f, 'ERR').astype(float)
    bad = ~np.isfinite(err) | (err <= 0)
    w = WCS(fits.getheader(f, 'SCI'))
    acc = Table.read(f.replace('.fits', '_resbgsub_m7_satstar_catalog.fits'))
    rej = Table.read(f.replace('.fits', '_resbgsub_m7_satstar_rejected.fits'))
    sat = (dq & 2) != 0
    dnu = (dq & 1) != 0
    lab, n = ndimage.label(sat)
    sizes = np.bincount(lab.ravel())
    dist, (iyn, ixn) = ndimage.distance_transform_edt(~sat, return_indices=True)
    axy = np.column_stack([np.asarray(acc['xcentroid'], float), np.asarray(acc['ycentroid'], float)])
    axy = axy[np.isfinite(axy).all(1)]
    atree = cKDTree(axy)
    gx = np.empty((0, 2))
    rrx = np.empty((0, 2)); rreason = np.array([], str)
    if len(rej):
        rx = np.column_stack([np.asarray(rej['xcentroid'], float), np.asarray(rej['ycentroid'], float)])
        rr = np.asarray(rej['reject_reason']).astype(str)
        okr = np.isfinite(rx).all(1)
        rrx, rreason = rx[okr], rr[okr]
        gx = rrx[rreason == 'implied_peak_gate']
    rtree = cKDTree(rrx) if len(rrx) else None
    # hand-off position sets: current (area 50) and variants
    def handoff(area, excl=1.5):
        xy = C._unaccepted_sat_component_xy(dq, acc, FWHM, sci=sci, data_floor=0.0, label='x',
                                            accepted_excl_fwhm=excl, peak_min_area=area)
        xy = np.empty((0, 2)) if xy is None else xy
        return np.vstack([xy, gx]) if len(gx) else xy
    import io, contextlib
    H = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for key, (area, excl) in {'cur': (50, 1.5), 'a0': (0, 1.5), 'a20': (20, 1.5), 'a10': (10, 1.5), 'a1': (1, 1.5),
                                  'e1.0': (50, 1.0), 'e0.5': (50, 0.5), 'e0': (50, 0.0), 'a10e0.5': (10, 0.5), 'a1e0': (1, 0.0)}.items():
            H[key] = handoff(area, excl)
        # all peaks of big components, no satstar exclusion
        big = np.where(sizes[1:] >= 50)[0] + 1
        allpk = C._sat_component_peak_xy(sci, dq, lab, big, FWHM)
    trees = {k: (cKDTree(v) if len(v) else None) for k, v in H.items()}
    pk_tree = cKDTree(allpk)
    restore = {k: C._handoff_restore_pixels(dq, sci, bad, v, acc, FWHM) if len(v) else np.zeros(sat.shape, bool)
               for k, v in H.items() if k in ('cur', 'a0', 'a10', 'a1', 'e1.0', 'e0.5', 'a10e0.5', 'a1e0')}
    x, y = w.world_to_pixel(rs[noro])
    ix, iy = np.rint(x).astype(int), np.rint(y).astype(int)
    inside = (ix >= 4) & (ix < sat.shape[1] - 4) & (iy >= 4) & (iy < sat.shape[0] - 4)
    for k in np.where(inside)[0]:
        xk, yk = x[k], y[k]
        L = lab[iyn[iy[k], ix[k]], ixn[iy[k], ix[k]]]
        r = dict(dolphot_idx=int(noro[k]), frame=f'{det}_{exp}', d_sat=float(dist[iy[k], ix[k]]), on_sat=bool(sat[iy[k], ix[k]]),
                 comp=int(sizes[L]), pix_nan=bool(not np.isfinite(sci[iy[k], ix[k]])), pix_dnu=bool(dnu[iy[k], ix[k]]),
                 nan_n3=int((~np.isfinite(sci[iy[k]-1:iy[k]+2, ix[k]-1:ix[k]+2])).sum()),
                 sci=float(sci[iy[k], ix[k]]), d_acc=float(atree.query([xk, yk])[0]),
                 d_rej=float(rtree.query([xk, yk])[0]) if rtree else np.inf)
        if rtree:
            _, jr = rtree.query([xk, yk]); r['rej_reason'] = str(rreason[jr])
        else:
            r['rej_reason'] = ''
        for key, tr in trees.items():
            r['d_' + key] = float(tr.query([xk, yk])[0]) if tr else np.inf
        r['d_allpk'] = float(pk_tree.query([xk, yk])[0])
        for key, rm in restore.items():
            r['rest_' + key] = bool(rm[iy[k], ix[k]])
        dr, jr = ptree.query([xk, yk])
        r['d_row'] = float(dr)
        r['row_mag'] = float(-2.5 * np.log10(pfl[jr])) if pfl[jr] > 0 else np.nan
        r['n_row_2px'] = int(len(ptree.query_ball_point([xk, yk], 2.0)))
        rows.append(r)
    print(f, len(rows), flush=True)
sf = Table(rows)
sf.write(f'sf_{b}.ecsv', overwrite=True)
print(len(sf), len(star), 'zp', zp)
