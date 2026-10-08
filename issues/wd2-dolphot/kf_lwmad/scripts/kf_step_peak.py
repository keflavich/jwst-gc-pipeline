"""Peak-pixel comparison: per matched satstar, max over its saturated component of the data the fit sees (off rewrite vs on keep),
and the peak pixel's number of usable groups.  usage: python kf_step_peak.py BAND  (writes kf_step_peak_<BAND>.fits)"""
import os, sys, glob
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
os.environ['NIRCAM_SATSTAR_RECOVERED_CAP'] = '1'
import jwst_gc_pipeline.reduction.saturated_star_finding as S
from stdatamodels.jwst.datamodels import dqflags
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
band = sys.argv[1]
SAT, DNU = dqflags.pixel['SATURATED'], dqflags.pixel['DO_NOT_USE']
GS, GD = dqflags.group['SATURATED'], dqflags.group['DO_NOT_USE']
out = []
for rampfn in sorted(glob.glob(f'/orange/adamginsburg/jwst/wd2/{band}/pipeline/*nrcblong_ramp.fits')):
    stem = os.path.basename(rampfn).replace('_ramp.fits', '')
    crf = rampfn.replace('_ramp.fits', '_align_o005_crf.fits')
    with fits.open(crf) as h:
        data = h['SCI'].data.astype(float); dq = h['DQ'].data.astype(np.int64); hdr = h[0].header
        ph = h['SCI'].header.get('PHOTMJSR', hdr.get('PHOTMJSR'))
    with fits.open(rampfn) as r:
        g0 = np.asarray(r['SCI'].data[0, 0], dtype=float); gdq = np.asarray(r['GROUPDQ'].data[0]); ff = np.asarray(r['ZEROFRAME'].data[0], dtype=float)
    g0sat = S._find_group0_saturation_for(crf, do_not_use=True)
    Rh = S.zeroframe_header_R(hdr, ph)
    res = {}
    for name, kf in (('off', '0'), ('on', '1')):
        os.environ['SATSTAR_ZF_KEEP_FINITE'] = kf
        res[name] = S.zeroframe_recover_saturated(data, dq, g0, group0_saturated=g0sat, first_frame=ff, R_header=Rh)[0]
    nuse = (((gdq & (GS | GD)) == 0)).sum(axis=0)
    sat = (dq & SAT) != 0
    lab, _ = ndi.label(sat, structure=np.ones((3, 3)))
    cats = {}
    for arm in ('main2', 'main2kf'):
        t = Table.read(f'{Q}/tree_{arm}/{band}/pipeline/{stem}_align_o005_crf_resbgsub_m7_satstar_catalog.fits')
        cats[arm] = (np.c_[np.ma.filled(t['xcentroid'], np.nan), np.ma.filled(t['ycentroid'], np.nan)].astype(float),
                     np.ma.filled(t['flux_fit'], np.nan).astype(float), np.ma.filled(t['flux_fit_precap'], np.nan).astype(float))
    d, i = cKDTree(cats['main2kf'][0]).query(cats['main2'][0], distance_upper_bound=0.5)
    for j in np.where(np.isfinite(d))[0]:
        x, y = cats['main2'][0][j]
        x0, y0 = int(round(x)), int(round(y))
        ys, xs = slice(max(y0 - 6, 0), min(y0 + 7, 2048)), slice(max(x0 - 6, 0), min(x0 + 7, 2048))
        yy, xx = np.mgrid[ys, xs]
        sub = lab[ys, xs]
        labs = np.unique(sub[(sub > 0) & ((yy - y) ** 2 + (xx - x) ** 2 <= 9)])
        comp = np.isin(sub, labs) if len(labs) else np.zeros(sub.shape, bool)
        if not comp.any():
            continue
        a = np.where(comp, res['off'][ys, xs], -np.inf); b = np.where(comp, res['on'][ys, xs], -np.inf)
        a = np.where(np.isfinite(a), a, -np.inf); b = np.where(np.isfinite(b), b, -np.inf)
        ia, ib = np.unravel_index(np.argmax(a), a.shape), np.unravel_index(np.argmax(b), b.shape)
        out.append(dict(frame=stem, x=x, y=y, f_off=cats['main2'][1][j], f_on=cats['main2kf'][1][i[j]],
                        pre_off=cats['main2'][2][j], pre_on=cats['main2kf'][2][i[j]],
                        peak_off=a[ia], peak_on=b[ib], nuse_pk_off=nuse[ys, xs][ia], nuse_pk_on=nuse[ys, xs][ib], ncomp=int(comp.sum())))
Table(out).write(f'{Q}/kf_lwmad/kf_step_peak_{band}.fits', overwrite=False)
