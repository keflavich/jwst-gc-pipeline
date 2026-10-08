"""Per-frame diagnosis of the KEEP_FINITE step.
usage: PYTHONPATH=<kf tree> python kf_step_diag.py BAND [BAND ...]
For each nrcblong frame: replay zeroframe_recover_saturated with KEEP_FINITE off and on,
match m7 satstar rows between arms, and tabulate per-star kept-pixel statistics.
Writes kf_step_diag_<BAND>.fits (one row per matched star)."""
import os
import sys
import glob
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

os.environ['NIRCAM_SATSTAR_RECOVERED_CAP'] = '1'
os.environ.pop('SATSTAR_ZF_KEEP_FINITE', None)
import jwst_gc_pipeline.reduction.saturated_star_finding as S  # noqa: E402
from stdatamodels.jwst.datamodels import dqflags  # noqa: E402

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
OUT = f'{Q}/kf_lwmad'
SAT = dqflags.pixel['SATURATED']
DNU = dqflags.pixel['DO_NOT_USE']
GSAT = dqflags.group['SATURATED']
GDNU = dqflags.group['DO_NOT_USE']


def cat(arm, band, stem):
    fn = f'{Q}/tree_{arm}/{band}/pipeline/{stem}_align_o005_crf_resbgsub_m7_satstar_catalog.fits'
    t = Table.read(fn)
    xy = np.c_[np.ma.filled(t['xcentroid'], np.nan), np.ma.filled(t['ycentroid'], np.nan)].astype(float)
    return t, xy, np.ma.filled(t['flux_fit'], np.nan).astype(float)


def run_frame(band, rampfn):
    stem = os.path.basename(rampfn).replace('_ramp.fits', '')
    crf = rampfn.replace('_ramp.fits', '_align_o005_crf.fits')
    with fits.open(crf) as h:
        data = h['SCI'].data.astype(float)
        dq = h['DQ'].data.astype(np.int64)
        hdr = h[0].header
        photmjsr = h['SCI'].header.get('PHOTMJSR', hdr.get('PHOTMJSR'))
    with fits.open(rampfn) as r:
        g0 = np.asarray(r['SCI'].data[0, 0], dtype=float)
        gdq = np.asarray(r['GROUPDQ'].data[0])  # (ngroup, y, x)
        ff = np.asarray(r['ZEROFRAME'].data[0], dtype=float)
    g0sat = S._find_group0_saturation_for(crf, do_not_use=True)
    Rh = S.zeroframe_header_R(hdr, photmjsr)
    res = {}
    for name, kf in (('off', '0'), ('on', '1')):
        os.environ['SATSTAR_ZF_KEEP_FINITE'] = kf
        res[name] = S.zeroframe_recover_saturated(data, dq, g0, group0_saturated=g0sat,
                                                  first_frame=ff, R_header=Rh)
    rec_off, rim_off, deep_off, R_off = res['off']
    rec_on, rim_on, deep_on, R_on = res['on']
    sat = (dq & SAT) != 0
    keep = sat & np.isfinite(data) & (data != 0) & ((dq & DNU) == 0)
    # usable groups seen by ramp_fit: groups with neither SATURATED nor DO_NOT_USE
    bad = (gdq & (GSAT | GDNU)) != 0
    nuse = (~bad).sum(axis=0)
    # first group flagged saturated
    issat = (gdq & GSAT) != 0
    firstsat = np.where(issat.any(axis=0), issat.argmax(axis=0), gdq.shape[0])
    g0flag = bad[0]
    lab, nl = ndi.label(sat, structure=np.ones((3, 3)))
    print(stem, 'R_off', R_off, 'R_on', R_on, 'Rheader', Rh, 'nsat', sat.sum(), 'nkeep', keep.sum(),
          'rim_off', rim_off.sum(), 'rim_on', rim_on.sum(), flush=True)
    to, xyo, fo = cat('main2', band, stem)
    tn, xyn, fn = cat('main2kf', band, stem)
    d, i = cKDTree(xyn).query(xyo, distance_upper_bound=0.5)
    ok = np.isfinite(d)
    rows = []
    for j in np.where(ok)[0]:
        x, y = xyo[j]
        x0, y0 = int(round(x)), int(round(y))
        ys, xs = slice(max(y0 - 6, 0), min(y0 + 7, 2048)), slice(max(x0 - 6, 0), min(x0 + 7, 2048))
        # saturated component(s) within 3 px of the centroid
        sub = lab[ys, xs]
        yy, xx = np.mgrid[ys, xs]
        near = (sub > 0) & ((yy - y) ** 2 + (xx - x) ** 2 <= 9)
        labs = np.unique(sub[near])
        if len(labs) == 0:
            comp = np.zeros_like(sub, bool)
        else:
            comp = np.isin(sub, labs)
        k = keep[ys, xs] & comp
        nk = int(k.sum())
        nsat_c = int(comp.sum())
        # off-arm rewrite at kept pixels (where off rewrote them) vs crf
        kr = k & rim_off[ys, xs]
        crf_k = data[ys, xs][k]
        rew = rec_off[ys, xs][kr]
        crf_kr = data[ys, xs][kr]
        fs = firstsat[ys, xs][k]
        nu = nuse[ys, xs][k]
        g0f = g0flag[ys, xs][k]
        ratio_pix = crf_kr / rew if kr.any() else np.array([])
        rows.append(dict(
            x=x, y=y, f_off=fo[j], f_on=fn[i[j]], sat_area=float(to['sat_area'][j]),
            nsat=nsat_c, nkeep=nk, nrew=int(kr.sum()),
            med_firstsat=float(np.median(fs)) if nk else np.nan,
            med_nuse=float(np.median(nu)) if nk else np.nan,
            frac_le1=float((nu <= 1).mean()) if nk else np.nan,
            frac_g0flag=float(g0f.mean()) if nk else np.nan,
            sum_crf_keep=float(crf_k.sum()) if nk else 0.0,
            sum_rew_keep=float(rew.sum()) if kr.any() else 0.0,
            sum_crf_rew=float(crf_kr.sum()) if kr.any() else 0.0,
            med_ratio=float(np.median(ratio_pix)) if kr.any() else np.nan,
            peak_crf=float(np.nanmax(data[ys, xs][comp])) if comp.any() else np.nan))
    t = Table(rows)
    t['frame'] = stem
    t['R_off'] = R_off
    t['R_on'] = R_on
    # also dump pixel-level table for kept pixels that off rewrote
    m = keep & rim_off
    pix = Table(dict(frame=[stem] * int(m.sum()), crf=data[m], rew=rec_off[m], g0=g0[m], ff=ff[m],
                     nuse=nuse[m], firstsat=firstsat[m], g0flag=g0flag[m]))
    return t, pix


if __name__ == '__main__':
    for band in sys.argv[1:]:
        ramps = sorted(glob.glob(f'/orange/adamginsburg/jwst/wd2/{band}/pipeline/*nrcblong_ramp.fits'))
        T, P = [], []
        for rf in ramps:
            t, p = run_frame(band, rf)
            T.append(t)
            P.append(p)
        from astropy.table import vstack
        vstack(T).write(f'{OUT}/kf_step_diag_{band}_stars.fits', overwrite=False)
        vstack(P).write(f'{OUT}/kf_step_diag_{band}_pix.fits', overwrite=False)
