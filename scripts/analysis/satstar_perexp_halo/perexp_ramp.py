"""Is the x~250-550 halo deficit of saturated NRCBLONG stars in the RAW ramp,
and does it build up along it (#1013 follow-up)?

    python perexp_ramp.py measure <rowdir> <saturation reffile> <cal key>...
    python perexp_ramp.py analyze <out prefix> <rowdir>

``measure`` fetches each ``_uncal`` and its ``_cal`` (an S3 key such as
``jwst/public/jw10678/jw10678041001/jw10678041001_02101_00001_nrcblong_cal.fits``,
or a local ``_cal`` path) and writes one small npz of per-star rows, then
deletes the frames.  For every saturated core (cal DQ SATURATED region of
>= 250 px whose equivalent radius is <= 12 px, so that 15-80 px is halo and
not core) it records, from the RAW ramp (before superbias, linearity, dark
and flat):

* ``H_k``: the median over r = 15-80 px of the group difference
  D_k = group k - group k-1 (k = 1..3, mean of the integrations), minus its
  median over r = 150-250 px.  Only pixels below the saturation threshold in
  every group and not DO_NOT_USE are used, and the other saturated regions
  are masked (7-px dilation);
* ``A_g``: the number of core pixels at or above the reffile threshold in
  group g (g = 0..3), i.e. the saturated-core area as the ramp proceeds;
* the GWCS sky position of the core (``frame_wcs``).

``analyze`` links each star between the dithers of its observation by sky
position (no catalog matching), divides every quantity by the star's mean
over its dithers, and compares the dithers that put the star in the band
(detector x = 250-550 px) with those that do not.  A deficit made by
non-linearity or saturation grows with the group (accumulated charge); one
made by fewer photons reaching the halo is the same in every D_k.
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from jwst_gc_pipeline.frame_wcs import frame_wcs

BASE = 'https://stpubdata.s3.amazonaws.com/'
NMIN = int(os.environ.get("RAMP_NMIN", 250))   # 20 for narrowband programs (2221)
RMAX_CORE = 12.0
BAND = (250, 550)
NULL_BAND = (1300, 1600)       # same width, where #1013 found no deficit
HALO = (15, 80)
SKY = (150, 250)
LINK_ARCSEC = 0.5
# RAMP_ZEROFRAME=1: prepend the first-frame terms (halo of G0 - ZEROFRAME,
# core area at or above the threshold in ZEROFRAME) to H and A
ZEROFRAME = bool(int(os.environ.get('RAMP_ZEROFRAME', 0)))


def fetch(key, tmpdir):
    if os.path.exists(key):
        return key, False
    out = os.path.join(tmpdir, os.path.basename(key))
    subprocess.run(['curl', '-sS', '--retry', '4', '-o', out, BASE + key], check=True)
    return out, True


def exposure(fn_uncal, fn_cal, thr):
    with fits.open(fn_uncal) as h:
        raw = h['SCI'].data.astype(np.float32)          # (nint, ngroup, y, x)
        zf = (h['ZEROFRAME'].data.astype(np.float32)
              if ZEROFRAME and 'ZEROFRAME' in [e.name for e in h] else None)
    with fits.open(fn_cal) as h:
        dq = h['DQ'].data.astype(np.int64)
    ramp = raw.mean(0)
    satg = raw >= thr
    satany = satg.any((0, 1))
    D = np.diff(ramp, axis=0)
    if zf is not None:
        # group 0 averages frames 1..NFRAMES and ZEROFRAME is frame 1, so
        # G0 - ZF is a bias-free signal from the first frame interval
        D = np.concatenate([(ramp[0] - zf.mean(0))[None], D])
        satz = (zf >= thr).mean(0)
    lab, nl = ndimage.label((dq & 2) > 0, structure=np.ones((3, 3)))
    sizes = ndimage.sum(np.ones_like(lab), lab, np.arange(1, nl + 1))
    keep = np.flatnonzero(sizes >= NMIN) + 1
    if keep.size == 0:
        return [], zf is not None
    cy, cx = np.array(ndimage.center_of_mass(np.ones_like(lab), lab, keep)).T
    good = ~satany & ((dq & 1) == 0)
    neigh = ndimage.binary_dilation(satany, iterations=7)
    yy, xx = np.mgrid[0:2048, 0:2048]
    out = []
    for L, x0, y0, n in zip(keep, cx, cy, sizes[keep - 1]):
        if np.sqrt(n / np.pi) > RMAX_CORE or not (90 < x0 < 1957 and 90 < y0 < 1957):
            continue
        y1, y2 = max(int(y0) - SKY[1], 0), min(int(y0) + SKY[1] + 1, 2048)
        x1, x2 = max(int(x0) - SKY[1], 0), min(int(x0) + SKY[1] + 1, 2048)
        sl = (slice(y1, y2), slice(x1, x2))
        r = np.hypot(xx[sl] - x0, yy[sl] - y0)
        own = ndimage.binary_dilation(lab[sl] == L, iterations=7)
        g = good[sl] & ~(neigh[sl] & ~own)
        h_in = g & (r >= HALO[0]) & (r <= HALO[1])
        h_sk = g & (r >= SKY[0]) & (r <= SKY[1])
        if h_in.sum() < 2000 or h_sk.sum() < 2000:
            continue
        H, He = [], []
        for Dk in D:
            v, s = Dk[sl][h_in], Dk[sl][h_sk]
            H.append(float(np.median(v) - np.median(s)))
            He.append(float(1.2533 * 1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(v.size)))
        core = (lab[sl] == L) | ((r < 2 * RMAX_CORE) & satany[sl])
        A = [float((satg[:, k, y1:y2, x1:x2] & core).sum() / satg.shape[0]) for k in range(ramp.shape[0])]
        if zf is not None:
            A = [float((satz[sl] * core).sum())] + A
        out.append((x0, y0, H, He, A))
    return out, zf is not None


def measure(rowdir, reffile, keys):
    os.makedirs(rowdir, exist_ok=True)
    with fits.open(reffile) as h:
        thr = h['SCI'].data.astype(np.float32)
    thr[~np.isfinite(thr) | (thr <= 0)] = np.inf
    for key in keys:
        root = os.path.basename(key).replace('_cal.fits', '')
        out = os.path.join(rowdir, root + '.npz')
        if os.path.exists(out):
            continue
        fc, tc = fetch(key, rowdir)
        fu, tu = fetch(key.replace('_cal.fits', '_uncal.fits'), rowdir)
        try:
            stars, used_zf = exposure(fu, fc, thr)
            w = frame_wcs(fc)
            x = np.array([s[0] for s in stars]); y = np.array([s[1] for s in stars])
            sky = w.pixel_to_world(x, y) if len(stars) else None
            np.savez(out, x=x, y=y,
                     ra=sky.ra.deg if sky is not None else x, dec=sky.dec.deg if sky is not None else y,
                     H=np.array([s[2] for s in stars]), He=np.array([s[3] for s in stars]),
                     A=np.array([s[4] for s in stars]), zf=used_zf)
            print(root, len(stars), flush=True)
        finally:
            for f, t in ((fc, tc), (fu, tu)):
                if t:
                    os.remove(f)


def analyze(outp, rowdir):
    rows = []                                   # (obs, exp, x, H[3], He[3], A[4], ra, dec)
    zf_flags = set()
    for fn in sorted(glob.glob(os.path.join(rowdir, '*.npz'))):
        d = np.load(fn)
        if d['x'].size == 0:
            continue
        b = os.path.basename(fn)
        # one star-visit = one activity (program, observation, visit, activity):
        # an observation can hold several filters
        obs, exp = b[:19], int(b[20:25])
        for i in range(d['x'].size):
            rows.append(dict(obs=obs, exp=exp, x=d['x'][i], H=d['H'][i], He=d['He'][i], A=d['A'][i],
                             ra=d['ra'][i], dec=d['dec'][i]))
        zf_flags.add(bool(d['zf']) if 'zf' in d.files else False)
    if len(zf_flags) > 1:
        # the ZEROFRAME column shifts every H/A index by one
        raise ValueError(f'{rowdir} mixes rows with and without the ZEROFRAME terms')
    has_zf = zf_flags.pop() if zf_flags else False
    # link within each star-visit: a row joins the first cluster within 0.5"
    # (small-angle offsets about the cluster's first position)
    clusters = []
    for obs in sorted({r['obs'] for r in rows}):
        cl, cra, cdec = [], [], []
        for r in (r for r in rows if r['obs'] == obs):
            j = None
            if cl:
                dra = (np.array(cra) - r['ra']) * np.cos(np.deg2rad(r['dec']))
                sep = 3600 * np.hypot(dra, np.array(cdec) - r['dec'])
                for i in np.flatnonzero(sep < LINK_ARCSEC):
                    if r['exp'] not in cl[i]['exps']:
                        j = i
                        break
            if j is None:
                cl.append(dict(rows=[r], exps={r['exp']})); cra.append(r['ra']); cdec.append(r['dec'])
            else:
                cl[j]['rows'].append(r); cl[j]['exps'].add(r['exp'])
        clusters += cl
    # per star: median over its exposures 1-2 / median over 3-6.  With the
    # 10678 pattern (+385 px per step in x) a star is in the band almost only
    # in exposures 1-2, so the CONTROL is the same ratio for the stars whose
    # exposures 1-2 are outside the band: a column effect appears only in the
    # first group, an exposure-order effect in both.
    groups = dict(band=[], control=[])
    for k in clusters:
        e = np.array([r['exp'] for r in k['rows']])
        first, later = np.isin(e, (1, 2)), np.isin(e, (3, 4, 5, 6))
        if first.sum() < 1 or later.sum() < 2:
            continue
        x = np.array([r['x'] for r in k['rows']])
        H = np.array([r['H'] for r in k['rows']]); A = np.array([r['A'] for r in k['rows']])
        Q = np.column_stack([H, H[:, int(has_zf):].sum(1), A])
        if np.any(np.median(Q[later], 0) <= 0):
            continue
        q = np.median(Q[first], 0) / np.median(Q[later], 0)
        # halo columns are kept only where the halo is positive in every exposure
        if np.any(H <= 0):
            q[:H.shape[1] + 1] = np.nan
        xin = (x[first] > BAND[0]) & (x[first] < BAND[1])
        if xin.all() and not ((x[later] > BAND[0]) & (x[later] < BAND[1])).any():
            groups['band'].append(q)
        elif not xin.any() and (x[first] > BAND[1] + 50).all():
            groups['control'].append(q)
    # direct split: a star's exposures with x in a band / those with x
    # outside it (for a dither pattern that moves stars through the band at
    # every exposure index), for BAND and for the null band
    for name, (lo, hi) in (('split_band', BAND), ('split_null', NULL_BAND)):
        groups[name] = []
        for k in clusters:
            x = np.array([r['x'] for r in k['rows']])
            inb = (x > lo) & (x < hi)
            out = (x < lo - 50) | (x > hi + 50)
            if inb.sum() < 1 or out.sum() < 2:
                continue
            H = np.array([r['H'] for r in k['rows']]); A = np.array([r['A'] for r in k['rows']])
            Q = np.column_stack([H, H[:, int(has_zf):].sum(1), A])
            if np.any(np.median(Q[out], 0) <= 0):
                continue
            q = np.median(Q[inb], 0) / np.median(Q[out], 0)
            if np.any(H <= 0):
                q[:H.shape[1] + 1] = np.nan
            groups[name].append(q)
    nD = len(clusters[0]['rows'][0]['H'])
    labels = [f'halo D{k + 1 - int(has_zf)}' for k in range(nD)] + ['halo sum'] + \
        (['core area ZF'] if has_zf else []) + \
        [f'core area g{g}' for g in range(len(clusters[0]['rows'][0]['A']) - int(has_zf))]
    rng = np.random.default_rng(0)

    def stat(v):
        v = v[np.isfinite(v)]
        if v.size < 5:
            return [np.nan, np.nan, int(v.size)]
        bs = [np.median(rng.choice(v, v.size)) for _ in range(2000)]
        return [float(np.median(v)), float(np.std(bs)), int(v.size)]

    res = dict(band=BAND, halo_r=HALO, quantities=labels,
               note='median over stars of (median over exposures 1-2) / (median over exposures 3-6); '
                    '[value, bootstrap error, n stars]')
    for g, rowsg in groups.items():
        Qg = np.array(rowsg).reshape(-1, len(labels))
        res[g] = {lab: stat(Qg[:, i]) for i, lab in enumerate(labels)}
    json.dump(res, open(outp + '.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xx = np.arange(len(labels))
    sets = (('band', f'exposures 1-2 / 3-6, 1-2 in x={BAND[0]}-{BAND[1]}'),
            ('control', f'exposures 1-2 / 3-6, 1-2 at x>{BAND[1] + 50}'),
            ('split_band', f'in x={BAND[0]}-{BAND[1]} / outside'),
            ('split_null', f'in x={NULL_BAND[0]}-{NULL_BAND[1]} / outside (null)'))
    for j, (g, lab) in enumerate(sets):
        v = np.array([res[g][q] for q in labels])
        if not np.isfinite(v[:, 0]).any():
            continue
        ax.errorbar(xx + 0.09 * (j - 1.5), v[:, 0], v[:, 1], fmt='osD^'[j], color=f'C{j}',
                    label=f'{lab} ({int(np.nanmax(v[:, 2]))} stars)')
    ax.axhline(1, color='k', lw=0.5)
    ax.set_xticks(xx); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('ratio of per-star medians (raw ramp)')
    ax.legend(fontsize=8)
    ax.set_title(f'{os.environ.get("RAMP_TITLE", "NRCBLONG F480M")} saturated stars, raw ramp (halo r = {HALO[0]}-{HALO[1]} px)', fontsize=9)
    fig.tight_layout(); fig.savefig(outp + '.png', dpi=int(os.environ.get('FIG_DPI', 130)))


if __name__ == '__main__':
    if sys.argv[1] == 'measure':
        measure(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        analyze(sys.argv[2], sys.argv[3])
