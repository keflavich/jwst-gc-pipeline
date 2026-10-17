"""Before/after comparison of the m2->m8 production rerun for one (field, band).

Reads the pre-rerun snapshot in ../prod_before/<field>/<band>/ (cutouts of the
data and m7 residual mosaics at 300 sampled starlike_left positions plus 100
random positions, and the pre-rerun m7 vetted catalog) and measures the same
positions in the rerun's in-place products:

* PSF-matched S/N left in the m7 residual (phase_loss.stamp_metrics), before
  (on the saved stamps) and after (on the new mosaic);
* whether the new m7 vetted catalog has a source within R_MATCH mosaic pixels.

Writes <out>.json (counts, S/N quantiles) and <out>.png: a histogram of the
residual S/N at the sampled positions (before/after, with random-position
controls), and a stamp gallery -- per row: data | before m7 residual + before
m7 vetted | after m7 residual + after m7 vetted.  Rows: NRAND_ROWS random rows
of the sampled starlike_left positions, then NWORST_ROWS rows with the largest
S/N still left after the rerun.  Red circle: the position; orange x: that
column's m7 vetted sources.  All panels of a row share one linear stretch
width set by the data stamp; each panel is centred on its own median.

usage: python compare.py <field> <FILT> <out-prefix>
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(S, '..', 'phaseloss'))
from phase_loss import FWHM_PIX, stamp_metrics, _image_hdu  # noqa: E402

HALF = 20
R_MATCH = 1.0
NRAND_ROWS, NWORST_ROWS, NNEG_ROWS = 6, 3, 4


def cat_xy(cat_path, wcs):
    t = Table.read(cat_path)
    sc = SkyCoord(t['skycoord'] if 'skycoord' in t.colnames else t['skycoord_avg'])
    ra, dec = sc.ra.deg, sc.dec.deg
    ok = np.isfinite(ra) & np.isfinite(dec)
    x, y = wcs.world_to_pixel_values(ra[ok], dec[ok])
    return np.c_[x, y]


def stamps_from(hdu, x, y):
    ny, nx = hdu.data.shape
    out = np.full((len(x), 2 * HALF + 1, 2 * HALF + 1), np.nan, np.float32)
    for k, (xi, yi) in enumerate(zip(np.round(x).astype(int), np.round(y).astype(int))):
        y0, y1 = max(yi - HALF, 0), min(yi + HALF + 1, ny)
        x0, x1 = max(xi - HALF, 0), min(xi + HALF + 1, nx)
        if y1 <= y0 or x1 <= x0:
            continue
        out[k, y0 - (yi - HALF):y1 - (yi - HALF), x0 - (xi - HALF):x1 - (xi - HALF)] = hdu.data[y0:y1, x0:x1]
    return out


def stamp_snr(stamps, x, y, fwhm):
    """stamp_metrics S/N at each stamp's own sub-pixel centre."""
    snr = np.full(len(stamps), np.nan)
    for k in range(len(stamps)):
        m = SimpleNamespace(data=stamps[k], shape=stamps[k].shape)
        cx = HALF + (x[k] - np.round(x[k]))
        cy = HALF + (y[k] - np.round(y[k]))
        snr[k] = stamp_metrics(m, np.array([cx]), np.array([cy]), fwhm)['snr'][0]
    return snr


def q(a):
    a = a[np.isfinite(a)]
    return {p: float(np.percentile(a, p)) for p in (10, 50, 90, 99)} if len(a) else {}


def main(field, filt, outp):
    f = filt.lower()
    snapdir = os.path.join(S, '..', 'prod_before', field, f)
    snap = json.load(open(os.path.join(snapdir, 'snapshot.json')))
    npz = np.load(os.path.join(snapdir, 'cutouts.npz'))
    ra, dec, is_left = npz['ra'], npz['dec'], npz['is_left']
    bx, by = npz['x'], npz['y']
    fwhm = FWHM_PIX[filt]

    after = {k: snap['files'][k]['path'] for k in ('data', 'resbgsub_m7_resid', 'resbgsub_m7_vetted')}
    # AFTER_REMAP=old_root=new_root reads the rerun products from another tree
    # (brick: the snapshot came from the /orange copy, the rerun wrote /blue)
    if os.environ.get('AFTER_REMAP'):
        old_root, new_root = os.environ['AFTER_REMAP'].split('=')
        after = {k: v.replace(old_root, new_root, 1) for k, v in after.items()}
        print(f'after products read from {new_root}', flush=True)
    # the m7 products must be new; the data mosaic is new only on fields the
    # rerun re-reduced (retie loop), and stays the same on catalog-only reruns
    for k in ('resbgsub_m7_resid', 'resbgsub_m7_vetted'):
        if os.path.getmtime(after[k]) <= snap['files'][k]['mtime']:
            raise RuntimeError(f'{after[k]} not rewritten since the snapshot -- rerun not finished?')
    data_rereduced = os.path.getmtime(after['data']) > snap['files']['data']['mtime']
    print(f'data mosaic re-reduced since snapshot: {data_rereduced}', flush=True)

    with fits.open(after['resbgsub_m7_resid'], memmap=True) as hr, \
            fits.open(after['data'], memmap=True) as hd:
        rh, dh = _image_hdu(hr), _image_hdu(hd)
        wcs = WCS(rh.header)
        ax, ay = wcs.world_to_pixel_values(ra, dec)
        grid_shift = float(np.nanmax(np.hypot(ax - bx, ay - by)))
        a_snr = stamp_metrics(rh, ax, ay, fwhm)['snr']
        a_res = stamps_from(rh, ax, ay)
        a_dat = stamps_from(dh, ax, ay)

    b_snr = stamp_snr(npz['resbgsub_m7_resid'], bx, by, fwhm)
    b_xy = cat_xy(os.path.join(snapdir, os.path.basename(snap['files']['resbgsub_m7_vetted']['path'])), wcs)
    a_xy = cat_xy(after['resbgsub_m7_vetted'], wcs)
    bt, at = cKDTree(b_xy), cKDTree(a_xy)
    b_d, _ = bt.query(np.c_[ax, ay])
    a_d, _ = at.query(np.c_[ax, ay])
    b_rec, a_rec = b_d <= R_MATCH, a_d <= R_MATCH

    L, C = is_left, ~is_left
    summ = dict(field=field, filt=filt, n_sampled_left=int(L.sum()), n_random=int(C.sum()),
                n_left_total_before=snap['n_left_total'],
                grid_shift_px_max=grid_shift, data_rereduced=bool(data_rereduced),
                after_paths=after,
                before_in_m7_vetted=int(b_rec[L].sum()), after_in_m7_vetted=int(a_rec[L].sum()),
                resid_snr_left_before=q(b_snr[L]), resid_snr_left_after=q(a_snr[L]),
                resid_snr_random_before=q(b_snr[C]), resid_snr_random_after=q(a_snr[C]),
                left_after_snr_gt5=int(np.sum(a_snr[L] > 5)), left_before_snr_gt5=int(np.sum(b_snr[L] > 5)),
                left_after_snr_lt_m5=int(np.sum(a_snr[L] < -5)), left_before_snr_lt_m5=int(np.sum(b_snr[L] < -5)),
                random_after_snr_lt_m5=int(np.sum(a_snr[C] < -5)), random_before_snr_lt_m5=int(np.sum(b_snr[C] < -5)),
                left_after_snr_lt_m5_in_cat=int(np.sum((a_snr < -5) & a_rec & L)))
    json.dump(summ, open(outp + '.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))

    rng = np.random.default_rng(1)
    li = np.flatnonzero(L)
    rows = list(rng.choice(li, size=min(NRAND_ROWS, len(li)), replace=False))
    worst = [i for i in li[np.argsort(-np.nan_to_num(a_snr[li], nan=-np.inf))] if i not in rows][:NWORST_ROWS]
    rows += worst
    neg = [i for i in li[np.argsort(np.nan_to_num(a_snr[li], nan=np.inf))] if i not in rows][:NNEG_ROWS]
    rows += neg
    nr = len(rows)
    fig = plt.figure(figsize=(9, 2.6 * nr + 3.2))
    gs = fig.add_gridspec(nr + 1, 3, height_ratios=[1.2] + [1] * nr)
    axh = fig.add_subplot(gs[0, :])
    bins = np.linspace(-5, 40, 46)
    clip = lambda a: np.clip(a[np.isfinite(a)], bins[0], bins[-1])  # noqa: E731
    axh.hist(clip(b_snr[L]), bins, histtype='step', color='C3', lw=2,
             label=f'before: sampled starlike_left (n={L.sum()})')
    axh.hist(clip(a_snr[L]), bins, histtype='step', color='C0', lw=2,
             label=f'after: same positions ({a_rec[L].sum()} now in m7 vetted)')
    axh.hist(clip(b_snr[C]), bins, histtype='step', color='C3', ls=':', label='before: random positions')
    axh.hist(clip(a_snr[C]), bins, histtype='step', color='C0', ls=':', label='after: random positions')
    axh.set_xlabel('PSF-matched S/N left in m7 residual (clipped to [-5, 40])')
    axh.set_ylabel('N')
    axh.legend(fontsize=7)
    axh.set_title(f'{field} {filt}: m7 residual at {L.sum()} of {snap["n_left_total"]} '
                  f'pre-rerun starlike_left positions', fontsize=9)
    cols = [('data (after)', a_dat, None), ('before m7 residual', npz['resbgsub_m7_resid'], b_xy),
            ('after m7 residual', a_res, a_xy)]
    for r, i in enumerate(rows):
        d = a_dat[i]
        med = np.nanmedian(d)
        peak = np.nanmax(d[HALF - 2:HALF + 3, HALF - 2:HALF + 3]) - med
        w = max(peak, 5 * 1.4826 * np.nanmedian(np.abs(d - med)), 1e-6)
        for c, (title, st, xy) in enumerate(cols):
            axs = fig.add_subplot(gs[r + 1, c])
            s = st[i]
            m = np.nanmedian(s)
            axs.imshow(s, origin='lower', cmap='gray', vmin=m - 0.5 * w, vmax=m + w, interpolation='nearest')
            cx = HALF + (ax[i] - np.round(ax[i]))
            cy = HALF + (ay[i] - np.round(ay[i]))
            axs.add_patch(plt.Circle((cx, cy), 2.5 * fwhm / 2, ec='r', fc='none', lw=1))
            if xy is not None:
                dxy = xy - np.array([np.round(ax[i]), np.round(ay[i])]) + HALF
                inb = (np.abs(dxy - HALF) <= HALF).all(axis=1)
                axs.plot(dxy[inb, 0], dxy[inb, 1], 'x', color='orange', ms=4, mew=1)
            axs.set_xticks([]), axs.set_yticks([])
            label = {1: f'S/N {b_snr[i]:.1f}', 2: f'S/N {a_snr[i]:.1f}' + (' in cat' if a_rec[i] else '')}.get(c, '')
            axs.text(0.03, 0.03, label, transform=axs.transAxes, color='yellow', fontsize=7)
            if r == 0:
                axs.set_title(title, fontsize=8)
            if c == 0:
                axs.set_ylabel('largest S/N left after' if i in worst else
                               'most negative after' if i in neg else 'random sample', fontsize=7)
    fig.tight_layout()
    fig.savefig(outp + '.png', dpi=110)


if __name__ == '__main__':
    main(*sys.argv[1:])
