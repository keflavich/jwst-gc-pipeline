"""Diagnostic cutouts for stars lost when SATSTAR_ZF_KEEP_FINITE is on (arm B)."""
import glob
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from astropy.io import fits
from astropy.table import Table, join
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
KL = f'{Q}/kf_lost'
TA, TB = f'{Q}/tree_mainfcbg', f'{Q}/tree_mainfcbgkf'
OUT = f'{KL}/fig/lost_cutouts_mainfcbg_mainfcbgkf.png'
HALF = 20  # 41x41 px
BANDS = ['f164n', 'f150w', 'f162m', 'f250m', 'f187n']


def sky_cols(tab, prefix):
    if prefix in tab.colnames:
        return tab[prefix]
    return SkyCoord(np.asarray(tab[prefix + '.ra'], float) * u.deg,
                    np.asarray(tab[prefix + '.dec'], float) * u.deg)


def frames(tree, band):
    """dict (det, exp) -> daophot path"""
    out = {}
    pat = f'{tree}/{band.upper()}/{band}_*_visit001_vgroup*_exp0000?_resbgsub_m7_daophot_basic.fits'
    for p in glob.glob(pat):
        m = re.search(rf'{band}_(\w+?)_visit001_vgroup\d+_exp0000(\d)_', os.path.basename(p))
        out[(m.group(1), int(m.group(2)))] = p
    return out


def pipe_stem(tree, band, det, exp):
    pat = f'{tree}/{band.upper()}/pipeline/jw*_*_0000{exp}_{det}_align_o005_crf_resbgsub_m7_satstar_catalog.fits'
    g = glob.glob(pat)
    return g[0].replace('_catalog.fits', '') if g else None


def load_daophot(path):
    t = Table.read(path)
    return t, sky_cols(t, 'skycoord_centroid')


def find_frame(band, star):
    """First (det, exp) where A has a row within 0.1" and B has none."""
    fa, fb = frames(TA, band), frames(TB, band)
    for key in sorted(fa):
        if key not in fb:
            continue
        ta, ca = load_daophot(fa[key])
        if not (ca.separation(star).arcsec < 0.1).any():
            continue
        tb, cb = load_daophot(fb[key])
        if (cb.separation(star).arcsec < 0.1).any():
            continue
        return key, ca, cb
    return None


def main():
    lost = Table.read(f'{KL}/lost_mainfcbg_mainfcbgkf.ecsv')
    trace = Table.read(f'{KL}/trace_mainfcbg_mainfcbgkf.ecsv')
    bead = Table.read(f'{KL}/spikebead_mainfcbg_mainfcbgkf.ecsv')['i', 'sep', 'sep_over_lam', 'pa', 'bead']
    bead.rename_columns(['sep', 'pa'], ['sepsat', 'pasat'])
    j = join(join(trace[trace['nframesB'] == 0], lost, keys='i'), bead, keys='i')
    j = j[(j['nframesA'] > 0) & (j['dsatB'] <= 1.5)]
    print('candidates', len(j), 'beads', int(np.sum(j['bead'])), 'by band', dict(zip(*np.unique(j['band'][j['bead']], return_counts=True))))
    slots = [('f164n', True), ('f150w', True), ('f162m', True), ('f250m', False), ('f187n', False)]
    rows, used = [], set()
    for band0, isbead in slots:
        for band in [band0] + [b for b in np.unique(j['band']) if b != band0]:
            cand = j[(j['band'] == band) & (j['bead'] == isbead) & ~np.isin(j['i'], list(used))]
            cand = cand[np.argsort(cand['dsatB'])]
            cand = [r for r in cand if r['dsatB'] > 0.3] + [r for r in cand if r['dsatB'] <= 0.3]
            hit = None
            for r in cand:
                star = SkyCoord(r['RA'] * u.deg, r['DEC'] * u.deg)
                f = find_frame(band, star)
                if f is not None:
                    hit = (band, r, star, f)
                    break
            if hit:
                rows.append(hit); used.add(int(hit[1]['i']))
                break
    n = len(rows)
    fig, axs = plt.subplots(n, 4, figsize=(15, 3.9 * n))
    titles = ['m7 input (A resid + A model)', 'A m7 satstar residual',
              'B m7 satstar residual', 'B model - A model']
    summary = []
    for irow, (band, r, star, (key, ca, cb)) in enumerate(rows):
        det, exp = key
        stems = {}
        for arm, tree in (('A', TA), ('B', TB)):
            stems[arm] = pipe_stem(tree, band, det, exp)
        data = {}
        for arm in 'AB':
            s = stems[arm]
            data[arm + 'res'] = fits.getdata(s + '_residual.fits')
            data[arm + 'mod'] = fits.getdata(s + '_model.fits')
            data[arm + 'cat'] = Table.read(s + '_catalog.fits')
        hdr = fits.getheader(stems['A'] + '_residual.fits')
        wcs = WCS(hdr)
        inp = data['Ares'] + data['Amod']
        diff = data['Bmod'] - data['Amod']
        x0, y0 = [float(v) for v in wcs.world_to_pixel(star)]

        def pix(c):
            x, y = wcs.world_to_pixel(c)
            return np.asarray(x), np.asarray(y)

        cat = {k: data[k + 'cat'] for k in 'AB'}
        sat = {k: pix(sky_cols(cat[k], 'skycoord_fit')) for k in 'AB'}  # x_fit/y_fit are not full-frame pixels
        pa, pb = pix(ca), pix(cb)
        pscale = np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600
        allx = np.concatenate([sat['A'][0], sat['B'][0]]); ally = np.concatenate([sat['A'][1], sat['B'][1]])
        dall = np.hypot(allx - x0, ally - y0)
        k0 = int(np.argmin(dall))
        sx, sy = allx[k0], ally[k0]   # nearest satstar (A or B) = spike source
        dist = dall[k0] * pscale
        dB = np.hypot(sat['B'][0] - sx, sat['B'][1] - sy)
        dA = np.hypot(sat['A'][0] - sx, sat['A'][1] - sy)
        ib, ia = int(np.argmin(dB)), int(np.argmin(dA))
        inB, inA = dB[ib] <= 0.5, dA[ia] <= 0.5
        if inB and inA:
            ratio = float(cat['B']['flux_fit'][ib]) / float(cat['A']['flux_fit'][ia])
            ratio_s = f'{ratio:.2f}'
        else:
            ratio = np.nan
            ratio_s = 'sat only in B' if inB else ('sat only in A' if inA else 'n/a')
        bflux = float(cat['B']['flux_fit'][ib]) if inB else np.nan
        cx, cy = 0.5 * (x0 + sx), 0.5 * (y0 + sy)
        half = int(min(max(np.hypot(x0 - sx, y0 - sy) / 2 + 14, 20), 60))
        cxi, cyi = int(round(cx)), int(round(cy))
        sl = (slice(max(cyi - half, 0), cyi + half + 1), slice(max(cxi - half, 0), cxi + half + 1))
        xo, yo = sl[1].start, sl[0].start
        ext = (xo - 0.5, sl[1].stop - 0.5, yo - 0.5, sl[0].stop - 0.5)
        satc = wcs.pixel_to_world(sx, sy)
        pa0 = float(r['pasat']) % 60 if r['bead'] else 20.0
        spikes = []
        for kk in range(6):
            p1 = satc.directional_offset_by((pa0 + 60 * kk) * u.deg, 3 * u.arcsec)
            xs, ys = wcs.world_to_pixel(p1)
            spikes.append((float(xs), float(ys)))
        ix, iy = int(round(x0)), int(round(y0))
        def med(a): return float(np.nanmedian(a[iy - 1:iy + 2, ix - 1:ix + 2]))
        def sig(a):
            c = a[sl]; m = np.nanmedian(c); return 1.4826 * np.nanmedian(np.abs(c - m))
        summary.append(dict(
            i=int(r['i']), band=band, det=det, exp=exp, bead=bool(r['bead']), dist_arcsec=round(float(dist), 2),
            tbl_sep_over_lam=round(float(r['sep_over_lam']), 2), tbl_pa=round(float(r['pasat']), 1),
            B_over_A=ratio_s, satB_flux=bflux, sat_in_A=bool(inA), sat_in_B=bool(inB), nsatA=len(cat['A']), nsatB=len(cat['B']),
            inp_3x3=round(med(inp), 2), Ares_3x3=round(med(data['Ares']), 2), Bres_3x3=round(med(data['Bres']), 2),
            Bmod_minus_Amod_3x3=round(med(diff), 2), cutout_sigma=round(sig(inp), 2),
            Bres_min_cutout=float(np.nanmin(data['Bres'][sl])), Ares_min_cutout=float(np.nanmin(data['Ares'][sl])),
            dolphot_flux=float(r['fluxA'])))
        vmax = np.nanpercentile(inp[sl], 99.5)
        vmin = np.nanpercentile(inp[sl], 1)
        rmax = np.nanpercentile(data['Ares'][sl], 99.5)
        dlim = np.nanmax(np.abs(diff[sl]))
        dlim = dlim if dlim > 0 else 1
        panels = [(inp, AsinhNorm(linear_width=max(abs(vmax) * 0.02, 1e-3), vmin=vmin, vmax=vmax), 'gray_r'),
                  (data['Ares'], AsinhNorm(linear_width=max(abs(rmax) * 0.02, 1e-3), vmin=vmin, vmax=rmax), 'gray_r'),
                  (data['Bres'], AsinhNorm(linear_width=max(abs(rmax) * 0.02, 1e-3), vmin=vmin, vmax=rmax), 'gray_r'),
                  (diff, matplotlib.colors.SymLogNorm(linthresh=dlim * 0.01, vmin=-dlim, vmax=dlim), 'RdBu_r')]
        for k, (arr, norm, cm) in enumerate(panels):
            ax = axs[irow, k]
            ax.imshow(arr[sl], origin='lower', extent=ext, norm=norm, cmap=cm, interpolation='nearest')
            ax.plot(x0, y0, 'o', mfc='none', mec='red', ms=14, mew=1.5)
            ax.plot(*pa, '+', color='lime', ms=10, mew=1.5)
            ax.plot(*pb, 'x', color='cyan', ms=9, mew=1.5)
            ax.plot(*sat['A'], 's', mfc='none', mec='magenta', ms=15, mew=1.3)
            ax.plot(*sat['B'], 'D', mfc='none', mec='orange', ms=12, mew=1.3)
            for xs, ys in spikes:
                ax.plot([sx, xs], [sy, ys], '-', color='yellow', lw=0.6, alpha=0.45)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_xticks([]); ax.set_yticks([])
            if irow == 0:
                ax.set_title(titles[k], fontsize=10)
        axs[irow, 0].set_ylabel(
            f'{band.upper()} {det} exp{exp}  ref #{int(r["i"])}\n'
            f'{"bead" if r["bead"] else "non-bead"}  sep/lam={float(r["sep_over_lam"]):.2f}\n'
            f'nearest satstar {dist:.2f}"\nB/A flux_fit: {ratio_s}', fontsize=8.5)
    handles = [
        plt.Line2D([], [], marker='o', mfc='none', mec='red', ls='', label='dolphot'),
        plt.Line2D([], [], marker='+', color='lime', ls='', label='A daophot'),
        plt.Line2D([], [], marker='x', color='cyan', ls='', label='B daophot'),
        plt.Line2D([], [], marker='s', mfc='none', mec='magenta', ls='', label='A satstar'),
        plt.Line2D([], [], marker='D', mfc='none', mec='orange', ls='', label='B satstar')]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=10)
    fig.suptitle('Stars lost in B (KEEP_FINITE on); yellow lines = 6 spike directions through nearest satstar', fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(OUT, dpi=110)
    for s in summary:
        print(s)


if __name__ == '__main__':
    main()
