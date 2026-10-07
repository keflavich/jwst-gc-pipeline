"""Per-detector (band - F212N) centre offsets on wd2 m6, measured vs the
scale-only prediction from the distortion refs.

Measured: c0 of datascale.py (explicit (frame - anchor) pair arithmetic,
per-(exposure, module) median removed, median over exposures 1-4).
Prediction: -s_mod * R, where s_mod is the module-mean within-detector scale
of the band's distortion ref relative to F212N's (refscale.txt) and R is the
on-sky vector from the module centre to the detector centre (F212N exposure
00001 crf).  The refs pin every detector centre to the same V2Ref/V3Ref, so a
band whose plate scale is larger by s places stars at the detector centre
s*R too close to the module centre."""
import glob
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

R = '/orange/adamginsburg/jwst/wd2'
DETS = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
BANDS = ['F162M', 'F164N', 'F150W', 'F200W', 'F182M', 'F115W', 'F187N']
ARROW = 4.0          # arcsec of plot per mas of offset


def c0_table(path, band):
    txt = open(path).read()
    m = re.search(rf'### {band.lower()} - f212n.*?\n.*?\n(.*?)(?:\n\s*\n|\n###|\Z)', txt, re.S)
    out = {}
    for line in m.group(1).splitlines():
        p = line.split()
        if p and p[0] in DETS:
            out[p[0]] = np.array([float(p[2]), float(p[3])])
    return out


def module_scales(band):
    txt = open('refscale.txt').read()
    m = re.search(rf'### {band} - F212N\n(.*?)(?:\n###|\Z)', txt, re.S)
    return {mm.group(1).lower(): float(mm.group(2)) * 1e-6
            for mm in re.finditer(r'module ([AB]) mean scale ([+-][\d.]+) ppm', m.group(1))}


def centres():
    out = {}
    for d in DETS:
        f = sorted(glob.glob(f'{R}/F212N/pipeline/jw03523005001_*_00001_{d}_align_o005_crf.fits'))[0]
        w = WCS(fits.getheader(f, 'SCI'))
        out[d] = (w.pixel_to_world(1023.5, 1023.5),
                  w.pixel_to_world(np.array([4, 2043, 2043, 4]), np.array([4, 4, 2043, 2043])))
    ref = out['nrca1'][0]
    xy, corners = {}, {}
    for d, (c, cc) in out.items():
        dra, ddec = ref.spherical_offsets_to(c)
        xy[d] = np.array([dra.to_value('arcsec'), ddec.to_value('arcsec')])
        cra, cdec = ref.spherical_offsets_to(cc)
        corners[d] = np.c_[cra.to_value('arcsec'), cdec.to_value('arcsec')]
    return xy, corners


def measured(band):
    src = 'datascale_f162m_run1.txt' if band == 'F162M' else 'datascale_other.txt'
    return c0_table(src, band)


def main():
    xy, corners = centres()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4))
    rows = []
    for ax, band in zip(axes.ravel(), BANDS):
        meas = measured(band)
        sc = module_scales(band)
        sq_m, sq_r = [], []
        for mod in 'ab':
            ds = [d for d in DETS if d[3] == mod]
            mc = np.mean([xy[d] for d in ds], axis=0)
            pred = {d: -sc[mod] * (xy[d] - mc) * 1e3 for d in ds}     # mas
            pm = np.mean([pred[d] for d in ds], axis=0)
            mm = np.mean([meas[d] for d in ds], axis=0)
            for d in ds:
                p = pred[d] - pm
                q = meas[d] - mm
                x, y = xy[d]
                ax.add_patch(plt.Polygon(corners[d], closed=True, fill=False, ec='0.75', lw=0.8))
                ax.annotate('', xy=(x + q[0] * ARROW, y + q[1] * ARROW), xytext=(x, y),
                            arrowprops=dict(arrowstyle='-|>', color='#1f5fbf', lw=2.0))
                ax.annotate('', xy=(x + p[0] * ARROW, y + p[1] * ARROW), xytext=(x, y),
                            arrowprops=dict(arrowstyle='-|>', color='#d0661b', lw=1.2,
                                            ls=(0, (3, 2))))
                ax.text(x + 18, y + 18, d[3:].upper(), ha='center', va='center', fontsize=7, color='0.45')
                sq_m.append(q @ q)
                sq_r.append((q - p) @ (q - p))
                rows.append((band, d, q[0], q[1], p[0], p[1]))
        rm, rr = np.sqrt(np.mean(sq_m)), np.sqrt(np.mean(sq_r))
        ax.set_title(f'{band} - F212N: rms {rm:.1f} mas\n'
                     f'ref scale A {sc["a"] * 1e6:+.0f}, B {sc["b"] * 1e6:+.0f} ppm; '
                     f'meas-pred {rr:.1f} mas', fontsize=8.5)
        allxy = np.concatenate(list(corners.values()))
        ax.set_xlim(allxy[:, 0].max() + 10, allxy[:, 0].min() - 10)
        ax.set_ylim(allxy[:, 1].min() - 10, allxy[:, 1].max() + 10)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)
        ax.set_xlabel('dRA* from NRCA1 centre (arcsec)', fontsize=7)
        ax.set_ylabel('dDec (arcsec)', fontsize=7)
    ax = axes.ravel()[-1]
    ax.axis('off')
    ax.annotate('', xy=(0.55, 0.62), xytext=(0.15, 0.62), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color='#1f5fbf', lw=2.0))
    ax.text(0.6, 0.62, 'measured (band - F212N)', va='center', fontsize=9, transform=ax.transAxes)
    ax.annotate('', xy=(0.55, 0.50), xytext=(0.15, 0.50), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color='#d0661b', lw=1.2, ls=(0, (3, 2))))
    ax.text(0.6, 0.50, 'prediction  -s R', va='center', fontsize=9, transform=ax.transAxes)
    ax.text(0.05, 0.30, f'arrow length: {ARROW:.0f} arcsec per mas\n'
            'module mean removed from both\n'
            'wd2 m6 per-frame catalogs, exposures 1-4\n'
            'anchor: F212N m6 merged vetted catalog',
            fontsize=8, va='top', transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig('fig_quiver.png', dpi=130)
    with open('fig_quiver_table.tsv', 'w') as fh:
        fh.write('band\tdet\tmeas_dra_mas\tmeas_ddec_mas\tpred_dra_mas\tpred_ddec_mas\n')
        for r in rows:
            fh.write('{}\t{}\t{:+.2f}\t{:+.2f}\t{:+.2f}\t{:+.2f}\n'.format(*r))


if __name__ == '__main__':
    main()
