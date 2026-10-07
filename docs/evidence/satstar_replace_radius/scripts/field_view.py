"""Whole evaluated box of one reference run in two variants.

Panels: data | A m7 residual + A m7 vetted | B m7 residual + B m7 vetted,
over the inner (evaluated) box.  Yellow star: saturated-star fits; orange x:
that panel's m7 vetted sources; red circle: B m7 vetted sources with no A
source within 1 px; blue circle: A m7 vetted sources with no B source within
1 px.  One linear stretch for all panels (data percentiles); each residual is
centred on its own median.

usage: python field_view.py <out.png> <varA> <varB> <field:band:seed>
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy import units as u  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ab_gallery_m7 import M7, Mosaic, RF, _files, _satstar_xy, cKDTree, read_cat  # noqa: E402


def main(outpng, var_a, var_b, spec_s):
    name, filt, seed = spec_s.split(':')
    filt, seed = filt.upper(), int(seed)
    _, fields = RF.load_config()
    spec = fields[name]
    fa, fb = _files(spec, var_a, seed, filt), _files(spec, var_b, seed, filt)
    data = Mosaic(fa['data'])
    va, vb = read_cat(fa[M7]['vetted'], data), read_cat(fb[M7]['vetted'], data)
    half_as = spec['size_arcsec'] / 2 - spec['inner_margin_arcsec']
    cx, cy = data.xy(SkyCoord([spec['ra']] * u.deg, [spec['dec']] * u.deg))
    cx, cy = float(cx[0]), float(cy[0])
    h = half_as / data.pix_as
    x0, x1 = int(np.floor(cx - h)), int(np.ceil(cx + h))
    y0, y1 = int(np.floor(cy - h)), int(np.ceil(cy + h))
    pa, pb = np.c_[va['_x'], va['_y']], np.c_[vb['_x'], vb['_y']]
    b_only = pb[cKDTree(pa).query(pb)[0] > 1.0]
    a_only = pa[cKDTree(pb).query(pa)[0] > 1.0]
    sat = {var_a: _satstar_xy(fa['satstar'], data), var_b: _satstar_xy(fb['satstar'], data)}
    dcut = np.asarray(data.data[y0:y1, x0:x1], float)
    lo, hi = np.nanpercentile(dcut, [1, 99.5])
    span = hi - lo
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9))
    panels = [('data', data, None, sat[var_b]),
              (f'{var_a} m7 resid + m7 vetted', Mosaic(fa[M7]['resid']), va, sat[var_a]),
              (f'{var_b} m7 resid + m7 vetted', Mosaic(fb[M7]['resid']), vb, sat[var_b])]
    for ax, (title, m, cat, st) in zip(axes, panels):
        cut = np.asarray(m.data[y0:y1, x0:x1], float)
        if m is data:
            vmin, vmax = lo, hi
        else:
            bg = np.nanmedian(cut)
            vmin, vmax = bg - 0.15 * span, bg + 0.35 * span
        ax.imshow(cut, origin='lower', cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
        if cat is not None:
            sel = (cat['_x'] > x0) & (cat['_x'] < x1) & (cat['_y'] > y0) & (cat['_y'] < y1)
            ax.plot(cat['_x'][sel] - x0, cat['_y'][sel] - y0, 'x', color='orange', ms=3, mew=0.7)
        for pts, col in ((b_only, 'r'), (a_only, 'deepskyblue')):
            for px, py in pts:
                if x0 < px < x1 and y0 < py < y1:
                    ax.add_patch(plt.Circle((px - x0, py - y0), 4, fill=False, color=col, lw=1.0))
        if len(st):
            sel = (st[:, 0] > x0) & (st[:, 0] < x1) & (st[:, 1] > y0) & (st[:, 1] < y1)
            ax.plot(st[sel, 0] - x0, st[sel, 1] - y0, '*', mfc='none', mec='yellow', ms=12, mew=1.0)
        ax.set_xlim(-0.5, x1 - x0 - 0.5)
        ax.set_ylim(-0.5, y1 - y0 - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        if m is not data:
            m.close()
    axes[0].set_ylabel(f'{name} {filt} seed {seed}: {2 * half_as:.1f}" box\n'
                       f'red: {var_b} only ({len(b_only)}), blue: {var_a} only ({len(a_only)})',
                       fontsize=8)
    fig.tight_layout()
    fig.savefig(outpng, dpi=110)
    plt.close(fig)
    print(outpng, 'b_only', len(b_only), 'a_only', len(a_only))


if __name__ == '__main__':
    main(*sys.argv[1:])
