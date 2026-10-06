"""Cutouts of stars vetted in an earlier pass and left in the final residual.

Per row (one lost star): data mosaic, residual of the last phase that vetted
the star (the star is subtracted there), residual of the next phase (blank
when that is the final phase) and the final (m7) residual.  Red circle: the
lost star; blue +: m7 seed sources; orange x: final (m7) vetted sources;
magenta diamond: for a ``companion_cut`` star, the brighter seed source the
cut measured it against.  The residual panels of a row share one linear
stretch set by the data panel.

usage: python lost_gallery.py <out.png> <spec.json>
spec.json: list of {"run": dir, "filt": F182M, "obstok": "_o001",
                    "lost": lost.fits, "rows": [i, ...], "title": "..."}
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table  # noqa: E402

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from phase_loss import PHASES, Mosaic, phase_files, read_cat  # noqa: E402

SHORT = {'m2': 'm2', 'm3': 'm3', 'm4': 'm4', 'resbgsub_m5': 'm5', 'resbgsub_m6': 'm6',
         'resbgsub_m7': 'm7'}


def stamp(mos, x, y, half):
    xi, yi = int(round(x)), int(round(y))
    ny, nx = mos.shape
    out = np.full((2 * half + 1, 2 * half + 1), np.nan)
    y0, y1 = max(yi - half, 0), min(yi + half + 1, ny)
    x0, x1 = max(xi - half, 0), min(xi + half + 1, nx)
    out[y0 - (yi - half):y1 - (yi - half), x0 - (xi - half):x1 - (xi - half)] = mos.data[y0:y1, x0:x1]
    return out, xi - half, yi - half


def companion_neighbour(files, data, filt, lost):
    """Pixel position of the brighter seed source nearest the lost star."""
    from astropy.coordinates import SkyCoord
    from phase_loss import FWHM_PIX
    bs = Table.read(files['m7_band_seed'])
    origin = np.asarray(bs['seed_origin']).astype(str)
    own = read_cat(files['resbgsub_m6']['vetted'], data)
    xb = bs[origin == 'crossband']
    xx, xy = data.xy(SkyCoord(xb['skycoord']))
    px = np.r_[xx, own['_x']]
    py = np.r_[xy, own['_y']]
    pf = np.r_[np.asarray(xb['flux'], float), np.asarray(own['flux'], float)]
    d = np.hypot(px - lost['x'], py - lost['y'])
    me = np.argmin(np.where(np.arange(len(px)) >= len(xx), d, np.inf))
    ok = (d > 0.2) & (d < float(bs.meta.get('COMPFWHM', 2.5)) * FWHM_PIX[filt.upper()]) & (pf > pf[me])
    if not ok.any():
        return None
    j = np.flatnonzero(ok)[np.argmin(d[ok])]
    return px[j], py[j]


def main(outpng, specpath, half_arcsec=0.5):
    spec = json.load(open(specpath))
    rows = [(s, r) for s in spec for r in s['rows']]
    fig, axes = plt.subplots(len(rows), 4, figsize=(9.6, 2.5 * len(rows)), squeeze=False)
    for k, (s, r) in enumerate(rows):
        files = phase_files(f"{s['run']}/catalogs", f"{s['run']}/{s['filt']}/pipeline", s['filt'],
                            'merged', s.get('obstok', ''))
        lost = Table.read(s['lost'])[r]
        data = Mosaic(files['data'])
        half = int(round(half_arcsec / data.pix_as))
        last, nxt = lost['last_phase'], lost['next_phase']
        panels = [('data', files['data']), (f'resid {SHORT[last]} (last vetted)', files[last]['resid']),
                  (f'resid {SHORT[nxt]}', files[nxt]['resid']), ('resid m7 (final)', files[PHASES[-1]]['resid'])]
        if nxt == PHASES[-1]:
            panels[2] = None
        nb = companion_neighbour(files, data, s['filt'], lost) if lost['seed_reason'] == 'companion_cut' else None
        seed = read_cat(files['m7_seed'], data) if files['m7_seed'] else None
        fin = read_cat(files[PHASES[-1]]['vetted'], data)
        ds, x0, y0 = stamp(data, lost['x'], lost['y'], half)
        lo, hi = np.nanpercentile(ds, [1, 99.5])
        for j, panel in enumerate(panels):
            ax = axes[k, j]
            if panel is None:
                ax.axis('off')
                continue
            name, path = panel
            m = data if path == files['data'] else Mosaic(path)
            st, _, _ = stamp(m, lost['x'], lost['y'], half)
            if j == 0:
                vmin, vmax = lo, hi
            else:
                bg = np.nanmedian(st)
                vmin, vmax = bg - 0.35 * (hi - lo), bg + 0.65 * (hi - lo)
            ax.imshow(st, origin='lower', cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.add_patch(plt.Circle((lost['x'] - x0, lost['y'] - y0), 3.5, fill=False, color='r', lw=1.2))
            if seed is not None:
                sel = (np.abs(seed['_x'] - lost['x']) < half) & (np.abs(seed['_y'] - lost['y']) < half)
                ax.plot(seed['_x'][sel] - x0, seed['_y'][sel] - y0, '+', color='deepskyblue', ms=7, mew=1.2)
            sel = (np.abs(fin['_x'] - lost['x']) < half) & (np.abs(fin['_y'] - lost['y']) < half)
            ax.plot(fin['_x'][sel] - x0, fin['_y'][sel] - y0, 'x', color='orange', ms=5, mew=1.0)
            if nb is not None:
                ax.plot(nb[0] - x0, nb[1] - y0, 'D', mfc='none', mec='magenta', ms=8, mew=1.2)
            ax.set_xlim(-0.5, 2 * half + 0.5); ax.set_ylim(-0.5, 2 * half + 0.5)
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0 or True:
                ax.set_title(name, fontsize=8)
            if m is not data:
                m.close()
        reason = lost['seed_reason'] if 'seed_reason' in lost.colnames else ''
        axes[k, 0].set_ylabel(f"{s.get('title', '')}\nS/N {lost['snr_last']:.0f} ({SHORT[last]}); "
                              f"{lost['category']}\n{reason}  final-resid S/N {lost['res_final_snr']:.0f}",
                              fontsize=7)
        data.close()
    fig.tight_layout()
    fig.savefig(outpng, dpi=110)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
