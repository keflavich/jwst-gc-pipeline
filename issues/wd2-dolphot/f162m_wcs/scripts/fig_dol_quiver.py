"""Per-detector (band - dolphot) centre offsets on wd2 m6 (datascale_dol.py,
MATCH=0.2, exposures 1-2): the dolphot catalog is an independent anchor whose
per-image alignment absorbs per-chip terms.  Dashed: the scale-only prediction
-s R of fig_quiver.py (zero for F212N)."""
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fig_quiver as fq

BANDS = ['F162M', 'F164N', 'F182M', 'F212N']


def c0_dol(path, band):
    txt = open(path).read()
    m = re.search(rf'### {band.lower()} - dolphot.*?\n.*?\n(.*?)(?:\n\s*\n|\n###|\Z)', txt, re.S)
    return {p[0]: np.array([float(p[2]), float(p[3])])
            for p in (ln.split() for ln in m.group(1).splitlines()) if p and p[0] in fq.DETS}


def main():
    xy, corners = fq.centres()
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    for ax, band in zip(axes, BANDS):
        meas = c0_dol(f'datascale_dol_m02_{band.lower()}.txt', band)
        sc = fq.module_scales(band) if band != 'F212N' else {'a': 0.0, 'b': 0.0}
        sq_m, sq_r = [], []
        for mod in 'ab':
            ds = [d for d in fq.DETS if d[3] == mod and d in meas]
            mc = np.mean([xy[d] for d in fq.DETS if d[3] == mod], axis=0)
            pred = {d: -sc[mod] * (xy[d] - mc) * 1e3 for d in ds}
            pm = np.mean([pred[d] for d in ds], axis=0)
            mm = np.mean([meas[d] for d in ds], axis=0)
            for d in ds:
                p, q = pred[d] - pm, meas[d] - mm
                x, y = xy[d]
                ax.annotate('', xy=(x + q[0] * fq.ARROW, y + q[1] * fq.ARROW), xytext=(x, y),
                            arrowprops=dict(arrowstyle='-|>', color='#1f5fbf', lw=2.0))
                if band != 'F212N':
                    ax.annotate('', xy=(x + p[0] * fq.ARROW, y + p[1] * fq.ARROW), xytext=(x, y),
                                arrowprops=dict(arrowstyle='-|>', color='#d0661b', lw=1.2, ls=(0, (3, 2))))
                sq_m.append(q @ q)
                sq_r.append((q - p) @ (q - p))
        for d in fq.DETS:
            ax.add_patch(plt.Polygon(corners[d], closed=True, fill=False, ec='0.75', lw=0.8))
            ax.text(xy[d][0] + 18, xy[d][1] + 18, d[3:].upper(), ha='center', va='center', fontsize=7,
                    color='0.45' if d in meas else '#c03030')
        allxy = np.concatenate(list(corners.values()))
        ax.set_xlim(allxy[:, 0].max() + 10, allxy[:, 0].min() - 10)
        ax.set_ylim(allxy[:, 1].min() - 10, allxy[:, 1].max() + 10)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)
        ax.set_xlabel('dRA* from NRCA1 centre (arcsec)', fontsize=7)
        extra = '' if band == 'F212N' else f'; meas-pred {np.sqrt(np.mean(sq_r)):.1f} mas'
        ax.set_title(f'{band} - dolphot: rms {np.sqrt(np.mean(sq_m)):.1f} mas ({len(meas)} det){extra}',
                     fontsize=9)
    axes[0].set_ylabel('dDec (arcsec)', fontsize=7)
    fig.suptitle(f'wd2 m6 per-frame catalogs vs dolphot positions; arrows {fq.ARROW:.0f} arcsec per mas, '
                 'module mean removed; blue measured, dashed orange -sR from the distortion refs', fontsize=9)
    fig.tight_layout()
    fig.savefig('fig_dol_quiver.png', dpi=120)


if __name__ == '__main__':
    main()
