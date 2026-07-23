"""Quiver figures of the CRDS->GDC astrometric distortion delta per detector.

For each SW detector of brick F212N (jw02221 o001) plus arches NRCA4
(jw02045 o001), build the affine-anchored :class:`GDCSkySolution` on a
representative crf frame and plot its CRDS-vs-GDC delta map
(``delta_xi_mas``/``delta_eta_mas``: affine-anchored STDGDC minus the frame's
existing CRDS WCS, mean 0 by construction) as a quiver field over the
detector, one panel per detector on a shared color/arrow scale, plus a
per-detector median/max |delta| summary chart.

Arrows are tangent-plane (xi, eta) = (East, North) offsets in mas at the raw
pixel grid positions; the NRCB4/F212N unmeasured STDGDC border cells (masked
out of the anchor fit, NaN in the delta map) are marked explicitly.

Figures are written as PNGs to ``--outdir``.  The repo bans tracked PNGs
(PR #73, ``.gitignore`` ``*.png``): do NOT commit the outputs -- publish them
via externally hosted URLs (e.g. a GitHub gist) as done for PR #154.

Usage::

    python -m jwst_gc_pipeline.astrometry_gdc.make_vector_figures \
        --outdir /path/to/plots [--grid-n 24]
"""
import argparse
import os

import numpy as np

from .gdc_wcs import GDCSkySolution, load_frame_wcs
from .stdgdc import STDGDC, detector_filter_from_header

SW_DETECTORS = ('nrca1', 'nrca2', 'nrca3', 'nrca4',
                'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4')

BRICK_F212N_TEMPLATE = ('/orange/adamginsburg/jwst/brick/F212N/pipeline/'
                        'jw02221001001_05101_00001_{det}_destreak_o001_crf.fits')
ARCHES_F212N_NRCA4 = ('/orange/adamginsburg/jwst/arches/F212N/pipeline/'
                      'jw02045001001_02101_00001_nrca4_destreak_o001_crf.fits')

# Sequential single-hue ramp for |delta| magnitude (dataviz: one hue,
# light->dark; no rainbow), truncated so the lightest arrows stay visible on
# a white surface.  Status red marks the invalid/unmeasured map cells.
CMAP_NAME = 'Blues'
CMAP_RANGE = (0.35, 1.0)
INVALID_COLOR = '#c4372c'
MEDIAN_BAR_COLOR = '#4269d0'
MAX_BAR_COLOR = '#efb118'


def default_frames():
    """[(panel_label, cal_file), ...] for brick F212N SW + arches NRCA4."""
    frames = [(f'brick F212N {det}', BRICK_F212N_TEMPLATE.format(det=det))
              for det in SW_DETECTORS]
    frames.append(('arches F212N nrca4', ARCHES_F212N_NRCA4))
    return frames


def solve_frame(cal_file, grid_n=24):
    """(GDCSkySolution, detector, filter) for one crf frame."""
    wcs_like, header = load_frame_wcs(cal_file)
    detector, filt = detector_filter_from_header(header)
    gdc = STDGDC.load(detector, filt)
    return GDCSkySolution(wcs_like, gdc, grid_n=grid_n), detector, filt


def collect(frames, grid_n=24):
    """Solve every frame; returns list of dicts with delta maps + stats."""
    out = []
    for label, cal_file in frames:
        sol, detector, filt = solve_frame(cal_file, grid_n=grid_n)
        gx, gy, dxi, deta = sol.delta_map()
        mag = np.hypot(dxi, deta)
        out.append({
            'label': label, 'cal_file': cal_file, 'detector': detector,
            'filter': filt, 'grid_x': gx, 'grid_y': gy,
            'dxi': dxi, 'deta': deta, 'mag': mag,
            'median_mas': float(np.nanmedian(mag)),
            'p95_mas': float(np.nanpercentile(mag, 95)),
            'max_mas': float(np.nanmax(mag)),
            'n_invalid': int(np.sum(~np.isfinite(mag))),
            'affine_rms_mas': sol.affine_rms_mas,
        })
    return out


def plot_vector_fields(results, outpath, dpi=150):
    """One quiver panel per detector, shared color + arrow scale."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colormaps, colors

    vmax = max(r['max_mas'] for r in results)
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    base = colormaps[CMAP_NAME]
    cmap = colors.LinearSegmentedColormap.from_list(
        'seq', base(np.linspace(*CMAP_RANGE, 256)))

    ncols = 3
    nrows = int(np.ceil(len(results) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.5 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    quiv = None
    any_invalid = False
    for ax, res in zip(axes, results):
        finite = np.isfinite(res['mag'])
        quiv = ax.quiver(
            res['grid_x'][finite], res['grid_y'][finite],
            res['dxi'][finite], res['deta'][finite], res['mag'][finite],
            cmap=cmap, norm=norm, angles='xy',
            scale_units='width', scale=vmax * 9.0, width=0.004)
        if (~finite).any():
            any_invalid = True
            ax.plot(res['grid_x'][~finite], res['grid_y'][~finite], 's',
                    ms=3.5, mfc='none', mec=INVALID_COLOR, mew=0.9)
        ax.set_title(f"{res['label']}\n"
                     f"med {res['median_mas']:.2f} / max {res['max_mas']:.2f} mas",
                     fontsize=10)
        ax.set_xlim(-60, 2107)
        ax.set_ylim(-60, 2107)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7, colors='0.45')
        for spine in ax.spines.values():
            spine.set_color('0.7')
    for ax in axes[len(results):]:
        ax.set_visible(False)
    if quiv is not None:
        axes[0].quiverkey(quiv, 0.14, 1.14, 2.0, '2 mas',
                          labelpos='E', coordinates='axes', fontproperties={'size': 9})
    cb = fig.colorbar(quiv, ax=axes.tolist(), shrink=0.6, pad=0.01)
    cb.set_label('|CRDS -> GDC delta|  (mas)')
    title = ('CRDS -> GDC distortion delta field (affine-anchored; '
             'mean 0 by construction)')
    if any_invalid:
        title += ('\nopen red squares = unmeasured STDGDC border cells '
                  '(NRCB4/F212N library holes; masked, stars fall back to CRDS)')
    fig.suptitle(title, fontsize=12)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return outpath


def plot_summary(results, outpath, dpi=150):
    """Per-detector median / p95 / max |delta| grouped bar chart."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = [r['label'].replace(' F212N', '\nF212N') for r in results]
    med = [r['median_mas'] for r in results]
    mx = [r['max_mas'] for r in results]
    x = np.arange(len(results))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    ax.bar(x - width / 2, med, width, label='median |delta|',
           color=MEDIAN_BAR_COLOR)
    ax.bar(x + width / 2, mx, width, label='max |delta|', color=MAX_BAR_COLOR)
    for xi_, v in zip(x + width / 2, mx):
        ax.annotate(f'{v:.1f}', (xi_, v), ha='center', va='bottom',
                    fontsize=8, color='0.25')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('|CRDS -> GDC delta|  (mas)')
    ax.set_title('Per-detector CRDS -> GDC delta magnitude (24x24 grid)',
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.yaxis.grid(True, color='0.9', lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return outpath


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--outdir', required=True,
                        help='Directory for the output PNGs (NOT in the repo; '
                             'tracked PNGs are banned)')
    parser.add_argument('--grid-n', type=int, default=24)
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    results = collect(default_frames(), grid_n=args.grid_n)

    print('| panel | median \\|delta\\| (mas) | p95 | max | invalid cells | '
          'affine rms (mas) |')
    print('|---|---|---|---|---|---|')
    for r in results:
        print(f"| {r['label']} | {r['median_mas']:.2f} | {r['p95_mas']:.2f} "
              f"| {r['max_mas']:.2f} | {r['n_invalid']} "
              f"| {r['affine_rms_mas']:.2f} |")

    f1 = plot_vector_fields(results,
                            os.path.join(args.outdir, 'gdc_delta_vecfield_f212n.png'),
                            dpi=args.dpi)
    f2 = plot_summary(results,
                      os.path.join(args.outdir, 'gdc_delta_summary_f212n.png'),
                      dpi=args.dpi)
    print(f'wrote {f1}')
    print(f'wrote {f2}')


if __name__ == '__main__':
    main()
