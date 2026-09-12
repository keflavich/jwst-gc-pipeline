#!/usr/bin/env python
"""Colour-colour (or colour-magnitude) diagram for a staged release field.

The release ships catalogues; until now the pages showed only imagery, so the
photometry could not be judged without downloading it.  This renders one
diagram per field from the SAME merged table the release distributes.

WHICH DIAGRAM
-------------
The requested one is F405N-F466N against F182M-F210M.  Naming those filters
literally would draw it for almost no field: the Galactic Center programmes use
F212N where this one says F210M, F480M where it says F466N, and so on.  Bands
are therefore chosen by WAVELENGTH SLOT, and the slot lists below are ordered by
preference within each slot:

    sw_blue  ~1.4-1.9 um   the "H" side
    sw_red   ~2.0-2.3 um   the "K" side
    lw_blue  ~3.0-4.1 um
    lw_red   ~4.4-4.9 um

* Both an SW pair and an LW pair present -> the requested diagram, x = SW
  colour, y = LW colour.  brick, cloudc and w51 land here; w51 literally has
  F182M/F210M and F405N/F480M.
* Otherwise three or more bands -> a colour-colour diagram from adjacent pairs
  (bluest-middle against middle-reddest).  sgra, sgrb2 and sgrc land here.
* Exactly two bands -> a colour-MAGNITUDE diagram.  gc2211 and quintuplet.
* Fewer -> nothing, and the caller says so rather than drawing an empty box.

QUALITY CUTS
------------
S/N > 10 in EVERY band the diagram uses, and a finite magnitude in every one --
so a point is a source measured in all of them, not one detected in two and
extrapolated.  S/N is flux/flux_err from the table's own columns.

Magnitudes are VEGA.  The AB zero points in these catalogues are ~1 mag off
(cloudc investigation, 2026-07), and a colour built from them is wrong by the
difference of two such errors.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

#: Wavelength slots, each ordered by preference.  A slot is satisfied by the
#: first of its filters the field actually has.
SLOTS = {
    'sw_blue': ('F182M', 'F187N', 'F162M', 'F150W', 'F140M', 'F115W'),
    'sw_red': ('F212N', 'F210M', 'F200W'),
    'lw_blue': ('F405N', 'F410M', 'F360M', 'F356W', 'F335M', 'F323N', 'F300M'),
    'lw_red': ('F466N', 'F470N', 'F480M', 'F444W'),
}

#: Approximate pivot wavelength, micron -- only ever used to ORDER bands, so
#: nearby values need not be exact.
_WAVE = {'F070W': 0.70, 'F090W': 0.90, 'F115W': 1.15, 'F140M': 1.40,
         'F150W': 1.50, 'F150W2': 1.66, 'F162M': 1.63, 'F164N': 1.64,
         'F182M': 1.85, 'F187N': 1.87, 'F200W': 2.00, 'F210M': 2.09,
         'F212N': 2.12, 'F250M': 2.50, 'F277W': 2.78, 'F300M': 2.99,
         'F322W2': 3.23, 'F323N': 3.24, 'F335M': 3.36, 'F356W': 3.57,
         'F360M': 3.62, 'F405N': 4.05, 'F410M': 4.08, 'F430M': 4.28,
         'F444W': 4.40, 'F466N': 4.65, 'F470N': 4.70, 'F480M': 4.82}

SNR_MIN = 10.0


#: ``..._resbgsub_m8_o001.fits`` -> ``('o001', 'resbgsub_m8')``; either may be
#: ``None``.
_TABLE_RE = re.compile(
    r'basic_merged.*?_((?:resbgsub_)?m\d+)(?:_(o[0-9]+(?:-[0-9]+)*))?\.fits$')


def select_tables(paths, iteration_rank):
    """The merged tables worth drawing, one per pointing.

    Two different things live in a field's ``catalogs/`` and only one of them
    means "another pointing":

    * brick v1.0 stages ``_m7.fits`` beside ``_m8.fits`` -- the SAME field at
      two merge iterations.  Drawing both wrote one file twice and silently
      kept whichever finished last.  Keep the best iteration.
    * gc2211 stages ``_m7_o023``, ``_m7_o050`` ... beside an untokened
      ``_m7.fits`` -- genuinely different pointings, plus a combined table that
      pools them.  Keep the per-observation ones and drop the untokened
      pooled table, which is the same rule `stage_release.discover_catalogs`
      applies to the shipped catalogues.

    Returns ``[(obs_or_None, path)]``, observation-ordered.
    """
    best = {}
    for path in paths:
        m = _TABLE_RE.search(os.path.basename(path))
        if not m:
            continue
        iteration, obs = m.group(1), m.group(2)
        rank = iteration_rank(iteration)
        if rank is None:
            continue
        prev = best.get(obs)
        if prev is None or rank > prev[0]:
            best[obs] = (rank, path)
    if len(best) > 1 and None in best:
        del best[None]
    return [(obs, path) for obs, (_r, path) in
            sorted(best.items(), key=lambda kv: (kv[0] is not None, kv[0] or ''))]


def bands_in(table):
    """Every band the merged table carries a Vega magnitude for, blue to red."""
    found = {m.group(1).upper() for c in table.colnames
             if (m := re.match(r'mag_vega_(f\d{3,4}[wmn]\d?)$', c))}
    return sorted(found, key=lambda b: (_WAVE.get(b, 99.0), b))


def _first(slot, available):
    for band in SLOTS[slot]:
        if band in available:
            return band
    return None


def choose_axes(bands):
    """``(kind, bands_used, xlabel, ylabel)`` or ``(None, ...)``.

    ``kind`` is ``'ccd'`` or ``'cmd'``; ``bands_used`` is ``(x_blue, x_red,
    y_blue, y_red)`` for a CCD and ``(blue, red)`` for a CMD.
    """
    have = set(bands)
    sw = (_first('sw_blue', have), _first('sw_red', have))
    lw = (_first('lw_blue', have), _first('lw_red', have))
    # Guard against one band satisfying both halves of a pair (F444W is in
    # lw_red, and a field with only F444W in the LW must not plot F444W-F444W).
    if all(sw) and all(lw) and sw[0] != sw[1] and lw[0] != lw[1]:
        return ('ccd', (sw[0], sw[1], lw[0], lw[1]),
                f'{sw[0]} - {sw[1]}', f'{lw[0]} - {lw[1]}')
    if len(bands) >= 3:
        b, m, r = bands[0], bands[len(bands) // 2], bands[-1]
        if len({b, m, r}) == 3:
            return ('ccd', (b, m, m, r), f'{b} - {m}', f'{m} - {r}')
    if len(bands) == 2:
        b, r = bands
        return ('cmd', (b, r), f'{b} - {r}', r)
    return (None, (), '', '')


def _mag_and_snr(table, band):
    key = band.lower()
    mag = np.asarray(table[f'mag_vega_{key}'], dtype=float)
    flux = np.asarray(table[f'flux_{key}'], dtype=float)
    err = np.asarray(table[f'flux_err_{key}'], dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(err > 0, flux / err, np.nan)
    return mag, snr


def select(table, bands_used, snr_min=SNR_MIN):
    """``(mags, keep)`` -- a source is kept only if EVERY band is finite and
    above the S/N floor.  That is the "matched in N bands" requirement: a row
    exists in the merged table if it was seen in ANY band."""
    mags, keep = {}, np.ones(len(table), dtype=bool)
    for band in dict.fromkeys(bands_used):
        mag, snr = _mag_and_snr(table, band)
        mags[band] = mag
        keep &= np.isfinite(mag) & np.isfinite(snr) & (snr > snr_min)
    return mags, keep


def render(table, field, out_png, snr_min=SNR_MIN, dpi=130):
    """Write the diagram; return an info dict (``None`` if undrawable)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    bands = bands_in(table)
    kind, used, xlabel, ylabel = choose_axes(bands)
    if kind is None:
        return None
    mags, keep = select(table, used, snr_min)
    n_keep = int(keep.sum())
    if n_keep < 20:
        return {'kind': kind, 'bands': list(used), 'n': n_keep,
                'n_total': len(table), 'drawn': False,
                'xlabel': xlabel, 'ylabel': ylabel, 'snr_min': snr_min}

    if kind == 'ccd':
        xb, xr, yb, yr = used
        x = mags[xb][keep] - mags[xr][keep]
        y = mags[yb][keep] - mags[yr][keep]
    else:
        b, r = used
        x = mags[b][keep] - mags[r][keep]
        y = mags[r][keep]

    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    # Hexbin, not scatter: at 10^4-10^5 points a scatter is a black blob whose
    # shape is set by the alpha and hides the structure the diagram is for.
    hb = ax.hexbin(x, y, gridsize=90, bins='log', mincnt=1, linewidths=0,
                   cmap='magma')
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label('stars per bin', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    # Percentile limits: a handful of extreme colours otherwise set the range
    # and squeeze the locus into a few pixels.
    ax.set_xlim(*np.percentile(x, [0.2, 99.8]))
    if kind == 'cmd':
        ax.set_ylim(*np.percentile(y, [99.8, 0.2]))     # brighter upward
    else:
        ax.set_ylim(*np.percentile(y, [0.2, 99.8]))
    ax.set_xlabel(xlabel + '  (Vega)', fontsize=9)
    ax.set_ylabel(ylabel + '  (Vega)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title('%s — %s' % (field, 'colour–colour' if kind == 'ccd'
                              else 'colour–magnitude'), fontsize=10)
    # Boxed: the hexbin's low-density corner is nearly black under `magma`, and
    # plain grey text on it was unreadable in exactly the corner the annotation
    # defaults to.
    ax.text(0.025, 0.975, 'S/N > %g in all %d bands\n%s of %s sources'
            % (snr_min, len(set(used)), f'{n_keep:,}', f'{len(table):,}'),
            transform=ax.transAxes, va='top', ha='left', fontsize=7.5,
            color='0.15',
            bbox=dict(facecolor='white', alpha=0.82, edgecolor='0.7',
                      linewidth=0.5, boxstyle='round,pad=0.32'))
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return {'kind': kind, 'bands': list(used), 'n': n_keep,
            'n_total': len(table), 'drawn': True,
            'xlabel': xlabel, 'ylabel': ylabel, 'snr_min': snr_min}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--field', required=True)
    ap.add_argument('--version', required=True)
    ap.add_argument('--release-root', default='/orange/adamginsburg/jwst/releases')
    ap.add_argument('--snr-min', type=float, default=SNR_MIN)
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from astropy.table import Table
    from stage_release import field_release_dir, iteration_rank
    import glob

    fdir = field_release_dir(args.field, args.version, args.release_root)
    cats = [p for p in sorted(glob.glob(str(fdir / 'catalogs' / 'basic_merged*.fits')))
            if 'qualcuts' not in os.path.basename(p)]
    if not cats:
        print(f'{args.field}: no merged catalog staged in {args.version}')
        return 1
    # A multi-observation field stages ONE TABLE PER POINTING, and they are not
    # interchangeable: brick ships o001 (2221: F182M/F187N/F212N/F405N/F410M/
    # F466N) beside o004 (1182: F115W/F200W/F356W/F444W).  Picking one -- by
    # size, or by being first -- silently decides which half of the field the
    # page describes, and picking by size chose the one whose bands are NOT the
    # requested diagram.  Draw each, and put the observation in the filename so
    # the page can say which pointing it is.
    out_dir = fdir / 'preview'
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = 1
    for obs, src in select_tables(cats, iteration_rank):
        table = Table.read(src)
        out = out_dir / ('%s_ccd%s.png' % (args.field, '_' + obs if obs else ''))
        info = render(table, args.field if not obs else f'{args.field} {obs}',
                      str(out), snr_min=args.snr_min)
        tag = f'{args.field}{" " + obs if obs else ""}'
        if info is None:
            print(f'{tag}: {len(bands_in(table))} band(s) -- '
                  f'not enough for any diagram')
            continue
        if not info['drawn']:
            print(f'{tag}: only {info["n"]} sources pass the cuts -- not drawn')
            continue
        # A sidecar beside the image.  The page needs the axes, the cuts and
        # the counts to caption it, and re-deriving them there would mean a
        # second implementation of the band choice that could disagree with
        # the one that drew the plot.
        info['observation'] = obs
        info['source_table'] = os.path.basename(src)
        out.with_suffix('.json').write_text(json.dumps(info, indent=1))
        print(f'{tag}: {info["kind"]} {info["xlabel"]} vs {info["ylabel"]}, '
              f'{info["n"]:,} of {info["n_total"]:,} sources -> {out}')
        rc = 0
    return rc


if __name__ == '__main__':
    sys.exit(main())
