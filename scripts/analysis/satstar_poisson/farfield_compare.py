"""Far-field chi floor of per-frame m7 residuals, STPSF versus hybrid PSF (issue #996).

Issue #996 measured a robust sigma(chi) of 1.1-1.3 away from the saturated target
with the leave-one-dither-out static-pattern model of ``loo_crop.py``, whose field
stars use neither STPSF nor the hybrid ePSF/STPSF library.  This script asks the
question that remains: does rendering the field stars with the hybrid library
(``photometry/epsf_hybrid.py``) lower sigma(chi) next to field stars?

For each frame it reads the m7 ``*_daophot_basic_residual.fits`` and ``*_model.fits``
products of two PSF variants and reports the robust sigma(chi) in bins of distance
to the nearest field star (model peak above ``--peak-threshold``, the definition
used in ``make_evidence_figures.py``, fig 10)::

    python farfield_compare.py --dir /orange/.../F480M/pipeline \\
        jw10678-o061_t001_nircam_clear-f480m-nrcblong_visit001_vgroup02101_exp00004_resbgsub \\
        jw10678-o061_t001_nircam_clear-f480m-nrcblong_visit001_vgroup02101_exp00005_resbgsub

Each frame stem is the filename up to and including ``_resbgsub``; the variants are
``<stem><suffix>_residual.fits`` with suffix ``_m7_daophot_basic`` (STPSF) and
``_hybpsf_m7_daophot_basic`` (hybrid).  Both suffixes can be overridden.

What the statistic is, and is not:

* chi = (residual - local median) / ERR, in sample, with the pipeline error image.
  The m7 residual keeps the sky, so a median filter (``--bg-size`` px) removes it.
  ``loo_crop.py`` used held-out dithers with Poisson plus read noise.  Absolute
  values differ from #996; only the change between variants in each bin is
  comparable.
* Both variants must come from the same frames and the same masks.  They are
  written on different dates, so other pipeline changes can enter the comparison.
  The DATE, GCTAG and GCPIPEV (pipeline commit) keywords of the primary header are
  printed to make that visible.  A residual header carries no keyword for the PSF
  variant itself, so the variant is known only from the file name.
* The robust sigma is 1.4826 * median(|chi|), uncentred, as in ``loo_crop.py``.
"""
import argparse
import json
import os
import sys

import numpy as np
from astropy.io import fits
from scipy import ndimage

#: DQ bits excluded from every bin: DO_NOT_USE (1), SATURATED (2), JUMP_DET (4).
BAD_DQ_BITS = 1 | 2 | 4
SATURATED_BIT = 2

#: distance-to-field-star bins [px], as in #996: core, ring, far.
DEFAULT_BINS = ((0.0, 6.0), (6.0, 10.0), (10.0, np.inf))

#: primary-header keywords printed per variant, to expose differing code revisions.
PROVENANCE_KEYS = ('DATE', 'GCTAG', 'GCPIPEV')

STPSF_SUFFIX = '_m7_daophot_basic'
HYBRID_SUFFIX = '_hybpsf_m7_daophot_basic'


def robust_sigma(x):
    """1.4826 * median(|x|) of the finite values; NaN for an empty selection."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(1.4826 * np.median(np.abs(x)))


def field_star_peaks(model, threshold=60.0, size=7):
    """Boolean map of local maxima of ``model`` above ``threshold`` (MJy/sr)."""
    model = np.nan_to_num(np.asarray(model, dtype=float))
    return (model == ndimage.maximum_filter(model, size)) & (model > threshold)


def distance_to_peaks(peaks):
    """Euclidean distance [px] to the nearest True pixel; inf if there is none."""
    if not peaks.any():
        return np.full(peaks.shape, np.inf)
    return ndimage.distance_transform_edt(~peaks)


def highpass_chi(resid, err, bad, bg_size=15):
    """(resid - local median) / err with ``bad`` pixels set to NaN.

    The median filter runs on an array whose bad pixels carry the global median, so
    that masked values do not enter the local estimate.
    """
    resid = np.asarray(resid, dtype=float)
    err = np.asarray(err, dtype=float)
    ok = np.isfinite(resid) & np.isfinite(err) & (err > 0) & ~bad
    if not ok.any():
        return np.full(resid.shape, np.nan)
    filled = np.where(ok, resid, np.median(resid[ok]))
    local = ndimage.median_filter(filled, size=bg_size)
    chi = np.full(resid.shape, np.nan)
    chi[ok] = (resid[ok] - local[ok]) / err[ok]
    return chi


def farfield_sigma(resid, err, dq, model, bins=DEFAULT_BINS, peak_threshold=60.0,
                   bg_size=15, sat_radius=15.0):
    """Robust sigma(chi) in bins of distance to the nearest field star.

    Pixels flagged in ``BAD_DQ_BITS`` and pixels within ``sat_radius`` px of a
    saturated pixel are excluded.  Returns a list of dicts with the bin edges, the
    pixel count and sigma.
    """
    dq = np.asarray(dq)
    saturated = (dq & SATURATED_BIT) > 0
    bad = (dq & BAD_DQ_BITS) > 0
    chi = highpass_chi(resid, err, bad, bg_size=bg_size)
    if saturated.any() and sat_radius > 0:
        near_sat = ndimage.distance_transform_edt(~saturated) <= sat_radius
        chi = np.where(near_sat, np.nan, chi)
    dist = distance_to_peaks(field_star_peaks(model, peak_threshold))
    rows = []
    for lo, hi in bins:
        # an open-ended bin also takes the all-inf distance map of a frame with no star
        below = (dist < hi) if np.isfinite(hi) else np.ones(dist.shape, bool)
        sel = np.isfinite(chi) & (dist >= lo) & below
        rows.append({'lo': float(lo), 'hi': float(hi), 'n': int(sel.sum()),
                     'sigma': robust_sigma(chi[sel])})
    return rows


def load_frame(path_residual, path_model):
    """(resid, err, dq, model, meta) from an m7 residual and model file pair.

    ``meta`` holds the DATE, GCTAG and GCPIPEV keywords of the residual primary header
    ('' where absent).
    """
    with fits.open(path_residual) as res:
        resid = res['SCI'].data.astype(float)
        err = res['ERR'].data.astype(float)
        dq = res['DQ'].data.astype(np.int64)
        meta = {key: str(res[0].header.get(key, '')) for key in PROVENANCE_KEYS}
    with fits.open(path_model) as mod:
        model = mod['SCI'].data.astype(float)
    return resid, err, dq, model, meta


def variant_paths(directory, stem, suffix):
    base = os.path.join(directory, stem + suffix)
    return base + '_residual.fits', base + '_model.fits'


def compare_frame(directory, stem, suffixes, **kwargs):
    """{variant label: {'date', 'gctag', 'gcpipev', 'bins'}} for one frame."""
    out = {}
    for label, suffix in suffixes.items():
        resid_path, model_path = variant_paths(directory, stem, suffix)
        for p in (resid_path, model_path):
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        resid, err, dq, model, meta = load_frame(resid_path, model_path)
        out[label] = {'date': meta['DATE'], 'gctag': meta['GCTAG'],
                      'gcpipev': meta['GCPIPEV'],
                      'bins': farfield_sigma(resid, err, dq, model, **kwargs)}
    return out


def _bin_label(row):
    hi = 'inf' if not np.isfinite(row['hi']) else f"{row['hi']:g}"
    return f"{row['lo']:g}-{hi} px"


def format_report(results):
    """Plain-text table: one line per frame and bin, variants side by side."""
    lines = []
    for stem, variants in results.items():
        labels = list(variants)
        lines.append(stem)
        for label in labels:
            v = variants[label]
            lines.append(f"  {label}: DATE {v['date']}  GCTAG {v['gctag']}  GCPIPEV {v['gcpipev']}")
        if len(labels) == 2 and variants[labels[0]]['gcpipev'] != variants[labels[1]]['gcpipev']:
            lines.append('  note: the variants come from different pipeline commits')
        head = '  {:<12s}'.format('bin') + ''.join(f'{lab:>14s}' for lab in labels)
        if len(labels) == 2:
            head += f'{"ratio":>10s}'
        lines.append(head)
        nbin = len(variants[labels[0]]['bins'])
        for i in range(nbin):
            row0 = variants[labels[0]]['bins'][i]
            vals = [variants[lab]['bins'][i]['sigma'] for lab in labels]
            line = '  {:<12s}'.format(_bin_label(row0)) + ''.join(f'{v:14.3f}' for v in vals)
            if len(labels) == 2:
                line += f'{vals[1] / vals[0]:10.3f}'
            lines.append(line + f'   n={row0["n"]}')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('stems', nargs='+', help='frame stems, up to and including _resbgsub')
    ap.add_argument('--dir', default='.', help='directory with the m7 products')
    ap.add_argument('--stpsf-suffix', default=STPSF_SUFFIX)
    ap.add_argument('--hybrid-suffix', default=HYBRID_SUFFIX)
    ap.add_argument('--peak-threshold', type=float, default=60.0,
                    help='model peak [MJy/sr] defining a field star (default 60)')
    ap.add_argument('--bg-size', type=int, default=15,
                    help='median-filter box [px] removing the sky (default 15)')
    ap.add_argument('--sat-radius', type=float, default=15.0,
                    help='exclude pixels within this distance [px] of a saturated pixel')
    ap.add_argument('--json', help='also write the numbers to this file')
    args = ap.parse_args(argv)

    suffixes = {'stpsf': args.stpsf_suffix, 'hybrid': args.hybrid_suffix}
    results = {}
    for stem in args.stems:
        results[stem] = compare_frame(args.dir, stem, suffixes,
                                      peak_threshold=args.peak_threshold,
                                      bg_size=args.bg_size, sat_radius=args.sat_radius)
    print(format_report(results))
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(results, fh, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
