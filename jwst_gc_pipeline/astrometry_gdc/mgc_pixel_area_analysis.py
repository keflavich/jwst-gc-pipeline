"""Quantify the STDGDC MGC (pixel-area magnitude correction) vs our pipeline.

Three questions, answered empirically (results in GDC_EXPERIMENT_REPORT.md,
section "MGC pixel-area magnitude correction"):

(a) How big is MGC?  Stats (median / p5 / p95 / max |mag|) of the MGC map for
    every SW detector, F212N + F182M, invalid (unmeasured-border) map cells
    masked via the forward-map sanity criterion plus an MGC sanity cut.

    NOTE (library finding, 2026-07-23): every **v2** STDGDC file stores MGC
    identically ZERO -- only the **v1** files carry the pixel-area map, so
    the peppar-style ``version='auto'`` (v2-preferred) load silently has no
    MGC.  This module therefore reads MGC from v1.  The v1 NRCB4/F212N MGC
    also has border-hole garbage (values to -3.2 mag), masked here.

(b) Does our PSF photometry already account for per-pixel area variation
    (which would make applying MGC double-counting)?  Checked two ways:
    - the PSF grid the fit uses: if every grid ePSF is normalized to unit
      sum, the model absorbs NO spatial throughput/pixel-area variation and
      ``flux_fit`` is the star's locally-summed surface brightness;
    - the flux calibration: whether a per-pixel AREA map is applied anywhere
      (vs one constant ``proj_plane_pixel_area()`` per frame).

(c) Is MGC the same information as the CRDS AREA reference file (the crf
    ``AREA`` extension = per-pixel area relative to nominal)?  Correlate
    MGC against ``2.5*log10(area_rel)`` per detector.

Usage::

    python -m jwst_gc_pipeline.astrometry_gdc.mgc_pixel_area_analysis \
        [--psf-grid /path/to/PSFgrid.fits]
"""
import argparse
import glob
import os

import numpy as np

from .stdgdc import STDGDC
from .gdc_wcs import MAX_SANE_CORRECTION_PIX

SW_DETECTORS = ('NRCA1', 'NRCA2', 'NRCA3', 'NRCA4',
                'NRCB1', 'NRCB2', 'NRCB3', 'NRCB4')

BRICK_F212N_TEMPLATE = ('/orange/adamginsburg/jwst/brick/F212N/pipeline/'
                        'jw02221001001_05101_00001_{det}_destreak_o001_crf.fits')
DEFAULT_PSF_GRID = ('/orange/adamginsburg/jwst/brick/psfs/'
                    'F212N_2221_001_merged_PSFgrid_oversample1_blur.fits')

#: A real MGC value is a few x 0.01 mag; the v1 border holes read up to
#: -3.2 mag.  Anything beyond this is an unmeasured/invalid MGC cell.
MAX_SANE_MGC_MAG = 0.5


def _valid_mask(gdc):
    """True where the forward map is measured (mirrors GDCSkySolution)."""
    ny, nx = gdc.xgc.shape
    xidx = np.arange(nx, dtype=float)[np.newaxis, :]
    yidx = np.arange(ny, dtype=float)[:, np.newaxis]
    # Map value is the corrected 1-BASED position of raw 1-based (i+1, j+1);
    # a sane correction is small, an unmeasured cell is stored as 0.
    return (np.isfinite(gdc.xgc) & np.isfinite(gdc.ygc)
            & (np.abs(gdc.xgc - 1.0 - xidx) < MAX_SANE_CORRECTION_PIX)
            & (np.abs(gdc.ygc - 1.0 - yidx) < MAX_SANE_CORRECTION_PIX))


def mgc_stats(filters=('F212N', 'F182M'), detectors=SW_DETECTORS,
              version='v1'):
    """Per-(filter, detector) MGC stats in mmag.  Returns list of dicts.

    ``version`` defaults to ``'v1'``: the v2 files store MGC == 0 everywhere
    (see module docstring).
    """
    rows = []
    for filt in filters:
        for det in detectors:
            gdc = STDGDC.load(det, filt, version=version)
            valid = _valid_mask(gdc) & (np.abs(gdc.mgc) < MAX_SANE_MGC_MAG)
            m = gdc.mgc[valid] * 1000.0  # mmag
            rows.append({
                'filter': filt, 'detector': det,
                'median': float(np.median(m)),
                'p5': float(np.percentile(m, 5)),
                'p95': float(np.percentile(m, 95)),
                'max_abs': float(np.max(np.abs(m))),
                'pk_pk': float(np.max(m) - np.min(m)),
                'n_invalid': int((~valid).sum()),
            })
    return rows


def compare_mgc_vs_area(detectors=SW_DETECTORS, template=BRICK_F212N_TEMPLATE,
                        filt='F212N', step=8, version='v1'):
    """Correlate MGC with the crf AREA extension (relative pixel area).

    The AREA extension of a cal/crf frame is the per-pixel area relative to
    the nominal ``PIXAR_SR``.  If MGC is (mostly) the pixel-area effect, then
    MGC ~ slope * 2.5*log10(area_rel) with |slope| ~ 1.  Returns per-detector
    Pearson r, best-fit slope, and the rms of the MGC residual after removing
    the area term.
    """
    from astropy.io import fits

    rows = []
    for det in detectors:
        fn = template.format(det=det.lower())
        if not os.path.exists(fn):
            rows.append({'detector': det, 'error': f'missing {fn}'})
            continue
        with fits.open(fn) as hdul:
            area = np.asarray(hdul['AREA'].data, dtype=float)
        gdc = STDGDC.load(det, filt, version=version)
        valid = _valid_mask(gdc) & (np.abs(gdc.mgc) < MAX_SANE_MGC_MAG)
        sl = (slice(None, None, step), slice(None, None, step))
        a = area[sl]
        m = gdc.mgc[sl] * 1000.0
        v = valid[sl] & np.isfinite(a) & (a > 0)
        area_mag = 2.5 * np.log10(a[v] / np.median(a[v])) * 1000.0  # mmag
        mg = m[v] - np.median(m[v])
        r = float(np.corrcoef(area_mag, mg)[0, 1])
        slope = float(np.polyfit(area_mag, mg, 1)[0])
        resid = mg - slope * area_mag
        rows.append({
            'detector': det, 'pearson_r': r, 'slope': slope,
            'mgc_rms': float(np.std(mg)),
            'area_mag_rms': float(np.std(area_mag)),
            'resid_rms': float(np.std(resid)),
            'area_pk_pk_mmag': float(np.max(area_mag) - np.min(area_mag)),
        })
    return rows


def psf_grid_normalization(grid_file=DEFAULT_PSF_GRID):
    """Sum of every ePSF in the gridded-PSF file used by the m1 fit.

    If the sums are constant (unit-normalized), the PSF model absorbs no
    spatial pixel-area/throughput variation: ``flux_fit`` is then the star's
    locally-summed surface brightness and inherits the local pixel-area error
    when calibrated with one constant pixel area.
    """
    from astropy.io import fits

    with fits.open(grid_file) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
    sums = data.reshape(data.shape[0], -1).sum(axis=1)
    return {
        'grid_file': grid_file, 'n_epsf': int(data.shape[0]),
        'sum_min': float(sums.min()), 'sum_max': float(sums.max()),
        'pk_pk_mmag': float(2.5 * np.log10(sums.max() / sums.min()) * 1000.0),
    }


def v2_mgc_max(filters=('F212N', 'F182M'), detectors=SW_DETECTORS):
    """max |MGC| (mag) over every v2 file -- 0.0 means MGC not populated."""
    from astropy.io import fits
    from .stdgdc import GDCFileNotFoundError, resolve_gdc_file

    worst = 0.0
    n = 0
    for filt in filters:
        for det in detectors:
            try:
                fn = resolve_gdc_file(det, filt, version='v2')
            except GDCFileNotFoundError:
                continue
            m = np.asarray(fits.getdata(fn, ext=3), dtype=float)
            worst = max(worst, float(np.max(np.abs(m))))
            n += 1
    return worst, n


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--psf-grid', default=None,
                        help='Gridded-PSF file to check (default: the brick '
                             'F212N oversample1 grid, plus every other brick '
                             'F212N/F182M oversample1 grid found)')
    args = parser.parse_args(argv)

    worst_v2, n_v2 = v2_mgc_max()
    print(f'v2 library MGC: max |MGC| = {worst_v2:.6f} mag over {n_v2} files '
          f'(0 => v2 MGC is unpopulated; stats below are from v1)')
    print()
    print('## (a) MGC map stats (v1; valid cells only; mmag)')
    print('| filter | detector | median | p5 | p95 | max|.| | pk-pk | '
          'invalid px |')
    print('|---|---|---|---|---|---|---|---|')
    for row in mgc_stats():
        print(f"| {row['filter']} | {row['detector']} | {row['median']:+.1f} "
              f"| {row['p5']:+.1f} | {row['p95']:+.1f} | {row['max_abs']:.1f} "
              f"| {row['pk_pk']:.1f} | {row['n_invalid']} |")

    print()
    print('## (b) PSF grid normalization (ePSF sums)')
    grids = ([args.psf_grid] if args.psf_grid else
             sorted(glob.glob('/orange/adamginsburg/jwst/brick/psfs/'
                              'F182M_2221_00?_merged_PSFgrid_oversample1*.fits')
                    + glob.glob('/orange/adamginsburg/jwst/brick/psfs/'
                                'F212N_2221_00?_merged_PSFgrid_oversample1*.fits')))
    print('| grid file | n ePSF | sum min | sum max | pk-pk (mmag) |')
    print('|---|---|---|---|---|')
    for gf in grids:
        g = psf_grid_normalization(gf)
        print(f"| {os.path.basename(g['grid_file'])} | {g['n_epsf']} "
              f"| {g['sum_min']:.6f} | {g['sum_max']:.6f} "
              f"| {g['pk_pk_mmag']:.3f} |")

    print()
    print('## (c) MGC vs crf AREA extension (brick F212N; mmag)')
    print('| detector | pearson r | slope (MGC / 2.5log10 area) | MGC rms | '
          'area-term rms | residual rms | area pk-pk (mmag) |')
    print('|---|---|---|---|---|---|---|')
    for row in compare_mgc_vs_area():
        if 'error' in row:
            print(f"| {row['detector']} | {row['error']} | | | | | |")
            continue
        print(f"| {row['detector']} | {row['pearson_r']:+.3f} "
              f"| {row['slope']:+.3f} | {row['mgc_rms']:.1f} "
              f"| {row['area_mag_rms']:.1f} | {row['resid_rms']:.1f} "
              f"| {row['area_pk_pk_mmag']:.1f} |")


if __name__ == '__main__':
    main()
