"""Survey coverage as a MOC (Multi-Order Coverage map), via ``mocpy``.

Fills the release's missing footprint tracker: a MOC per field/filter, unioned
into a survey coverage map that (a) shows "what's covered so far" as new programs
(10678) roll in, (b) drives incremental HiPS rebuilds (the MOC of a new field =
the tiles to regenerate), and (c) overlays natively in Aladin.

``mocpy`` is an optional dependency; :func:`_require_mocpy` raises a clear
install hint if it is absent.
"""
import os


def _require_mocpy():
    try:
        from mocpy import MOC
        return MOC
    except ImportError as exc:
        raise ImportError(
            "coverage_moc needs 'mocpy' (pip install mocpy). It is an optional "
            "dependency of the CMZ release tooling.") from exc


def footprint_moc_from_i2d(i2d_path, max_depth=10, sci_ext='SCI'):
    """MOC of a mosaic's sky footprint from its WCS boundary polygon.

    Uses the outer footprint (``WCS.calc_footprint``); this is the coverage
    outline, not the per-pixel valid-data mask (good enough for a survey coverage
    map and far cheaper).  ``max_depth`` sets the MOC HEALPix order.
    """
    MOC = _require_mocpy()
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    with fits.open(i2d_path) as hdul:
        hdu = hdul[sci_ext] if sci_ext in [h.name for h in hdul] else hdul[0]
        w = WCS(hdu.header)
    corners = w.calc_footprint()  # (N,2) RA,Dec deg, in order around the boundary
    sky = SkyCoord(corners[:, 0] * u.deg, corners[:, 1] * u.deg)
    return MOC.from_polygon_skycoord(sky, max_depth=max_depth)


def union_moc(mocs):
    """Union a list of MOCs into one coverage MOC."""
    _require_mocpy()
    if not mocs:
        raise ValueError('no MOCs to union')
    out = mocs[0]
    for m in mocs[1:]:
        out = out.union(m)
    return out


def build_survey_coverage(i2d_paths, max_depth=10, sci_ext='SCI'):
    """Union coverage MOC over many ``_i2d`` mosaics."""
    mocs = [footprint_moc_from_i2d(p, max_depth=max_depth, sci_ext=sci_ext)
            for p in i2d_paths]
    return union_moc(mocs)


def write_moc(moc, path):
    """Write a MOC to ``.fits`` (Aladin-readable) or ``.json``."""
    if path.endswith('.json'):
        moc.save(path, format='json', overwrite=True)
    else:
        moc.save(path, format='fits', overwrite=True)
    return path


def build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog='python -m jwst_gc_pipeline.cmz.coverage_moc',
        description='Build a survey coverage MOC from _i2d mosaics.')
    p.add_argument('--i2d', nargs='+', required=True, help='_i2d file(s)/glob(s)')
    p.add_argument('--out', required=True, help='output MOC (.fits or .json)')
    p.add_argument('--max-depth', type=int, default=10)
    return p


def main(argv=None):
    import glob
    args = build_parser().parse_args(argv)
    paths = []
    for pat in args.i2d:
        paths += sorted(glob.glob(pat)) if any(c in pat for c in '*?[') else [pat]
    if not paths:
        raise SystemExit('no _i2d files matched')
    moc = build_survey_coverage(paths, max_depth=args.max_depth)
    write_moc(moc, args.out)
    frac = moc.sky_fraction if hasattr(moc, 'sky_fraction') else float('nan')
    print(f"coverage MOC over {len(paths)} mosaic(s) -> {args.out} "
          f"(sky fraction {frac:.3e})")


if __name__ == '__main__':
    main()
