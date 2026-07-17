"""Update a field's bulk astrometry WITHOUT re-cataloging.

When the cross-exposure (relative) astrometry of a field is already correct, a
residual tie error is a SINGLE RIGID on-sky offset for the whole field.  The PSF
fits live in detector (x_fit/y_fit) space and every source's RELATIVE position is
unchanged, so correcting the tie is a pure COORDINATE RELABEL: shift every
catalog's sky positions by one (dRA, dDec) vector.  No detection, no PSF fit, no
re-seed -- the expensive cataloging steps are skipped entirely.

This is the operational form of the versioning design's ``REPROJECT`` verdict
(data facet unchanged, WCS facet changed).  Re-stamping provenance afterwards
(best-effort, if the versioning layer is installed) records exactly that.

⚠️  VALIDITY PRECONDITION -- the residual must be a UNIFORM rigid offset.
If the per-tile offset map varies across the field (some tiles need a different
shift than others), the relative astrometry is NOT globally consistent and a
single rigid shift is WRONG for part of the field (the brick-1182 failure mode:
a bulk ~0 hiding a shifted half-mosaic).  :func:`measure_bulk_offset` runs the
per-tile gate and REFUSES a non-uniform field -- do not bypass it; that field
needs a per-exposure re-tie + re-catalog, not this path.

⛔  Offsets are measured ONLY with the sanctioned density-immune estimator
(``astrometry_offsets.measure_offset``, offset-histogram stacking, swept window).
NN-median against a dense catalog is banned (see repo CLAUDE.md).
"""
import glob as _glob
import os

import numpy as np
from astropy import units as u

# on-sky offset provenance cards written into the catalog's ``.meta`` (mirrors the
# merge_catalogs APL* convention: an APPLIED bulk shift, in mas, plus when/how).
_META_CARDS = ('ABULKDRA', 'ABULKDDE', 'ABULKUTC', 'ABULKMTH', 'ABULKREF')


def _sky_from_radec(ra_deg, dec_deg):
    from astropy.coordinates import SkyCoord
    return SkyCoord(np.asarray(ra_deg, float) * u.deg,
                    np.asarray(dec_deg, float) * u.deg)


def _coordinate_targets(cat):
    """Discover the sky-coordinate representations present in ``cat``.

    Returns a list of targets, each either ``('skycoord', colname)`` for a
    SkyCoord mixin column or ``('radec', ra_col, dec_col)`` for a float degree
    pair.  Every representation of the SAME physical positions is shifted by the
    same rigid offset (so a catalog carrying both a skycoord mixin and ra/dec
    columns stays internally consistent).
    """
    from astropy.coordinates import SkyCoord
    targets = []
    # SkyCoord mixin columns.
    for c in cat.colnames:
        col = cat[c]
        if isinstance(col, SkyCoord) or getattr(col, 'info', None) is not None and \
                col.__class__.__name__ == 'SkyCoord':
            targets.append(('skycoord', c))
    # float RA/Dec pairs, by common naming.
    lower = {c.lower(): c for c in cat.colnames}
    pairs = [('ra', 'dec'), ('skycoord_ra', 'skycoord_dec'),
             ('skycoord_centroid_ra', 'skycoord_centroid_dec'),
             ('ra_deg', 'dec_deg')]
    for rlo, dlo in pairs:
        if rlo in lower and dlo in lower:
            targets.append(('radec', lower[rlo], lower[dlo]))
    return targets


def apply_rigid_offset_to_catalog(cat, dra_mas, ddec_mas):
    """Shift every sky position in ``cat`` by the on-sky ``(dra, ddec)`` in mas.

    Mutates ``cat`` in place and returns the list of columns touched.  Detector
    ``x_fit``/``y_fit`` and all relative geometry are left untouched -- this is a
    rigid coordinate relabel.  The offset is applied with
    ``SkyCoord.spherical_offsets_by`` so the cos(dec) handling is astropy's, never
    hand-rolled (the repo's ~69 mas cos(dec) bug came from hand-rolling it).

    ``(dra_mas, ddec_mas)`` follow ``measure_offset``: the on-sky vector that moves
    the catalog ONTO the reference (i.e. add the measured ``dra``/``ddec``).
    """
    dlon = float(dra_mas) * u.mas
    dlat = float(ddec_mas) * u.mas
    touched = []
    for target in _coordinate_targets(cat):
        if target[0] == 'skycoord':
            c = target[1]
            new = cat[c].spherical_offsets_by(dlon, dlat)
            cat[c] = new
            touched.append(c)
        else:
            _, rcol, dcol = target
            sc = _sky_from_radec(cat[rcol], cat[dcol]).spherical_offsets_by(dlon, dlat)
            cat[rcol] = sc.ra.deg
            cat[dcol] = sc.dec.deg
            touched += [rcol, dcol]
    return touched


def _stamp_meta(cat, dra_mas, ddec_mas, utc, method, reference):
    cat.meta['ABULKDRA'] = (float(dra_mas), 'applied bulk on-sky dRA (mas)')
    cat.meta['ABULKDDE'] = (float(ddec_mas), 'applied bulk on-sky dDec (mas)')
    cat.meta['ABULKUTC'] = (utc, 'bulk astrometry update UTC')
    cat.meta['ABULKMTH'] = (method, 'offset measurement method')
    cat.meta['ABULKREF'] = (reference, 'astrometric reference')


class NonUniformResidualError(RuntimeError):
    """The per-tile residual is not a single rigid offset -- rigid update invalid."""


def measure_bulk_offset(cat_coords, ref_coords, *, uniformity_tol_mas=15.0,
                        min_contrast=None, nx=6, ny=6, maxsep=3.0 * u.arcsec,
                        context=""):
    """Measure the field's bulk residual AND gate on spatial uniformity.

    Returns ``dict(dra, ddec, off, contrast, ok, window_arcsec, swept, grid)``
    (mas) -- the sanctioned ``measure_offset`` global tie plus the per-tile map.

    Raises :class:`NonUniformResidualError` if any OK tile's offset deviates from
    the global offset by more than ``uniformity_tol_mas`` (the rigid-update
    precondition fails -> this field needs a per-exposure re-tie + re-catalog, not
    a bulk relabel).  Also raises if no coherent global tie is found.
    """
    from jwst_gc_pipeline.photometry import astrometry_offsets as ao
    glob = ao.measure_offset(cat_coords, ref_coords, maxsep=maxsep,
                             min_contrast=min_contrast, sweep=True,
                             raise_on_low_contrast=True, context=context)
    if glob is None or not glob.get('ok'):
        raise NonUniformResidualError(
            f"no coherent global tie ({context or 'field'}); cannot bulk-update.")
    grid = ao.measure_offset_grid(cat_coords, ref_coords, nx=nx, ny=ny,
                                  maxsep=maxsep, min_contrast=min_contrast,
                                  context=context)
    g_dra, g_ddec = glob['dra'], glob['ddec']
    worst = 0.0
    for cell in grid.get('cells', []):
        if not cell or not cell.get('ok'):
            continue
        dev = float(np.hypot(cell['dra'] - g_dra, cell['ddec'] - g_ddec))
        worst = max(worst, dev)
    if worst > uniformity_tol_mas:
        raise NonUniformResidualError(
            f"per-tile residual is NON-UNIFORM ({context or 'field'}): worst tile "
            f"deviates {worst:.1f} mas from the global ({g_dra:.1f},{g_ddec:.1f}) "
            f"> tol {uniformity_tol_mas:.1f}. A single rigid shift is invalid for "
            f"part of this field -- it needs a per-exposure re-tie + re-catalog, "
            f"not a bulk relabel.")
    return {**glob, 'grid': grid, 'worst_tile_dev_mas': worst}


def _restamp_provenance(path):
    """Best-effort provenance re-stamp (data facet unchanged, WCS facet changed)."""
    try:
        from jwst_gc_pipeline.versioning import stamping as _vstamp
        from jwst_gc_pipeline.versioning import prov_sidecar as _vps
    except ImportError:
        return
    rec = _vps.read_sidecar(path)
    stage = rec.get('stage') if rec else None
    if stage:
        # preserve stage + any recorded upstream/params; the wcs facet moves.
        inp = rec.get('inputs', {})
        _vstamp.try_stamp_catalog(
            path, stage, params=inp.get('params'),
            upstream=inp.get('upstream') or None)


def apply_offset_to_catalog_file(path, dra_mas, ddec_mas, *, utc='',
                                 method='measure_offset (histogram, swept)',
                                 reference='', backup=True, dry_run=False,
                                 restamp=True):
    """Load, rigid-shift, and rewrite one catalog file; return columns touched."""
    from astropy.table import Table
    cat = Table.read(path)
    touched = apply_rigid_offset_to_catalog(cat, dra_mas, ddec_mas)
    if not touched:
        return []
    _stamp_meta(cat, dra_mas, ddec_mas, utc, method, reference)
    if dry_run:
        return touched
    if backup and not os.path.exists(path + '.pre_bulkastrom'):
        import shutil
        shutil.copy2(path, path + '.pre_bulkastrom')
    cat.write(path, overwrite=True)
    if restamp:
        _restamp_provenance(path)
    return touched


def apply_offset_to_field(catalog_paths, dra_mas, ddec_mas, **kwargs):
    """Apply the same rigid ``(dra, ddec)`` mas offset to many catalog files.

    Returns ``{path: columns_touched}``.  ``catalog_paths`` may include globs.
    """
    paths = []
    for p in catalog_paths:
        paths += sorted(_glob.glob(p)) if any(ch in p for ch in '*?[') else [p]
    out = {}
    for p in paths:
        out[p] = apply_offset_to_catalog_file(p, dra_mas, ddec_mas, **kwargs)
    return out


def _load_coords(path, ra_col, dec_col):
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    t = Table.read(path)
    if ra_col and dec_col:
        return _sky_from_radec(t[ra_col], t[dec_col])
    for target in _coordinate_targets(t):
        if target[0] == 'skycoord':
            return SkyCoord(t[target[1]])
        return _sky_from_radec(t[target[1]], t[target[2]])
    raise SystemExit(f"could not find sky coordinates in {path} "
                     f"(pass --ra-col/--dec-col)")


def build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog='python -m jwst_gc_pipeline.photometry.bulk_astrometry_update',
        description='Apply a bulk rigid astrometry correction to catalogs '
                    'WITHOUT re-cataloging (valid only for a uniform residual).')
    p.add_argument('--catalogs', nargs='+', required=True,
                   help='catalog file(s) or glob(s) to update in place')
    p.add_argument('--dra-mas', type=float,
                   help='apply this on-sky dRA (mas); with --ddec-mas, skip measuring')
    p.add_argument('--ddec-mas', type=float, help='apply this on-sky dDec (mas)')
    p.add_argument('--measure', action='store_true',
                   help='MEASURE the offset (histogram-stacked, swept) against '
                        '--reference and gate on per-tile uniformity')
    p.add_argument('--reference', help='reference catalog (VIRAC2/Gaia-only) for --measure')
    p.add_argument('--measure-catalog',
                   help='catalog whose positions are tied to the reference '
                        '(default: first of --catalogs)')
    p.add_argument('--ra-col', help='RA column in the measured catalog')
    p.add_argument('--dec-col', help='Dec column in the measured catalog')
    p.add_argument('--ref-ra-col', help='RA column in the reference')
    p.add_argument('--ref-dec-col', help='Dec column in the reference')
    p.add_argument('--uniformity-tol-mas', type=float, default=15.0,
                   help='max per-tile deviation from the global offset (default 15)')
    p.add_argument('--dry-run', action='store_true',
                   help='report what would change; write nothing')
    p.add_argument('--no-backup', action='store_true')
    p.add_argument('--no-restamp', action='store_true')
    return p


def main(argv=None):
    import datetime
    args = build_parser().parse_args(argv)
    if args.dra_mas is not None and args.ddec_mas is not None:
        dra, ddec = args.dra_mas, args.ddec_mas
        method, ref = 'explicit --dra-mas/--ddec-mas', (args.reference or 'n/a')
    elif args.measure:
        if not args.reference:
            raise SystemExit('--measure requires --reference')
        meas_cat = args.measure_catalog or args.catalogs[0]
        a = _load_coords(meas_cat, args.ra_col, args.dec_col)
        b = _load_coords(args.reference, args.ref_ra_col, args.ref_dec_col)
        res = measure_bulk_offset(a, b, uniformity_tol_mas=args.uniformity_tol_mas,
                                  context=os.path.basename(meas_cat))
        dra, ddec = res['dra'], res['ddec']
        print(f"measured bulk offset: dRA={dra:.2f} dDec={ddec:.2f} mas "
              f"(off={res['off']:.2f}, contrast={res['contrast']:.1f}, "
              f"window={res['window_arcsec']:.0f}\", swept={res['swept']}, "
              f"worst tile dev={res['worst_tile_dev_mas']:.1f} mas)")
        if res['swept']:
            print("  WARNING: offset found only after widening the window -- the "
                  "field may be grossly shifted; inspect before applying.")
        method = 'measure_offset (histogram, swept)'
        ref = os.path.basename(args.reference)
    else:
        raise SystemExit('provide --dra-mas/--ddec-mas, or --measure --reference.')

    utc = datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    results = apply_offset_to_field(
        args.catalogs, dra, ddec, utc=utc, method=method, reference=ref,
        backup=not args.no_backup, dry_run=args.dry_run,
        restamp=not args.no_restamp)
    verb = 'WOULD update' if args.dry_run else 'updated'
    for path, cols in results.items():
        print(f"{verb} {os.path.basename(path)}: {len(cols)} coord col(s) "
              f"{cols}" if cols else f"skipped {os.path.basename(path)} (no coords)")
    print(f"{verb} {sum(1 for c in results.values() if c)} catalog(s) by "
          f"dRA={dra:.2f} dDec={ddec:.2f} mas.")


if __name__ == '__main__':
    main()
