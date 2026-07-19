#!/usr/bin/env python
"""Orchestrate the CMZ-wide release products from a spec.

Ties together ``jwst_gc_pipeline.cmz``:
  1. assemble the CMZ-wide catalog (FITS/ECSV/Parquet)
  2. survey coverage MOC (optional; needs mocpy)
  3. incremental mono HiPS per band (F212N + long) + derived two-color HiPS
  4. HATS export (optional; needs hats-import)

Driven by a JSON spec so it re-runs cheaply as fields (10678) roll in -- the HiPS
step is incremental (only tiles a new field overlaps rewrite), and the catalog /
MOC re-assemble from the current member set.

Each step is independent and fail-soft on a missing OPTIONAL dependency (it logs
and skips, so a minimal env still produces the catalog + HiPS).  Nothing here runs
during a normal pipeline reduction; it is release-time tooling.

Spec (JSON)::

    {
      "version": "v1.2-2026.08",
      "out_dir": "/orange/adamginsburg/jwst/releases/v1.2-2026.08/cmz",
      "dedup_radius_arcsec": 0.2,
      "hats": false,
      "fields": [
        {"field": "brick", "program": "2221", "obsid": "001",
         "catalog": ".../basic_merged_..._resbgsub_m7...fits",
         "f212n_i2d": ".../jw02221-o001_..._f212n-merged_i2d.fits",
         "long_i2d":  ".../jw02221-o001_..._f405n-merged_i2d.fits",
         "long_band": "F405N"},
        ...
      ]
    }

``long_band`` is F480M for 10678-era fields, F405N for legacy fields (the
two-color derive uses the F480M HiPS where present, F405N to fill legacy gaps).
"""
import argparse
import json
import os

STEPS = ('catalog', 'moc', 'hips', 'hats')


def _log(msg):
    print(f"[build_cmz] {msg}", flush=True)


def build_catalog(spec, out_dir, dry_run=False):
    from jwst_gc_pipeline.cmz import catalog_assembly as CA
    specs = [{'path': f['catalog'], 'field': f['field'],
              'program': f.get('program', ''), 'obsid': f.get('obsid', ''),
              'tag': f.get('tag')}
             for f in spec['fields'] if f.get('catalog')]
    if not specs:
        _log('no catalog paths in spec; skipping catalog step')
        return None
    stem = os.path.join(out_dir, 'cmz_catalog')
    _log(f"assembling catalog from {len(specs)} field(s) -> {stem}.*")
    if dry_run:
        return stem
    table = CA.assemble_from_paths(
        specs, dedup_radius_arcsec=spec.get('dedup_radius_arcsec', 0.2))
    written = CA.write_outputs(table, stem)
    _log(f"catalog: {len(table)} sources ({table.meta['CMZNFLD']} fields) "
         f"-> {', '.join(os.path.basename(w) for w in written)}")
    return stem


def build_moc(spec, out_dir, dry_run=False):
    try:
        from jwst_gc_pipeline.cmz import coverage_moc as CM
        CM._require_mocpy()
    except ImportError as exc:
        _log(f"MOC step SKIPPED ({exc})")
        return None
    for band, key in (('f212n', 'f212n_i2d'), ('long', 'long_i2d')):
        paths = [f[key] for f in spec['fields'] if f.get(key)]
        if not paths:
            continue
        out = os.path.join(out_dir, f'cmz_{band}_coverage.fits')
        _log(f"coverage MOC ({band}) over {len(paths)} mosaic(s) -> {out}")
        if not dry_run:
            CM.write_moc(CM.build_survey_coverage(paths), out)
    return out_dir


def build_hips(spec, out_dir, dry_run=False):
    try:
        from jwst_gc_pipeline.cmz import hips as HP
    except ImportError as exc:
        _log(f"HiPS step SKIPPED ({exc})")
        return None
    hips_dir = os.path.join(out_dir, 'hips')
    blue = os.path.join(hips_dir, 'F212N')
    # long band(s): keep F480M and F405N as separate mono HiPS; two-color prefers
    # F480M and fills legacy gaps from F405N.
    long_dirs = {}
    for f in spec['fields']:
        if f.get('f212n_i2d'):
            _log(f"HiPS F212N += {f['field']}")
            if not dry_run:
                HP.add_field_to_mono_hips(blue, [f['f212n_i2d']], f['field'],
                                          tag=f.get('tag'))
        if f.get('long_i2d'):
            band = (f.get('long_band') or 'LONG').upper()
            ld = os.path.join(hips_dir, band)
            long_dirs.setdefault(band, ld)
            _log(f"HiPS {band} += {f['field']}")
            if not dry_run:
                HP.add_field_to_mono_hips(ld, [f['long_i2d']], f['field'],
                                          tag=f.get('tag'))
    # derive two-color from F212N + preferred long band (F480M if built, else F405N)
    red = long_dirs.get('F480M') or long_dirs.get('F405N') or next(
        iter(long_dirs.values()), None)
    if red is not None:
        color = os.path.join(hips_dir, 'CMZ_color')
        _log(f"derive two-color HiPS (B={blue}, R={red}) -> {color}")
        if not dry_run:
            HP.derive_two_color_hips(blue, red, color)
        if long_dirs.get('F480M') and long_dirs.get('F405N'):
            _log("note: both F480M and F405N present; color uses F480M. Fill "
                 "legacy-only regions from F405N in a follow-up merge if needed.")
    return hips_dir


def build_hats(spec, out_dir, catalog_stem, dry_run=False):
    if not spec.get('hats'):
        _log("HATS step not requested (spec.hats=false)")
        return None
    try:
        from jwst_gc_pipeline.cmz import hats_export as HE
        HE._require_hats_import()
    except ImportError as exc:
        _log(f"HATS step SKIPPED ({exc})")
        return None
    if not catalog_stem:
        _log("HATS step needs the catalog; skipping")
        return None
    parquet = catalog_stem + '.parquet'
    if not os.path.exists(parquet) and not dry_run:
        _log(f"HATS step needs {parquet} (pyarrow); skipping")
        return None
    out = os.path.join(out_dir, 'hats')
    _log(f"HATS export {parquet} -> {out}/cmz_jwst")
    if not dry_run:
        HE.to_hats(parquet, out, 'cmz_jwst', ra_col='ra', dec_col='dec')
    return out


def run(spec, only=None, dry_run=False):
    out_dir = spec['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    steps = only or STEPS
    catalog_stem = None
    if 'catalog' in steps:
        catalog_stem = build_catalog(spec, out_dir, dry_run=dry_run)
    if 'moc' in steps:
        build_moc(spec, out_dir, dry_run=dry_run)
    if 'hips' in steps:
        build_hips(spec, out_dir, dry_run=dry_run)
    if 'hats' in steps:
        if catalog_stem is None:
            catalog_stem = os.path.join(out_dir, 'cmz_catalog')
        build_hats(spec, out_dir, catalog_stem, dry_run=dry_run)
    _log(f"done ({'dry-run' if dry_run else 'built'}): {out_dir}")
    return out_dir


def build_parser():
    p = argparse.ArgumentParser(
        prog='python scripts/release/build_cmz_products.py',
        description='Build the CMZ-wide release products from a spec.')
    p.add_argument('--spec', required=True, help='JSON spec (see module docstring)')
    p.add_argument('--only', help='comma-separated subset of: ' + ','.join(STEPS))
    p.add_argument('--dry-run', action='store_true',
                   help='log the plan; build nothing')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    only = tuple(s.strip() for s in args.only.split(',')) if args.only else None
    if only:
        bad = set(only) - set(STEPS)
        if bad:
            raise SystemExit(f'--only unknown step(s): {sorted(bad)}; '
                             f'valid: {STEPS}')
    with open(args.spec) as fh:
        spec = json.load(fh)
    run(spec, only=only, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
