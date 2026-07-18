"""Export the assembled CMZ catalog to HATS (partitioned Parquet) for LSDB.

HATS (Hierarchical Adaptive Tiling Scheme, formerly hipscat) is the LINCC
Frameworks format: HEALPix-partitioned Parquet with ~equal objects per partition,
read by LSDB for out-of-core cross-matching and analytics (the format Rubin /
STScI / IPAC / CDS / ESA are standardizing on).  This is the SCALABLE
distribution path; the flat FITS/ECSV/Parquet from ``catalog_assembly`` remain
the "one file" deliverables.

``hats_import`` is an optional dependency (pip install hats-import).
"""
import os


def _require_hats_import():
    try:
        import hats_import  # noqa: F401
        return True
    except ImportError as exc:
        raise ImportError(
            "hats_export needs 'hats-import' (pip install hats-import) and "
            "'lsdb' to query. Optional dependency of the CMZ release tooling."
        ) from exc


def to_hats(parquet_path, out_dir, catalog_name, ra_col='ra', dec_col='dec',
            lowest_order=0, highest_order=7, pixel_threshold=1_000_000,
            overwrite=True):
    """Convert a flat Parquet catalog to a HATS catalog directory.

    Parameters mirror ``hats_import.catalog.arguments.ImportArguments``.  Returns
    ``out_dir/catalog_name``.  Run this on the ``.parquet`` written by
    ``catalog_assembly.write_outputs`` (its coordinate columns are flat floats;
    for a ``skycoord`` mixin catalog the parquet writer expands it to
    ``skycoord_ra``/``skycoord_dec``).
    """
    _require_hats_import()
    from hats_import.catalog.arguments import ImportArguments
    from hats_import.pipeline import pipeline

    args = ImportArguments(
        output_artifact_name=catalog_name,
        input_file_list=[parquet_path],
        file_reader='parquet',
        ra_column=ra_col,
        dec_column=dec_col,
        output_path=out_dir,
        lowest_healpix_order=lowest_order,
        highest_healpix_order=highest_order,
        pixel_threshold=pixel_threshold,
        overwrite=overwrite,
    )
    pipeline(args)
    return os.path.join(out_dir, catalog_name)


def build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog='python -m jwst_gc_pipeline.cmz.hats_export',
        description='Export a flat Parquet CMZ catalog to HATS (for LSDB).')
    p.add_argument('--parquet', required=True, help='flat parquet catalog')
    p.add_argument('--out', required=True, help='output directory (HATS root)')
    p.add_argument('--name', required=True, help='catalog name')
    p.add_argument('--ra-col', default='ra')
    p.add_argument('--dec-col', default='dec')
    p.add_argument('--highest-order', type=int, default=7)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = to_hats(args.parquet, args.out, args.name, ra_col=args.ra_col,
                  dec_col=args.dec_col, highest_order=args.highest_order)
    print(f"HATS catalog written -> {out}")


if __name__ == '__main__':
    main()
