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
            overwrite=True, client=None, threaded=False, **import_kwargs):
    """Convert a flat Parquet catalog to a HATS catalog directory.

    Parameters mirror ``hats_import.catalog.arguments.ImportArguments`` (verified
    against hats-import 0.10.x, which has no ``overwrite`` kwarg -- so ``overwrite``
    here clears an existing output dir first).  Extra ``import_kwargs`` pass through
    to ``ImportArguments`` (e.g. ``dask_n_workers``).  Returns
    ``out_dir/catalog_name``.  Run on the ``.parquet`` from
    ``catalog_assembly.write_outputs``; a ``skycoord`` mixin is expanded to
    ``<name>_ra``/``<name>_dec`` there (e.g. ``skycoord_ref_ra``).
    """
    _require_hats_import()
    from hats_import.catalog.arguments import ImportArguments
    from hats_import.pipeline import pipeline, pipeline_with_client

    dest = os.path.join(out_dir, catalog_name)
    if overwrite and os.path.isdir(dest):
        import shutil
        shutil.rmtree(dest)

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
        **import_kwargs,
    )
    # A process-based Dask Nanny won't start on many login/compute nodes; a
    # THREADED client (processes=False, no Nanny) is the robust default there.
    if client is not None:
        pipeline_with_client(args, client)
    elif threaded:
        from dask.distributed import Client
        with Client(n_workers=1, threads_per_worker=4, processes=False,
                    dashboard_address=None) as _c:
            pipeline_with_client(args, _c)
    else:
        pipeline(args)
    return dest


def build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog='python -m jwst_gc_pipeline.cmz.hats_export',
        description='Export a flat Parquet CMZ catalog to HATS (for LSDB).')
    p.add_argument('--parquet', required=True, help='flat parquet catalog')
    p.add_argument('--out', required=True, help='output directory (HATS root)')
    p.add_argument('--name', required=True, help='catalog name')
    # The CMZ catalog's coordinate is skycoord_ref -> parquet 'skycoord_ref_ra'.
    p.add_argument('--ra-col', default='skycoord_ref_ra')
    p.add_argument('--dec-col', default='skycoord_ref_dec')
    # The CMZ is extremely crowded (a single order-7 pixel can hold >600k
    # sources), so partitions must subdivide deep; 12 keeps them under threshold.
    p.add_argument('--highest-order', type=int, default=12)
    p.add_argument('--pixel-threshold', type=int, default=1_000_000)
    p.add_argument('--threaded', action='store_true',
                   help='use a threaded Dask client (no process Nanny) -- needed '
                        'on login/compute nodes where the Nanny fails to start')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = to_hats(args.parquet, args.out, args.name, ra_col=args.ra_col,
                  dec_col=args.dec_col, highest_order=args.highest_order,
                  pixel_threshold=args.pixel_threshold, threaded=args.threaded)
    print(f"HATS catalog written -> {out}")


if __name__ == '__main__':
    main()
