"""CMZ-wide release tooling: growing two-color HiPS + giant-catalog sharing.

Implements ``scripts/release/CMZ_HIPS_AND_CATALOG_SHARING_PLAN.md``.

Pure-Python is the default path everywhere.  The two-color pair is **F212N (blue)
+ F480M (red)** — both CMZ-wide with program 10678; F405N is a legacy per-field
fallback.

Modules:
  * ``catalog_assembly`` -- assemble a CMZ-wide catalog from per-field combined
    catalogs (vstack + cross-field dedup + provenance + coverage); write
    FITS/ECSV/Parquet.  (pure-Python; no optional deps)
  * ``hips`` -- mono-per-filter HiPS substrate + **pure-Python incremental merge**
    (per-order tile combine, no pyramid re-derivation) + derived two-color HiPS
    (R=F480M, B=F212N, G=0.5*(R+B), global stretch).  (needs ``reproject`` +
    ``Pillow``)
  * ``coverage_moc`` -- survey coverage MOC from _i2d footprints.  (needs ``mocpy``)
  * ``hats_export`` -- HATS/partitioned-Parquet export for LSDB.  (needs
    ``hats-import``)
  * ``hipsgen`` -- OPTIONAL CDS Hipsgen.jar / Hipsgen-cat.jar drivers; used only
    for the progressive CATALOG HiPS (no pure-Python equivalent) or as an
    alternative image builder.  (needs Java + the jars)

The three catalog layers are complementary, not alternatives:
  coverage_moc = WHERE data is · hipsgen catalog HiPS = VIEW in Aladin ·
  hats_export = ANALYSE/distribute/cross-match at scale.
"""

__all__ = ['catalog_assembly', 'hips', 'coverage_moc', 'hats_export', 'hipsgen']
