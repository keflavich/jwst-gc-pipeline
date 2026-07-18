"""CMZ-wide release tooling: growing two-color HiPS + giant-catalog sharing.

Implements ``scripts/release/CMZ_HIPS_AND_CATALOG_SHARING_PLAN.md``.

Modules:
  * ``catalog_assembly`` -- assemble a CMZ-wide catalog from per-field combined
    catalogs (vstack + cross-field dedup + provenance + coverage); write
    FITS/ECSV/Parquet.  (pure-Python; no optional deps)
  * ``hips`` -- mono-per-filter HiPS substrate (``reproject.hips``) + derived
    two-color HiPS (R=long, B=F212N, G=0.5*(R+B), global stretch).  (needs
    ``reproject`` + ``Pillow``)
  * ``coverage_moc`` -- survey coverage MOC from _i2d footprints.  (needs ``mocpy``)
  * ``hats_export`` -- HATS/partitioned-Parquet export for LSDB.  (needs
    ``hats-import``)
  * ``hipsgen`` -- CDS Hipsgen.jar / Hipsgen-cat.jar drivers (native incremental
    image HiPS, RGB HiPS, progressive catalog HiPS).  (needs Java + the jars)

The three catalog layers are complementary, not alternatives:
  coverage_moc = WHERE data is · hipsgen catalog HiPS = VIEW in Aladin ·
  hats_export = ANALYSE/distribute/cross-match at scale.
"""

__all__ = ['catalog_assembly', 'hips', 'coverage_moc', 'hats_export', 'hipsgen']
