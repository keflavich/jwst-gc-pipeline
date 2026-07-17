"""GCTAG is stamped alongside GCPIPEV by the global FITS-write hook."""
import numpy as np
from astropy.io import fits

from jwst_gc_pipeline import provenance
from jwst_gc_pipeline.versioning import tags


def test_stamp_header_sets_both_keys():
    h = fits.Header()
    provenance.stamp_header(h)
    assert provenance.GCPIPEV_KEY in h
    assert provenance.GCTAG_KEY in h
    # GCTAG must be a parseable release or dev tag (or the commit fallback).
    val = h[provenance.GCTAG_KEY]
    assert tags.parse_tag(val) is not None or val == provenance.get_pipeline_commit()


def test_global_hook_stamps_gctag_on_write(tmp_path):
    provenance.install_fits_provenance_hook()  # idempotent
    p = str(tmp_path / 'hooked.fits')
    fits.HDUList([fits.PrimaryHDU(np.zeros((2, 2)))]).writeto(p)
    with fits.open(p) as hdul:
        assert provenance.GCTAG_KEY in hdul[0].header
        assert provenance.GCPIPEV_KEY in hdul[0].header
