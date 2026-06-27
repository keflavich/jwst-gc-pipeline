"""Every FITS file written in-process must carry the pipeline git commit in the
primary-header keyword ``GCPIPEV`` (provenance).  Verifies the import-time
``HDUList.writeto`` hook catches all the write paths the pipeline uses.
"""
import numpy as np
import pytest

from astropy.io import fits
from astropy.table import Table

# importing the package installs the provenance hook (jwst_gc_pipeline/__init__)
from jwst_gc_pipeline import get_pipeline_commit
from jwst_gc_pipeline.provenance import GCPIPEV_KEY


def test_commit_is_nonempty_string():
    c = get_pipeline_commit()
    assert isinstance(c, str) and c  # never empty; 'unknown' at worst


def test_hdulist_writeto_stamps(tmp_path):
    p = str(tmp_path / 'hdulist.fits')
    fits.HDUList([fits.PrimaryHDU(np.zeros((3, 3)))]).writeto(p)
    assert fits.getheader(p)[GCPIPEV_KEY] == get_pipeline_commit()


def test_single_hdu_writeto_stamps(tmp_path):
    # PrimaryHDU.writeto wraps itself in an HDUList -> same hook.
    p = str(tmp_path / 'single.fits')
    fits.PrimaryHDU(np.zeros((3, 3))).writeto(p)
    assert fits.getheader(p)[GCPIPEV_KEY] == get_pipeline_commit()


def test_fits_writeto_convenience_stamps(tmp_path):
    p = str(tmp_path / 'conv.fits')
    fits.writeto(p, np.zeros((3, 3)))
    assert fits.getheader(p)[GCPIPEV_KEY] == get_pipeline_commit()


def test_table_write_fits_stamps_primary_header(tmp_path):
    # Table.write(format='fits') builds an HDUList (table in ext 1) and calls
    # HDUList.writeto -> GCPIPEV lands in the primary (ext 0) header.
    p = str(tmp_path / 'table.fits')
    Table({'x': [1, 2, 3], 'y': [4.0, 5.0, 6.0]}).write(p)
    assert fits.getheader(p, ext=0)[GCPIPEV_KEY] == get_pipeline_commit()


def test_multiext_stamps_primary_only(tmp_path):
    p = str(tmp_path / 'multi.fits')
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(np.zeros((2, 2)), name='SCI')]).writeto(p)
    assert fits.getheader(p, ext=0)[GCPIPEV_KEY] == get_pipeline_commit()
