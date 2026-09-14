"""`build_mono_hips` must hand reproject a directory it is allowed to create.

`reproject_to_hips` does `os.makedirs(output_directory, exist_ok=False)` --
its own comment reads "Create output directory (and error if it already
exists)" (reproject 0.19).  `build_mono_hips` pre-created that directory, so
every call raised `FileExistsError` before any reprojection happened.  Caught
on the treasury two-colour build: three fields, three failures, ~6 s each.
"""
import os

import pytest

from jwst_gc_pipeline.cmz import hips


class _Recorder:
    """Stands in for reproject_to_hips, with the same directory contract."""

    def __init__(self):
        self.called_with = None

    def __call__(self, input_data, **kwargs):
        out = kwargs['output_directory']
        os.makedirs(out, exist_ok=False)      # the real contract
        self.called_with = kwargs
        return None


def _run(tmp_path, out_dir, monkeypatch):
    rec = _Recorder()
    # `build_mono_hips` imports the name INSIDE the function, so patching the
    # attribute on this module never intercepts it -- patch where it is looked
    # up from.
    import reproject.hips
    monkeypatch.setattr(reproject.hips, 'reproject_to_hips', rec)
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    src = tmp_path / 'in_i2d.fits'
    hdu = fits.PrimaryHDU(np.zeros((4, 4)))
    for k, v in (('CTYPE1', 'RA---TAN'), ('CTYPE2', 'DEC--TAN'),
                 ('CRVAL1', 266.0), ('CRVAL2', -29.0), ('CRPIX1', 2),
                 ('CRPIX2', 2), ('CDELT1', -1e-5), ('CDELT2', 1e-5)):
        hdu.header[k] = v
    hdu.writeto(src)
    hips.build_mono_hips(str(src), str(out_dir))
    return rec


def test_the_output_directory_is_left_for_reproject_to_create(tmp_path, monkeypatch):
    """The bug: pre-creating it made reproject raise every single time."""
    out = tmp_path / 'nested' / 'GC_132_hips'
    rec = _run(tmp_path, out, monkeypatch)
    assert rec.called_with['output_directory'] == str(out)
    assert out.is_dir(), 'reproject should have created it'


def test_a_missing_parent_is_still_created_for_the_caller(tmp_path, monkeypatch):
    """Callers pass paths inside a scratch dir; only the LEAF belongs to
    reproject."""
    out = tmp_path / 'a' / 'b' / 'c_hips'
    _run(tmp_path, out, monkeypatch)
    assert out.parent.is_dir()


def test_an_empty_leftover_directory_does_not_block_a_rebuild(tmp_path, monkeypatch):
    """An interrupted run can leave the directory behind with nothing in it;
    that must not wedge every later build."""
    out = tmp_path / 'GC_132_hips'
    out.mkdir()
    rec = _run(tmp_path, out, monkeypatch)
    assert rec.called_with is not None


def test_a_populated_tree_is_refused_rather_than_overwritten(tmp_path, monkeypatch):
    """A tree with tiles in it is someone's product.  Refuse, and name the path
    -- reproject's own error names a temp directory that tells a reader nothing
    about which field failed."""
    out = tmp_path / 'GC_132_hips'
    (out / 'Norder3').mkdir(parents=True)
    with pytest.raises(FileExistsError) as e:
        _run(tmp_path, out, monkeypatch)
    assert 'GC_132_hips' in str(e.value)
    assert 'not empty' in str(e.value)


def test_a_png_request_fails_loudly_rather_than_writing_fits(tmp_path, monkeypatch):
    """0.19 infers the format, so a png request cannot be honoured -- and
    silently writing FITS would hand the caller a tree of the wrong type with
    no error."""
    import numpy as np
    from astropy.io import fits
    src = tmp_path / 'in_i2d.fits'
    hdu = fits.PrimaryHDU(np.zeros((4, 4)))
    for k, v in (('CTYPE1', 'RA---TAN'), ('CTYPE2', 'DEC--TAN'),
                 ('CRVAL1', 266.0), ('CRVAL2', -29.0), ('CRPIX1', 2),
                 ('CRPIX2', 2), ('CDELT1', -1e-5), ('CDELT2', 1e-5)):
        hdu.header[k] = v
    hdu.writeto(src)
    with pytest.raises(ValueError) as e:
        hips.build_mono_hips(str(src), str(tmp_path / 'o'), tile_format='png')
    assert 'png' in str(e.value)


def test_tile_format_is_not_forwarded_to_reproject(tmp_path, monkeypatch):
    """It is not a reproject_to_hips parameter in 0.19; forwarding it lands it
    in **kwargs and reproject_interp rejects it."""
    out = tmp_path / 'x_hips'
    rec = _run(tmp_path, out, monkeypatch)
    assert 'tile_format' not in rec.called_with
