"""Pairwise MIRI sky levels (reduction/miri_skymatch.py).

The failure these guard against: jwst skymatch 'match' on a half-dark-cloud,
half-nebula field (10678 o078) applied up to 130 MJy/sr between frames that
agree to <2 MJy/sr, because each frame's own sky estimate inside an overlap
landed on a different peak of a bimodal scene.
"""
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from jwst_gc_pipeline.reduction.miri_skymatch import (
    DNU_BIT, pairwise_sky_levels, write_skylist)

pytest.importorskip('reproject')

SKY_NY, SKY_NX = 200, 360
PIX_DEG = 0.11 / 3600


def _scene():
    """Bimodal sky: dark cloud (20) on the left, bright nebula (150) on the
    right, plus stars."""
    rng = np.random.default_rng(1)
    sky = np.where(np.arange(SKY_NX)[None, :] < SKY_NX // 2, 20.0, 150.0)
    sky = np.broadcast_to(sky, (SKY_NY, SKY_NX)).copy()
    for _ in range(60):
        y, x = rng.integers(0, SKY_NY), rng.integers(0, SKY_NX)
        sky[y, x] += 5000.0
    return sky


def _write_frame(path, sky, x0, offset, bad_frac, seed, width=200):
    """Cut a width-wide frame starting at sky column x0, add a sky offset and
    DNU-flag a random pixel subset (differing masks between frames)."""
    rng = np.random.default_rng(seed)
    data = sky[:, x0:x0 + width] + offset + rng.normal(0, 0.5, (SKY_NY, width))
    dq = np.zeros(data.shape, dtype=np.uint32)
    dq[rng.random(data.shape) < bad_frac] = DNU_BIT
    # masked pixels carry garbage, which must be ignored
    data[dq > 0] = 1e4
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [266.5, -28.7]
    w.wcs.cdelt = [-PIX_DEG, PIX_DEG]
    # the frame's pixel (0, 0) sits at sky column x0
    w.wcs.crpix = [1 - x0 + SKY_NX / 2, SKY_NY / 2]
    hdr = w.to_header()
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(data.astype('float32'), hdr, name='SCI'),
                  fits.ImageHDU(dq, hdr, name='DQ')]).writeto(path)
    return str(path)


def test_recovers_known_offsets_on_a_bimodal_scene(tmp_path):
    sky = _scene()
    offsets = [0.0, -7.0, 12.0]
    # overlapping frames straddling the cloud/nebula edge by different amounts,
    # with different bad-pixel masks
    files = [_write_frame(tmp_path / f'f{k}.fits', sky, x0, off, bad, seed=k)
             for k, (x0, off, bad) in enumerate(zip((40, 100, 150), offsets,
                                                    (0.02, 0.2, 0.05)))]
    levels, pairs = pairwise_sky_levels(files, bin_factor=2, min_overlap=50)
    expect = np.array(offsets) - max(offsets)
    np.testing.assert_allclose(levels, expect, atol=0.5)
    assert levels.max() == 0.0
    assert len(pairs) == 3
    assert max(abs(p[4]) for p in pairs) < 0.5


def test_disjoint_frames_get_zero_levels(tmp_path):
    sky = _scene()
    files = [_write_frame(tmp_path / 'a.fits', sky, 0, 0.0, 0.0, 1, width=100),
             _write_frame(tmp_path / 'b.fits', sky, 250, 30.0, 0.0, 2, width=100)]
    levels, pairs = pairwise_sky_levels(files, bin_factor=2, min_overlap=50)
    assert pairs == []
    np.testing.assert_array_equal(levels, 0.0)


def test_skylist_rows_match_jwst_stem_convention(tmp_path):
    from jwst.lib.suffix import remove_suffix
    files = ['/x/y/jw10678078001_02201_00001_mirimage_align.fits',
             '/x/y/jw10678078001_02201_00002_mirimage_align.fits']
    path = write_skylist(files, [-1.5, 0.0], tmp_path / 'sky.txt')
    tbl = np.genfromtxt(path, dtype=[('fname', '<S128'), ('sky', 'f')])
    stems = [remove_suffix(Path(f).stem)[0] for f in tbl['fname'].astype(str)]
    # skymatch compares these to remove_suffix(Path(model.meta.filename).stem);
    # meta.filename of a frame read from disk is its basename
    assert stems == [remove_suffix(Path(Path(f).name).stem)[0] for f in files]
    np.testing.assert_allclose(tbl['sky'], [-1.5, 0.0])


def test_pipeline_uses_pairwise_skylist_by_default():
    src = Path(__file__).parents[1] / 'reduction' / 'PipelineMIRI.py'
    text = src.read_text()
    assert "os.getenv('MIRI_SKYMATCH', 'pairwise') == 'pairwise'" in text
    assert "'skymethod': 'user'" in text
