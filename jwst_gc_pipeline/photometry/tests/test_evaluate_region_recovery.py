"""Unit tests for the region-file recovery benchmark.

Pins the two properties the Brick selected-stars scoring depends on:
region parsing (points + box centers) and ONE-TO-ONE greedy matching --
in a sub-FWHM clump a single catalog row must never "recover" two targets.
"""
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.evaluate_region_recovery import (
    load_region_targets, match_targets)

REG = """# Region file format: DS9
global color=green
icrs
box(266.530800, -28.737287, 1.0000", 1.0000", 0) # color=#2EE6D6
point(266.543579, -28.712754) # width=2
point(266.543599, -28.712703)
"""


def _write(tmp_path, text=REG):
    p = tmp_path / 't.reg'
    p.write_text(text)
    return str(p)


def test_load_region_targets(tmp_path):
    sc, kinds = load_region_targets(_write(tmp_path))
    assert kinds == ['box', 'point', 'point']
    assert np.isclose(sc[1].ra.deg, 266.543579)
    sc2, kinds2 = load_region_targets(_write(tmp_path), include_boxes=False)
    assert kinds2 == ['point', 'point']


def test_match_targets_one_to_one(tmp_path):
    """Two targets 0.18\" apart, ONE catalog source between them: exactly one
    target may claim it (the closer one)."""
    t1 = SkyCoord(266.0 * u.deg, -28.0 * u.deg)
    t2 = t1.directional_offset_by(0 * u.deg, 0.18 * u.arcsec)
    src = t1.directional_offset_by(0 * u.deg, 0.05 * u.arcsec)
    targets = SkyCoord([t1.ra, t2.ra], [t1.dec, t2.dec])
    cat = SkyCoord([src.ra], [src.dec])
    idx, sep = match_targets(targets, cat, tolerance_arcsec=0.3)
    assert (idx >= 0).sum() == 1, "one source cannot recover two targets"
    assert idx[0] == 0 and idx[1] == -1, "the closer target wins"
    assert np.isclose(sep[0], 0.05, atol=0.01)


def test_match_targets_tolerance(tmp_path):
    t1 = SkyCoord(266.0 * u.deg, -28.0 * u.deg)
    far = t1.directional_offset_by(0 * u.deg, 1.0 * u.arcsec)
    idx, sep = match_targets(SkyCoord([t1.ra], [t1.dec]),
                             SkyCoord([far.ra], [far.dec]),
                             tolerance_arcsec=0.5)
    assert idx[0] == -1
