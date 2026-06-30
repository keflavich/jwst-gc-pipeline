"""Unit tests for post-merge de-duplication (dedup_catalog)."""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry import dedup_catalog as dc


def _row(ra, dec, **bandvals):
    """bandvals like f405n=(mag, detected) -> fills mag/mask/emag/flux_jy."""
    return ra, dec, bandvals


def _build(rows, bands=('f405n', 'f187n')):
    n = len(rows)
    t = Table()
    ras = np.array([r[0] for r in rows])
    decs = np.array([r[1] for r in rows])
    t['skycoord_ref'] = SkyCoord(ra=ras * u.deg, dec=decs * u.deg)
    t['skycoord_ref_filtername'] = ['f405n'] * n
    for b in bands:
        mag = np.full(n, np.nan)
        mask = np.ones(n, bool)
        emag = np.full(n, np.nan)
        flux = np.full(n, np.nan)
        for i, r in enumerate(rows):
            if b in r[2]:
                m, det = r[2][b]
                mag[i] = m
                if det:
                    mask[i] = False
                    emag[i] = 0.02
                    flux[i] = 10 ** (-0.4 * m) * 1e-7
        t[f'mag_vega_{b}'] = mag
        t[f'mag_ab_{b}'] = mag + 0.5
        t[f'mask_{b}'] = mask
        t[f'emag_ab_{b}'] = emag
        t[f'flux_jy_{b}'] = flux
    return t


def test_complementary_pair_merges(tmp_path):
    # one star split: row A has f187n only, row B (0.04") has f405n only
    rows = [
        _row(0.0, 0.0, f187n=(15.0, True)),
        _row(0.0, 0.04 / 3600., f405n=(16.0, True)),
        _row(1.0, 1.0, f187n=(14.0, True), f405n=(14.5, True)),  # isolated
    ]
    t = _build(rows)
    p = tmp_path / 'in.fits'
    o = tmp_path / 'out.fits'
    t.write(p)
    dc.dedup_merged_catalog(str(p), str(o), verbose=False)
    out = Table.read(o)
    assert len(out) == 2  # the split pair collapsed to one
    # the merged primary carries BOTH bands
    merged = out[out['n_merged'] > 1][0]
    assert np.isfinite(merged['mag_vega_f187n'])
    assert np.isfinite(merged['mag_vega_f405n'])
    assert not bool(merged['mask_f405n'])
    assert not bool(merged['mask_f187n'])


def test_resolved_binary_preserved(tmp_path):
    # two real stars 0.05" apart, both detected in f187n with 2-mag difference
    rows = [
        _row(0.0, 0.0, f187n=(14.0, True), f405n=(15.0, True)),
        _row(0.0, 0.05 / 3600., f187n=(16.0, True)),
    ]
    t = _build(rows)
    p = tmp_path / 'in.fits'
    o = tmp_path / 'out.fits'
    t.write(p)
    dc.dedup_merged_catalog(str(p), str(o), verbose=False)
    out = Table.read(o)
    assert len(out) == 2  # NOT merged -- binary protected


def test_near_identical_collision_merges(tmp_path):
    # same star detected in f187n in both rows with ~identical mag -> merge
    rows = [
        _row(0.0, 0.0, f187n=(15.00, True), f405n=(16.0, True)),
        _row(0.0, 0.03 / 3600., f187n=(15.03, True)),
    ]
    t = _build(rows)
    p = tmp_path / 'in.fits'
    o = tmp_path / 'out.fits'
    t.write(p)
    dc.dedup_merged_catalog(str(p), str(o), dmag_collision=0.1, verbose=False)
    out = Table.read(o)
    assert len(out) == 1


def test_far_pair_untouched(tmp_path):
    rows = [
        _row(0.0, 0.0, f187n=(15.0, True)),
        _row(0.0, 0.5 / 3600., f405n=(16.0, True)),  # 0.5" away
    ]
    t = _build(rows)
    p = tmp_path / 'in.fits'
    o = tmp_path / 'out.fits'
    t.write(p)
    dc.dedup_merged_catalog(str(p), str(o), verbose=False)
    out = Table.read(o)
    assert len(out) == 2
