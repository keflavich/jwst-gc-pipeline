"""Reading the ZEROFRAME fit-treatment column of the satstar catalogs (#681).

Catalogs written before the column existed must read as UNKNOWN, never as
"rim masked" (0): a mixed-vintage archive is the normal case, and an analysis
that counts old catalogs as 0 reports a treatment nobody measured.
"""
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
import astropy.units as u

from jwst_gc_pipeline.photometry import merge_catalogs as mc
from jwst_gc_pipeline.photometry.satstar_zf_status import (
    satstar_zf_status, satstar_zf_summary, zf_fit_fraction)

RA0, DEC0 = 266.5, -28.9


def _cat(zf=None, n=3, exposure='a'):
    t = Table()
    t['skycoord_fit'] = SkyCoord((RA0 + 0.01 * np.arange(n)) * u.deg,
                                 np.full(n, DEC0) * u.deg, frame='icrs')
    t['flux_fit'] = np.full(n, 1.0e4)
    t[mc._SATSTAR_EXPKEY_COL] = np.array([exposure] * n, dtype=str)
    if zf is not None:
        t['zf_fit'] = np.full(n, zf, dtype=np.int32)
    return t


def test_absent_column_reads_unknown_not_zero():
    st = satstar_zf_status(_cat(zf=None))
    assert np.ma.getmaskarray(st).all()
    assert not (np.ma.getdata(st)[~np.ma.getmaskarray(st)] == 0).any()


def test_present_column_reads_its_values():
    st = satstar_zf_status(_cat(zf=1))
    assert not np.ma.getmaskarray(st).any()
    assert (st == 1).all()
    assert (satstar_zf_status(_cat(zf=0)) == 0).all()


def test_mixed_vintages_stay_unknown_per_row():
    new = _cat(zf=0, n=2, exposure='new')
    old = _cat(zf=None, n=3, exposure='old')
    both = vstack([new, old], metadata_conflicts='silent')
    st = satstar_zf_status(both)
    assert list(np.ma.getmaskarray(st)) == [False, False, True, True, True]
    assert int((st == 0).sum()) == 2


def test_unknown_survives_a_fits_round_trip(tmp_path):
    """The consolidated cache is a FITS file.  An integer column keeps its
    mask through the write; a boolean one reads masked rows back as True."""
    both = vstack([_cat(zf=1, n=2, exposure='new'),
                   _cat(zf=None, n=2, exposure='old')],
                  metadata_conflicts='silent')
    path = str(tmp_path / 'cache.fits')
    both.write(path, format='fits')
    back = Table.read(path)
    st = satstar_zf_status(back)
    assert list(np.ma.getmaskarray(st)) == [False, False, True, True]
    assert list(np.ma.getdata(st)[:2]) == [1, 1]


def test_summary_counts_the_three_states():
    both = vstack([_cat(zf=1, n=2, exposure='a'), _cat(zf=0, n=1, exposure='b'),
                   _cat(zf=None, n=4, exposure='c')],
                  metadata_conflicts='silent')
    msg = satstar_zf_summary(both)
    assert '2 deep-core' in msg and '1 blob' in msg and '4 unknown' in msg


def test_fraction_ignores_unknown_and_unusable_members():
    st = np.ma.array([1, 0, 1, 1, 0], mask=[False, False, False, True, False])
    group = np.array([0, 0, 0, 0, 1])
    frac = zf_fit_fraction(st, group, 3)
    assert frac[0] == pytest.approx(2 / 3)       # the masked member is skipped
    assert frac[1] == 0.0
    assert np.isnan(frac[2])                      # empty group
    usable = np.array([True, False, True, True, True])
    assert zf_fit_fraction(st, group, 3, usable=usable)[0] == 1.0


def test_dedup_carries_the_fraction_and_keeps_the_representative_value():
    """Three exposures of one star: two fitted on the deep core, one on the
    blob.  The brightest (the representative) was a blob fit; ``flux_fit`` is
    the median, so ``zf_fit_frac`` reports how the median was built."""
    t = Table()
    t['skycoord_fit'] = SkyCoord(np.full(3, RA0) * u.deg,
                                 np.full(3, DEC0) * u.deg, frame='icrs')
    t['flux_fit'] = np.array([3.0e4, 2.0e4, 2.1e4])
    t[mc._SATSTAR_EXPKEY_COL] = np.array(['e0', 'e1', 'e2'])
    t['zf_fit'] = np.array([0, 1, 1], dtype=np.int32)
    out = mc._dedup_satstar_catalog(t)
    assert len(out) == 1
    assert int(out['zf_fit'][0]) == 0
    assert float(out['zf_fit_frac'][0]) == pytest.approx(2 / 3)


def test_dedup_without_the_column_adds_nothing():
    t = _cat(zf=None, n=1)
    out = mc._dedup_satstar_catalog(vstack([t, t], metadata_conflicts='silent'))
    assert 'zf_fit_frac' not in out.colnames


def test_dedup_of_mixed_vintages_counts_only_known_members():
    """One star seen in an old and a new catalog, and a second star seen only
    in old ones: the first counts its known member, the second is NaN."""
    def _tab(ra, exps):
        tab = Table()
        tab['skycoord_fit'] = SkyCoord(np.full(len(exps), ra) * u.deg,
                                       np.full(len(exps), DEC0) * u.deg,
                                       frame='icrs')
        tab['flux_fit'] = np.full(len(exps), 1.0e4)
        tab[mc._SATSTAR_EXPKEY_COL] = np.array(exps)
        return tab
    new = _tab(RA0, ['n0'])
    new['zf_fit'] = np.array([1], dtype=np.int32)
    out = mc._dedup_satstar_catalog(vstack(
        [_tab(RA0, ['o0']), new, _tab(RA0 + 0.01, ['o0', 'o1'])],
        metadata_conflicts='silent'))
    assert len(out) == 2
    frac = {round(float(r.ra.deg), 3): float(f)
            for r, f in zip(out['skycoord_fit'], out['zf_fit_frac'])}
    assert frac[round(RA0, 3)] == 1.0
    assert np.isnan(frac[round(RA0 + 0.01, 3)])


def test_summary_of_a_catalog_with_no_column_and_of_an_empty_one():
    assert '3 unknown' in satstar_zf_summary(_cat(zf=None))
    assert '0 unknown' in satstar_zf_summary(_cat(zf=1, n=0))
