"""A table with fewer than 100 sources must not crash the sanity check.

`sanity_check_individual_table` printed the 100th-brightest source by indexing
`[-100]` unconditionally.  Any shorter table raised IndexError out of a *print*,
and `merge_daophot` re-raises, so a diagnostic line took the whole stage down.

sickle's joint MIRI m6 finalize died exactly there (job 41038582): F1130W has
88 sources, and m12 through m5 had all already succeeded.
"""
import numpy as np
import pytest
import astropy.units as u
from astropy.table import Table

from jwst_gc_pipeline.photometry.merge_catalogs import (
    ABMAG_OFFSET, sanity_check_individual_table)


def _tbl(n, filtername='f1130w'):
    """A table whose mag_ab is consistent with its flux_jy, as the real one is."""
    flux = np.linspace(1e-3, 1e-1, n) if n else np.array([])
    t = Table()
    t['flux_jy'] = u.Quantity(flux, u.Jy)
    with np.errstate(divide='ignore'):
        t['mag_ab'] = u.Quantity(-2.5 * np.log10(flux) + ABMAG_OFFSET, u.mag)
    t['flux'] = flux * 1e6
    t.meta['filter'] = filtername
    return t


@pytest.mark.parametrize('n', [88, 1, 99])
def test_a_table_shorter_than_100_does_not_raise(n, capsys):
    sanity_check_individual_table(_tbl(n))
    out = capsys.readouterr().out
    assert 'fewer than 100' in out, out
    assert f'only {n} source' in out, out


def test_a_table_of_100_or_more_still_reports_the_100th(capsys):
    sanity_check_individual_table(_tbl(150))
    out = capsys.readouterr().out
    assert '100th brightest' in out, out


def test_an_all_nonpositive_table_does_not_raise(capsys):
    """The filter is `flux_jy > 0`, so a table can be non-empty and still leave
    nothing to report."""
    t = _tbl(5)
    t['flux_jy'] = u.Quantity(np.full(5, -1.0), u.Jy)
    sanity_check_individual_table(t)
    assert 'no positive-flux sources' in capsys.readouterr().out
