"""The overshoot forced-refit flag must survive the merge (#932, #931).

``forced_refit`` marks a fit redone because a brighter neighbour's model
overshot the source.  Those magnitudes read too bright: injected stars 4-14 px
from a bright neighbour come back 0.31/0.44/0.76 mag bright at 14.5-15.5,
15.5-16.5 and 16.5-17.5 on refit rows, against 0.01-0.05 mag on rows the refit
left alone (measured on o132 F480M).  #931 raises the share of rows taking that
path near a recovered saturated star, so the merged catalog has to say which
rows it is.  The flag was written per frame only; both merged products dropped
it, and the per-frame position scatter cannot stand in for it (the merge NaNs
an exactly-zero scatter, which is what seeded/forced rows produce).
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.photometry import merge_catalogs


def _sci_file(tmp_path):
    path = tmp_path / 'fr_sci.fits'
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(data=np.zeros((4, 4)), name='SCI')]
                 ).writeto(path, overwrite=True)
    return str(path)


def _frames(tmp_path, forced_per_frame, n_src=12, n_frames=3, with_col=True):
    """``n_frames`` single-exposure catalogs on a 10" grid (sparse: no guard).

    ``forced_per_frame[f]`` is the boolean ``forced_refit`` column of frame f.
    """
    filename = _sci_file(tmp_path)
    ra = 266.5 + np.arange(n_src) * (10.0 / 3600.0)
    dec = np.full(n_src, -28.8)
    tbls = []
    for f in range(n_frames):
        t = Table()
        t['id'] = np.arange(1, n_src + 1)
        t['skycoord'] = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
        t['flux'] = np.full(n_src, 1000.0, dtype='float32')
        t['dflux'] = np.full(n_src, 10.0, dtype='float32')
        t['qf'] = np.full(n_src, 1.0, dtype='float32')
        t['fracflux'] = np.full(n_src, 0.9, dtype='float32')
        if with_col:
            t['forced_refit'] = np.asarray(forced_per_frame[f], dtype=bool)
        t.meta.update(exposure=f + 1, MODULE='nrca', filter='f405n',
                      FILENAME=filename, ra_offset=0.0 * u.arcsec,
                      dec_offset=0.0 * u.arcsec)
        tbls.append(t)
    return tbls


def _combine(tbls):
    return merge_catalogs.combine_singleframe(
        tbls, nanaverage=merge_catalogs.nanaverage_numpy)


def test_fraction_counts_the_frames_that_took_the_refit(tmp_path):
    n_src, n_frames = 12, 3
    forced = np.zeros((n_frames, n_src), dtype=bool)
    forced[:, 0] = True           # refit in all 3 frames
    forced[:2, 1] = True          # 2 of 3
    forced[0, 2] = True           # 1 of 3
    out = _combine(_frames(tmp_path, forced, n_src=n_src, n_frames=n_frames))
    frac = np.asarray(out['forced_refit_frac'], dtype=float)
    nfr = np.asarray(out['forced_refit_nframes'], dtype=int)
    assert frac[0] == pytest.approx(1.0)
    assert frac[1] == pytest.approx(2 / 3)
    assert frac[2] == pytest.approx(1 / 3)
    assert np.allclose(frac[3:], 0.0)
    assert list(nfr[:3]) == [3, 2, 1]
    assert nfr[3:].sum() == 0


def test_the_fraction_is_unweighted(tmp_path):
    """An inverse-variance mean would let one precise frame outvote the rest;
    the fraction has to be a plain count over contributing frames."""
    n_src, n_frames = 6, 2
    forced = np.zeros((n_frames, n_src), dtype=bool)
    forced[0, 0] = True
    tbls = _frames(tmp_path, forced, n_src=n_src, n_frames=n_frames)
    tbls[0]['dflux'] = np.full(n_src, 0.1, dtype='float32')   # 10^4x the weight
    out = _combine(tbls)
    assert float(out['forced_refit_frac'][0]) == pytest.approx(0.5)


def test_absent_column_leaves_the_merge_alone(tmp_path):
    """Catalogs from before the flag existed must merge unchanged."""
    out = _combine(_frames(tmp_path, np.zeros((3, 12), dtype=bool),
                           with_col=False))
    assert 'forced_refit_frac' not in out.colnames
    assert 'forced_refit_nframes' not in out.colnames


def test_frames_without_the_column_leave_the_denominator(tmp_path):
    """A partial re-run can leave one frame from an older vintage without the
    column.  That frame recorded no refits, so counting it as "not refit" would
    dilute the fraction and understate the contamination; it is excluded and
    the fraction reads over the frames that could report."""
    n_src = 8
    forced = np.zeros((2, n_src), dtype=bool)
    forced[0, 0] = True
    tbls = _frames(tmp_path, forced, n_src=n_src, n_frames=2)
    del tbls[1]['forced_refit']
    out = _combine(tbls)
    assert float(out['forced_refit_frac'][0]) == pytest.approx(1.0)
    assert int(out['forced_refit_nframes'][0]) == 1


def test_carried_into_the_shipped_minimal_table():
    """``merge_individual_frames`` ships a minimal table built from an
    allowlist, so a new column reaches the release only if it is named there."""
    import inspect
    src = inspect.getsource(merge_catalogs.merge_individual_frames)
    block = src[src.index("for key in ('dra_avg'"):]
    block = block[:block.index("minimal_table = Table(")]
    assert "'forced_refit_frac'" in block
    assert "'forced_refit_nframes'" in block


def test_reaches_the_crossband_table_with_a_filter_suffix(tmp_path):
    """``gather_crossfilter_columns`` suffixes every column of the per-filter
    table, so the m8 catalog gets ``forced_refit_frac_{filter}``."""
    n_src = 5
    t = Table()
    t['skycoord'] = SkyCoord(ra=(266.5 + np.arange(n_src) * 1e-3) * u.deg,
                             dec=np.full(n_src, -28.8) * u.deg, frame='icrs')
    t['flux'] = np.full(n_src, 1.0)
    t['forced_refit_frac'] = np.linspace(0, 1, n_src).astype('float32')
    t['forced_refit_nframes'] = np.arange(n_src, dtype='int16')
    matches = np.arange(n_src)
    sep = np.zeros(n_src) * u.arcsec
    out = merge_catalogs.gather_crossfilter_columns(
        t, matches, sep, np.ones(n_src, dtype=bool), 0.1 * u.arcsec, 'f405n')
    assert 'forced_refit_frac_f405n' in out.colnames
    assert 'forced_refit_nframes_f405n' in out.colnames
    assert np.allclose(np.asarray(out['forced_refit_frac_f405n'], dtype=float),
                       np.linspace(0, 1, n_src), atol=1e-6)
