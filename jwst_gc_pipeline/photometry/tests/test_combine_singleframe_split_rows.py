"""`combine_singleframe` must not split one star into two merged rows.

Loop 1 adds a frame's source as a new base row when it lies more than
``min_offset`` from every existing row, judged at that frame's fitted position.
One drifted fit therefore creates a second row for the same star, and Phase 1's
mutual nearest-neighbour match hands each later frame's single fit of the star
to only one of the two rows.  Both rows average to the full flux at converging
positions, and the merged-catalog residual subtracts the star twice in every
frame: brick m7 F405N vetted pairs closer than 0.5 FWHM have
nmatch_a + nmatch_b equal to the local isolated nmatch (median ratio 1.000) and
leave a crater in 97-100% of cases.

The rows of such a star have disjoint frame sets.  A real close pair that the
frames resolve contributes to both rows in the same frames and stays two rows.
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.photometry import merge_catalogs as MC

RA0, DEC0 = 266.5, -28.8
COSD = np.cos(np.deg2rad(DEC0))


def _sci_file(tmp_path):
    path = tmp_path / 'split_sci.fits'
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(data=np.zeros((4, 4)), name='SCI')]
                 ).writeto(path, overwrite=True)
    return str(path)


def _frame(tmp_path, f, offsets_as, fluxes, dao=False, forced=None):
    """One exposure catalog; ``offsets_as`` are (east, north) arcsec offsets
    from (RA0, DEC0).  A far-away reference star rides along in every frame."""
    off = np.asarray(list(offsets_as) + [(20.0, 0.0)], dtype=float)
    flux = np.asarray(list(fluxes) + [500.0], dtype='float32')
    ra = RA0 + off[:, 0] / 3600.0 / COSD
    dec = DEC0 + off[:, 1] / 3600.0
    n = len(off)
    t = Table()
    t['id'] = np.arange(1, n + 1)
    sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    if dao:
        t['skycoord_centroid'] = sc
        t['flux_fit'] = flux
        t['flux_err'] = np.full(n, 10.0, dtype='float32')
        t['qfit'] = np.full(n, 0.1, dtype='float32')
        t['cfit'] = np.zeros(n, dtype='float32')
        t['flux_init'] = flux
        t['flags'] = np.zeros(n, dtype='int32')
        t['forced_refit'] = (np.zeros(n, dtype=bool) if forced is None
                             else np.asarray(list(forced) + [False]))
    else:
        t['skycoord'] = sc
        t['flux'] = flux
        t['dflux'] = np.full(n, 10.0, dtype='float32')
        t['qf'] = np.full(n, 1.0, dtype='float32')
        t['fracflux'] = np.full(n, 0.9, dtype='float32')
    t.meta.update(exposure=f + 1, MODULE='nrca', filter='f405n',
                  FILENAME=_sci_file(tmp_path),
                  ra_offset=0.0 * u.arcsec, dec_offset=0.0 * u.arcsec)
    return t


# One star at the origin.  Frame 1's fit drifted 0.11" east (> min_offset), so
# Loop 1 gives it a second row; frames 2-5 fit it near the midpoint, a little
# closer to one row or the other.
SPLIT_X = [0.0, 0.11, 0.05, 0.06, 0.05, 0.06]
SPLIT_FLUX = [1000.0, 1010.0, 990.0, 1005.0, 995.0, 1000.0]


def _split_frames(tmp_path, dao=False):
    return [_frame(tmp_path, f, [(x, 0.0)], [fl], dao=dao)
            for f, (x, fl) in enumerate(zip(SPLIT_X, SPLIT_FLUX))]


def _combine(tbls):
    return MC.combine_singleframe(tbls, nanaverage=MC.nanaverage_numpy,
                                  verbose=False)


def _near_origin(out, colname='skycoord_avg', radius_as=0.3):
    sep = out[colname].separation(SkyCoord(RA0 * u.deg, DEC0 * u.deg))
    return out[sep.arcsec < radius_as]


def test_the_input_splits_without_the_fold(tmp_path, monkeypatch):
    """Guard on the fixture: with the fold disabled, Loop 1 + Phase 1 produce
    the two-row split this module is about."""
    monkeypatch.setattr(MC, 'split_row_owners',
                        lambda arr_ra, arr_dec, radius: np.arange(arr_ra.shape[0]))
    rows = _near_origin(_combine(_split_frames(tmp_path)))
    assert len(rows) == 2
    assert sorted(rows['nmatch']) == [3, 3]
    # each half averages to the full flux at positions inside min_offset
    assert np.all(np.abs(rows['flux_avg'] - 1000.0) < 20.0)
    assert rows['skycoord_avg'][0].separation(rows['skycoord_avg'][1]).arcsec < 0.1


def test_split_star_becomes_one_row(tmp_path):
    out = _combine(_split_frames(tmp_path))
    rows = _near_origin(out)
    assert len(rows) == 1
    row = rows[0]
    assert row['nmatch'] == 6
    # the position clip drops the two outlying fits (0.0" and 0.11"); the
    # average is over the four central frames
    assert row['nmatch_good'] == 4
    np.testing.assert_allclose(row['flux_avg'], np.mean(SPLIT_FLUX[2:]), rtol=1e-5)
    expect = SkyCoord((RA0 + np.mean(SPLIT_X[2:]) / 3600.0 / COSD) * u.deg, DEC0 * u.deg)
    assert row['skycoord_avg'].separation(expect).to_value(u.mas) < 0.1
    # the reference star is untouched
    assert len(out) == 2
    assert int(out['nmatch'][1]) == 6


def test_split_star_is_logged(tmp_path, capsys):
    _combine(_split_frames(tmp_path))
    assert "merged 1 split-star row(s)" in capsys.readouterr().out


def test_dao_columns_follow_the_merged_row(tmp_path):
    """Phase 2 columns and the forced-refit tally read the remapped matches."""
    tbls = _split_frames(tmp_path, dao=True)
    tbls[1]['forced_refit'][0] = True
    out = MC.combine_singleframe(tbls, nanaverage=MC.nanaverage_numpy, verbose=False)
    rows = _near_origin(out)
    assert len(rows) == 1
    row = rows[0]
    assert row['nmatch'] == 6
    np.testing.assert_allclose(row['qfit_avg'], 0.1, rtol=1e-5)
    # frame 1 fed the absorbed row; its flag now counts toward the kept row,
    # over all six contributing frames
    assert row['forced_refit_nframes'] == 1
    assert row['forced_refit_frac'] == pytest.approx(1 / 6)


def test_resolved_close_pair_stays_two_rows(tmp_path):
    """Two stars 0.08" apart (< min_offset) fitted as two sources in every
    frame share every frame and are never merged."""
    tbls = [_frame(tmp_path, f, [(0.0, 0.0), (0.08, 0.0)], [1000.0, 400.0])
            for f in range(5)]
    rows = _near_origin(_combine(tbls))
    assert len(rows) == 2
    assert list(rows['nmatch']) == [5, 5]


def test_faint_companion_seen_in_some_frames_stays_separate(tmp_path):
    """A faint companion 0.076" away, detected in two of five frames (frame 0,
    where both enter the base, and frame 3), co-occurs with the primary in
    those frames, so it stays its own row."""
    tbls = []
    for f in range(5):
        offs, fl = [(0.0, 0.0)], [1000.0]
        if f in (0, 3):
            offs.append((0.07, 0.03))
            fl.append(80.0)
        tbls.append(_frame(tmp_path, f, offs, fl))
    rows = _near_origin(_combine(tbls))
    assert len(rows) == 2
    assert sorted(rows['nmatch']) == [2, 5]


def test_disjoint_rows_beyond_min_offset_stay_separate(tmp_path):
    """Disjoint frame sets alone do not merge: the averaged positions must lie
    within min_offset."""
    tbls = [_frame(tmp_path, f, [((0.0 if f % 2 == 0 else 0.15), 0.0)], [1000.0])
            for f in range(4)]
    rows = _near_origin(_combine(tbls))
    assert len(rows) == 2


# --- split_row_owners / fold_split_rows on bare arrays ---------------------

def _arrays(rows):
    """``rows`` maps row -> {frame: east offset in arcsec}; 4 frames."""
    ra = np.full((len(rows), 4), np.nan)
    dec = np.full((len(rows), 4), np.nan)
    for i, fr in enumerate(rows):
        for f, x in fr.items():
            ra[i, f] = RA0 + x / 3600.0 / COSD
            dec[i, f] = DEC0
    return ra, dec


def test_owners_merge_disjoint_close_rows_into_the_larger():
    ra, dec = _arrays([{0: 0.0}, {1: 0.02, 2: 0.03, 3: 0.02}])
    owner = MC.split_row_owners(ra, dec, 0.1 * u.arcsec)
    assert list(owner) == [1, 1]


def test_owners_keep_co_occurring_rows():
    ra, dec = _arrays([{0: 0.0, 1: 0.0}, {1: 0.05, 2: 0.05}])
    owner = MC.split_row_owners(ra, dec, 0.1 * u.arcsec)
    assert list(owner) == [0, 1]


def test_owners_do_not_chain_beyond_the_radius():
    """A, B, C on a line 0.09" apart, all disjoint: B joins one neighbour, and
    the other stays out because it lies more than 0.1" from a group member."""
    ra, dec = _arrays([{0: 0.0}, {1: 0.09}, {2: 0.18, 3: 0.18}])
    owner = MC.split_row_owners(ra, dec, 0.1 * u.arcsec)
    assert len(set(owner.tolist())) == 2
    assert owner[0] != owner[2]


def test_owners_group_is_never_wider_than_the_radius():
    """Rows at 0", 0.08" (3 frames) and 0.155": both outer rows lie within
    0.1" of the middle one, and the group centroid would lie within 0.1" of
    each, but the outer rows are 0.155" apart, so only one joins."""
    ra, dec = _arrays([{0: 0.0}, {1: 0.08, 2: 0.08, 3: 0.08}, {}])
    ra = np.c_[ra, np.full(3, np.nan)]
    dec = np.c_[dec, np.full(3, np.nan)]
    ra[2, 4] = RA0 + 0.155 / 3600.0 / COSD
    dec[2, 4] = DEC0
    owner = MC.split_row_owners(ra, dec, 0.1 * u.arcsec)
    assert owner[1] == 1
    assert owner[0] != owner[2]
    # every group's rows lie within the radius of each other
    x = np.array([0.0, 0.08, 0.155])
    for o in set(owner.tolist()):
        xs = x[owner == o]
        assert xs.max() - xs.min() <= 0.1


def test_owners_ignore_rows_without_frames():
    ra, dec = _arrays([{0: 0.0}, {}, {1: 0.01}])
    owner = MC.split_row_owners(ra, dec, 0.1 * u.arcsec)
    assert owner[1] == 1
    assert owner[0] == owner[2]


def test_fold_moves_frames_and_remaps_matches():
    ra, dec = _arrays([{0: 0.0}, {3: 5.0}, {1: 0.02, 2: 0.03}])
    flux = np.where(np.isfinite(ra), 1.0, np.nan)
    flux[0, 0] = 7.0
    owner = MC.split_row_owners(ra, dec, 0.1 * u.arcsec)
    assert list(owner) == [2, 1, 2]
    # frame f's catalog row k matched base row mi[f][k]
    mi = [np.array([0], np.int32), np.array([2], np.int32),
          np.array([2, 1], np.int32), np.array([1], np.int32)]
    (ra2, dec2, flux2), mi2 = MC.fold_split_rows(owner, [ra, dec, flux], mi)
    assert ra2.shape == (2, 4)
    # old row 1 -> new row 0, old rows 0+2 -> new row 1
    assert [list(m) for m in mi2] == [[1], [1], [1, 0], [0]]
    assert np.isfinite(ra2[1, :3]).all() and np.isnan(ra2[1, 3])
    assert flux2[1, 0] == 7.0
    assert np.isfinite(ra2[0, 3]) and np.isnan(ra2[0, :3]).all()


def test_chunked_combine_merges_the_split(tmp_path):
    """The spatial-chunk wrapper runs combine_singleframe per tile, so it
    inherits the fold."""
    tbls = _split_frames(tmp_path)
    out = MC.combine_singleframe_chunked(tbls, n_chunks=2, workers=1)
    rows = _near_origin(out)
    assert len(rows) == 1
    assert rows[0]['nmatch'] == 6
