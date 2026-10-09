"""`combine_singleframe` must not drop a source whose frames agree.

The Phase-1 mask is the union of three per-axis `sigma_clip` masks (flux, ra,
dec) with `stdfunc='mad_std'`.  With four frames the MAD comes from two or three
near-equal values, so each axis can clip frames that agree to a few per cent in
flux or 1-2 mas in position, and the union can cover every frame.  That source
then had nmatch_good=0 and a NaN position, and the caller dropped it as "nan
coordinates" (wd2 m7 F277W: 49 rows, 39 of them good against dolphot).

The per-frame values below are those of one such F277W star (19.97 mag, dolphot
dm +0.004 from the unclipped mean): flux clips frames 2 and 3, dec clips frames
0 and 1.
"""
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
from astropy.table import Table
import astropy.units as u

from jwst_gc_pipeline.photometry import merge_catalogs

# wd2 F277W, per-frame m7 daophot rows of one star (4 frames)
STAR_RA = np.array([155.894726688, 155.894726603, 155.894726516, 155.894726218])
STAR_DEC = np.array([-57.773432621, -57.773433066, -57.773432523, -57.773432539])
STAR_FLUX = np.array([46.67, 46.55, 45.88, 44.22])

# a second star 2" away whose fourth frame is a flux outlier: the clip must
# still remove that frame
OUTLIER_FLUX = np.array([100.0, 101.0, 102.0, 1000.0])


def _frames():
    tbls = []
    for f in range(4):
        ra = np.array([STAR_RA[f], STAR_RA[0] + 2.0 / 3600.0])
        dec = np.array([STAR_DEC[f], STAR_DEC[0]])
        t = Table()
        t['id'] = np.arange(1, 3)
        t['skycoord'] = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
        t['flux'] = np.array([STAR_FLUX[f], OUTLIER_FLUX[f]], dtype='float32')
        t['dflux'] = np.array([1.0, 1.0], dtype='float32')
        t['qf'] = np.ones(2, dtype='float32')
        t['fracflux'] = np.full(2, 0.9, dtype='float32')
        t.meta.update(exposure=f + 1, MODULE='nrcb', filter='f277w',
                      ra_offset=0.0 * u.arcsec, dec_offset=0.0 * u.arcsec)
        tbls.append(t)
    return tbls


def _combine():
    """Rows come out in base-catalog order, and the base is frame 0's table:
    row 0 is the star, row 1 the outlier star."""
    out = merge_catalogs.combine_singleframe(
        _frames(), nanaverage=merge_catalogs.nanaverage_numpy, verbose=False)
    assert len(out) == 2
    return out


def test_the_star_is_all_clipped_by_the_union_mask():
    """Guard on the input: if astropy's sigma_clip changes so that these values
    no longer mask every frame, the tests below stop testing the fallback."""
    flux = STAR_FLUX[None, :].astype('float32')
    m = (sigma_clip(flux, stdfunc='mad_std', axis=1).mask
         | sigma_clip(STAR_RA[None, :], stdfunc='mad_std', axis=1).mask
         | sigma_clip(STAR_DEC[None, :], stdfunc='mad_std', axis=1).mask)
    assert m.all()
    for a in (flux, STAR_RA[None, :], STAR_DEC[None, :]):
        assert not sigma_clip(a, stdfunc='mad_std', axis=1).mask.all()


def test_all_clipped_source_keeps_a_finite_position_and_flux():
    row = _combine()[0]
    assert np.isfinite(row['skycoord_avg'].ra.deg)
    assert np.isfinite(row['skycoord_avg'].dec.deg)
    expected = SkyCoord(STAR_RA.mean() * u.deg, STAR_DEC.mean() * u.deg)
    assert row['skycoord_avg'].separation(expected).to_value(u.mas) < 0.01
    # equal dflux -> the inverse-variance mean is the plain mean
    np.testing.assert_allclose(row['flux_avg'], STAR_FLUX.mean(), rtol=1e-5)
    assert row['nmatch'] == 4
    assert row['nmatch_good'] == 4
    assert np.isfinite(row['dflux_prop'])


def test_the_fallback_is_logged(capsys):
    merge_catalogs.combine_singleframe(
        _frames(), nanaverage=merge_catalogs.nanaverage_numpy, verbose=False)
    log = capsys.readouterr().out
    assert "masked every frame of 1 source(s)" in log, log


def test_a_flux_outlier_frame_is_still_clipped():
    """The fallback applies only when nothing survives the clip."""
    row = _combine()[1]
    assert row['skycoord_avg'].separation(
        SkyCoord((STAR_RA[0] + 2.0 / 3600.0) * u.deg, STAR_DEC[0] * u.deg)
    ).to_value(u.mas) < 0.01
    assert row['nmatch'] == 4
    assert row['nmatch_good'] == 3
    np.testing.assert_allclose(row['flux_avg'], OUTLIER_FLUX[:3].mean(), rtol=1e-5)
