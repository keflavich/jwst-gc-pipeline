"""In-field satstar rows and the post-merge off-FOV cleanup (#972).

``_clean_offfov_dups_and_offfield`` mechanism A collapses ``replaced_saturated``
rows within 1.0" to one row.  It exists for OFF-FOV stars, whose per-frame fits
scatter wider than the 0.15" satstar dedup.  Run over in-field rows it deleted
distinct stars: gc-treasury o111 m6 lost 8 F480M and ~1450 F212N rows per module
run, among them F480M #12 of missed_satstars_20260925 (0.71" from #11; the two
are fit separately in 5 of 6 exposures), so #12 was rendered in no frame whose
satstar fit had been rejected.

These tests pin:
* the default collapses off-FOV rows only, so an in-field close pair survives;
* ``dedup_offfov_only=False`` still gives the old all-rows collapse;
* with per-exposure co-fit positions, a pair that one exposure fits side by
  side is kept, and a pair no exposure ever fits together (an edge-truncated
  re-fit of one star) collapses to its best-supported member;
* without a readable data i2d the collapse falls back to all rows, as before;
* an empty co-fit list keeps every in-field row (with a WARNING) instead of
  collapsing every close pair;
* the co-fit test is pairwise: a co-fit pair can still collapse onto a
  better-supported third row between them (current behaviour, documented);
* ``_infield_dedup_settings`` (the ``SATSTAR_INFIELD_DEDUP`` parsing that
  ``run_manual_pipeline`` calls): NIRCam defaults to 'cofit', MIRI and NIRISS
  keep the legacy collapse, and the extended-emission targets keep their
  off-FOV-only path in every mode, with the same kept rows as before.
"""
import inspect

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry import cataloging as C

PIXSCALE = 0.063 / 3600.0      # deg/px, NIRCam LW
SHAPE = (400, 400)             # ~25" on a side
RA0, DEC0 = 266.70, -28.55


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [200.5, 200.5]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.cdelt = [-PIXSCALE, PIXSCALE]
    return w


def _data_i2d(tmp_path):
    path = tmp_path / 'data_i2d.fits'
    hdu = fits.ImageHDU(np.zeros(SHAPE, dtype='float32'), name='SCI')
    hdu.header.update(_wcs().to_header())
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    return str(path)


def _sky(offsets_arcsec):
    off = np.atleast_2d(np.asarray(offsets_arcsec, float))
    return SkyCoord(RA0 * u.deg + off[:, 0] * u.arcsec / np.cos(np.deg2rad(DEC0)),
                    DEC0 * u.deg + off[:, 1] * u.arcsec)


def _merged(offsets, flux, nframes, replaced=None):
    n = len(offsets)
    return Table({'skycoord': _sky(offsets),
                  'flux': np.asarray(flux, float),
                  'satstar_nframes': np.asarray(nframes, float),
                  'replaced_saturated': (np.ones(n, bool) if replaced is None
                                         else np.asarray(replaced, bool))})


# in-field pair 0.71" apart (the #11/#12 geometry) + one off-FOV star fit 3x,
# scattered by 0.4" (the case the collapse was written for) + an isolated star
IN_A, IN_B = (0.0, 0.0), (0.64, 0.30)
OFF = [(20.0, 0.0), (20.3, 0.1), (20.1, -0.3)]   # 20" > 12.6"/2 + pad: off the i2d
ISO = (5.0, 5.0)


def _table():
    return _merged([IN_A, IN_B] + OFF + [ISO],
                   flux=[64743., 37324., 1e6, 2e6, 3e6, 5e4],
                   nframes=[5, 6, 1, 1, 1, 3])


def _positions(out):
    return SkyCoord(out['skycoord'])


def _has_row(out, offset, tol=0.01):
    return bool(np.any(_sky([offset])[0].separation(_positions(out)).arcsec < tol))


def test_default_keeps_infield_pair_and_collapses_offfov(tmp_path):
    out, ndup, noff = C._clean_offfov_dups_and_offfield(
        _table(), 'F480M', _data_i2d(tmp_path), str(tmp_path))
    assert ndup == 2 and noff == 0
    assert _has_row(out, IN_A) and _has_row(out, IN_B) and _has_row(out, ISO)
    offsel = _positions(out).separation(_sky([OFF[0]])[0]).arcsec < 2
    assert offsel.sum() == 1
    # the representative of the off-FOV cluster is still the median-flux member
    assert float(out['flux'][offsel][0]) == 2e6


def test_legacy_mode_collapses_the_infield_pair(tmp_path):
    out, ndup, _ = C._clean_offfov_dups_and_offfield(
        _table(), 'F480M', _data_i2d(tmp_path), str(tmp_path),
        dedup_offfov_only=False)
    assert ndup == 3
    assert _has_row(out, IN_A) != _has_row(out, IN_B)


def test_cofit_pair_is_kept(tmp_path):
    # one exposure fits both members side by side -> two stars
    runs = [_sky([IN_A, IN_B]), _sky([(0.01, 0.0)])]
    out, ndup, _ = C._clean_offfov_dups_and_offfield(
        _table(), 'F480M', _data_i2d(tmp_path), str(tmp_path),
        cofit_positions=runs)
    assert ndup == 2
    assert _has_row(out, IN_A) and _has_row(out, IN_B)


def test_never_cofit_pair_keeps_best_supported_member(tmp_path):
    # every exposure fits ONE of the two (edge-truncated re-fit of one star):
    # collapse, keeping the member with more per-exposure support (IN_B, 6
    # frames) even though IN_A is brighter and first in the table
    runs = [_sky([IN_A]), _sky([IN_B]), _sky([(0.64, 0.31)])]
    out, ndup, _ = C._clean_offfov_dups_and_offfield(
        _table(), 'F480M', _data_i2d(tmp_path), str(tmp_path),
        cofit_positions=runs)
    assert ndup == 3
    assert _has_row(out, IN_B) and not _has_row(out, IN_A)
    assert _has_row(out, ISO)


def test_one_run_row_between_two_members_is_not_a_cofit(tmp_path):
    # a single fit halfway between the two members must not count for both
    runs = [_sky([(0.30, 0.14)]), _sky([IN_A]), _sky([IN_B])]
    drop = C._never_cofit_duplicates(_sky([IN_A, IN_B]), [5, 6], [64743., 37324.],
                                     runs, dedup_arcsec=1.0, match_arcsec=0.5)
    assert drop.tolist() == [True, False]


def test_nonsatstar_rows_are_never_collapsed(tmp_path):
    tbl = _merged([IN_A, IN_B], flux=[1e4, 2e4], nframes=[1, 1],
                  replaced=[False, False])
    i2d = _data_i2d(tmp_path)
    for only in (True, False):
        out, ndup, _ = C._clean_offfov_dups_and_offfield(
            tbl.copy(), 'F480M', i2d, str(tmp_path),
            dedup_offfov_only=only)
        assert ndup == 0 and len(out) == 2


def test_without_data_i2d_falls_back_to_all_rows(tmp_path, capsys):
    out, ndup, noff = C._clean_offfov_dups_and_offfield(
        _table(), 'F480M', str(tmp_path / 'missing_i2d.fits'), str(tmp_path))
    assert ndup == 3 and noff == 0
    assert 'cannot be told apart' in capsys.readouterr().out


def _write_run(pdir, obs, x_acc, x_rej, stale_arcsec=0.5):
    """One frame plus its m6 accepted and rejected satstar files; the stored
    sky positions are deliberately STALE by ``stale_arcsec`` in Dec."""
    w = _wcs()
    frame = pdir / f'jw10678{obs}001_02101_00001_nrcalong_destreak_o{obs}_crf.fits'
    hdu = fits.ImageHDU(np.zeros(SHAPE, dtype='float32'), name='SCI')
    hdu.header.update(w.to_header())
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(frame)
    for kind, xs in (('catalog', x_acc), ('rejected', x_rej)):
        xs = np.asarray(xs, float)
        ys = np.full(len(xs), 200.0)
        sky = w.pixel_to_world(xs, ys)
        stale = SkyCoord(sky.ra, sky.dec + stale_arcsec * u.arcsec)
        tbl = Table({'xcentroid': xs, 'ycentroid': ys, 'flux_fit': np.full(len(xs), 1e4)})
        tbl['skycoord_fit'] = stale
        tbl.write(str(frame).replace('.fits', f'_resbgsub_m6_satstar_{kind}.fits'))
    return w


def test_cofit_positions_pair_accepted_and_rejected_on_the_current_wcs(tmp_path):
    pdir = tmp_path / 'pipeline'
    pdir.mkdir()
    w = _write_run(pdir, '001', x_acc=[100.0], x_rej=[111.0])
    _write_run(pdir, '002', x_acc=[50.0], x_rej=[60.0])       # another observation
    runs = C._satstar_cofit_positions(str(pdir), proposal_id='10678', field='001')
    assert len(runs) == 1
    got = runs[0]
    want = w.pixel_to_world(np.array([100.0, 111.0]), np.array([200.0, 200.0]))
    _, sep, _ = want.match_to_catalog_sky(got)
    # both the accepted AND the rejected row are re-projected (not 0.5" stale)
    assert np.all(sep.to_value(u.arcsec) < 1e-3)


@pytest.mark.parametrize('runs', [[], [None], [SkyCoord([], [], unit='deg')]],
                         ids=['empty-list', 'none-entry', 'empty-skycoord'])
def test_empty_cofit_list_keeps_infield_rows_with_a_warning(tmp_path, capsys, runs):
    # No per-exposure satstar file found: an empty list would make every close
    # pair "never co-fit" and collapse it like the legacy FoF.  Fall back to
    # keeping the in-field rows ('none'), and say so.
    out, ndup, noff = C._clean_offfov_dups_and_offfield(
        _table(), 'F480M', _data_i2d(tmp_path), str(tmp_path),
        cofit_positions=runs)
    assert ndup == 2 and noff == 0          # the off-FOV cluster only
    assert _has_row(out, IN_A) and _has_row(out, IN_B) and _has_row(out, ISO)
    log = capsys.readouterr().out
    assert 'WARNING' in log and "falls back to 'none'" in log


def test_cofit_pair_can_collapse_to_a_better_supported_midpoint():
    # A and B (0.6" apart) are fit side by side in one exposure; a blended fit
    # at their midpoint C is the only fit in three others.  The co-fit test is
    # pairwise, so A-C and B-C are "never co-fit" links and the greedy keep
    # order decides: C (support 3) survives and BOTH co-fit members are
    # dropped.  This pins current behaviour (the legacy FoF collapses the
    # same three rows); it is documented in _never_cofit_duplicates.
    A, B, M = (0.0, 0.0), (0.6, 0.0), (0.3, 0.0)
    runs = [_sky([A, B]), _sky([M]), _sky([M]), _sky([M])]
    drop = C._never_cofit_duplicates(_sky([A, B, M]), [1, 1, 3],
                                     [5e4, 4e4, 9e4], runs)
    assert drop.tolist() == [True, True, False]
    # with A and B better supported, the midpoint row is the one dropped
    drop = C._never_cofit_duplicates(_sky([A, B, M]), [3, 3, 1],
                                     [5e4, 4e4, 9e4], runs)
    assert drop.tolist() == [False, False, True]


# ---- SATSTAR_INFIELD_DEDUP parsing (run_manual_pipeline's call site) -------

_INFIELD_ENVS = ('SATSTAR_INFIELD_DEDUP', 'SATSTAR_INFIELD_DEDUP_MIRI',
                 'SATSTAR_INFIELD_DEDUP_NIRISS', 'GC_INSTRUMENT_OVERRIDE')


@pytest.fixture
def clean_env(monkeypatch):
    for name in _INFIELD_ENVS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.mark.parametrize('value, expected', [
    (None, ('cofit', True, True)),
    ('', ('cofit', True, True)),
    ('cofit', ('cofit', True, True)),
    ('none', ('none', True, False)),
    ('legacy', ('legacy', False, False)),
    ('  Legacy \n', ('legacy', False, False)),
    ('NONE', ('none', True, False)),
])
def test_nircam_settings_parse_satstar_infield_dedup(clean_env, value, expected):
    if value is not None:
        clean_env.setenv('SATSTAR_INFIELD_DEDUP', value)
    for module in ('nrca', 'nrcb', 'merged', 'nrcalong'):
        mode, env, offfov_only, use_cofit = C._infield_dedup_settings(
            'gc-treasury', module, 'F480M')
        assert env == 'SATSTAR_INFIELD_DEDUP'
        assert (mode, offfov_only, use_cofit) == expected, module


@pytest.mark.parametrize('env, value', [
    ('SATSTAR_INFIELD_DEDUP', 'cofit2'),
    ('SATSTAR_INFIELD_DEDUP', '1'),
    ('SATSTAR_INFIELD_DEDUP_MIRI', 'true'),
])
def test_invalid_infield_dedup_value_raises(clean_env, env, value):
    clean_env.setenv(env, value)
    module, filt = (('mirimage', 'F770W') if env.endswith('_MIRI')
                    else ('merged', 'F480M'))
    with pytest.raises(ValueError, match=env):
        C._infield_dedup_settings('gc-treasury', module, filt)


@pytest.mark.parametrize('module, filt', [('mirimage', 'F770W'),
                                          ('merged', 'F770W'),
                                          ('nrcb', 'F1130W')])
def test_miri_keeps_the_legacy_collapse_by_default(clean_env, module, filt):
    # The co-fit rule was validated on NIRCam only; on F770W o132 it changes
    # decisions nobody has checked.  MIRI keeps main's behaviour (all-rows
    # FoF on a non-extended target) and the NIRCam switch does not reach it.
    assert C._infield_dedup_settings('gc-treasury', module, filt) == (
        'legacy', 'SATSTAR_INFIELD_DEDUP_MIRI', False, False)
    clean_env.setenv('SATSTAR_INFIELD_DEDUP', 'cofit')
    assert C._infield_dedup_settings('gc-treasury', module, filt)[0] == 'legacy'
    # an explicit MIRI opt-in is honoured (for a MIRI validation run)
    clean_env.setenv('SATSTAR_INFIELD_DEDUP_MIRI', 'cofit')
    assert C._infield_dedup_settings('gc-treasury', module, filt) == (
        'cofit', 'SATSTAR_INFIELD_DEDUP_MIRI', True, True)


def test_niriss_keeps_the_legacy_collapse_by_default(clean_env):
    assert C._infield_dedup_settings('sgrc', 'nis', 'F200W') == (
        'legacy', 'SATSTAR_INFIELD_DEDUP_NIRISS', False, False)
    clean_env.setenv('GC_INSTRUMENT_OVERRIDE', 'NIRISS')
    assert C._infield_dedup_settings('sgrc', 'merged', 'F200W')[1] == (
        'SATSTAR_INFIELD_DEDUP_NIRISS')


def test_miri_default_reproduces_the_legacy_kept_rows(clean_env, tmp_path):
    # the settings MIRI gets, fed to the cleanup, give the legacy result
    _, _, offfov_only, use_cofit = C._infield_dedup_settings(
        'gc-treasury', 'mirimage', 'F770W')
    out, ndup, _ = C._clean_offfov_dups_and_offfield(
        _table(), 'F770W', _data_i2d(tmp_path), str(tmp_path),
        dedup_offfov_only=offfov_only,
        cofit_positions=([_sky([IN_A, IN_B])] if use_cofit else None))
    assert ndup == 3
    assert _has_row(out, IN_A) != _has_row(out, IN_B)


@pytest.mark.parametrize('target', C._EXTENDED_EMISSION_TARGETS)
@pytest.mark.parametrize('value', [None, 'cofit', 'none', 'legacy'])
@pytest.mark.parametrize('module, filt', [('merged', 'F480M'),
                                          ('nrcb', 'F470N'),
                                          ('mirimage', 'F770W')])
def test_extended_targets_keep_the_offfov_only_path(clean_env, target, value,
                                                     module, filt):
    if value is not None:
        clean_env.setenv('SATSTAR_INFIELD_DEDUP', value)
        clean_env.setenv('SATSTAR_INFIELD_DEDUP_MIRI', value)
    _, _, offfov_only, use_cofit = C._infield_dedup_settings(
        target.upper(), module, filt)
    assert offfov_only is True and use_cofit is False


# Kept-row masks that origin/main (c48ff2aa) gives on this table for the
# extended-target call (dedup_offfov_only=True, no co-fit): an in-field pair,
# a 3-row off-FOV cluster, an isolated satstar, two in-field non-satstar rows
# and one non-satstar row 40" off the i2d.  Measured by running main's
# _clean_offfov_dups_and_offfield on the same rows.
_EXT_OFFS = [(0, 0), (0.64, 0.30), (20, 0), (20.3, 0.1), (20.1, -0.3), (5, 5),
             (-4, 3), (40, 40), (4.9, 5.3)]
_EXT_RS = [True, True, True, True, True, True, False, False, False]
_EXT_FLUX = [64743., 37324., 1e6, 2e6, 3e6, 5e4, 1e4, 2e4, 3e3]
_MAIN_KEPT = {'with_i2d': ([1, 1, 0, 1, 0, 1, 1, 0, 1], 2, 1),
              'no_i2d': ([1, 0, 0, 1, 0, 1, 1, 1, 1], 3, 0)}


@pytest.mark.parametrize('case', sorted(_MAIN_KEPT))
def test_extended_target_kept_rows_match_main(clean_env, tmp_path, case):
    tbl = _merged(_EXT_OFFS, _EXT_FLUX, [5, 6, 1, 1, 1, 3, 0, 0, 0],
                  replaced=_EXT_RS)
    tbl['rowid'] = np.arange(len(tbl))
    i2d = (_data_i2d(tmp_path) if case == 'with_i2d'
           else str(tmp_path / 'missing_i2d.fits'))
    for target in C._EXTENDED_EMISSION_TARGETS:
        _, _, offfov_only, use_cofit = C._infield_dedup_settings(
            target, 'merged', 'F480M')
        out, ndup, noff = C._clean_offfov_dups_and_offfield(
            tbl.copy(), 'F480M', i2d, str(tmp_path),
            dedup_offfov_only=offfov_only,
            cofit_positions=(_sky(_EXT_OFFS) if use_cofit else None))
        kept = np.isin(tbl['rowid'], out['rowid']).astype(int).tolist()
        assert (kept, ndup, noff) == _MAIN_KEPT[case], target


def test_run_manual_pipeline_routes_the_cleanup_through_the_settings():
    # run_manual_pipeline is not driven by any test; pin that its cleanup call
    # takes both arguments from _infield_dedup_settings and that no second
    # parse of the env var sits next to it.
    src = inspect.getsource(C.run_manual_pipeline)
    assert '_infield_dedup_settings(target, module, filt)' in src
    start = src.index('_clean_offfov_dups_and_offfield(')
    call = src[start:src.index('cofit_positions=', start) + 40]
    assert 'dedup_offfov_only=_offfov_only' in call
    assert 'cofit_positions=_cofit' in call
    assert "'SATSTAR_INFIELD_DEDUP'" not in src
    assert 'if _want_cofit:' in src
