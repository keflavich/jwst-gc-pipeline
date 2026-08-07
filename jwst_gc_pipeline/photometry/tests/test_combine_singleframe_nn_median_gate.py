"""`combine_singleframe`'s dense-NN-median guard must cover the DIAGNOSTIC path
too (issue #314).

`:444` medians a mutual nearest-neighbour match into `tbl.meta['ra_offset']` /
`['dec_offset']`.  Against a dense internal base catalogue that is the
dense-NN-median method that corrupted brick-1182 -- ASTROMETRY RULE #1.

The guard was keyed on `realign`, which covers only the branch that APPLIES the
median.  With `realign=False` and `MERGE_REMATCH_DIAGNOSTICS=1` the same number
is computed against the same base and written to the same metadata key,
ungated.  "It is only a diagnostic" is not a property of the number: it is
stored under a name a reader can consume as a correction, and the allowlist
entry that used to cover this asserted the opposite.
"""
import inspect

from jwst_gc_pipeline.photometry import merge_catalogs


def test_the_guard_is_keyed_on_rematch_not_on_realign():
    src = inspect.getsource(merge_catalogs.combine_singleframe)
    assert "if rematch and basecrds is not None:" in src, \
        "the guard must cover the diagnostic path, not only realign=True"
    assert "if realign and basecrds is not None:\n        assert_sparse" not in src


def test_realign_TRUE_still_RAISES_on_a_dense_base():
    """The median would move the coordinates; nothing else is acceptable."""
    src = inspect.getsource(merge_catalogs.combine_singleframe)
    blk = src[src.index("except DenseNNMedianAstrometryError:"):]
    blk = blk[:blk.index("if not rematch:")]
    assert "if realign:" in blk and "raise" in blk, blk


def test_realign_FALSE_refuses_to_RECORD_rather_than_refusing_to_RUN():
    """The diagnostic's other outputs are legitimate and some callers want them
    on a dense base, so the loop still runs and only the offset metadata is
    NaN'd.  Refusing outright would make a forbidden-method guard the reason an
    unrelated diagnostic cannot run -- it broke
    test_combine_singleframe_rematch_without_module_meta, whose subject is a
    KeyError on missing MODULE meta and has nothing to do with astrometry."""
    src = inspect.getsource(merge_catalogs.combine_singleframe)
    assert "dense_nn_refused = True" in src
    blk = src[src.index("if dense_nn_refused:"):]
    blk = blk[:blk.index("else:")]
    for key in ("ra_offset", "dec_offset", "dra_offset", "ddec_offset"):
        assert f"tbl.meta['{key}'] = np.nan" in blk, key


def test_the_guard_runs_before_the_median_is_taken():
    """Order matters: asserting after the loop would let the forbidden value be
    computed and stored first."""
    src = inspect.getsource(merge_catalogs.combine_singleframe)
    guard = src.index("if rematch and basecrds is not None:")
    median = src.index("medsep_ra, medsep_dec = np.median(")
    assert guard < median, (guard, median)


def test_the_diagnostic_flag_is_named_in_the_refusal():
    """Whoever trips this set an env var; the message has to say which one, or
    the refusal is unactionable."""
    src = inspect.getsource(merge_catalogs.combine_singleframe)
    ctx = src[src.index("if rematch and basecrds is not None:"):]
    ctx = ctx[:ctx.index("if not rematch:")]
    assert "MERGE_REMATCH_DIAGNOSTICS" in ctx, ctx
    assert "realign={realign}" in ctx, ctx
    assert "not measured" in ctx, ctx


def test_the_production_default_still_records_not_measured():
    """`realign=False` without the flag never reaches the loop and records NaN.
    The guard must not have made the default path refuse instead -- that would
    turn a silent no-op into a stop on every merge."""
    src = inspect.getsource(merge_catalogs.combine_singleframe)
    assert "rematch = realign or os.environ.get('MERGE_REMATCH_DIAGNOSTICS', '') == '1'" in src
    assert "tbl.meta['ra_offset'] = np.nan" in src


# ---------------------------------------------------------------------------
# BEHAVIOURAL coverage.
#
# Everything above pins the SOURCE of the guard.  That is not enough on its own
# and it was not enough here: `basecrds[:2]` or `min_nn_spacing=0.0` in the
# guard call both read like routine tuning, silently turn it into a no-op (the
# guard returns early on n < 3), and leave every source-level test green.  The
# tests below call `combine_singleframe` and assert on what comes out, so a
# guard that does not FIRE fails them regardless of how the call is spelled.
#
# This is ASTROMETRY RULE #1 -- the method that corrupted brick-1182 twice --
# so the source-greps stay (the ordering one is hard to state behaviourally)
# but they are not the only thing standing here.
# ---------------------------------------------------------------------------
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

from jwst_gc_pipeline.photometry.measure_offsets import DenseNNMedianAstrometryError


def _sci_file(tmp_path):
    """A minimal SCI HDU: the re-match loop opens meta['FILENAME']."""
    path = tmp_path / "gate_sci.fits"
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(data=np.zeros((4, 4)), name='SCI')]
                 ).writeto(path, overwrite=True)
    return str(path)


def _frames(tmp_path, spacing_arcsec, shift_mas=50.0, n_src=40):
    """Two single-exposure catalogs on a grid of the requested spacing, the
    second bodily shifted in RA.

    `spacing_arcsec=1.0` -> median NN spacing 1", below the 3" floor, so the
    base catalog is DENSE and the guard must bite.  `spacing_arcsec=10.0` ->
    sparse, and the offset must still be measured.
    """
    filename = _sci_file(tmp_path)
    base_ra = 266.5 + np.arange(n_src) * (spacing_arcsec / 3600.0)
    base_dec = np.full(n_src, -28.8)
    tbls = []
    for f in range(2):
        ra = base_ra + (shift_mas / 1000.0 / 3600.0 if f else 0.0)
        t = Table()
        t['id'] = np.arange(1, n_src + 1)
        t['skycoord'] = SkyCoord(ra=ra * u.deg, dec=base_dec * u.deg,
                                 frame='icrs')
        t['flux'] = np.full(n_src, 1000.0, dtype='float32')
        t['dflux'] = np.full(n_src, 10.0, dtype='float32')
        t['qf'] = np.full(n_src, 1.0, dtype='float32')
        t['fracflux'] = np.full(n_src, 0.9, dtype='float32')
        t.meta.update(exposure=f + 1, MODULE='nrca', filter='f405n',
                      FILENAME=filename,
                      ra_offset=0.0 * u.arcsec, dec_offset=0.0 * u.arcsec)
        tbls.append(t)
    return tbls


def _nanaverage():
    return merge_catalogs.nanaverage_numpy


def test_DENSE_base_records_NO_offset_on_the_diagnostic_path(tmp_path, monkeypatch):
    """The bug (#314): with realign=False and MERGE_REMATCH_DIAGNOSTICS=1 the
    dense-NN-median was computed against the dense internal base and stored in
    meta['ra_offset'] -- the key a reader consumes as a correction.  It must
    come back NaN.  On `main` exposure 2 here records ~+57 mas.
    """
    monkeypatch.setenv('MERGE_REMATCH_DIAGNOSTICS', '1')
    out = merge_catalogs.combine_singleframe(
        _frames(tmp_path, spacing_arcsec=1.0), nanaverage=_nanaverage())
    offsets = out.meta['offsets']
    assert len(offsets) == 2
    for exp, (ra_off, dec_off) in offsets.items():
        assert np.isnan(np.asarray(ra_off, dtype=float)), (exp, ra_off)
        assert np.isnan(np.asarray(dec_off, dtype=float)), (exp, dec_off)


def test_SPARSE_base_still_MEASURES_the_offset(tmp_path, monkeypatch):
    """The guard must refuse the dense case without swallowing the sparse one:
    a 10" grid is above the 3" floor, so the diagnostic this PR is protecting
    has to still produce a real number.  Without this, NaN-ing unconditionally
    would pass the test above."""
    monkeypatch.setenv('MERGE_REMATCH_DIAGNOSTICS', '1')
    out = merge_catalogs.combine_singleframe(
        _frames(tmp_path, spacing_arcsec=10.0, shift_mas=50.0),
        nanaverage=_nanaverage())
    # exposure 1 is the base itself (all self-match -> NaN); exposure 2 is the
    # shifted one and is what carries the measurement.
    ra_off, dec_off = out.meta['offsets'][2]
    ra_mas = float(u.Quantity(ra_off).to(u.marcsec).value)
    assert np.isfinite(ra_mas), ra_off
    assert 30.0 < ra_mas < 70.0, f"the injected 50 mas shift was not recovered: {ra_mas}"


@pytest.mark.parametrize("flag", ["1", None],
                         ids=["with_diagnostics_flag", "without_diagnostics_flag"])
def test_DENSE_base_with_realign_TRUE_raises(tmp_path, monkeypatch, flag):
    """realign=True APPLIES the median to the coordinates, so a dense base has
    to stop the run outright -- in both flag states, since `rematch` is
    `realign or <flag>` and realign already forces it."""
    if flag is None:
        monkeypatch.delenv('MERGE_REMATCH_DIAGNOSTICS', raising=False)
    else:
        monkeypatch.setenv('MERGE_REMATCH_DIAGNOSTICS', flag)
    with pytest.raises(DenseNNMedianAstrometryError):
        merge_catalogs.combine_singleframe(
            _frames(tmp_path, spacing_arcsec=1.0), realign=True,
            nanaverage=_nanaverage())


def test_the_refused_value_is_not_printed_as_a_measurement(tmp_path, monkeypatch, capsys):
    """A log line reading `was offset by X +/- Y` is the same hazard as storing
    the number, one step removed: it is the form someone reads off a log and
    types into an offsets table.  On the refused path the value may appear only
    as evidence, never in measurement grammar."""
    monkeypatch.setenv('MERGE_REMATCH_DIAGNOSTICS', '1')
    merge_catalogs.combine_singleframe(
        _frames(tmp_path, spacing_arcsec=1.0), nanaverage=_nanaverage())
    out = capsys.readouterr().out
    assert "was offset by" not in out, out
    assert "REFUSED" in out and "NOT MEASURED" in out, out
