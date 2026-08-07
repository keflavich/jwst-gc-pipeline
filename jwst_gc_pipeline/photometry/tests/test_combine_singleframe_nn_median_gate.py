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
