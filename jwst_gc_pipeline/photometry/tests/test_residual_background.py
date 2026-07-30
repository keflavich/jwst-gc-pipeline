"""Per-source residual-footprint background (Jay Anderson 3x3 convention).

The point of these columns is that they are measured on the STAR-SUBTRACTED
residual, not on the data being fit -- so they trace the extended emission at
the source rather than the neighbouring stars that photutils' LocalBackground
annulus necessarily includes.  The tests below pin that distinction, the
weighting used to combine frames, and the edge/mask bookkeeping.
"""
import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry.residual_background import (
    FOOTPRINT_BOX, MERGED_RESBKG_COLUMNS, RESBKG_COLUMNS, combine_frames,
    measure_footprint_background)


# ---------------------------------------------------------------- per frame --
def test_measures_the_footprint_not_the_star():
    """The semantic claim: run on the residual, the value is the BACKGROUND.

    Same field measured two ways -- on the data (star present) and on the
    residual (star subtracted).  Only the residual gives the background level.
    If this ever passes on both, the caller is handing in the wrong image.
    """
    ny = nx = 41
    yy, xx = np.mgrid[0:ny, 0:nx]
    background = 7.0
    star = 500.0 * np.exp(-((xx - 20) ** 2 + (yy - 20) ** 2) / (2 * 1.5 ** 2))
    data = background + star
    residual = data - star                     # star-only model removed

    m_res, r_res, n_res = measure_footprint_background(residual, [20.0], [20.0])
    m_dat, _, _ = measure_footprint_background(data, [20.0], [20.0])

    assert n_res[0] == FOOTPRINT_BOX ** 2
    assert m_res[0] == pytest.approx(background)
    assert r_res[0] == pytest.approx(0.0, abs=1e-9)
    # the point of the test: the SAME footprint on the data is dominated by the
    # star, ~70x higher.  If both agreed, the caller is passing the wrong image.
    assert m_dat[0] > 10 * background
    assert m_dat[0] / m_res[0] > 50


def test_mean_and_rms_are_the_footprint_statistics():
    rng = np.random.default_rng(0)
    img = rng.normal(3.0, 2.0, size=(60, 60))
    x, y = 30.0, 25.0
    mean, rms, npix = measure_footprint_background(img, [x], [y], box=3)
    patch = img[int(round(y)) - 1:int(round(y)) + 2,
                int(round(x)) - 1:int(round(x)) + 2]
    assert npix[0] == 9
    assert mean[0] == pytest.approx(patch.mean())
    assert rms[0] == pytest.approx(patch.std(ddof=1))


def test_box_is_forced_odd_so_the_footprint_stays_centred():
    img = np.zeros((21, 21))
    img[10, 10] = 9.0                          # only the exact centre is hot
    mean, _, npix = measure_footprint_background(img, [10.0], [10.0], box=4)
    assert npix[0] == 25                       # 4 -> 5, not 4
    assert mean[0] == pytest.approx(9.0 / 25)


def test_edges_and_off_array_are_distinguishable():
    """npix must separate 'measured on a clipped footprint' from 'no data'."""
    img = np.ones((20, 20))
    mean, _, npix = measure_footprint_background(
        img, [0.0, 19.0, 10.0, -50.0], [0.0, 19.0, 10.0, -50.0])
    assert npix[0] == 4 and npix[1] == 4       # corners: quarter footprint
    assert npix[2] == 9                        # interior: full
    assert npix[3] == 0 and np.isnan(mean[3])  # off the array entirely
    assert np.isfinite(mean[:3]).all()


def test_masked_and_nonfinite_pixels_are_excluded():
    img = np.ones((21, 21))
    img[10, 10] = np.nan                       # e.g. an un-interpolated bad px
    mask = np.zeros_like(img, dtype=bool)
    mask[9, 9] = True                          # e.g. a DQ-flagged pixel
    mean, _, npix = measure_footprint_background(img, [10.0], [10.0], mask=mask)
    assert npix[0] == 7                        # 9 - 1 NaN - 1 masked
    assert mean[0] == pytest.approx(1.0)       # and the NaN did not propagate


def test_single_valid_pixel_reports_nan_rms_not_zero():
    """One pixel has no scatter; reporting 0 would be false precision."""
    img = np.full((21, 21), np.nan)
    img[10, 10] = 4.0
    mean, rms, npix = measure_footprint_background(img, [10.0], [10.0])
    assert npix[0] == 1 and mean[0] == pytest.approx(4.0) and np.isnan(rms[0])


def test_nonfinite_positions_are_skipped_not_crashed():
    img = np.ones((20, 20))
    mean, _, npix = measure_footprint_background(img, [np.nan, 10.0],
                                                 [10.0, np.nan])
    assert npix.tolist() == [0, 0] and np.isnan(mean).all()


def test_rejects_a_non_2d_image():
    with pytest.raises(ValueError, match='must be 2-D'):
        measure_footprint_background(np.ones((3, 3, 3)), [1.0], [1.0])


# ------------------------------------------------------------- combination --
def test_combine_is_inverse_variance_weighted_by_npix_over_rms_squared():
    """A quieter footprint must dominate; flux error must not enter."""
    mean = np.array([[10.0, 20.0]])
    rms = np.array([[1.0, 10.0]])              # frame 0 is 10x quieter
    npix = np.array([[9.0, 9.0]])
    out = combine_frames(mean, rms, npix)
    w0, w1 = 9 / 1.0 ** 2, 9 / 10.0 ** 2
    assert out['resbkg_mean_avg'][0] == pytest.approx(
        (w0 * 10 + w1 * 20) / (w0 + w1))
    assert out['resbkg_mean_err'][0] == pytest.approx(1 / np.sqrt(w0 + w1))
    assert out['resbkg_nframes'][0] == 2


def test_combine_sigma_clips_an_outlier_frame():
    """One frame where a neighbour was badly subtracted must not drag the mean.

    Uses frames with REALISTIC scatter, not identical values: nine identical
    values give ``mad_std == 0``, so the outlier sits at infinite significance
    and *any* sigma -- including 1e6 -- would clip it.  That version of this
    test passed for a broken threshold.
    """
    rng = np.random.default_rng(7)
    good = rng.normal(5.0, 1.0, 9)
    mean = np.array([list(good) + [500.0]])
    rms = np.ones((1, 10))
    npix = np.full((1, 10), 9.0)

    out = combine_frames(mean, rms, npix, sigma=3.0)
    assert out['resbkg_nframes'][0] < 10                  # the outlier is gone
    assert out['resbkg_mean_avg'][0] == pytest.approx(good.mean(), abs=1.0)

    # and the threshold is load-bearing: a huge sigma keeps the outlier
    loose = combine_frames(mean, rms, npix, sigma=1e6)
    assert loose['resbkg_nframes'][0] == 10
    assert loose['resbkg_mean_avg'][0] > 50


def test_combine_reports_nan_scatter_for_a_single_frame():
    """Scatter across one frame is undefined, not 0 -- same trap as std_ra."""
    out = combine_frames(np.array([[3.0]]), np.array([[1.0]]),
                         np.array([[9.0]]))
    assert out['resbkg_nframes'][0] == 1
    assert out['resbkg_mean_avg'][0] == pytest.approx(3.0)
    assert np.isnan(out['resbkg_mean_std'][0])


def test_combine_scatter_is_the_across_frame_rms():
    """Pins the POPULATION (biased-low) weighted form deliberately, matching the
    existing std_*_avg convention in merge_catalogs.  The unbiased value for
    [4, 6] would be 1.414; 1.0 is the population form."""
    mean = np.array([[4.0, 6.0]])
    rms = np.ones((1, 2))
    npix = np.full((1, 2), 9.0)
    out = combine_frames(mean, rms, npix)
    assert out['resbkg_mean_avg'][0] == pytest.approx(5.0)
    assert out['resbkg_mean_std'][0] == pytest.approx(1.0)


def test_combine_drops_edge_clipped_and_unmeasured_frames():
    """A 1-pixel footprint is not a background; neither is a NaN."""
    mean = np.array([[5.0, 99.0, np.nan]])
    rms = np.array([[1.0, 1.0, 1.0]])
    npix = np.array([[9.0, 1.0, 0.0]])         # frame 1 edge-clipped to 1 px
    out = combine_frames(mean, rms, npix, min_npix=2)
    assert out['resbkg_nframes'][0] == 1
    assert out['resbkg_mean_avg'][0] == pytest.approx(5.0)


def test_combine_source_with_no_usable_frame_is_nan_not_zero():
    out = combine_frames(np.array([[np.nan, np.nan]]),
                         np.array([[np.nan, np.nan]]),
                         np.array([[0.0, 0.0]]))
    assert out['resbkg_nframes'][0] == 0
    for k in ('resbkg_mean_avg', 'resbkg_mean_std', 'resbkg_mean_err'):
        assert np.isnan(out[k][0]), k


def test_combine_rms_avg_is_the_weighted_mean_of_the_per_frame_rms():
    """Its VALUE was previously unasserted -- a mutant returning the sum passed."""
    mean = np.array([[5.0, 5.0]])
    rms = np.array([[1.0, 3.0]])
    npix = np.full((1, 2), 9.0)
    w = 9 / np.array([1.0, 9.0])
    out = combine_frames(mean, rms, npix)
    assert out['resbkg_rms_avg'][0] == pytest.approx(
        (w * np.array([1.0, 3.0])).sum() / w.sum(), rel=1e-5)


def test_combine_rejects_1d_input_instead_of_silently_transposing():
    """np.atleast_2d would read 3 sources x 1 frame as 1 source x 3 frames and
    return a single meaningless value."""
    with pytest.raises(ValueError, match='must be 2-D'):
        combine_frames([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [9.0, 9.0, 9.0])


def test_combine_does_not_blow_the_memory_budget():
    """combine_singleframe streams columns to stay inside a documented budget
    (its Phase 1 peaks ~27 GB on brick F200W).  An earlier version of this
    function upcast all three float32 stacks to float64 and held ~8 temporaries,
    peaking at ~7.8x the input size -- ~79 GB at that scale, which would OOM the
    serial merge path.  Pin the ratio, not the wall-clock size.
    """
    import tracemalloc

    n_src, n_frames = 40000, 16
    rng = np.random.default_rng(3)
    mean = rng.normal(4.0, 1.0, (n_src, n_frames)).astype(np.float32)
    rms = np.abs(rng.normal(1.0, 0.2, (n_src, n_frames))).astype(np.float32)
    npix = np.full((n_src, n_frames), 9.0, dtype=np.float32)
    inputs = mean.nbytes + rms.nbytes + npix.nbytes

    tracemalloc.start()
    combine_frames(mean, rms, npix)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak / inputs < 3.0, (
        f"combine_frames peaked at {peak / inputs:.1f}x its input size; the "
        f"float32/free-as-you-go handling has regressed")


def test_combine_returns_every_declared_column():
    out = combine_frames(np.array([[1.0, 2.0]]), np.ones((1, 2)),
                         np.full((1, 2), 9.0))
    assert set(out) == set(MERGED_RESBKG_COLUMNS)


def test_combine_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match='shapes differ'):
        combine_frames(np.ones((2, 3)), np.ones((2, 4)), np.ones((2, 3)))


# ------------------------------------------------------- expected behaviour --
def test_traces_extended_emission():
    """The stated expectation: the column should track diffuse structure.

    Build a residual with a smooth emission gradient and stars on top of it;
    subtract the stars; the recovered per-source background must reproduce the
    gradient at each position.
    """
    ny = nx = 80
    yy, xx = np.mgrid[0:ny, 0:nx]
    emission = 2.0 + 0.05 * xx                 # smooth gradient
    xs = np.array([15.0, 35.0, 55.0, 70.0])
    ys = np.full_like(xs, 40.0)
    stars = np.zeros_like(emission, dtype=float)
    for x0, y0 in zip(xs, ys):
        stars += 300.0 * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / 2.0)
    # NOT the trivial (emission + stars) - stars: that leaves an exactly
    # constant-per-footprint array and asserts nothing about the estimator.
    # Add noise and a slightly imperfect model so the test measures recovery
    # under realistic subtraction residual.  The model error has to be SMALL
    # (0.1%): on a 300-amplitude star even 0.5% biases the 3x3 mean by ~0.8,
    # which is the contamination pinned in
    # test_bright_source_subtraction_residual_biases_the_value.
    rng = np.random.default_rng(5)
    noise = rng.normal(0.0, 0.05, emission.shape)
    imperfect = stars * 1.001
    residual = (emission + stars + noise) - imperfect

    mean, rms, _ = measure_footprint_background(residual, xs, ys)
    truth = 2.0 + 0.05 * xs
    assert np.max(np.abs(mean - truth)) < 0.35, (mean, truth)
    assert np.all(np.diff(mean) > 0)                 # monotonic with emission
    assert np.corrcoef(mean, truth)[0, 1] > 0.99
    assert np.all(rms > 0)


def test_bright_source_subtraction_residual_biases_the_value():
    """PIN the documented limitation: this is NOT a pure sky measurement.

    On real data (brick F182M nrca1 m7) the brightest 1% of sources read
    resbkg_mean ~14.3 where the subtracted background map says ~5.5 -- i.e.
    ~+9 units, ~2.6x the true background, is PSF-subtraction residual rather
    than sky, and Spearman r(resbkg_mean, flux_fit) = 0.32.

    Reproduce the mechanism synthetically: the same fractional model error on a
    bright and a faint star biases the bright one far more, because the error
    scales with the flux while the background does not.  If this test ever
    fails because the bias vanished, the estimator changed and the docstring
    warning needs revisiting.
    """
    ny = nx = 60
    yy, xx = np.mgrid[0:ny, 0:nx]
    background = 4.0
    faint = 100.0 * np.exp(-((xx - 15) ** 2 + (yy - 30) ** 2) / 2.0)
    bright = 20000.0 * np.exp(-((xx - 45) ** 2 + (yy - 30) ** 2) / 2.0)
    stars = faint + bright
    residual = (background + stars) - stars * 1.01     # 1% under-subtraction

    mean, _, _ = measure_footprint_background(residual, [15.0, 45.0],
                                              [30.0, 30.0])
    bias_faint = mean[0] - background
    bias_bright = mean[1] - background
    assert abs(bias_faint) < 1.0                       # negligible when faint
    assert abs(bias_bright) > 5 * background           # dominant when bright
    assert abs(bias_bright) > 50 * abs(bias_faint)


def test_column_name_contracts_are_stable():
    """Downstream (merge_catalogs) keys off these names."""
    assert RESBKG_COLUMNS == ('resbkg_mean', 'resbkg_rms', 'resbkg_npix')
    assert 'resbkg_mean_avg' in MERGED_RESBKG_COLUMNS


# ------------------------------------------------------------- integration --
def test_attach_is_fail_soft_and_optional():
    """A diagnostic column set must never cost a frame its photometry."""
    from jwst_gc_pipeline.photometry.cataloging import _attach_residual_background
    import types

    result = Table({'x_fit': [10.0, 12.0], 'y_fit': [10.0, 12.0],
                    'flux_fit': [1.0, 2.0]})
    ctx = types.SimpleNamespace(mask=None)
    opts = types.SimpleNamespace(manual_residual_background=True,
                                 manual_residual_background_box=3)

    out = _attach_residual_background(result.copy(), np.ones((30, 30)), ctx, opts)
    assert all(c in out.colnames for c in RESBKG_COLUMNS)
    assert out['resbkg_mean'][0] == pytest.approx(1.0)

    # a broken residual must not raise -- the frame keeps its photometry
    broken = _attach_residual_background(result.copy(), np.ones((3, 3, 3)),
                                         ctx, opts)
    assert 'flux_fit' in broken.colnames
    assert not any(c in broken.colnames for c in RESBKG_COLUMNS)

    # and it can be switched off
    off = types.SimpleNamespace(manual_residual_background=False,
                                manual_residual_background_box=3)
    disabled = _attach_residual_background(result.copy(), np.ones((30, 30)),
                                           ctx, off)
    assert not any(c in disabled.colnames for c in RESBKG_COLUMNS)


def test_columns_survive_the_bad_xfit_row_drop_in_alignment():
    """The PR's own safety property: the columns are attached BEFORE
    save_photutils_results drops rows with non-finite x_fit, so they are
    filtered with those rows rather than left misaligned by one.

    Reproduces that drop (crowdsource_catalogs_long.py: ``result = result[~bad]``)
    and checks each surviving row still carries ITS OWN background value.
    """
    from jwst_gc_pipeline.photometry.cataloging import _attach_residual_background
    import types

    # a residual whose value encodes the column, so a shift shows up immediately
    residual = np.tile(np.arange(40, dtype=float), (40, 1))
    result = Table({'x_fit': [5.0, np.nan, 20.0, 30.0],
                    'y_fit': [5.0, 10.0, 20.0, 30.0],
                    'flux_fit': [1.0, 2.0, 3.0, 4.0]})
    ctx = types.SimpleNamespace(mask=None)
    opts = types.SimpleNamespace(manual_residual_background=True,
                                 manual_residual_background_box=3)
    _attach_residual_background(result, residual, ctx, opts)

    bad = ~np.isfinite(np.asarray(result['x_fit'], dtype=float))
    kept = result[~bad]
    assert len(kept) == 3
    for row in kept:
        assert row['resbkg_mean'] == pytest.approx(row['x_fit'])


def test_merge_block_combines_through_combine_singleframe(monkeypatch):
    """The merge integration was entirely untested: the stacking, the npix
    nan_to_num, the already_done exclusion and the emitted column set.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from jwst_gc_pipeline.photometry import merge_catalogs as mc

    rng = np.random.default_rng(11)
    n = 30
    ra = 266.5 + rng.normal(0, 1e-4, n)
    dec = -28.7 + rng.normal(0, 1e-4, n)

    tbls = []
    for j in range(4):
        t = Table()
        # dao branch: combine_singleframe keys off 'qfit' and reads
        # skycoord_centroid
        t['skycoord_centroid'] = SkyCoord(ra * u.deg, dec * u.deg)
        t['flux_fit'] = np.full(n, 100.0)
        t['flux_err'] = np.full(n, 1.0)
        t['qfit'] = np.full(n, 0.1)
        t['resbkg_mean'] = np.full(n, 4.0) + j * 0.0   # same in every frame
        t['resbkg_rms'] = np.full(n, 1.0)
        t['resbkg_npix'] = np.full(n, 9.0)
        t.meta = {'exposure': j, 'ra_offset': 0.0, 'dec_offset': 0.0,
                  'filter': 'F182M'}
        tbls.append(t)

    out = mc.combine_singleframe(tbls, nanaverage=mc.nanaverage_numpy)

    for col in MERGED_RESBKG_COLUMNS:
        assert col in out.colnames, f"{col} missing from the merged table"
    assert np.nanmedian(out['resbkg_mean_avg']) == pytest.approx(4.0, abs=1e-3)
    assert np.nanmedian(out['resbkg_nframes']) == 4
    # and the per-frame names must NOT also appear with Phase 2's flux-weighted
    # _avg suffix -- that would mean they were combined the wrong way too
    assert 'resbkg_mean_avg' in out.colnames
    assert 'std_resbkg_mean_avg' not in out.colnames


def test_attach_is_fail_soft_on_a_missing_y_fit_or_mask():
    """The guard checked only x_fit while the body read y_fit, so a table
    without y_fit raised KeyError straight out of a function whose contract is
    'never costs a frame its photometry'.  Same for a ctx without .mask.
    """
    from jwst_gc_pipeline.photometry.cataloging import _attach_residual_background
    import types

    opts = types.SimpleNamespace(manual_residual_background=True,
                                 manual_residual_background_box=3)
    no_y = Table({'x_fit': [10.0], 'flux_fit': [1.0]})
    out = _attach_residual_background(no_y, np.ones((30, 30)),
                                      types.SimpleNamespace(mask=None), opts)
    assert 'flux_fit' in out.colnames
    assert not any(c in out.colnames for c in RESBKG_COLUMNS)

    full = Table({'x_fit': [10.0], 'y_fit': [10.0], 'flux_fit': [1.0]})
    out2 = _attach_residual_background(full, np.ones((30, 30)),
                                       types.SimpleNamespace(), opts)
    assert 'flux_fit' in out2.colnames          # no AttributeError escaped


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
