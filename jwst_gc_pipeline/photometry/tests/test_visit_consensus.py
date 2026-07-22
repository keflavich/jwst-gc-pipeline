"""Tests for the visit-consensus astrometry failsafe (visit_consensus.py).

Synthetic star fields only — no data dependencies.  The key behaviors under
test are the ones that have historically failed silently:
  * a single misaligned exposure is FOUND against the visit consensus,
    including when the shift is huge (the brick-1182 20" class, via sweep);
  * an aligned visit does NOT produce false corrections;
  * the reference tie refuses to sign off on a single check.
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry.visit_consensus import (
    ConsensusBuildError, build_visit_consensus, filter_wavelength_um,
    measure_reference_tie, pick_reference_anchor_filter, select_reliable_stars,
)

RA0, DEC0 = 266.5, -28.7
COSD = np.cos(np.radians(DEC0))
RNG_SEED = 42


def _field(n=400, extent_arcsec=90.0, rng=None):
    rng = rng or np.random.default_rng(RNG_SEED)
    ra = RA0 + rng.uniform(0, extent_arcsec, n) / 3600.0 / COSD
    dec = DEC0 + rng.uniform(0, extent_arcsec, n) / 3600.0
    return ra, dec


def _exposure_table(ra, dec, visit="001", exposure=1, module="nrcb1",
                    filtername="F212N", dra_mas=0.0, ddec_mas=0.0,
                    noise_mas=1.0, rng=None, raoffset=0.1, deoffset=-0.05):
    """Synthetic per-frame catalog: true positions + centroid noise + an
    optional rigid offset (an im0 alignment error)."""
    rng = rng or np.random.default_rng(RNG_SEED + exposure)
    n = len(ra)
    ra_obs = ra + (dra_mas + rng.normal(0, noise_mas, n)) / 3.6e6 / COSD
    dec_obs = dec + (ddec_mas + rng.normal(0, noise_mas, n)) / 3.6e6
    tbl = Table()
    tbl["skycoord"] = SkyCoord(ra=ra_obs * u.deg, dec=dec_obs * u.deg, frame="icrs")
    tbl["flux_fit"] = rng.uniform(1e3, 1e5, n)
    tbl["flux_err"] = tbl["flux_fit"] / 100.0
    tbl["qfit"] = rng.uniform(0.01, 0.05, n)
    tbl.meta.update(VISIT=visit, EXPOSURE=f"{exposure:05d}", MODULE=module,
                    FILTER=filtername, RAOFFSET=raoffset, DEOFFSET=deoffset)
    return tbl


def _visit_tables(n_exp=4, misaligned=None, **kwargs):
    """misaligned: dict exposure_number -> (dra_mas, ddec_mas)."""
    misaligned = misaligned or {}
    ra, dec = _field()
    tables = []
    for e in range(1, n_exp + 1):
        dra, ddec = misaligned.get(e, (0.0, 0.0))
        tables.append(_exposure_table(ra, dec, exposure=e, dra_mas=dra,
                                      ddec_mas=ddec, **kwargs))
    return tables


def test_aligned_visit_no_false_misalignment():
    cons = build_visit_consensus(_visit_tables(), context="test-aligned")
    assert cons["consensus_ok"]
    assert len(cons["coords"]) >= 50
    for exp in cons["exposures"]:
        assert not exp["misaligned"], exp
        assert exp["vs_consensus"]["off"] < 2.0


def test_small_misalignment_detected():
    # 6 mas: comfortably above the 2 mas tolerance, far below any window
    cons = build_visit_consensus(
        _visit_tables(misaligned={2: (6.0, 0.0)}), context="test-6mas")
    flagged = [e for e in cons["exposures"] if e["misaligned"]]
    assert len(flagged) == 1
    assert flagged[0]["key"][1] == 2
    res = flagged[0]["vs_consensus"]
    # measured correction = consensus - exposure = the full injected offset,
    # negated (median re-centring keeps the bad exposure out of the frame)
    assert res["dra"] == pytest.approx(-6.0, abs=1.5)
    assert res["off"] == pytest.approx(6.0, abs=1.5)
    assert not res["swept"]


def test_huge_misalignment_found_by_sweep():
    # the brick-1182 class: ~20" rigid shift.  The narrow window contains zero
    # true pairs; only the sweep finds it.  It must be flagged, never absorbed.
    cons = build_visit_consensus(
        _visit_tables(misaligned={3: (20000.0, 4000.0)}), context="test-20as")
    flagged = [e for e in cons["exposures"] if e["misaligned"]]
    assert len(flagged) == 1
    assert flagged[0]["key"][1] == 3
    res = flagged[0]["vs_consensus"]
    assert res["swept"]
    assert res["off"] > 10000.0


def test_consensus_positions_recover_truth():
    ra, dec = _field()
    tables = [
        _exposure_table(ra, dec, exposure=e, noise_mas=1.5)
        for e in range(1, 5)
    ]
    cons = build_visit_consensus(tables, context="test-recover")
    truth = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    idx, sep, _ = cons["coords"].match_to_catalog_sky(truth)
    # consensus of 4 exposures with 1.5 mas noise -> per-star ~0.75 mas
    assert np.median(sep.mas) < 1.5


def test_scatter_reads_true_noise_not_float_cancellation():
    """Per-star scatter was computed as sum(x^2)/n - mean^2 on RAW ~266 deg
    coordinates: catastrophic float64 cancellation fabricated a ~10-15 mas
    scatter floor.  On synthetic 3 mas centroid noise the reported scatter_mas
    must read ~3-4 mas (RA+Dec combined, /n variance), never ~15."""
    cons = build_visit_consensus(_visit_tables(noise_mas=3.0),
                                 context="test-scatter")
    med = float(np.median(cons["scatter_mas"]))
    # 4 exposures, sigma=3 mas per coordinate, biased /n variance:
    # E[scatter] ~ sqrt(2 * 9 * 3/4) ~ 3.7 mas.  The cancellation bug read >10.
    assert 2.0 < med < 6.0, f"median scatter {med} mas; expected ~3.7 mas"


def test_too_few_exposures_raises():
    with pytest.raises(ConsensusBuildError):
        build_visit_consensus(_visit_tables(n_exp=1), context="test-single")


def test_select_reliable_stars_cuts():
    tbl = _visit_tables(n_exp=2)[0]
    tbl["qfit"][:10] = 0.5           # bad fits
    tbl["flux_err"][10:20] = tbl["flux_fit"][10:20]  # snr=1
    keep = select_reliable_stars(tbl)
    assert not keep[:20].any()
    assert keep[20:].all()


def test_filter_wavelengths_and_anchor():
    assert filter_wavelength_um("F212N") == pytest.approx(2.12)
    assert filter_wavelength_um("F410M") == pytest.approx(4.10)
    assert filter_wavelength_um("f770w") == pytest.approx(7.70)
    # F212N is the closest to VIRAC2 Ks (2.149 um)
    assert pick_reference_anchor_filter(
        ["F115W", "F182M", "F212N", "F410M", "F480M"]) == "F212N"
    assert pick_reference_anchor_filter(["F115W", "F410M"]) == "F115W"


# ---------------------------------------------------------------------------
# reference tie
# ---------------------------------------------------------------------------

def _reference_sets(ra, dec, dense_extra=3000, rng=None):
    """Reference = the same stars (as VIRAC2 would see the bright ones) plus a
    dense unrelated filler population; sparse = every 10th real star (Gaia)."""
    rng = rng or np.random.default_rng(7)
    fr, fd = _field(n=dense_extra, rng=rng)
    ref_all = SkyCoord(ra=np.concatenate([ra, fr]) * u.deg,
                       dec=np.concatenate([dec, fd]) * u.deg, frame="icrs")
    ref_sparse = SkyCoord(ra=ra[::10] * u.deg, dec=dec[::10] * u.deg, frame="icrs")
    return ref_all, ref_sparse


def test_reference_tie_measures_and_signs_off():
    ra, dec = _field()
    # consensus sits 10 mas off the reference frame
    cons = SkyCoord(ra=(ra - 10.0 / 3.6e6 / COSD) * u.deg, dec=dec * u.deg,
                    frame="icrs")
    ref_all, ref_sparse = _reference_sets(ra, dec)
    tie = measure_reference_tie(cons, ref_all, ref_sparse, context="test-tie",
                                grid_nx=2, grid_ny=2)
    assert tie["vs_full"]["ok"]
    assert tie["dra_mas"] == pytest.approx(10.0, abs=2.0)
    assert abs(tie["ddec_mas"]) < 2.0
    assert tie["cross_reference"]["agree"]
    assert tie["apply_ok"]


def test_reference_tie_bulk_is_samestar_refined():
    """A small, verified tie -> the reported bulk comes from the SAME-STAR
    refinement (not the histogram peak, biased against a dense reference; memory
    histogram-vs-samestar-offset-bias). Total = histogram + matched-pair residual."""
    ra, dec = _field()
    cons = SkyCoord(ra=(ra - 10.0 / 3.6e6 / COSD) * u.deg, dec=dec * u.deg,
                    frame="icrs")
    ref_all, ref_sparse = _reference_sets(ra, dec)
    tie = measure_reference_tie(cons, ref_all, ref_sparse, context="ss",
                                grid_nx=2, grid_ny=2)
    assert tie["bulk_source"] == "same-star"
    assert tie["same_star"] is not None
    # reported bulk == the same-star total (histogram offset + residual)
    assert tie["dra_mas"] == pytest.approx(tie["same_star"]["dra"], abs=1e-9)
    assert tie["dra_mas"] == pytest.approx(10.0, abs=2.0)


def test_reference_tie_large_offset_keeps_histogram():
    """A gross offset (found only by the window SWEEP) cannot be same-star
    refined (pairs ambiguous) -> the bulk falls back to the histogram value."""
    ra, dec = _field()
    cons = SkyCoord(ra=(ra - 20.0 / 3600.0 / COSD) * u.deg, dec=dec * u.deg,
                    frame="icrs")   # 20" -> res_a swept
    ref_all, ref_sparse = _reference_sets(ra, dec)
    tie = measure_reference_tie(cons, ref_all, ref_sparse, context="big",
                                grid_nx=2, grid_ny=2)
    assert tie["vs_full"]["swept"]
    assert tie["same_star"] is None
    assert tie["bulk_source"] == "histogram"
    assert tie["off_mas"] > 15000.0   # ~20" kept, not collapsed


def test_fine_sparse_gaia_disagreement_does_not_block():
    """GC policy (gc-gaia-frame-not-catalog): a fine (~30 mas) sparse-Gaia split
    is the sparse-noise regime, NOT a catalog conflict -- it is RECORDED (fine
    cross_reference agree=False) but must NOT block a coherent VIRAC tie."""
    ra, dec = _field()
    cons = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    ref_all, _ = _reference_sets(ra, dec)
    # sparse reference shifted 30 mas -> below the gross tol (100 mas)
    ref_sparse_bad = SkyCoord(ra=(ra[::10] + 30.0 / 3.6e6 / COSD) * u.deg,
                              dec=dec[::10] * u.deg, frame="icrs")
    tie = measure_reference_tie(cons, ref_all, ref_sparse_bad,
                                context="test-fine-disagree", grid_nx=2, grid_ny=2)
    assert not tie["cross_reference"]["agree"]   # fine check flags it (diagnostic)
    assert tie["cross_reference_gross_ok"]        # gross check is fine
    assert tie["apply_ok"]                        # ... so the VIRAC tie still applies


def test_gross_sparse_disagreement_still_blocks():
    """A GROSS sparse split (spurious/window-limited VIRAC peak, the brick-1182
    v001 ~700 mas tell) MUST still block -- the gross cross-check is retained."""
    ra, dec = _field()
    cons = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    ref_all, _ = _reference_sets(ra, dec)
    # sparse reference shifted 300 mas -> above the gross tol (100 mas)
    ref_sparse_bad = SkyCoord(ra=(ra[::10] + 300.0 / 3.6e6 / COSD) * u.deg,
                              dec=dec[::10] * u.deg, frame="icrs")
    tie = measure_reference_tie(cons, ref_all, ref_sparse_bad,
                                context="test-gross-disagree", grid_nx=2, grid_ny=2)
    assert not tie["cross_reference_gross_ok"]
    assert not tie["apply_ok"]


def test_unmeasurable_sparse_does_not_block():
    """Extreme-sparse GC regime (arches/quintuplet/sgra): too few Gaia stars to
    form a coherent sparse peak -> sep_mas is nan.  An UNMEASURABLE cross-check
    must NOT block a coherent VIRAC tie -- gating on nan would re-block exactly
    the tie this policy keeps.  Only a FINITE gross split can block."""
    ra, dec = _field()
    cons = SkyCoord(ra=(ra - 10.0 / 3.6e6 / COSD) * u.deg, dec=dec * u.deg,
                    frame="icrs")
    ref_all, _ = _reference_sets(ra, dec)
    # 3 Gaia stars -> far below min_pairs -> measure_offset returns None ->
    # agree_across_references -> sep_mas = nan
    ref_sparse_tiny = SkyCoord(ra=ra[:3] * u.deg, dec=dec[:3] * u.deg, frame="icrs")
    tie = measure_reference_tie(cons, ref_all, ref_sparse_tiny,
                                context="test-unmeasurable-sparse",
                                grid_nx=2, grid_ny=2)
    assert not np.isfinite(tie["cross_reference"]["sep_mas"])  # sparse unmeasurable
    assert tie["cross_reference_gross_ok"]   # nan does NOT block
    assert tie["vs_full"]["ok"]
    assert tie["apply_ok"]                    # VIRAC-coherent tie still applies


# ---------------------------------------------------------------------------
# mosaic visits (2026-07-12): a visit's exposures span DISJOINT pointing tiles
# and two modules -- "no tie to the anchor" is geometry, not misalignment
# ---------------------------------------------------------------------------

def _tile_tables(ra_offset_arcsec=0.0, n_exp=2, exp0=1, n=300, seed=13,
                 misaligned=None, **kwargs):
    rng = np.random.default_rng(seed)
    ra = RA0 + (rng.uniform(0, 60.0, n) + ra_offset_arcsec) / 3600.0 / COSD
    dec = DEC0 + rng.uniform(0, 60.0, n) / 3600.0
    misaligned = misaligned or {}
    return [
        _exposure_table(ra, dec, exposure=e,
                        dra_mas=misaligned.get(e, (0, 0))[0],
                        ddec_mas=misaligned.get(e, (0, 0))[1], **kwargs)
        for e in range(exp0, exp0 + n_exp)
    ]


def test_mosaic_disjoint_tiles_build_components_not_failures():
    # two disjoint tiles (200" apart), 2 exposures each: the old anchor-seeded
    # build raised ConsensusBuildError; now = 2 components, all verified
    tables = (_tile_tables(0.0, exp0=1, seed=21)
              + _tile_tables(200.0, exp0=3, seed=22))
    cons = build_visit_consensus(tables, context="test-mosaic")
    assert cons["n_components"] == 2
    assert cons["consensus_ok"]
    for exp in cons["exposures"]:
        assert exp["internal_tie"]
        assert not exp["unverified"]
        assert not exp["misaligned"]


def test_mosaic_misalignment_detected_within_component():
    # 3 exposures in the second tile: the component median isolates the one
    # bad exposure.  (With only TWO exposures in a component the blame is
    # physically ambiguous -- the median splits it +-off/2 and BOTH get
    # flagged, which is the safe direction: over-flag, never hide.)
    tables = (_tile_tables(0.0, exp0=1, seed=23)
              + _tile_tables(200.0, n_exp=3, exp0=3, seed=24,
                             misaligned={4: (7.0, 0.0)}))
    cons = build_visit_consensus(tables, context="test-mosaic-bad")
    flagged = [e for e in cons["exposures"] if e["misaligned"]]
    assert len(flagged) == 1
    assert flagged[0]["key"][1] == 4
    assert flagged[0]["vs_consensus"]["off"] == pytest.approx(7.0, abs=1.5)


def test_two_exposure_component_ambiguity_overflags_never_hides():
    # n=2 component with one bad exposure: cannot attribute -> both read
    # +-off/2 from the component frame and both are flagged (>2 mas).  The
    # failure must NEVER be absorbed into a silent pass.
    tables = _tile_tables(0.0, n_exp=2, exp0=1, seed=27,
                          misaligned={2: (8.0, 0.0)})
    cons = build_visit_consensus(tables, context="test-n2-ambiguity")
    flagged = [e for e in cons["exposures"] if e["misaligned"]]
    assert len(flagged) == 2
    for e in flagged:
        assert e["vs_consensus"]["off"] == pytest.approx(4.0, abs=1.5)


def test_isolated_exposure_is_unverified_not_misaligned():
    # 2 overlapping exposures at tile A + ONE exposure alone at a far tile:
    # the loner has no >=2-exposure consensus coverage -> UNVERIFIED, never
    # silently passed, never called misaligned
    tables = (_tile_tables(0.0, exp0=1, seed=25)
              + _tile_tables(300.0, n_exp=1, exp0=5, seed=26))
    cons = build_visit_consensus(tables, context="test-island")
    assert not cons["consensus_ok"]
    lone = [e for e in cons["exposures"] if e["key"][1] == 5][0]
    assert lone["unverified"]
    assert not lone["misaligned"]
    others = [e for e in cons["exposures"] if e["key"][1] != 5]
    assert all(not e["unverified"] for e in others)


def test_large_visit_parity_halves_detects_misalignment():
    """>16 exposures triggers the O(n) parity-halves tie (the 9-hour
    union-growth fix); a single misaligned exposure must still be isolated."""
    tables = _visit_tables(n_exp=20, misaligned={7: (8.0, 0.0)})
    cons = build_visit_consensus(tables, context="test-parity")
    assert cons["n_components"] == 1
    flagged = [e for e in cons["exposures"] if e["misaligned"]]
    assert len(flagged) == 1
    assert flagged[0]["key"][1] == 7
    assert flagged[0]["vs_consensus"]["off"] == pytest.approx(8.0, abs=2.0)


def test_footprint_crop_is_lossless_for_the_measured_offset():
    """_crop_to_footprint removes only reference stars no sweep window could
    pair with the target; the measured offset must be unchanged."""
    from jwst_gc_pipeline.photometry.visit_consensus import _crop_to_footprint
    from jwst_gc_pipeline.photometry.astrometry_offsets import measure_offset
    rng = np.random.default_rng(7)
    # wide mosaic-scale reference; target covers a small corner of it
    ref = SkyCoord(ra=(266.5 + rng.uniform(0, 0.2, 40000)) * u.deg,
                   dec=(-28.7 + rng.uniform(0, 0.2, 40000)) * u.deg,
                   frame="icrs")
    sel = (ref.ra.deg < 266.55) & (ref.dec.deg < -28.65)
    # target = that corner's stars shifted by a known 12 mas
    tgt = SkyCoord(ra=(ref.ra.deg[sel] - 12.0 / 3.6e6 / np.cos(np.radians(-28.7))) * u.deg,
                   dec=ref.dec.deg[sel] * u.deg, frame="icrs")
    cropped = _crop_to_footprint(ref, tgt)
    assert len(cropped) < len(ref)
    full = measure_offset(tgt, ref)
    fast = measure_offset(tgt, cropped)
    assert fast["ok"]
    assert fast["dra"] == pytest.approx(full["dra"], abs=0.5)
    assert fast["ddec"] == pytest.approx(full["ddec"], abs=0.5)
    assert fast["off"] == pytest.approx(12.0, abs=2.0)


def test_footprint_crop_falls_back_on_wrap_and_no_overlap():
    from jwst_gc_pipeline.photometry.visit_consensus import _crop_to_footprint
    rng = np.random.default_rng(8)
    ref = SkyCoord(ra=rng.uniform(100, 100.1, 500) * u.deg,
                   dec=rng.uniform(0, 0.1, 500) * u.deg, frame="icrs")
    # RA-wrap-straddling target: box test invalid -> full reference returned
    wrap = SkyCoord(ra=np.array([359.99, 0.01]) * u.deg,
                    dec=np.array([0.0, 0.0]) * u.deg, frame="icrs")
    assert len(_crop_to_footprint(ref, wrap)) == len(ref)
    # disjoint target: <100 boxed stars -> full reference (caller's
    # too-few-pairs/unverified path must behave exactly as uncropped)
    far = SkyCoord(ra=np.array([200.0, 200.01]) * u.deg,
                   dec=np.array([50.0, 50.01]) * u.deg, frame="icrs")
    assert len(_crop_to_footprint(ref, far)) == len(ref)


def test_cap_stars_deterministic_and_preserves_peak():
    from jwst_gc_pipeline.photometry.visit_consensus import _cap_stars
    from jwst_gc_pipeline.photometry.astrometry_offsets import measure_offset
    rng = np.random.default_rng(9)
    ra = 266.5 + rng.uniform(0, 0.05, 30000)
    dec = -28.7 + rng.uniform(0, 0.05, 30000)
    big = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    small = SkyCoord(ra=(ra[:8000] + 10.0 / 3.6e6) * u.deg,
                     dec=dec[:8000] * u.deg, frame="icrs")
    capped1 = _cap_stars(big, n_max=10000)
    capped2 = _cap_stars(big, n_max=10000)
    assert len(capped1) == 10000
    assert np.array_equal(capped1.ra.deg, capped2.ra.deg)
    res = measure_offset(small, capped1)
    assert res["ok"]
    assert res["ddec"] == pytest.approx(0.0, abs=1.0)


def test_kdtree_reference_identical_to_plain_path():
    """measure_offset against a KDTreeReference must reproduce the plain
    (astropy search_around_sky) path exactly: same deterministic subsample
    RNG, exact within-radius pair sets, same histogram."""
    from jwst_gc_pipeline.photometry.astrometry_offsets import (
        KDTreeReference, measure_offset)
    rng = np.random.default_rng(11)
    ra = 266.5 + rng.uniform(0, 0.08, 60000)
    dec = -28.7 + rng.uniform(0, 0.08, 60000)
    ref_sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    tgt = SkyCoord(ra=(ra[:15000] + 25.0 / 3.6e6 / np.cos(np.radians(-28.7))) * u.deg,
                   dec=(dec[:15000] - 8.0 / 3.6e6) * u.deg, frame="icrs")
    plain = measure_offset(tgt, ref_sc)
    tree = measure_offset(tgt, KDTreeReference(ref_sc))
    assert tree["ok"] and plain["ok"]
    for k in ("dra", "ddec", "off", "npairs", "contrast", "n_peak",
              "window_arcsec"):
        assert tree[k] == pytest.approx(plain[k], rel=1e-9), k
    assert tree["off"] == pytest.approx(np.hypot(25.0, 8.0), abs=2.0)
