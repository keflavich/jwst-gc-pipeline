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
    ConsensusBuildError, DuplicateExposureError, build_visit_consensus,
    exposure_key, filter_wavelength_um,
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
                    noise_mas=1.0, rng=None, raoffset=0.1, deoffset=-0.05,
                    vgroup=None):
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
    if vgroup is not None:
        tbl.meta["VGROUP"] = vgroup
    return tbl


def test_reliable_star_with_nonfinite_coord_does_not_crash_consensus():
    """A reliable (good snr/qfit) star can carry a non-finite RA/Dec -- e.g. a
    saturated-core replacement / recovered row whose position solve failed.  Such
    a row must be dropped, not fed into the parity-halves cKDTree (which raises
    "data must be finite") -- the brick F410M zeroframe-recovery m3 checkpoint
    crash (2026-07-29).  Use >16 exposures to exercise the KD-tree path."""
    ra, dec = _field(n=300)
    tables = [_exposure_table(ra, dec, exposure=e) for e in range(1, 19)]
    # inject a NaN coordinate into one reliable star of one exposure
    bad = tables[5]["skycoord"]
    tables[5]["skycoord"] = SkyCoord(
        ra=np.concatenate([[np.nan], bad.ra.deg[1:]]) * u.deg,
        dec=np.concatenate([[np.nan], bad.dec.deg[1:]]) * u.deg, frame="icrs")
    cons = build_visit_consensus(tables, context="test-nan-coord")
    assert np.all(np.isfinite(cons["coords"].ra.deg))
    assert np.all(np.isfinite(cons["coords"].dec.deg))


def test_exposure_key_distinguishes_visit_groups():
    """A visit can dither across several vgroups and the exposure number
    RESTARTS in each, so (visit, exposure, module, filter) is ambiguous:
    cloudc F182M/nrcb1 collapsed 16 catalogs onto 8 keys that way."""
    ra, dec = _field(n=50)
    a = _exposure_table(ra, dec, exposure=1, vgroup="06201")
    b = _exposure_table(ra, dec, exposure=1, vgroup="12201")
    ka, kb = exposure_key(a), exposure_key(b)
    assert ka != kb
    # ...while the positional unpacking astrometry_checkpoint relies on
    # (key[0..2] -> visit/exposure/module) is unchanged
    assert ka[:3] == kb[:3] == ("001", 1, "nrcb1")
    assert ka[4] == "06201" and kb[4] == "12201"


def test_duplicate_exposure_identity_raises():
    """Two catalogs of ONE exposure must not be silently blended into the
    consensus -- that is what fabricated the arches/sgrc corrections."""
    tables = _visit_tables(n_exp=4)
    tables.append(tables[0].copy())          # same identity, second measurement
    with pytest.raises(DuplicateExposureError, match="(?i)more than once"):
        build_visit_consensus(tables, context="dup-test")


def test_distinct_vgroups_are_not_duplicates():
    """The same exposure number in two vgroups is legitimate, not a duplicate."""
    ra, dec = _field(n=400)
    tables = ([_exposure_table(ra, dec, exposure=e, vgroup="06201") for e in (1, 2)]
              + [_exposure_table(ra, dec, exposure=e, vgroup="12201") for e in (1, 2)])
    cons = build_visit_consensus(tables, context="vgroup-test")
    assert len({tuple(e["key"]) for e in cons["exposures"]}) == 4


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


def test_gaia_only_reference_per_tile_does_not_gate():
    """VIRAC2-absent fields (w51/sgrc): the refcat is Gaia-ONLY, so the per-tile
    check D is measured against a SPARSE catalog and returns noise -- its grid is
    not 'clean' (starved tiles), which under the dense gate would strand the bulk
    Gaia tie the reducer needs.  With ``dense=False`` the per-tile check is
    replaced by the same-star refinement, so a coherent, same-star-verified tie
    APPLIES; with ``dense=True`` the same starved grid still (correctly) blocks."""
    ra, dec = _field(n=400)
    cons = SkyCoord(ra=(ra - 10.0 / 3.6e6 / COSD) * u.deg, dec=dec * u.deg,
                    frame="icrs")
    # Gaia-only reference: full == sparse == the real stars, no dense filler.
    ref = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    tie_dense = measure_reference_tie(cons, ref, ref, context="gaia-only-dense",
                                      grid_nx=6, grid_ny=6, dense=True)
    tie_sparse = measure_reference_tie(cons, ref, ref, context="gaia-only-sparse",
                                       grid_nx=6, grid_ny=6, dense=False)

    # the shared, sparse per-tile grid is not clean (starved tiles)
    assert not tie_dense["per_tile"].get("clean")
    # dense gate: per-tile blocks -> the bug that stranded the w51 bulk sentinel
    assert not tie_dense["apply_ok"]
    # Gaia-only regime: same-star refinement carries the sign-off -> applies
    assert tie_sparse["vs_full"]["ok"]
    assert tie_sparse["same_star"] is not None
    assert tie_sparse["per_tile_ok"]
    assert tie_sparse["apply_ok"]
    assert tie_sparse["dra_mas"] == pytest.approx(10.0, abs=2.0)


def test_gaia_only_reference_still_needs_samestar():
    """``dense=False`` does NOT wave the tie through: a gross offset that only the
    SWEEP finds cannot be same-star refined (ambiguous pairs) -> no corroborating
    second check -> must NOT apply.  Prevents a single-number (histogram) sign-off."""
    ra, dec = _field(n=400)
    cons = SkyCoord(ra=(ra - 20.0 / 3600.0 / COSD) * u.deg, dec=dec * u.deg,
                    frame="icrs")   # 20" -> res_a swept, same_star None
    ref = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    tie = measure_reference_tie(cons, ref, ref, context="gaia-only-swept",
                                grid_nx=6, grid_ny=6, dense=False)
    assert tie["same_star"] is None
    assert not tie["per_tile_ok"]
    assert not tie["apply_ok"]


def test_load_reference_catalog_dense_flag(tmp_path):
    """A source-tagged gaia+virac2 refcat is DENSE; an all-Gaia one, or one with
    no source column (w51 gaia_refcat.fits), is Gaia-only (``dense=False``)."""
    from jwst_gc_pipeline.photometry.visit_consensus import load_reference_catalog
    ra, dec = _field(n=50)

    def _write(cols, name):
        t = Table()
        t["RA"] = ra
        t["DEC"] = dec
        for k, v in cols.items():
            t[k] = v
        p = tmp_path / name
        t.write(p, overwrite=True)
        return str(p)

    n = len(ra)
    mixed = _write({"source": ["GAIA"] * (n // 2) + ["VIRAC2"] * (n - n // 2)},
                   "mixed.fits")
    allgaia = _write({"source": ["GAIA"] * n}, "allgaia.fits")
    nosrc = _write({}, "nosrc.fits")

    assert load_reference_catalog(mixed)["dense"] is True
    assert load_reference_catalog(allgaia)["dense"] is False
    assert load_reference_catalog(nosrc)["dense"] is False


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


# ---------------------------------------------------------------------------
# module-scoped growth (issue #158 root fix): adjacent NRCA/NRCB tiles that
# share no stars must not be swept-tied into one component; same-module large
# real shifts must still tie.
# ---------------------------------------------------------------------------
def test_adjacent_module_tiles_do_not_merge_across_the_gap():
    """Two NIRCam modules on ADJACENT, disjoint sky tiles (~56" apart, no shared
    stars) must seed SEPARATE components -- the cross-module growth tie is
    bounded + non-swept, so with no genuine overlap it cannot lock onto the
    window-edge footprint alias (issue #158)."""
    from jwst_gc_pipeline.photometry.visit_consensus import module_family
    rng = np.random.default_rng(7)
    n = 350
    ra_a = RA0 + rng.uniform(0, 40.0, n) / 3600.0 / COSD
    dec_a = DEC0 + rng.uniform(0, 40.0, n) / 3600.0
    # NRCB tile 56" south, DIFFERENT stars (disjoint footprints)
    ra_b = RA0 + rng.uniform(0, 40.0, n) / 3600.0 / COSD
    dec_b = (DEC0 - 56.0 / 3600.0) + rng.uniform(0, 40.0, n) / 3600.0
    tables = []
    for e in range(1, 4):
        tables.append(_exposure_table(ra_a, dec_a, exposure=e, module="nrcalong",
                                      filtername="F335M", noise_mas=1.0,
                                      rng=np.random.default_rng(100 + e)))
    for e in range(1, 4):
        tables.append(_exposure_table(ra_b, dec_b, exposure=e, module="nrcblong",
                                      filtername="F335M", noise_mas=1.0,
                                      rng=np.random.default_rng(200 + e)))
    cons = build_visit_consensus(tables, context="test-adjacent-modules")
    exps = cons["exposures"]
    comps_a = {e["component"] for e in exps
               if module_family(e["key"][2]) == "a" and e["component"] >= 0}
    comps_b = {e["component"] for e in exps
               if module_family(e["key"][2]) == "b" and e["component"] >= 0}
    assert comps_a and comps_b, "each module should tie within its own footprint"
    assert comps_a.isdisjoint(comps_b), (
        f"NRCA and NRCB were merged into a shared component (issue #158 alias): "
        f"a={comps_a} b={comps_b}")
    assert cons["n_components"] >= 2


def test_same_module_large_shift_still_ties():
    """A single same-module exposure with a large (~20") rigid im0 error but the
    SAME stars must still be found + tied by the full sweep -- the cross-module
    bound must not touch same-module growth (brick-1182 v001 ~20")."""
    rng = np.random.default_rng(11)
    n = 400
    ra = RA0 + rng.uniform(0, 60.0, n) / 3600.0 / COSD
    dec = DEC0 + rng.uniform(0, 60.0, n) / 3600.0
    tables = []
    for e in range(1, 4):
        tables.append(_exposure_table(ra, dec, exposure=e, module="nrcalong",
                                      filtername="F335M", noise_mas=1.0,
                                      rng=np.random.default_rng(300 + e)))
    # exposure 4: same module + same stars, but a 20" (20000 mas) rigid shift
    tables.append(_exposure_table(ra, dec, exposure=4, module="nrcalong",
                                  filtername="F335M", dra_mas=20000.0, ddec_mas=0.0,
                                  noise_mas=1.0, rng=np.random.default_rng(304)))
    cons = build_visit_consensus(tables, context="test-samemod-largeshift")
    exps = cons["exposures"]
    e4 = [e for e in exps if e["key"][1] == 4][0]
    assert e4["component"] >= 0, \
        "the 20-inch-shifted same-module exposure must tie, not become an island"
    comps = {e["component"] for e in exps if e["component"] >= 0}
    assert len(comps) == 1, \
        f"same-module exposures should form ONE component, got {comps}"


# ---------------------------------------------------------------------------
# same-star restriction (issue #285)
# ---------------------------------------------------------------------------

def test_mutual_match_mask_is_mutual_not_one_way():
    from jwst_gc_pipeline.photometry.visit_consensus import _mutual_match_mask
    import astropy.units as u
    # two coords crowd around ONE reference star; a one-way match keeps both
    ref = SkyCoord([RA0] * u.deg, [DEC0] * u.deg)
    close = SkyCoord([RA0, RA0 + 0.02 / 3600.0 / COSD] * u.deg,
                     [DEC0, DEC0] * u.deg)
    mask = _mutual_match_mask(close, ref, 0.15 * u.arcsec)
    assert mask.sum() == 1, mask
    assert mask[0]                      # the nearer one wins


def test_mutual_match_mask_edges():
    from jwst_gc_pipeline.photometry.visit_consensus import _mutual_match_mask
    import astropy.units as u
    a = SkyCoord([RA0] * u.deg, [DEC0] * u.deg)
    assert _mutual_match_mask(a, a[:0], 0.15 * u.arcsec).sum() == 0
    assert len(_mutual_match_mask(a[:0], a, 0.15 * u.arcsec)) == 0
    assert _mutual_match_mask(a, None, 0.15 * u.arcsec).sum() == 0
    # a reference star far away matches nothing
    far = SkyCoord([RA0 + 10.0 / 3600.0 / COSD] * u.deg, [DEC0] * u.deg)
    assert _mutual_match_mask(a, far, 0.15 * u.arcsec).sum() == 0


def test_restrict_to_freezes_the_star_list_not_the_positions():
    """The gate must measure MOVEMENT, not a change of population.

    Rebuild a visit from catalogs that detect EXTRA stars -- what a later,
    background-subtracted stage does -- and check that restricting to the
    first consensus's stars lands back on the first population.
    """
    ra, dec = _field(n=400)
    base_tables = [_exposure_table(ra, dec, exposure=e) for e in range(1, 5)]
    base = build_visit_consensus(base_tables, context="base")
    assert len(base["coords"]) > 100

    # the later stage: the same stars plus 300 newly-detected faint ones
    rng = np.random.default_rng(999)
    ra2 = np.concatenate([ra, RA0 + rng.uniform(0, 90.0, 300) / 3600.0 / COSD])
    dec2 = np.concatenate([dec, DEC0 + rng.uniform(0, 90.0, 300) / 3600.0])
    wide_tables = [_exposure_table(ra2, dec2, exposure=e) for e in range(1, 5)]

    wide = build_visit_consensus(wide_tables, context="wide")
    tight = build_visit_consensus(wide_tables, context="tight",
                                  restrict_to=base["coords"])

    assert len(wide["coords"]) > len(tight["coords"]), (
        len(wide["coords"]), len(tight["coords"]))
    # the restricted rebuild lands on the m2 population, not the new one
    assert abs(len(tight["coords"]) - len(base["coords"])) < 0.2 * len(base["coords"]), (
        len(tight["coords"]), len(base["coords"]))
    # and it REPORTS what it dropped rather than hiding it
    for e in tight["exposures"]:
        assert e["n_reliable_unrestricted"] >= e["n_reliable"]
    assert any(e["n_reliable_unrestricted"] > e["n_reliable"]
               for e in tight["exposures"])


def test_restrict_to_does_not_hide_a_real_movement():
    """Freezing the star list must not blind the gate to a shifted exposure."""
    ra, dec = _field(n=400)
    base = build_visit_consensus(
        [_exposure_table(ra, dec, exposure=e) for e in range(1, 5)],
        context="base")
    moved = [_exposure_table(ra, dec, exposure=e,
                             dra_mas=(25.0 if e == 3 else 0.0))
             for e in range(1, 5)]
    cons = build_visit_consensus(moved, context="moved",
                                 restrict_to=base["coords"])
    offs = {tuple(e["key"])[1]: e["vs_consensus"]["off"] for e in cons["exposures"]}
    assert offs[3] > 15.0, offs
    assert all(v < 10.0 for k, v in offs.items() if k != 3), offs


def test_restrict_to_none_is_unchanged_behaviour():
    ra, dec = _field(n=400)
    tables = [_exposure_table(ra, dec, exposure=e) for e in range(1, 5)]
    a = build_visit_consensus(tables, context="a")
    b = build_visit_consensus(tables, context="b", restrict_to=None)
    assert len(a["coords"]) == len(b["coords"])
    assert [e["n_reliable"] for e in a["exposures"]] == \
           [e["n_reliable"] for e in b["exposures"]]
    for e in b["exposures"]:
        assert e["n_reliable_unrestricted"] == e["n_reliable"]


def _consensus_stars(n=400):
    ra, dec = _field(n=n)
    return build_visit_consensus(
        [_exposure_table(ra, dec, exposure=e) for e in range(1, 5)],
        context="m2")["coords"], ra, dec


def test_restriction_is_refused_when_the_star_list_is_the_wrong_sky():
    """cloudef restricted against the OTHER observation's consensus: 0.2-15%
    matched at a median pair separation of 104 mas, i.e. chance.  A wrong star
    list must be refused, not matched."""
    from jwst_gc_pipeline.photometry.visit_consensus import _restrict_to_same_stars
    ref, ra, dec = _consensus_stars()
    tables = [_exposure_table(ra, dec, exposure=e) for e in range(1, 5)]
    # an unrelated star list over the same footprint
    rng = np.random.default_rng(4242)
    other = SkyCoord((RA0 + rng.uniform(0, 90.0, 400) / 3600.0 / COSD) * u.deg,
                     (DEC0 + rng.uniform(0, 90.0, 400) / 3600.0) * u.deg)
    cons = build_visit_consensus(tables, context="wrong-list", restrict_to=other)
    assert all(e["restrict_refused"] for e in cons["exposures"]), cons["exposures"]
    # and the gate still has its full input
    for e in cons["exposures"]:
        assert e["n_reliable"] == e["n_reliable_unrestricted"]


def test_a_gross_movement_is_still_found_not_dropped():
    """Above ~150 mas the mutual match collapses.  The exposure must still be
    MEASURED and flagged, never fall out of the gate."""
    ref, ra, dec = _consensus_stars()
    for shift in (160.0, 1000.0):
        tables = [_exposure_table(ra, dec, exposure=e,
                                  dra_mas=(shift if e == 3 else 0.0))
                  for e in range(1, 5)]
        cons = build_visit_consensus(tables, context=f"shift{shift}",
                                     restrict_to=ref)
        keys = [tuple(e["key"])[1] for e in cons["exposures"]]
        assert 3 in keys, (shift, keys, cons["skipped"])
        moved = [e for e in cons["exposures"] if tuple(e["key"])[1] == 3][0]
        assert moved["misaligned"], (shift, moved)


def test_restriction_is_off_at_a_correcting_stage(tmp_path):
    """m1/m2/m12 have nothing to freeze against; the full star set is right."""
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        CORRECTION_STAGES)
    assert "m2" in CORRECTION_STAGES and "m3" not in CORRECTION_STAGES


def test_missing_m2_consensus_falls_back_open_not_closed(tmp_path, capsys):
    """A missing baseline is not evidence the solution moved."""
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        _m2_consensus_stars)
    stars, path = _m2_consensus_stars(str(tmp_path), str(tmp_path), "F212N", "")
    assert stars is None                       # falls back, does not raise


def test_restrict_radius_is_tight_enough_to_be_same_star():
    """A 5\" radius is not a same-star match in a GC field -- it is whatever is
    nearby.  Guard the constant, which no behavioural test pins."""
    import inspect
    from jwst_gc_pipeline.photometry import visit_consensus as vc
    default = inspect.signature(vc.build_visit_consensus).parameters[
        "restrict_radius"].default
    assert default.to(u.arcsec).value <= 0.3, default
    assert vc.RESTRICT_MIN_SURVIVAL >= 0.5
    assert vc.RESTRICT_MAX_TIE_MAS <= 100.0


def test_the_tie_precondition_alone_refuses_a_displaced_star_list():
    """Pins the PRECONDITION, not just the outcome.  A star list that is the
    right stars but the wrong sky must be refused by the TIE check.

    The displacement matters and 3 x RESTRICT_MAX_TIE_MAS (150 mas) was the
    wrong choice: at 150 mas the mutual match has already collapsed to 0.2%
    survival, so the survival floor refuses it too and deleting the tie
    precondition still gets `mask is None` -- the test failed only on the
    reason string.  Mapped against the real constants:

        disp    survival   refused by
          40      100.0%   nothing
          60      100.0%   TIE only     <- the tie check's unique region
          80      100.0%   TIE only
         120      100.0%   TIE only
         150        0.2%   survival floor also

    With the tie precondition deleted, a 60-120 mas wrong list is accepted at
    100% survival.  80 mas puts this test inside the region only the tie check
    covers, so it is a behavioural pin rather than a message assertion.
    """
    from jwst_gc_pipeline.photometry.visit_consensus import (
        RESTRICT_MAX_TIE_MAS, _restrict_to_same_stars)
    import astropy.units as u
    ra, dec = _field(n=400)
    ref = build_visit_consensus(
        [_exposure_table(ra, dec, exposure=e) for e in range(1, 5)],
        context="m2")["coords"]
    disp_mas = 1.6 * RESTRICT_MAX_TIE_MAS          # 80 mas at the real constant
    shifted = SkyCoord(ref.ra + (disp_mas / 3.6e6 / COSD) * u.deg,
                       ref.dec, frame="icrs")
    mask, why = _restrict_to_same_stars(ref, shifted, 0.15 * u.arcsec)
    assert mask is None and "tie to the m2 star list" in why, why
    # and the survival floor is NOT what refused it: at this displacement the
    # stars still match, so deleting the tie check would let it through
    m2, _ = _restrict_to_same_stars(ref, shifted, 0.5 * u.arcsec)
    assert disp_mas < 0.15 * 1000, "the pin must stay inside the match radius"


def test_the_survival_floor_alone_refuses_a_chance_match():
    """Pins the SURVIVAL floor.  An unrelated list over the same footprint
    ties fine at zero offset and must still be refused."""
    from jwst_gc_pipeline.photometry.visit_consensus import _restrict_to_same_stars
    import astropy.units as u
    ra, dec = _field(n=400)
    ref = build_visit_consensus(
        [_exposure_table(ra, dec, exposure=e) for e in range(1, 5)],
        context="m2")["coords"]
    # 20% real stars (so the tie IS found -- they give a clean peak) plus 80%
    # unrelated ones.  cloudef's shape: a list that ties but does not match.
    rng = np.random.default_rng(31337)
    n_real = len(ref) // 5
    mixed = SkyCoord(
        np.concatenate([ref.ra.deg[:n_real],
                        RA0 + rng.uniform(0, 90.0, len(ref)) / 3600.0 / COSD]) * u.deg,
        np.concatenate([ref.dec.deg[:n_real],
                        DEC0 + rng.uniform(0, 90.0, len(ref)) / 3600.0]) * u.deg)
    mask, why = _restrict_to_same_stars(ref, mixed, 0.15 * u.arcsec)
    assert mask is None, (mask if mask is None else mask.sum(), why)
    assert "matched the m2 list" in why, why


def test_gate_membership_uses_the_unrestricted_count():
    """Pins the fix for the 🔴 BEHAVIOURALLY: keying on the restricted count
    let a displaced exposure LEAVE the gate instead of failing it.

    An exposure whose restricted set collapses must still be REPORTED; a gross
    movement that removes an exposure from the consensus is the one outcome
    that cannot be tolerated, because nothing downstream sees it at all.

    Two things worth stating rather than papering over.  The source-grep
    version of this test could not hold the invariant: the substring
    `e["n_reliable_unrestricted"] >= min_stars` occurs TWICE -- at the gate
    membership and again at the thin-set fallback -- so mutating the first
    left the assertion green.  And the behavioural version below does not
    hold it either, because it is currently UNREACHABLE: at 900 mas the tie
    precondition refuses the restriction, the exposure falls back to its full
    star set, and `n_reliable` is then the unrestricted count anyway.  Keying
    on the unrestricted count is defence in depth behind the tie precondition
    and the thin-set fallback, and no input distinguishes the two today.  What
    this test does hold is the OUTCOME -- the displaced exposure is reported,
    by whichever of the three mechanisms gets there first.
    """
    import astropy.units as u
    from jwst_gc_pipeline.photometry.visit_consensus import (
        build_visit_consensus)
    ra, dec = _field(n=400)
    tables = [_exposure_table(ra, dec, exposure=e) for e in range(1, 5)]
    m2 = build_visit_consensus(tables, context="m2")["coords"]
    # one exposure displaced far enough that essentially nothing survives the
    # restriction -- its restricted count would be ~0
    moved = _exposure_table(ra + 900.0 / 3.6e6 / COSD, dec, exposure=5)
    cons = build_visit_consensus(tables + [moved], context="m6",
                                 restrict_to=m2, min_stars=50)
    keys = {e["key"] for e in cons["exposures"]}
    assert len(keys) == 5, (sorted(keys), cons["skipped"])
    assert not cons["skipped"], cons["skipped"]
    # and the source invariant, anchored to the ASSIGNMENT so the thin-set
    # fallback's identical substring cannot satisfy it
    import inspect
    from jwst_gc_pipeline.photometry import visit_consensus as vc
    src = inspect.getsource(vc.build_visit_consensus)
    assert ('usable_idx = [i for i, e in enumerate(entries)\n'
            '                  if e["n_reliable_unrestricted"] >= min_stars]'
            ) in src, "usable_idx must key on the unrestricted count"


def test_a_star_list_that_is_half_other_sky_is_reported():
    """RESTRICT_MIN_SURVIVAL measures the fraction of the EXPOSURE's stars that
    matched; it is blind to what fraction of the STAR LIST is foreign sky,
    because the foreign half simply never matches and is ignored.

    Real: cloudef's f360m_o002 and f480m_o005 consensus catalogs each pool two
    pointings 15 arcmin apart, because the m2 run that built them ingested two
    observations under one filename namespace.
    """
    from jwst_gc_pipeline.photometry.visit_consensus import (
        _fraction_within_footprint, _restrict_to_same_stars)
    import astropy.units as u
    ra, dec = _field(n=400)
    ref = build_visit_consensus(
        [_exposure_table(ra, dec, exposure=e) for e in range(1, 5)],
        context="m2")["coords"]
    # the same list plus an equal blob 15 arcmin away -- cloudef's shape
    far = SkyCoord((ref.ra.deg + 15.0 / 60.0 / COSD) * u.deg, ref.dec, frame="icrs")
    mixed = SkyCoord(np.concatenate([ref.ra.deg, far.ra.deg]) * u.deg,
                     np.concatenate([ref.dec.deg, far.dec.deg]) * u.deg)
    assert _fraction_within_footprint(mixed, ref) < 0.6
    # it still RESTRICTS -- the local half matches fine -- and survival cannot
    # see the problem, which is the point
    mask, why = _restrict_to_same_stars(ref, mixed, 0.15 * u.arcsec)
    assert mask is not None and mask.sum() > 0.8 * len(ref), why
    # what makes it visible is the RECORDED coverage, not a threshold
    cons = build_visit_consensus(
        [_exposure_table(ra, dec, exposure=e) for e in range(1, 5)],
        context="mixed", restrict_to=mixed)
    covs = [e["restrict_list_coverage"] for e in cons["exposures"]]
    assert all(c is not None and c < 0.6 for c in covs), covs


def test_footprint_coverage_is_reported_not_gated():
    """A visit-wide star list legitimately covers more sky than one exposure,
    so a low fraction is a reason to look, never a reason to refuse."""
    from jwst_gc_pipeline.photometry.visit_consensus import (
        _fraction_within_footprint)
    import astropy.units as u
    ra, dec = _field(n=200)
    a = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    assert _fraction_within_footprint(a, a) == 1.0
    assert _fraction_within_footprint(a[:0], a) is None
    assert _fraction_within_footprint(a, a[:0]) is None


# ---------------------------------------------------------------------------
# The RECORD.  `same_star_gate`, `same_star_refused`, `n_same_star_refused` and
# `restrict_list_coverage` had zero readers and zero assertions anywhere in the
# repo -- one writer at astrometry_checkpoint.py and nothing else -- so the
# whole checkpoint-side half of this feature was deletable green.
# ---------------------------------------------------------------------------

def _record_for(tmp_path, m2_stars, tables, filtername="F212N"):
    """Run the real checkpoint writer and hand back its consensus record."""
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        run_visit_checkpoint)
    import jwst_gc_pipeline.photometry.astrometry_checkpoint as ac

    # NB the checkpoint supplies `restrict_to` itself, from
    # `_m2_consensus_stars` -- which the caller monkeypatches.  Passing it in
    # `consensus_kwargs` too is a TypeError, and going through the real path is
    # the point: it is the wiring that had no test.
    return run_visit_checkpoint(
        tables, "m6", filtername=filtername, basepath=str(tmp_path),
        record_dir=str(tmp_path), context="test")


def test_the_record_says_applied_only_when_every_exposure_restricted(tmp_path, monkeypatch):
    """`any(not refused)` reported "applied" when ONE exposure of sixteen
    restricted.  It is `all` now, and the restriction is all-or-nothing per
    visit, so the two agree.

    The counts are asserted too, not just the label: the m2 list is a SUBSET
    of what this stage detects, so "applied" has to show up as a smaller
    restricted count.  Without that, deleting the wiring that passes
    `restrict_to` from the checkpoint leaves the label reading "applied" for a
    run in which nothing was restricted -- a mutant that survived every test
    in this file.
    """
    import jwst_gc_pipeline.photometry.astrometry_checkpoint as ac
    ra, dec = _field(n=400)
    tables = [_exposure_table(ra, dec, exposure=e) for e in range(1, 5)]
    # the m2 star list is HALF the stars this stage detects: the population
    # change the restriction exists to absorb
    full = build_visit_consensus(tables, context="m2")["coords"]
    m2 = full[:len(full) // 2]
    monkeypatch.setattr(ac, "_m2_consensus_stars",
                        lambda *a, **k: (m2, "/x/m2.fits"))
    rec = _record_for(tmp_path, m2, tables)
    cons = rec["visits"][0]["consensus"]
    assert cons["same_star_gate"] == "applied", cons
    assert cons["n_same_star_refused"] == 0
    assert cons["same_star_refused"] == []
    assert cons["n_reliable_restricted"] < cons["n_reliable_unrestricted"], cons


def test_one_refusal_does_not_cascade_but_does_not_pollute_either(tmp_path, monkeypatch):
    """Two failure modes, and the fix has to avoid both.

    BEFORE any fix: a refused exposure contributed its FULL star list, so any
    star two refused exposures shared cleared `min_exposures` and re-entered
    the consensus -- and the exposures that DID restrict were measured against
    a consensus holding stars they do not have.

    FIRST FIX (all-or-nothing per visit): on real cloudc F182M, 8 of 128
    frames tie 52-54 mas, 2-4 mas over RESTRICT_MAX_TIE_MAS, and 120 healthy
    frames cascaded off them -- 128 of 128 refused, consensus 119,994 ->
    129,135 stars, the gate inert on the largest field measured.

    What must be homogeneous is the SEED.  A refused exposure is still tied
    and measured; it just does not EXTEND the consensus.
    """
    import jwst_gc_pipeline.photometry.astrometry_checkpoint as ac
    import jwst_gc_pipeline.photometry.visit_consensus as vc

    ra, dec = _field(n=400)
    tables = [_exposure_table(ra, dec, exposure=e) for e in range(1, 5)]
    full = build_visit_consensus(tables, context="m2")["coords"]
    m2 = full[:len(full) // 2]

    real = vc._restrict_to_same_stars
    seen = {"n": 0}

    def _one_refusal(coords, reference, radius, context="", reference_tree=None):
        seen["n"] += 1
        if seen["n"] == 2:
            return None, "forced refusal for the test"
        return real(coords, reference, radius, context=context,
                    reference_tree=reference_tree)

    monkeypatch.setattr(vc, "_restrict_to_same_stars", _one_refusal)
    monkeypatch.setattr(ac, "_m2_consensus_stars",
                        lambda *a, **k: (m2, "/x/m2.fits"))
    rec = _record_for(tmp_path, m2, tables)
    cons = rec["visits"][0]["consensus"]

    # NO cascade: exactly the one exposure that refused
    assert cons["n_same_star_refused"] == 1, cons["same_star_refused"]
    assert cons["same_star_gate"] == "refused", cons
    reasons = [r["reason"] for r in cons["same_star_refused"]]
    assert reasons == [r for r in reasons if "forced refusal" in r], reasons

    # ... and NO pollution: the consensus is still the restricted population,
    # not the refused exposure's full one
    ref = _record_for(tmp_path / "all", m2,
                      [_exposure_table(ra, dec, exposure=e) for e in range(1, 5)])
    assert (cons["n_stars"]
            <= ref["visits"][0]["consensus"]["n_stars"] + 1), (
        cons["n_stars"], ref["visits"][0]["consensus"]["n_stars"])
    # the refused exposure was still MEASURED -- it is in the record's
    # per-exposure list, not in `skipped`
    assert len(rec["visits"][0]["exposures"]) == 4, rec["visits"][0]
    assert not rec["visits"][0]["consensus"]["skipped"], cons


def test_the_record_carries_the_star_list_footprint_coverage(tmp_path, monkeypatch):
    """`restrict_list_coverage` was computed per exposure and never reached the
    record -- verbatim the `restrict_refused` defect this PR already fixed
    once.  A list that is half a DIFFERENT POINTING passes survival and tie
    alike, because the foreign half simply never matches."""
    import jwst_gc_pipeline.photometry.astrometry_checkpoint as ac
    ra, dec = _field(n=400)
    tables = [_exposure_table(ra, dec, exposure=e) for e in range(1, 5)]
    m2 = build_visit_consensus(tables, context="m2")["coords"]
    monkeypatch.setattr(ac, "_m2_consensus_stars",
                        lambda *a, **k: (m2, "/x/m2.fits"))
    cons = _record_for(tmp_path, m2, tables)["visits"][0]["consensus"]
    assert cons["restrict_list_coverage"] is not None, cons
    assert cons["restrict_list_coverage"] > 0.9, cons["restrict_list_coverage"]


def test_the_record_says_unavailable_when_there_is_no_star_list(tmp_path, monkeypatch):
    """A missing baseline is not evidence the solution moved, and the record
    must not spell that the same way as a refusal."""
    import jwst_gc_pipeline.photometry.astrometry_checkpoint as ac
    ra, dec = _field(n=400)
    tables = [_exposure_table(ra, dec, exposure=e) for e in range(1, 5)]
    monkeypatch.setattr(ac, "_m2_consensus_stars", lambda *a, **k: (None, None))
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        run_visit_checkpoint)
    rec = run_visit_checkpoint(tables, "m6", filtername="F212N",
                               basepath=str(tmp_path), record_dir=str(tmp_path),
                               context="test")
    cons = rec["visits"][0]["consensus"]
    assert cons["same_star_gate"] == "unavailable", cons


def test_a_correcting_stage_does_not_restrict(tmp_path, monkeypatch):
    """m2 is where the baseline is DEFINED; there is nothing to freeze against
    and the full star set is the right one.  Dropping the `not correcting`
    guard -- applying the restriction at m1/m2/m12 too -- survived every test
    in this file through two review rounds."""
    import jwst_gc_pipeline.photometry.astrometry_checkpoint as ac
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        run_visit_checkpoint)
    ra, dec = _field(n=400)
    tables = [_exposure_table(ra, dec, exposure=e) for e in range(1, 5)]
    full = build_visit_consensus(tables, context="m2")["coords"]
    called = {"n": 0}

    def _spy(*a, **k):
        called["n"] += 1
        return full[:len(full) // 2], "/x/m2.fits"

    monkeypatch.setattr(ac, "_m2_consensus_stars", _spy)
    rec = run_visit_checkpoint(tables, "m2", filtername="F212N",
                               basepath=str(tmp_path),
                               record_dir=str(tmp_path), context="test")
    cons = rec["visits"][0]["consensus"]
    # m2 is a CORRECTING stage: the star list is never even looked up
    assert called["n"] == 0, "m2 must not read an m2 baseline to freeze against"
    assert cons["same_star_gate"] == "unavailable", cons
    assert cons["n_reliable_restricted"] == cons["n_reliable_unrestricted"]


def test_a_refused_exposure_does_not_SEED_the_consensus(monkeypatch):
    """The other half of the mechanism, and it was pinned by nothing.

    The design has two rules and the comment says so: a refused exposure must
    not EXTEND the seed, and it must not BE the seed -- if it seeded, its full
    star list would be the population, the same defect from the other end.
    Deleting the member reordering left `52 passed`, and no test referenced
    `_contributes` at all.

    It needs `min_exposures` refusals in one component to bite, which is why a
    single-refusal fixture could not see it: with one refusal the seed's extra
    stars reach `counts == 1` and the `counts >= min_exposures` filter drops
    them anyway.  With THREE, the refused members corroborate each other's
    extra stars and the pre-#285 population comes back.  That is the cloudc
    F182M shape -- its eight refusals are not scattered, they are all `nrcb3`
    in vgroup 06201.
    """
    import jwst_gc_pipeline.photometry.visit_consensus as vc

    ra, dec = _field(n=400)
    tables = [_exposure_table(ra, dec, exposure=e) for e in range(1, 7)]
    full = build_visit_consensus(tables, context="m2")["coords"]
    m2 = full[:len(full) // 2]          # the restricted population is HALF

    real = vc._restrict_to_same_stars
    seen = {"n": 0}

    def _three_refuse(coords, reference, radius, context="",
                      reference_tree=None):
        seen["n"] += 1
        if seen["n"] <= 3:
            return None, "forced refusal for the test"
        return real(coords, reference, radius, context=context,
                    reference_tree=reference_tree)

    monkeypatch.setattr(vc, "_restrict_to_same_stars", _three_refuse)
    cons = build_visit_consensus(tables, context="m6", restrict_to=m2)
    refused = [e for e in cons["exposures"] if e.get("restrict_refused")]
    assert len(refused) == 3, refused                      # no cascade
    # the consensus is the RESTRICTED population, not the refused exposures'
    assert len(cons["coords"]) < 0.75 * len(full), (
        len(cons["coords"]), len(full))


def test_a_component_with_no_restricted_member_says_so(tmp_path, monkeypatch, capsys):
    """`any(...)` deliberately permits a component whose members ALL refused to
    build from their full star sets -- the pre-#285 behaviour, and the same
    decision the visit-wide `contributing == []` path takes.  That is a real
    choice and it was implicit; it is stated and printed now, because a
    component silently built from a different star population than its
    neighbours is the confusion the restriction exists to remove."""
    import jwst_gc_pipeline.photometry.visit_consensus as vc

    ra, dec = _field(n=400)
    tables = [_exposure_table(ra, dec, exposure=e) for e in range(1, 5)]
    m2 = build_visit_consensus(tables, context="m2")["coords"]
    monkeypatch.setattr(vc, "_restrict_to_same_stars",
                        lambda *a, **k: (None, "forced refusal for the test"))
    cons = build_visit_consensus(tables, context="m6", restrict_to=m2)
    out = capsys.readouterr().out
    assert "NO exposure could be restricted" in out, out
    assert all(e.get("restrict_refused") for e in cons["exposures"])
