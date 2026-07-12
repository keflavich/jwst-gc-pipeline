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


def test_reference_tie_refuses_on_reference_disagreement():
    ra, dec = _field()
    cons = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    ref_all, _ = _reference_sets(ra, dec)
    # sparse reference shifted 30 mas -> the two references DISAGREE -> no sign-off
    ref_sparse_bad = SkyCoord(ra=(ra[::10] + 30.0 / 3.6e6 / COSD) * u.deg,
                              dec=dec[::10] * u.deg, frame="icrs")
    tie = measure_reference_tie(cons, ref_all, ref_sparse_bad,
                                context="test-disagree", grid_nx=2, grid_ny=2)
    assert not tie["cross_reference"]["agree"]
    assert not tie["apply_ok"]
