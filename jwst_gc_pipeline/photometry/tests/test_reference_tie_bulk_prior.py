"""The opt-in ``bulk_prior`` of ``measure_reference_tie`` (issue #819).

crowded_l3 (9438/o006) F150W/F182M/F210M read a zero-prior same-star offset of
51-59 mas on 1600-1700 pairs while their dense-reference histograms peaked at
contrast 3.7-8 and were refused, so the noisy histogram locked out the
estimator that would have replaced it.  A hand-recorded prior seeds the pairing
when check A verifies nothing, and four checks stand in for the histogram.

The histogram legs are replaced by stubs here: the real estimator takes minutes
per call on a field big enough for the region map, and what is under test is
the decision made on top of it.  The same-star pairing, the pair-excess
control, the magnitude check and the region map all run for real.
"""
import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry import visit_consensus as vc
from jwst_gc_pipeline.photometry.astrometry_checkpoint import attach_bulk_prior
from jwst_gc_pipeline.reduction.alignment_config import (
    ALIGNMENT_CONFIG, FieldAlignment, TABLE_CONSENSUS, VIRAC2)

_RA0, _DEC0 = 266.4, -28.9
_COSD = float(np.cos(np.radians(_DEC0)))
_W = 150.0
_TRUE = (40.0, -35.0)          # mas, 53 mas total
_PRIOR = dict(dra_mas=38.0, ddec_mas=-30.0, source="test",
              coherence=[dict(label="F182M", dra_mas=43.0, ddec_mas=-38.0)])


def _sky(ra_off_arcsec, dec_off_arcsec):
    return SkyCoord((_RA0 + ra_off_arcsec / 3600.0 / _COSD) * u.deg,
                    (_DEC0 + dec_off_arcsec / 3600.0) * u.deg)


def _field(seed=3, n=4000, shift=_TRUE, mag_slope=0.0, scatter_mas=20.0):
    """Consensus plus a denser reference holding 80% of its stars shifted by
    ``shift`` (+ optional magnitude-dependent shift in RA) and as many chance
    stars again."""
    rng = np.random.RandomState(seed)
    x = (rng.rand(n) - 0.5) * _W
    y = (rng.rand(n) - 0.5) * _W
    mag = rng.uniform(14.0, 20.0, n)
    cons = _sky(x, y)
    keep = rng.rand(n) < 0.8
    sx = shift[0] + mag_slope * (mag - 17.0)
    rx = x[keep] + (sx[keep] + rng.randn(keep.sum()) * scatter_mas) / 1000.0
    ry = y[keep] + (shift[1] + rng.randn(keep.sum()) * scatter_mas) / 1000.0
    extra = int(0.8 * n)
    ex = (rng.rand(extra) - 0.5) * _W
    ey = (rng.rand(extra) - 0.5) * _W
    ref = _sky(np.concatenate([rx, ex]), np.concatenate([ry, ey]))
    return cons, mag, ref


def _failed_histogram(*_a, **_k):
    return dict(ok=False, swept=False, dra=3.0, ddec=2.0, off=3.6, contrast=3.7,
                n_peak=40, window_consistent=None)


@pytest.fixture
def noisy_histograms(monkeypatch):
    """Dense and sparse histogram legs that verify nothing."""
    monkeypatch.setattr(vc, "measure_offset", _failed_histogram)
    monkeypatch.setattr(
        vc, "agree_across_references",
        lambda *a, **k: dict(a=None, b=None, ddra_mas=float("nan"),
                             dddec_mas=float("nan"), sep_mas=float("nan"),
                             agree=False))
    monkeypatch.setattr(
        vc, "measure_offset_grid",
        lambda *a, **k: dict(cells=[], n_ok=0, n_total=0, worst_off_mas=float("nan"),
                             worst_off_cell=None, min_contrast_seen=float("nan"),
                             clean=False))


def _tie(cons, mag, ref, prior=None, consensus_mag="mag", **kw):
    sparse = ref[:5]
    return vc.measure_reference_tie(
        cons, ref, sparse, filtername="F210M",
        consensus_mag=mag if consensus_mag == "mag" else consensus_mag,
        dense=True, bulk_prior=prior, context="prior-test", **kw)


def test_prior_recovers_the_tie_the_histogram_cannot(noisy_histograms):
    cons, mag, ref = _field()
    out = _tie(cons, mag, ref, _PRIOR)
    bp = out["bulk_prior"]
    assert bp["engaged"] and bp["passed"], bp["reasons"]
    assert out["bulk_source"] == "same-star-prior"
    assert out["apply_ok"] is True
    assert abs(out["dra_mas"] - _TRUE[0]) < 3.0
    assert abs(out["ddec_mas"] - _TRUE[1]) < 3.0
    # The failed histogram stays in the record; the seed is recorded apart.
    assert out["vs_full"]["ok"] is False
    assert bp["vs_prior"]["prior"] is True
    assert bp["pair_excess"]["ratio"] >= vc.PRIOR_PAIR_EXCESS_MIN_RATIO
    assert bp["magnitude"]["n_bins"] >= vc.PRIOR_MIN_MAGNITUDE_BINS


def test_without_a_prior_the_same_field_stays_refused(noisy_histograms):
    cons, mag, ref = _field()
    out = _tie(cons, mag, ref, None)
    assert out["apply_ok"] is False
    assert out["bulk_source"] == "histogram"
    assert "bulk_prior" not in out


def test_a_verified_histogram_ignores_the_prior(monkeypatch):
    cons, mag, ref = _field()
    good = dict(ok=True, swept=False, dra=_TRUE[0], ddec=_TRUE[1],
                off=float(np.hypot(*_TRUE)), contrast=40.0, n_peak=900,
                window_consistent=True)
    monkeypatch.setattr(vc, "measure_offset", lambda *a, **k: dict(good))
    monkeypatch.setattr(
        vc, "agree_across_references",
        lambda *a, **k: dict(a=good, b=good, ddra_mas=0.0, dddec_mas=0.0,
                             sep_mas=0.0, agree=True))
    monkeypatch.setattr(
        vc, "measure_offset_grid",
        lambda *a, **k: dict(cells=[], n_ok=0, n_total=0, worst_off_mas=0.0,
                             worst_off_cell=None, min_contrast_seen=40.0,
                             clean=True))
    with_prior = _tie(cons, mag, ref, dict(_PRIOR, dra_mas=-90.0, ddec_mas=0.0))
    without = _tie(cons, mag, ref, None)
    assert with_prior["bulk_source"] == without["bulk_source"] == "same-star"
    assert with_prior["dra_mas"] == without["dra_mas"]
    assert with_prior["bulk_prior"]["engaged"] is False
    assert with_prior["vs_full"] == without["vs_full"]


def test_a_tie_beyond_the_pairing_limit_is_refused(noisy_histograms):
    """A 250 mas truth with a zero prior is the nearest-neighbour collapse case:
    the pairs are found, but the result is outside what a seed within the match
    radius / 3 can verify."""
    cons, mag, ref = _field(shift=(250.0, 0.0))
    out = _tie(cons, mag, ref, dict(_PRIOR, dra_mas=0.0, ddec_mas=0.0,
                                    coherence=[dict(label="F182M", dra_mas=250.0,
                                                    ddec_mas=0.0)]))
    assert out["apply_ok"] is False
    assert out["bulk_prior"]["passed"] is False
    assert "unambiguous pairing" in out["bulk_prior"]["reasons"][0]


def test_noise_only_reference_is_refused(noisy_histograms):
    cons, mag, _ref = _field()
    _c2, _m2, unrelated = _field(seed=99)
    out = _tie(cons, mag, unrelated, _PRIOR)
    assert out["apply_ok"] is False
    assert out["bulk_prior"]["passed"] is False


def test_pair_excess_control_sees_chance_pairs():
    cons, _mag, ref = _field()
    real = vc._prior_pair_excess(cons, ref, dict(dra=_TRUE[0], ddec=_TRUE[1]))
    wrong = vc._prior_pair_excess(cons, ref, dict(dra=_TRUE[0] + 600.0,
                                                  ddec=_TRUE[1]))
    assert real["passed"] and real["ratio"] > 10.0
    assert not wrong["passed"] and wrong["ratio"] < 2.0


def test_a_magnitude_dependent_offset_is_refused(noisy_histograms):
    cons, mag, ref = _field(mag_slope=15.0)
    out = _tie(cons, mag, ref, _PRIOR)
    assert out["apply_ok"] is False
    assert out["bulk_prior"]["magnitude"]["passed"] is False


def test_missing_magnitudes_refuse(noisy_histograms):
    cons, _mag, ref = _field()
    out = _tie(cons, None, ref, _PRIOR, consensus_mag=None)
    assert out["apply_ok"] is False
    assert any("magnitude" in r for r in out["bulk_prior"]["reasons"])


def test_cross_filter_coherence_is_required(noisy_histograms):
    cons, mag, ref = _field()
    off_sibling = dict(_PRIOR, coherence=[dict(label="F182M", dra_mas=70.0,
                                               ddec_mas=-35.0)])
    out = _tie(cons, mag, ref, off_sibling)
    assert out["apply_ok"] is False
    assert out["bulk_prior"]["coherence"]["passed"] is False
    none_given = _tie(cons, mag, ref, dict(_PRIOR, coherence=[]))
    assert none_given["apply_ok"] is False
    assert any("sibling" in r for r in none_given["bulk_prior"]["reasons"])


def test_prior_beyond_the_pairing_radius_is_a_configuration_error():
    cons, mag, ref = _field()
    with pytest.raises(ValueError, match="match radius / 3"):
        vc._bulk_prior_seed(dict(dra_mas=90.0, ddec_mas=60.0))
    with pytest.raises(ValueError):
        vc._bulk_prior_seed(dict(dra_mas=float("nan"), ddec_mas=0.0))


# ---------------------------------------------------------------------------
# alignment_config plumbing
# ---------------------------------------------------------------------------

def test_no_field_carries_a_prior_yet():
    """The policy question of issue #819 (Q23) is open, so every shipped entry
    keeps an empty prior table; setting one is a deliberate change that must
    update this test."""
    assert [c.proposal for c in ALIGNMENT_CONFIG if c.bulk_prior_mas] == []


def test_field_alignment_defaults_to_no_prior():
    cfg = FieldAlignment(proposal="X", fields=None, reference_frame=VIRAC2,
                         source=TABLE_CONSENSUS)
    assert cfg.bulk_prior_mas == {}


def test_attach_bulk_prior_is_identity_without_an_entry():
    refcat = dict(all=None, sparse=None)
    assert attach_bulk_prior(refcat, "9438", "006", "F210M") is refcat
    assert attach_bulk_prior(None, "9438", "006", "F210M") is None


def test_attach_bulk_prior_lists_the_siblings_for_coherence(monkeypatch):
    from jwst_gc_pipeline.reduction import alignment_config as ac
    cfg = FieldAlignment(
        proposal="X", fields=("001",), reference_frame=VIRAC2,
        source=TABLE_CONSENSUS,
        bulk_prior_mas={"F150W": (40.0, -30.0), "F182M": (42.0, -33.0),
                        "F210M": (41.0, -31.0)})
    monkeypatch.setattr(ac, "resolve", lambda proposal, field: cfg)
    refcat = dict(all="A", sparse="S")
    out = attach_bulk_prior(refcat, "X", "001", "f210m")
    assert out is not refcat and "bulk_prior" not in refcat
    bp = out["bulk_prior"]
    assert (bp["dra_mas"], bp["ddec_mas"]) == (41.0, -31.0)
    assert [c["label"] for c in bp["coherence"]] == ["F150W", "F182M"]
    # A filter without an entry is untouched.
    assert attach_bulk_prior(refcat, "X", "001", "F277W") is refcat


# ---------------------------------------------------------------------------
# sparse cross-check of the adopted tie, and entry-point routing
# ---------------------------------------------------------------------------

def _with_sparse_peak(monkeypatch, dra, ddec):
    """Dense leg refuses; the sparse leg returns a measured peak at (dra, ddec)."""
    def _measure(coords, ref, **_k):
        if len(ref) > 100:
            return _failed_histogram()
        return dict(ok=True, swept=False, dra=dra, ddec=ddec,
                    off=float(np.hypot(dra, ddec)), contrast=60.0, n_peak=40,
                    window_consistent=True, alias_rejected=False)
    monkeypatch.setattr(vc, "measure_offset", _measure)


def test_a_measured_sparse_peak_far_from_the_adopted_tie_blocks(
        noisy_histograms, monkeypatch):
    _with_sparse_peak(monkeypatch, -400.0, 300.0)
    cons, mag, ref = _field()
    out = _tie(cons, mag, ref, _PRIOR)
    assert out["bulk_prior"]["passed"], out["bulk_prior"]["reasons"]
    assert out["bulk_prior"]["cross_reference_sep_mas"] > 400.0
    assert out["cross_reference_gross_ok"] is False
    assert out["apply_ok"] is False


def test_a_sparse_peak_at_the_adopted_tie_confirms_it(noisy_histograms, monkeypatch):
    _with_sparse_peak(monkeypatch, _TRUE[0], _TRUE[1])
    cons, mag, ref = _field()
    out = _tie(cons, mag, ref, _PRIOR)
    assert out["bulk_prior"]["cross_reference_sep_mas"] < 10.0
    assert out["cross_reference_gross_ok"] is True
    assert out["apply_ok"] is True


def test_no_pairs_around_the_prior_is_refused(noisy_histograms):
    cons, mag, _ref = _field()
    far = _sky(np.array([5000.0, 5001.0]), np.array([5000.0, 5001.0]))
    out = _tie(cons, mag, far, _PRIOR)
    assert out["apply_ok"] is False
    assert out["bulk_prior"]["passed"] is False


def test_both_checkpoint_entry_points_attach_the_prior():
    """The CLI is the second production caller of run_visit_checkpoint; a filter
    that carries a prior must be measured the same from either."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[3]
    for rel in ("jwst_gc_pipeline/photometry/cataloging.py",
                "scripts/reduction/run_astrometry_checkpoint.py"):
        assert "attach_bulk_prior(" in (root / rel).read_text(), rel
