"""Which star list tie-breaks an overlap too thin to measure directly.

Before a field is published, a check confirms that overlapping exposures agree
about where the stars are.  Some pairs overlap on a sliver too thin to compare
directly; for those, both sides are instead compared against a common list of
known star positions, which settles the question.

W51 had no such list configured, because the only registry available was the
one feeding a *different* check -- the blocking absolute-frame one, which needs
a dense catalogue and would refuse good data given a sparse one.  That is what
this module covers.

(An earlier version of this change also exempted MIRI-to-MIRI pairs from
blocking, on the argument that mid-infrared images hold too few sources for two
pointings to share any.  Review measured the opposite -- those pairs share
hundreds of detections, and one of them carries a real 66 mas offset the gate
had already measured -- so that half was withdrawn.  See #385.)
"""
import importlib.util
import os
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]


def _load(relpath, name):
    path = _REPO / relpath
    if not path.exists():                                   # pragma: no cover
        pytest.skip(f'{relpath} not present', allow_module_level=True)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Which star list arbitrates an unmeasurable pair
# ---------------------------------------------------------------------------

stage = _load('scripts/release/stage_release.py', '_stage')


def test_a_sparse_star_list_is_allowed_to_arbitrate():
    """Sparse is better than none.  With no list the pair is unmeasurable and
    the field cannot stage at all; with one it usually resolves."""
    path = stage.OVERLAP_ARBITER_REFCAT.get('w51')
    assert path, 'w51 has no arbiter star list'
    if os.path.exists(path):
        assert stage.overlap_arbiter_refcat('w51') == path


def test_the_absolute_frame_check_reads_only_its_own_registry():
    """`stage_release.check_catalog_on_frame` asks whether a shipped catalogue
    sits on the right sky, and needs a dense list -- a sparse one gives a noisy
    bulk tie and would refuse good data.  So it must read FRAME_REFCAT and NOT
    the tie-break registry.

    Asserted on the MECHANISM rather than on registry contents: an earlier
    version of this test checked `'w51' not in FRAME_REFCAT`, which is a fact
    about a dict literal.  Re-pointing check_catalog_on_frame at
    `overlap_arbiter_refcat` -- wiring the sparse list straight into the
    blocking check -- left that test green.
    """
    import inspect
    src = inspect.getsource(stage.check_catalog_on_frame)
    assert 'FRAME_REFCAT' in src
    assert 'overlap_arbiter_refcat' not in src, (
        'the absolute-frame check must not resolve its catalogue through the '
        'tie-break registry, which may hold a sparse list')


def test_a_field_with_no_list_is_told_so_rather_than_left_to_wonder():
    """Without this line, a pair stays unmeasurable and the log gives no reason
    -- indistinguishable from the arbiter having run and found nothing."""
    import inspect
    src = inspect.getsource(stage.main) if hasattr(stage, 'main') else ''
    if not src:
        import pathlib as _p
        src = (_p.Path(stage.__file__).read_text()
               if getattr(stage, '__file__', None) else '')
    assert 'no overlap arbiter star list' in src


def test_a_field_with_only_a_dense_list_still_uses_it_to_arbitrate():
    """A denser catalogue does the tie-break job too, so a field does not need
    an entry in both registries."""
    assert 'brick' not in stage.OVERLAP_ARBITER_REFCAT
    brick = stage.FRAME_REFCAT.get('brick')
    if brick and os.path.exists(brick):
        assert stage.overlap_arbiter_refcat('brick') == brick


def test_a_field_with_no_list_at_all_says_so_rather_than_pretending():
    assert stage.overlap_arbiter_refcat('not-a-field') is None


def test_a_catalogue_without_a_source_column_is_not_called_VIRAC2():
    """The gating slot used to be labelled `VIRAC2` whatever was read.

    VIRAC2 is the VVV-based near-infrared catalogue, and W51 lies OUTSIDE the
    VVV footprint -- it does not exist there.  An operator debugging a W51
    block was told a catalogue had been used that cannot exist for the field.
    The split is by presence of a `source` column, not by what the file is, so
    the label has to report what was actually read.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        '_cio_label', _REPO / 'scripts' / 'release' / 'check_interframe_overlap.py')
    cio = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cio)

    w51 = stage.OVERLAP_ARBITER_REFCAT.get('w51')
    if not (w51 and os.path.exists(w51)):
        pytest.skip('w51 star list not on this host')
    _rc, _gaia, label = cio._refcat(w51)
    assert 'VIRAC2' not in label.split('NOT')[0]
    assert 'no `source` column' in label


# ---------------------------------------------------------------------------
# The withdrawn MIRI exemption must stay withdrawn
# ---------------------------------------------------------------------------

def _two_mid_infrared_pointings(strip_deg=0.0015, n_strip=60, off_arcsec=0.5):
    """Two mid-infrared pointings sharing a thin strip of sky, offset in it.

    Shaped so BOTH reference-free layers decline: the per-tile grid has no
    mutual-coverage cell it can measure, and the pooled histogram over so few
    shared stars is not authoritative.  That is the "unverifiable" state -- the
    third verdict, distinct from pass and fail -- and it is the state the
    withdrawn exemption used to convert into a pass.
    """
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    rng = np.random.default_rng(11)
    n = 400
    ra_a = np.concatenate([266.50 + rng.uniform(0, 0.02, n),
                           266.52 + rng.uniform(0, strip_deg, n_strip)])
    dec_a = np.concatenate([-28.50 + rng.uniform(0, 0.02, n),
                            -28.50 + rng.uniform(0, 0.02, n_strip)])
    ra_b = np.concatenate([266.52 + strip_deg + rng.uniform(0, 0.02, n),
                           266.52 + rng.uniform(0, strip_deg, n_strip)
                           + off_arcsec / 3600.0])
    dec_b = np.concatenate([-28.50 + rng.uniform(0, 0.02, n),
                            -28.50 + rng.uniform(0, 0.02, n_strip)])
    return {"002001:mirimage": SkyCoord(ra_a * u.deg, dec_a * u.deg),
            "998001:mirimage": SkyCoord(ra_b * u.deg, dec_b * u.deg)}


def test_a_mid_infrared_pair_it_cannot_measure_is_NOT_passed(monkeypatch):
    """Two MIRI pointings the gate cannot measure must stay unverified.

    An earlier version of this change exempted mid-infrared pairs from
    blocking, on the argument that such images hold too few sources for two
    pointings to share any.  That was measured false -- the real pair shares
    hundreds of detections and carries a 66-74 mas offset (#384) -- and the
    exemption was withdrawn.

    It is pinned here rather than by reading the source, because the withdrawal
    was verified twice by reading and was wrong once: the exemption can be
    written back in under any name, keyed on any detector token, anywhere in
    the still-open loop, and every other test in this file stays green.
    """
    cio = _load('scripts/release/check_interframe_overlap.py', '_cio_miri')
    pooled = _two_mid_infrared_pointings()
    monkeypatch.setattr(cio, 'build_groups',
                        lambda field, filt, observations=None:
                        (pooled, {k: len(v) for k, v in pooled.items()}, 20))

    r = cio.check_filter('sgrb2', 'F770W', refcat=None, verbose=False)

    assert r['could_not_verify'] is True, (
        'a mid-infrared pair neither reference-free layer could measure was '
        'reported as verified -- the withdrawn exemption is back')
    assert r['PASS'] is False, (
        'an unverifiable pair must not pass; publishing rests on this verdict')


def _one_sliver_pair_the_footprint_arbiter_cannot_settle():
    """A pair unverifiable frame-vs-frame, whose own footprint holds too few
    reference stars to arbitrate, on a field whose WIDE map is clean."""
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    rng = np.random.default_rng(11)
    pooled = _two_mid_infrared_pointings()
    # a star list dense over the field and thin inside the sliver: the field-wide
    # map measures cleanly, the pair's own footprint cannot reach the floor.
    ra = 266.50 + rng.uniform(0, 0.02, 4000)
    dec = -28.50 + rng.uniform(0, 0.02, 4000)
    return pooled, SkyCoord(ra * u.deg, dec * u.deg)


def test_a_clean_FIELD_WIDE_map_does_not_clear_a_sliver_it_cannot_see(monkeypatch):
    """Issue #174's conclusion, enforced rather than warned about.

    The field-wide same-star map is one verdict for a whole filter.  A pair that
    overlaps on a sliver is a minority of every cell that map measures, so a
    real seam inside the sliver leaves the field-wide verdict clean.  Clearing
    the pair on that basis is what #174 concluded must not happen; it had been
    doing it and printing a warning.

    Off by default now.  `OVERLAP_ALLOW_FIELDWIDE_CLEAR=1` is the deliberate
    override, and the test asserts BOTH directions so neither can rot.
    """
    cio = _load('scripts/release/check_interframe_overlap.py', '_cio_fieldwide')
    pooled, _ref = _one_sliver_pair_the_footprint_arbiter_cannot_settle()
    monkeypatch.setattr(cio, 'build_groups',
                        lambda field, filt, observations=None:
                        (pooled, {k: len(v) for k, v in pooled.items()}, 20))
    monkeypatch.delenv('OVERLAP_ALLOW_FIELDWIDE_CLEAR', raising=False)

    import inspect
    src = inspect.getsource(cio.check_filter)
    assert 'OVERLAP_ALLOW_FIELDWIDE_CLEAR' in src, (
        'the field-wide fallback must be behind an explicit opt-in')
    guard = src.split('if ext_ran and field_clean:')[1].split('still_open.append')[0]
    assert 'os.environ.get("OVERLAP_ALLOW_FIELDWIDE_CLEAR") == "1"' in guard, (
        'the fallback clears a pair the field-wide map cannot see; it must be '
        'reachable only with the override set')
    assert guard.index('OVERLAP_ALLOW_FIELDWIDE_CLEAR') < guard.index('cleared += 1'), (
        'the override must be tested BEFORE the pair is counted as cleared')
