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

def _two_mid_infrared_pointings(strip_deg=0.0015, n_strip=60, off_arcsec=0.5,
                                with_reference=False):
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
    pooled = {"002001:mirimage": SkyCoord(ra_a * u.deg, dec_a * u.deg),
              "998001:mirimage": SkyCoord(ra_b * u.deg, dec_b * u.deg)}
    if not with_reference:
        return pooled
    # A reference drawn from THESE positions, dense over the two footprints and
    # thin inside the shared strip.  Drawn independently it shares no stars, the
    # arbiter never runs, and an exemption gated on "the arbiter measured
    # nothing" is unreachable -- so that shape of re-insertion survives.
    body_ra = np.concatenate([ra_a[:n], ra_b[:n]])
    body_dec = np.concatenate([dec_a[:n], dec_b[:n]])
    strip_ra, strip_dec = ra_a[n:n + 6], dec_a[n:n + 6]      # only 6, under the floor
    ref = SkyCoord(np.concatenate([body_ra, strip_ra]) * u.deg,
                   np.concatenate([body_dec, strip_dec]) * u.deg)
    return pooled, ref


@pytest.mark.parametrize('field', ['sgrb2', 'w51'])
def test_a_mid_infrared_pair_it_cannot_measure_is_NOT_passed(tmp_path,
                                                             monkeypatch,
                                                             field):
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
    from astropy.table import Table

    cio = _load('scripts/release/check_interframe_overlap.py', '_cio_miri')
    # WITH a reference list, so `ext_pair` is a real unmeasurable verdict rather
    # than None.  Called with refcat=None -- what this test used to do -- an
    # exemption gated on "the arbiter measured nothing" is never reached, so
    # that whole shape of re-insertion survived.  It is also w51 F560W's live
    # state.
    pooled, ref = _two_mid_infrared_pointings(with_reference=True)
    refcat = tmp_path / 'ref.fits'
    Table({'ra': ref.ra.deg, 'dec': ref.dec.deg}).write(refcat)
    monkeypatch.setattr(cio, 'build_groups',
                        lambda f, filt, observations=None:
                        (pooled, {k: len(v) for k, v in pooled.items()}, 20))
    monkeypatch.delenv('OVERLAP_ALLOW_FIELDWIDE_CLEAR', raising=False)

    r = cio.check_filter(field, 'F770W', refcat=str(refcat), verbose=False)
    assert r['could_not_verify'] is True, (
        'a mid-infrared pair neither reference-free layer could measure was '
        'reported as verified -- the withdrawn exemption is back')
    assert r['PASS'] is False, (
        'an unverifiable pair must not pass; publishing rests on this verdict')

    # ...and on the EXIT CODE, which is what stage_release consumes.  An
    # exemption written into main()'s aggregation instead of the still-open loop
    # leaves the verdict dict untouched and turns exit 2 into exit 0.
    #
    # The stub replays the REAL verdict, pair records included.  Returning
    # `pairs=[]` -- what this did first -- means a main()-level exemption keyed
    # on the pair labels (which is how the withdrawn one was keyed) finds
    # nothing to inspect and is never reached, so it survived while this
    # assertion passed.
    monkeypatch.setattr(cio, 'check_filter', lambda *a, **k: dict(r))
    rc = cio.main(['--field', field, '--filter', 'F770W'])
    assert rc == 2, (
        f'an unverifiable pair must leave the gate at exit 2, not {rc}; '
        f'stage_release refuses on the exit code, not on the dict')
    assert r.get('pairs'), (
        'the replayed verdict carries no pair records, so an exemption keyed '
        'on the pair labels would never be reached by this assertion')


# ---------------------------------------------------------------------------
# The field-wide fallback, driven through the real gate in both directions
# ---------------------------------------------------------------------------

def _sliver_with_a_seam(seed=31, n=9000, sliver_arcsec=7.2, seam_mas=500.0,
                        sliver_ref_stars=12):
    """Two groups overlapping in a thin strip, with a seam only inside it.

    The reference is drawn from the SAME truth positions as the detections --
    an independently drawn one gives no same-star tie at all, so the field-wide
    map reads `measurable=False` and the branch under test is never entered in
    either direction.  An earlier version of this fixture made that mistake and
    was never wired up, so it could not have failed.

    Dense over the field and thinned inside the strip: that is the shape that
    makes the field-wide map read clean while the pair's own footprint holds too
    few reference stars to arbitrate, which is the only way to reach the
    fallback.
    """
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    ra0, dec0 = 266.5, -28.7
    cosd = float(np.cos(np.deg2rad(dec0)))
    rng = np.random.default_rng(seed)
    ra = ra0 + rng.uniform(-0.02, 0.02, n) / cosd
    dec = dec0 + rng.uniform(-0.02, 0.02, n)

    half = sliver_arcsec / 2.0 / 3600.0
    in_a, in_b = dec <= dec0 + half, dec >= dec0 - half
    jit = lambda v, k: v + rng.normal(0, 5.0 / 3.6e6, k)      # 5 mas jitter
    a_ra, a_dec = jit(ra[in_a], in_a.sum()), jit(dec[in_a], in_a.sum())
    b_ra, b_dec = jit(ra[in_b], in_b.sum()), jit(dec[in_b], in_b.sum())
    band = b_dec <= dec0 + half                      # the seam, inside the strip
    b_ra[band] += seam_mas / 3.6e6 / cosd

    # reference: every star outside the strip, only a handful inside it
    outside = np.abs(dec - dec0) > half
    inside = np.where(~outside)[0]
    keep = np.concatenate([np.where(outside)[0],
                           rng.choice(inside, size=min(sliver_ref_stars,
                                                       len(inside)),
                                      replace=False)])
    ref = SkyCoord(ra[keep] * u.deg, dec[keep] * u.deg)
    pooled = {"001001:nrca": SkyCoord(a_ra * u.deg, a_dec * u.deg),
              "001002:nrcb": SkyCoord(b_ra * u.deg, b_dec * u.deg)}
    return pooled, ref


@pytest.mark.parametrize('allow,expect_pass', [
    (None, False),
    ('1', True),
    # '0' must NOT enable it: loosening the test to `is not None` survives every
    # other assertion here, and an operator who sets the variable to 0 to turn
    # the fallback OFF would turn it on.
    ('0', False),
])
def test_a_clean_FIELD_WIDE_map_does_not_clear_a_sliver_it_cannot_see(
        tmp_path, monkeypatch, allow, expect_pass):
    """Issue #174's conclusion, enforced rather than warned about.

    The field-wide same-star map is one verdict for a whole filter.  A pair
    overlapping on a thin strip is a minority of every cell that map measures,
    so a real seam inside the strip leaves the field-wide verdict clean.  The
    gate used to clear the pair on that basis while printing a warning saying
    the map could not resolve it.

    Driven through the real `check_filter`, both directions.  The previous
    version of this test asserted three substrings of `check_filter`'s SOURCE:
    re-inserting the clearing with the words in a different order left it green
    while the 500 mas seam passed the gate.
    """
    from astropy.table import Table

    cio = _load('scripts/release/check_interframe_overlap.py', '_cio_fieldwide')
    pooled, ref = _sliver_with_a_seam()

    refcat = tmp_path / 'sliver_ref.fits'
    Table({'ra': ref.ra.deg, 'dec': ref.dec.deg}).write(refcat)
    monkeypatch.setattr(cio, 'build_groups',
                        lambda field, filt, observations=None:
                        (pooled, {k: len(v) for k, v in pooled.items()}, 20))
    if allow is None:
        monkeypatch.delenv('OVERLAP_ALLOW_FIELDWIDE_CLEAR', raising=False)
    else:
        monkeypatch.setenv('OVERLAP_ALLOW_FIELDWIDE_CLEAR', allow)

    r = cio.check_filter('brick', 'F405N', refcat=str(refcat), verbose=False)

    assert r['PASS'] is expect_pass, (
        f"with OVERLAP_ALLOW_FIELDWIDE_CLEAR={allow!r} a 500 mas seam confined "
        f"to the strip gave PASS={r['PASS']}; the field-wide map cannot see it")
    if not expect_pass:
        assert r['could_not_verify'] is True


# ---------------------------------------------------------------------------
# What may FAIL a field on an absolute-frame tie
# ---------------------------------------------------------------------------

def _whole_field_shift(seed=5, n=9000, shift_mas=500.0):
    """One filter's detections, and a reference the whole field is offset from."""
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    rng = np.random.default_rng(seed)
    ra0, dec0 = 266.5, -28.7
    cosd = float(np.cos(np.deg2rad(dec0)))
    ra = ra0 + rng.uniform(-0.02, 0.02, n) / cosd
    dec = dec0 + rng.uniform(-0.02, 0.02, n)
    ref = SkyCoord(ra * u.deg, dec * u.deg)
    half = n // 2
    det = SkyCoord((ra + shift_mas / 3.6e6 / cosd) * u.deg, dec * u.deg)
    pooled = {"001001:nrca": det[:half], "001002:nrcb": det[half:]}
    return pooled, ref


@pytest.mark.parametrize('with_source_column', [True, False])
def test_a_DENSE_reference_fails_the_field_whatever_its_columns(
        tmp_path, monkeypatch, with_source_column):
    """A real 500 mas absolute offset must block, and the column must not decide.

    The first version of this guard keyed on whether the catalogue carried a
    `source` column.  That is not a density test and it inverts at the boundary:
    omegacen's 115,009-row list has no such column, ngc6334's 23,639-row list
    has one.  Same table, same offset, different column name gave opposite
    verdicts.
    """
    from astropy.table import Table

    cio = _load('scripts/release/check_interframe_overlap.py', '_cio_dense')
    pooled, ref = _whole_field_shift()
    cols = {'ra': ref.ra.deg, 'dec': ref.dec.deg}
    if with_source_column:
        cols['source'] = [b'VIRAC2'] * len(ref)
    path = tmp_path / 'dense.fits'
    Table(cols).write(path)
    monkeypatch.setattr(cio, 'build_groups',
                        lambda f, filt, observations=None:
                        (pooled, {k: len(v) for k, v in pooled.items()}, 20))
    monkeypatch.delenv('OVERLAP_ALLOW_FIELDWIDE_CLEAR', raising=False)

    # The two exposure groups are shifted TOGETHER, so they agree with each
    # other and the pair is unmeasurable -- `PASS is False` would be true from
    # `could_not_verify` alone, which is how the first version of this test let
    # `if may_gate and False:` survive.  Assert on the absolute arm itself.
    r = cio.check_filter('brick', 'F405N', refcat=str(path), verbose=False)
    assert r['PASS'] is False
    assert r.get('ext_fail') is True, (
        f'a dense reference measuring a 500 mas absolute offset did not fail '
        f'the field on that arm (source column present: {with_source_column})')


def test_a_SPARSE_reference_does_not_fail_the_field_by_itself(tmp_path,
                                                              monkeypatch):
    """The w51 case: a list matching a few hundred stars over a mosaic cannot
    support the claim "this whole filter is in the wrong place".

    Supplying w51's Gaia-only list turned F140M, F150W and F162M from pass to
    FAIL on this arm alone, while each pair's own frame-against-frame
    measurement stayed clean at 3 mas over 18 of 18 tiles.
    """
    from astropy.table import Table

    cio = _load('scripts/release/check_interframe_overlap.py', '_cio_sparse')
    pooled, ref = _whole_field_shift()
    thin = ref[::40]                      # ~225 stars, under the floor
    path = tmp_path / 'sparse.fits'
    Table({'ra': thin.ra.deg, 'dec': thin.dec.deg}).write(path)
    monkeypatch.setattr(cio, 'build_groups',
                        lambda f, filt, observations=None:
                        (pooled, {k: len(v) for k, v in pooled.items()}, 20))
    monkeypatch.delenv('OVERLAP_ALLOW_FIELDWIDE_CLEAR', raising=False)

    r = cio.check_filter('w51', 'F140M', refcat=str(path), verbose=False)
    assert r.get('ext_fail') is not True, (
        'a sparse list failed the field on an absolute tie by itself')


def test_the_gating_floor_is_a_measured_match_count(tmp_path):
    """Not the catalogue's provenance, and not a column name."""
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    cio = _load('scripts/release/check_interframe_overlap.py', '_cio_floor')
    rng = np.random.default_rng(3)
    ra = 266.5 + rng.uniform(-0.02, 0.02, 4000)
    dec = -28.7 + rng.uniform(-0.02, 0.02, 4000)
    src = SkyCoord(ra * u.deg, dec * u.deg)

    may, n, _ = cio._may_gate_absolute_frame(src, src)
    assert may and n >= cio.MIN_GATING_MATCHES
    may, n, _ = cio._may_gate_absolute_frame(src, src[::100])
    assert not may and n < cio.MIN_GATING_MATCHES
    assert cio._may_gate_absolute_frame(src, None)[0] is False

    # ...and it must NOT collapse when the field is genuinely misaligned, which
    # is the case the absolute arm exists for.  A match-based count does: the
    # same 4000-star reference reads 4000 in-footprint and only a few hundred
    # matched once a 500 mas shift is applied.
    shifted = SkyCoord((ra + 500 / 3.6e6 / np.cos(np.deg2rad(-28.7))) * u.deg,
                       dec * u.deg)
    may_shift, n_shift, _ = cio._may_gate_absolute_frame(shifted, src)
    assert may_shift, (
        f'a misaligned field made its own reference look too sparse to gate '
        f'({n_shift} in footprint) -- the arm would switch off exactly when it '
        f'is needed')
