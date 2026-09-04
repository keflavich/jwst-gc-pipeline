"""The arbiter of last resort: a field's own merged photometry.

Before a field is published, a check confirms that overlapping exposures agree
about where the stars are.  A pair that overlaps on a sliver too thin to compare
directly is settled instead by matching both sides against a common list of
star positions -- and for two fields no such list is dense enough.  w51's
inter-module F187N pair reaches 15 matches against its 9,454-row Gaia list and
needs 20; wd1 has five deferred F200W pairs and no list at all.  Both fields
have their own merged photometry on disk, several times denser, and using it is
not circular for THIS job: the arbiter differences the two groups' residuals
star by star, so the reference cancels out.

It IS circular for the absolute frame, where nothing cancels -- a field
measured against a catalogue built from that same field agrees by construction.
So the fallback comes with a bar, and the bar is what most of this module pins.
Issue #263.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

_REPO = Path(__file__).resolve().parents[3]


def _load(relpath, name):
    path = _REPO / relpath
    if not path.exists():                                   # pragma: no cover
        pytest.skip(f'{relpath} not present', allow_module_level=True)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stage = _load('scripts/release/stage_release.py', '_stage_internal_arbiter')
cio = _load('scripts/release/check_interframe_overlap.py', '_cio_internal_arbiter')


def _write_internal(path, coords):
    """A stand-in for a field's merged catalogue: positions in `skycoord_ref`."""
    Table({'skycoord_ref.ra': coords.ra.deg,
           'skycoord_ref.dec': coords.dec.deg,
           'skycoord_ref_filtername': ['f187n'] * len(coords)}).write(path)
    return str(path)


def _write_external(path, coords):
    Table({'ra': coords.ra.deg, 'dec': coords.dec.deg}).write(path)
    return str(path)


# ---------------------------------------------------------------------------
# Which list is reached, and in what order
# ---------------------------------------------------------------------------

def test_a_field_no_registry_names_falls_back_to_its_own_merged_catalogue(
        tmp_path, monkeypatch):
    """Without this, w51 stays blocked on a pair 5 reference stars short and
    wd1 cannot stage at all -- the registries have no entry that would help."""
    catdir = tmp_path / 'catalogs'
    catdir.mkdir()
    merged = catdir / stage.INTERNAL_ARBITER_CATALOGS[0]
    merged.write_text('')
    monkeypatch.setitem(stage.FIELDS, 'fakefield', {'data_dir': tmp_path})
    assert stage.internal_arbiter_refcat('fakefield') == str(merged)
    assert stage.overlap_arbiter_refcat('fakefield') == str(merged)


def test_an_older_merge_stage_is_used_when_the_newest_is_absent(tmp_path,
                                                                monkeypatch):
    """wd1's merged catalogue is m7, not m8.  Naming only the newest stage
    would leave the second field this fallback exists for uncovered."""
    catdir = tmp_path / 'catalogs'
    catdir.mkdir()
    older = catdir / stage.INTERNAL_ARBITER_CATALOGS[-1]
    older.write_text('')
    monkeypatch.setitem(stage.FIELDS, 'fakefield', {'data_dir': tmp_path})
    assert stage.internal_arbiter_refcat('fakefield') == str(older)


def test_a_registered_external_list_still_wins(tmp_path, monkeypatch):
    """An external list is independent evidence and an internal one is not, so
    the internal list is reached only where nothing else can arbitrate."""
    catdir = tmp_path / 'catalogs'
    catdir.mkdir()
    (catdir / stage.INTERNAL_ARBITER_CATALOGS[0]).write_text('')
    registered = tmp_path / 'gaia_refcat.fits'
    registered.write_text('')
    monkeypatch.setitem(stage.FIELDS, 'fakefield', {'data_dir': tmp_path})
    monkeypatch.setitem(stage.OVERLAP_ARBITER_REFCAT, 'fakefield',
                        str(registered))
    assert stage.overlap_arbiter_refcat('fakefield') == str(registered)


# ---------------------------------------------------------------------------
# Recognising the list for what it is
# ---------------------------------------------------------------------------

def test_an_internal_list_is_recognised_by_its_position_columns(tmp_path):
    """Provenance is read off the FILE, not off its name: a path can be renamed
    or registered by hand, and the bar below has to hold for whatever arrives."""
    sc = SkyCoord(np.linspace(266.5, 266.51, 50) * u.deg,
                  np.linspace(-28.7, -28.69, 50) * u.deg)
    _rc, gaia, label, internal = cio._refcat(
        _write_internal(tmp_path / 'merged.fits', sc))
    assert internal is True
    assert gaia is None
    assert 'INTERNAL' in label

    _rc, _gaia, _label, internal = cio._refcat(
        _write_external(tmp_path / 'ext.fits', sc))
    assert internal is False


def test_the_internal_list_keeps_its_positions(tmp_path):
    sc = SkyCoord(np.linspace(266.5, 266.51, 50) * u.deg,
                  np.linspace(-28.7, -28.69, 50) * u.deg)
    rc, _gaia, _label, _internal = cio._refcat(
        _write_internal(tmp_path / 'merged.fits', sc))
    assert len(rc) == 50
    assert float(np.max(rc.separation(sc).mas)) < 1.0


def test_rows_with_no_position_are_dropped(tmp_path):
    """A merged catalogue carries a row per SOURCE, and a source absent from
    the reference filter has no position.  Kept, those NaNs propagate into the
    arbiter's matching."""
    ra = np.linspace(266.5, 266.51, 50)
    dec = np.linspace(-28.7, -28.69, 50)
    ra[3], dec[7] = np.nan, np.nan
    Table({'skycoord_ref.ra': ra, 'skycoord_ref.dec': dec}).write(
        tmp_path / 'merged.fits')
    rc, _g, _l, internal = cio._refcat(str(tmp_path / 'merged.fits'))
    assert internal is True
    assert len(rc) == 48


# ---------------------------------------------------------------------------
# The bar: an internal list may not judge the absolute frame
# ---------------------------------------------------------------------------

def _dense_field(n=4000, seed=5):
    rng = np.random.default_rng(seed)
    ra = 266.5 + rng.uniform(-0.02, 0.02, n)
    dec = -28.7 + rng.uniform(-0.02, 0.02, n)
    return SkyCoord(ra * u.deg, dec * u.deg)


def test_the_fields_own_catalogue_may_not_fail_the_field_on_the_absolute_frame():
    """The density floor does not stop it: an internal list has ~57,000 stars
    against a floor of 1000, so it sails through and would gate on a check that
    passes by construction -- including on a field whose absolute tie is wrong,
    which for w51 is open (#257).  The bar has to be by provenance."""
    src = _dense_field()
    ref = _dense_field()
    assert len(ref) > cio.MIN_GATING_MATCHES

    may_gate, n_inside, _n_ref = cio._may_gate_absolute_frame(src, ref)
    assert may_gate is True, 'the fixture is not dense enough to test the bar'
    assert n_inside > cio.MIN_GATING_MATCHES

    may_gate, n_inside_internal, _n_ref = cio._may_gate_absolute_frame(
        src, ref, internal=True)
    assert may_gate is False, (
        "a list built from the field's own photometry was allowed to fail the "
        "field on its absolute frame, which it agrees with by construction")
    assert n_inside_internal == n_inside, (
        'the star count is still reported, so the log can say how dense the '
        'arbiter was')


def _one_pointing_off_the_reference(off_mas=70.0, n=6000, n_off=1500, seed=9):
    """Two pointings on disjoint sky, one of them sitting off the reference.

    Disjoint, so the gate finds no overlapping pair and the only verdict under
    test is the ABSOLUTE-frame arm.  The displaced pointing is the minority, so
    the global tie still lands on zero and the mis-tie shows up where the arm
    looks for it -- per cell, in that pointing's own cells.  A uniform
    whole-field offset would not: the global tie absorbs it and every cell then
    reads clean.
    """
    rng = np.random.default_rng(seed)
    dec0 = -28.7
    cosd = float(np.cos(np.deg2rad(dec0)))
    n_on = n - n_off
    ra = 266.5 + rng.uniform(-0.02, 0.02, n) / cosd
    # separated by 72" in declination, so the two footprints do not intersect
    dec = np.concatenate([dec0 - 0.02 + rng.uniform(0, 0.01, n_on),
                          dec0 + 0.01 + rng.uniform(0, 0.01, n_off)])
    ref = SkyCoord(ra * u.deg, dec * u.deg)
    jit = rng.normal(0, 3.0 / 3.6e6, (2, n))
    sra, sdec = ra + jit[0], dec + jit[1]
    sra[n_on:] += off_mas / 3.6e6 / cosd
    pooled = {'001001:nrca': SkyCoord(sra[:n_on] * u.deg, sdec[:n_on] * u.deg),
              '002001:nrcb': SkyCoord(sra[n_on:] * u.deg, sdec[n_on:] * u.deg)}
    return pooled, ref


@pytest.mark.parametrize('writer,expect_fail', [
    (_write_external, True),
    (_write_internal, False),
])
def test_only_an_external_list_fails_a_field_on_the_absolute_tie(
        tmp_path, monkeypatch, writer, expect_fail):
    """Driven through the real gate, on IDENTICAL numbers: the same 6,000 stars
    and the same 60 mas whole-field offset, written once with plain `ra`/`dec`
    and once as a merged catalogue's `skycoord_ref`.  The external file refuses
    the filter; the internal one reports the same map and refuses nothing.
    """
    pooled, ref = _one_pointing_off_the_reference()
    path = writer(tmp_path / 'ref.fits', ref)
    monkeypatch.setattr(cio, 'build_groups',
                        lambda f, filt, observations=None:
                        (pooled, {k: len(v) for k, v in pooled.items()}, 8))
    r = cio.check_filter('w51', 'F187N', refcat=path, verbose=False)
    assert r['n_overlapping'] == 0, 'the fixture must isolate the absolute arm'
    assert r['ext_fail'] is expect_fail, (
        'an internal list refused a field on its absolute frame'
        if not expect_fail else
        'the fixture no longer produces a measurable absolute mis-tie')
    assert r['PASS'] is not expect_fail


# ---------------------------------------------------------------------------
# ...while still doing the job it was added for
# ---------------------------------------------------------------------------

def _sliver_pair(seam_mas=0.0, seed=31, n=9000, sliver_arcsec=7.2):
    """Two groups overlapping in a thin strip, with the reference drawn from
    the same truth positions -- the shape a merged catalogue has.

    ``seam_mas`` is a real inter-module offset applied inside the strip.
    """
    ra0, dec0 = 266.5, -28.7
    cosd = float(np.cos(np.deg2rad(dec0)))
    rng = np.random.default_rng(seed)
    ra = ra0 + rng.uniform(-0.02, 0.02, n) / cosd
    dec = dec0 + rng.uniform(-0.02, 0.02, n)
    half = sliver_arcsec / 2.0 / 3600.0
    in_a, in_b = dec <= dec0 + half, dec >= dec0 - half
    jit = lambda v, k: v + rng.normal(0, 3.0 / 3.6e6, k)
    a_ra, a_dec = jit(ra[in_a], in_a.sum()), jit(dec[in_a], in_a.sum())
    b_ra, b_dec = jit(ra[in_b], in_b.sum()), jit(dec[in_b], in_b.sum())
    b_ra[b_dec <= dec0 + half] += seam_mas / 3.6e6 / cosd
    pooled = {'001001:nrca': SkyCoord(a_ra * u.deg, a_dec * u.deg),
              '001001:nrcb': SkyCoord(b_ra * u.deg, b_dec * u.deg)}
    return pooled, SkyCoord(ra * u.deg, dec * u.deg)


@pytest.mark.parametrize('seam_mas,expect_pass', [(0.0, True), (500.0, False)])
def test_the_internal_list_arbitrates_the_pair_it_was_added_for(
        tmp_path, monkeypatch, seam_mas, expect_pass):
    """Both directions, because a fallback that only ever says "clean" is worse
    than none: a registered pair with a 500 mas seam inside the sliver must
    still be refused by the same list that clears the aligned one.
    """
    pooled, ref = _sliver_pair(seam_mas=seam_mas)
    path = _write_internal(tmp_path / 'merged.fits', ref)
    monkeypatch.setattr(cio, 'build_groups',
                        lambda f, filt, observations=None:
                        (pooled, {k: len(v) for k, v in pooled.items()}, 64))
    monkeypatch.delenv('OVERLAP_ALLOW_FIELDWIDE_CLEAR', raising=False)
    r = cio.check_filter('w51', 'F187N', refcat=path, verbose=False)
    assert r['n_overlapping'] == 1
    assert r['PASS'] is expect_pass
    if expect_pass:
        assert r['could_not_verify'] is False, (
            'the pair the fallback exists for was still not arbitrated')


def test_a_deferred_pair_is_not_cleared_by_an_internal_field_wide_map(
        tmp_path, monkeypatch):
    """The field-wide map against an internal list is clean by construction, so
    it must not clear a pair its own footprint could not arbitrate -- not even
    under OVERLAP_ALLOW_FIELDWIDE_CLEAR, which exists for an INDEPENDENT map
    that could not see the sliver.
    """
    pooled, ref = _sliver_pair(seam_mas=500.0)
    # thin the reference inside the sliver so the pair's own footprint cannot
    # arbitrate, leaving only the field-wide route
    dec0, half = -28.7, 7.2 / 2.0 / 3600.0
    outside = np.abs(ref.dec.deg - dec0) > half
    rng = np.random.default_rng(3)
    inside = np.where(~outside)[0]
    keep = np.concatenate([np.where(outside)[0],
                           rng.choice(inside, size=6, replace=False)])
    path = _write_internal(tmp_path / 'merged.fits', ref[keep])
    monkeypatch.setattr(cio, 'build_groups',
                        lambda f, filt, observations=None:
                        (pooled, {k: len(v) for k, v in pooled.items()}, 64))
    monkeypatch.setenv('OVERLAP_ALLOW_FIELDWIDE_CLEAR', '1')
    r = cio.check_filter('w51', 'F187N', refcat=path, verbose=False)
    assert r['PASS'] is False, (
        'a pair nothing could arbitrate in its own footprint was cleared by a '
        'field-wide map against the field\'s own catalogue, which reads clean '
        'whatever the field does')
