"""Builder behaviour on a SINGLE-MODULE field.

sickle (3958/007) is nrcb-only.  Three separate places in
``build_virac2_offsets`` assumed a two-module field and refused the whole build
for a field that is perfectly measurable:

* ``coarse_from_i2d`` required a ``-merged`` mosaic, which a single-module field
  never produces;
* ``_gather``'s ``exp*`` glob also matched the grouped-fit per-frame catalogs
  sickle carries, so two products claimed the same frame and the duplicate check
  aborted;
* ``lock_filter`` let the "no per-frame catalogs" refusal from the module the
  field never observed abort the build for the module it did.

These test the first two directly, and the third's guard rail: skipping empty
modules must not degrade into silently locking nothing.
"""
import os

import pytest

import pytest

from jwst_gc_pipeline.reduction import build_virac2_offsets as b
from jwst_gc_pipeline.reduction.build_virac2_offsets import (
    MODULE_MOSAICS, NoPerFrameCatalogsError, REGION, mosaic_candidates,
    perframe_matches)


STEM = '/base/F210M/pipeline/jw03958-o007_t001_nircam_clear-f210m'


def _mosaic(tmp_path, label, mtime):
    p = tmp_path / f'jw03958-o007_t001_nircam_clear-f210m-{label}_i2d.fits'
    p.touch()
    os.utime(p, (mtime, mtime))
    return p


def test_merged_wins_on_a_two_module_field(tmp_path):
    """A two-module field must tie on merged -- the seam lives nowhere else --
    even when a per-module mosaic is newer."""
    stem = str(tmp_path / 'jw03958-o007_t001_nircam_clear-f210m')
    _mosaic(tmp_path, 'merged', 1_000_000)
    _mosaic(tmp_path, 'nrcb', 2_000_000)          # newer, and still must not win
    assert mosaic_candidates(stem, n_modules=2)[0][0] == 'merged'


def test_single_module_field_takes_the_newest_not_merged(tmp_path):
    """On a single-module field a `-merged` product cannot BE a merge, so it can
    only be a leftover generation.  sickle: f210m-merged is 2026-04-19 against a
    2026-08-04 nrcb, and ~17 mas of the F210M answer rides on which seeds it."""
    stem = str(tmp_path / 'jw03958-o007_t001_nircam_clear-f210m')
    _mosaic(tmp_path, 'merged', 1_000_000)        # older leftover
    _mosaic(tmp_path, 'nrcb', 2_000_000)          # current
    assert mosaic_candidates(stem, n_modules=1)[0][0] == 'nrcb'


def test_single_module_field_with_only_a_module_mosaic(tmp_path):
    """The original case: no merged product exists at all, build still measurable."""
    stem = str(tmp_path / 'jw03958-o007_t001_nircam_clear-f210m')
    _mosaic(tmp_path, 'nrcb', 2_000_000)
    assert [l for l, _ in mosaic_candidates(stem, n_modules=1)] == ['nrcb']


def test_no_mosaic_at_all_is_still_refused(tmp_path):
    stem = str(tmp_path / 'jw03958-o007_t001_nircam_clear-f210m')
    assert mosaic_candidates(stem, n_modules=1) == []
    assert mosaic_candidates(stem, n_modules=2) == []


def test_module_mosaic_labels_cover_both_channels():
    assert set(MODULE_MOSAICS) == {'nrca', 'nrcb', 'nrcalong', 'nrcblong'}


@pytest.mark.parametrize('mtag', ['_m3', '_m6'])
def test_grouped_variant_is_not_mistaken_for_the_plain_per_frame(mtag):
    """The grouped fit is a legitimate SECOND product of the same exposure.

    Matching both is what made the duplicate check report them as a stale
    pre-observation-token duplicate, which they are not.
    """
    plain = f'f210m_nrcb1_visit001_vgroup0310e_exp00001{mtag}_daophot_basic.fits'
    grouped = (f'f210m_nrcb1_visit001_vgroup0310e_exp00001_group{mtag}'
               f'_daophot_basic.fits')
    assert perframe_matches(plain, mtag)
    assert not perframe_matches(grouped, mtag)


def test_the_grouped_variant_is_matched_when_it_is_what_was_asked_for():
    """Nothing here should make the grouped products unreachable."""
    grouped = ('f210m_nrcb1_visit001_vgroup0310e_exp00001_group_m3'
               '_daophot_basic.fits')
    assert perframe_matches(grouped, '_group_m3')


def test_a_different_merge_tag_does_not_match():
    name = 'f210m_nrcb1_visit001_vgroup0310e_exp00001_m3_daophot_basic.fits'
    assert not perframe_matches(name, '_m6')


def test_sickle_declares_its_modules_for_both_channels():
    """The declaration is what authorises skipping module A. It must name the LW
    key too -- module_key() keeps the full detector name for long-wavelength, so
    'nrcb' alone would skip three of sickle's five filters."""
    assert REGION['sickle']['modules'] == ('nrcb', 'nrcblong')
    for det in ('nrcb1', 'nrcb4'):
        assert b.module_key(det) in REGION['sickle']['modules']
    assert b.module_key('nrcblong') in REGION['sickle']['modules']
    # and module A is genuinely excluded
    assert b.module_key('nrca1') not in REGION['sickle']['modules']


def test_sickle_region_is_registered_single_module_and_obs_007():
    """Step 0 refuses to measure a fresh tie for an already-tied field, so the
    route to VIRAC2 is to BUILD the table -- which needs a REGION entry."""
    rc = REGION['sickle']
    assert rc['proposal'] == '3958'
    # NIRCam is observation 007; the 3958 MIRI data are observation 001 and are
    # NOT tied by this entry.
    assert rc['field'] == '007'
    assert set(rc['filts']) == {'f187n', 'f210m', 'f335m', 'f470n', 'f480m'}
    # One epoch for the whole field: all five bands were taken together.
    assert {v[1] for v in rc['filts'].values()} == {2024.643}
    assert {v[2] for v in rc['filts'].values()} == {'_m3'}


def test_sgra_region_is_registered():
    """sgra 1939/001 had NO region, and that was a hard block (issue #409).

    Its m12 finalize dies every iteration on ``OffsetsTableUpdateError: cannot
    pool corrections ... 8 corrections spanning module families ['nrca','nrcb']
    land on the same row(s)`` -- correct, because the live table is 36 rows keyed
    (Visit, Exposure, Filter) with NO Module column.  The guard's message names
    ``build_virac2_offsets --per-module`` as the remedy, and without a region key
    that remedy could not be run at all.
    """
    rc = REGION['sgra']
    assert rc['proposal'] == '1939'
    assert rc['field'] == '001'
    assert set(rc['filts']) == {'f115w', 'f212n', 'f405n'}
    # one visit, one epoch: DATE-OBS 2022-09-19
    assert {v[1] for v in rc['filts'].values()} == {2022.715}
    assert {v[2] for v in rc['filts'].values()} == {'_m3'}


def test_every_virac2_locked_field_has_a_builder_region():
    """The gap sgra fell into is structural, not a typo.

    A field routed to ``TABLE_LOCKED`` against VIRAC2 can only get its table from
    this builder, so a locked field with no region key is a field whose table
    cannot be rebuilt -- and the m2 checkpoint's own remedy message points at a
    command that will refuse.  Keyed on (proposal, obsid) because that is what
    both registries agree on; the region KEY itself is free-form (``1182``,
    ``cloudef2``, ``gc2211_023``).
    """
    from jwst_gc_pipeline.reduction import alignment_config as ac

    have = {(rc['proposal'], rc['field']) for rc in REGION.values()}
    have_prop = {rc['proposal'] for rc in REGION.values()}
    missing = []
    for fa in ac.ALIGNMENT_CONFIG:
        if fa.source != ac.TABLE_LOCKED or fa.reference_frame != ac.VIRAC2:
            continue
        if fa.fields is None:
            # a proposal-wide entry (4147, 5365, 2211): any region for the
            # proposal can rebuild it, and gc2211 deliberately has one per
            # observation.
            if fa.proposal not in have_prop:
                missing.append(f'{fa.proposal}/<all>')
            continue
        for fld in fa.fields:
            if (fa.proposal, fld) not in have:
                missing.append(f'{fa.proposal}/{fld}')
    assert not missing, (
        'VIRAC2 TABLE_LOCKED field(s) with no build_virac2_offsets region, so '
        'their offsets tables cannot be rebuilt: ' + ', '.join(sorted(missing)))


def test_a_declared_module_with_no_catalogs_is_refused(monkeypatch, tmp_path):
    """DECLARED and empty means cataloging has not finished for a module the
    field DOES have.  Locking then writes a table missing it entirely, so it must
    raise rather than skip -- the failure would otherwise resurface much later as
    a match=0 raise at apply time."""
    rc = dict(REGION['sickle'], basepath=str(tmp_path),
              filts={'f210m': ('F210M', 2024.643, '_m3')})
    monkeypatch.setattr(b, 'virac2', lambda ep, cache: None)
    monkeypatch.setattr(b, 'coarse_from_i2d', lambda *a, **k: (0.0, 0.0))

    def empty(*a, **k):
        raise NoPerFrameCatalogsError('f210m: no per-frame catalogs for x matched y')
    monkeypatch.setattr(b, '_gather', empty)

    with pytest.raises(NoPerFrameCatalogsError):
        b.lock_filter('f210m', rc, per_module=True)


def test_skipping_every_module_cannot_silently_lock_nothing(monkeypatch, tmp_path):
    """With no declaration the skip is allowed, but a build that skipped EVERY
    module must still raise -- otherwise a field whose catalogs are all missing
    produces a zero-row table that reads as 'tied'."""
    rc = dict(REGION['sickle'], basepath=str(tmp_path),
              filts={'f210m': ('F210M', 2024.643, '_m3')})
    rc.pop('modules')                       # undeclared: skipping is permitted
    monkeypatch.setattr(b, 'virac2', lambda ep, cache: None)
    monkeypatch.setattr(b, 'coarse_from_i2d', lambda *a, **k: (0.0, 0.0))

    def empty(*a, **k):
        raise NoPerFrameCatalogsError('f210m: no per-frame catalogs for x matched y')
    monkeypatch.setattr(b, '_gather', empty)

    with pytest.raises(NoPerFrameCatalogsError, match='no module produced a tie'):
        b.lock_filter('f210m', rc, per_module=True)


def test_the_empty_module_exception_is_its_own_type():
    """Sniffing a message substring made a two-module field's unfinished catalog
    run look like an unobserved module."""
    assert issubclass(NoPerFrameCatalogsError, RuntimeError)
    assert not issubclass(NoPerFrameCatalogsError, b.WrongObservationError)


# ---------------------------------------------------------------------------
# coord_shift: same-star, not nearest-neighbour
# ---------------------------------------------------------------------------

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


def _dense_field(n=400, seed=7, spacing_arcsec=1.1):
    """A reference with sickle's real crowding: median NN ~1.1", below the 3"
    the dense-reference guard draws the line at."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    step = spacing_arcsec / 3600.0
    ra0, dec0 = 266.5733, -28.8009
    g = np.array([(i, j) for i in range(side) for j in range(side)][:n], float)
    jit = rng.normal(0, step * 0.15, g.shape)
    ra = ra0 + (g[:, 0] * step + jit[:, 0]) / np.cos(np.radians(dec0))
    dec = dec0 + g[:, 1] * step + jit[:, 1]
    return SkyCoord(ra * u.deg, dec * u.deg)


def test_the_counterpart_is_chosen_by_expected_offset_not_sky_distance():
    """The whole difference between same-star and nearest-neighbour.

    Each source has its TRUE counterpart 100 mas away in Dec and a DECOY 40 mas
    away in the opposite direction.  Nearest-neighbour picks the decoy, because it
    is nearer on the sky.  Selecting the pair nearest the expected offset picks
    the true one.  Both survive the CLIP_MAS=60 clip, which is why the clip does
    not save the NN form here.
    """
    ref = _dense_field(n=300, spacing_arcsec=3.0)
    true_mas, decoy_mas = 100.0, -40.0
    src = SkyCoord(ref.ra, ref.dec - (true_mas / 3.6e6) * u.deg)
    decoys = SkyCoord(src.ra, src.dec + (decoy_mas / 3.6e6) * u.deg)
    ref_plus = SkyCoord(np.concatenate([ref.ra.deg, decoys.ra.deg]) * u.deg,
                        np.concatenate([ref.dec.deg, decoys.dec.deg]) * u.deg)

    got = b.coord_shift(src.ra.deg, src.dec.deg, ref_plus,
                        peak=(0.0, true_mas / 1000.0))
    assert got is not None
    assert abs(got[1] * 1000.0 - true_mas) < 2.0, (
        f"recovered {got[1] * 1000:.1f} mas; the decoy at {decoy_mas} mas won")

    # and with no peak supplied it does fall for the nearer decoy -- which is what
    # the old nearest-neighbour form did unconditionally.
    naive = b.coord_shift(src.ra.deg, src.dec.deg, ref_plus)
    assert naive is not None
    assert abs(naive[1] * 1000.0 - decoy_mas) < 2.0


@pytest.mark.parametrize('inject_mas', [0.0, 50.0, 100.0, 200.0])
def test_a_clean_shift_is_recovered_at_every_scale(inject_mas):
    """Sanity: with the peak supplied, the injected shift comes back exactly."""
    ref = _dense_field()
    shift_deg = inject_mas / 1000.0 / 3600.0
    src = SkyCoord(ref.ra, ref.dec - shift_deg * u.deg)
    got = b.coord_shift(src.ra.deg, src.dec.deg, ref,
                        peak=(0.0, shift_deg * 3600.0))
    assert got is not None, f"no tie recovered at {inject_mas} mas"
    assert abs(got[1] * 1000.0 - inject_mas) < 2.0


def test_one_pair_per_source():
    """Without the uniqueness pass a crowded source contributes several pairs and
    the median is weighted by local density rather than by star."""
    ref = _dense_field()
    src = SkyCoord(ref.ra, ref.dec)
    got = b.coord_shift(src.ra.deg, src.dec.deg, ref)
    assert got is not None
    assert got[4] <= len(src), (
        f"n={got[4]} exceeds the {len(src)} sources -- a source paired twice")


def test_too_few_pairs_returns_none():
    ref = _dense_field(n=400)
    far = SkyCoord(ref.ra + 1.0 * u.deg, ref.dec)
    assert b.coord_shift(far.ra.deg, far.dec.deg, ref) is None
