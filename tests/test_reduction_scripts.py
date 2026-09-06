"""Unit tests for the operational scripts in scripts/reduction/ (not part of
the package; imported by path)."""
import importlib.util
import json
import os
import re
import subprocess
import time

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'reduction')


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(SCRIPTS, f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RELEASE = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'release')


def _load_release(name):
    """Same by-path import, for the release-side scripts."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(RELEASE, f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rename_stale_band_token():
    m = _load('rename_stale_mosaics')
    assert m.band_of('jw02221-o001_t001_nircam_clear-f182m-merged-reproject-vvv_i2d.fits') == 'f182m'
    assert m.band_of('jw02221-o002_t001_miri_f2550w_realigned-to-vvv.fits') == 'f2550w'
    assert m.band_of('jw02221-o001_t001_nircam_clear-F405N-merged_realigned-to-vvv.fits') == 'f405n'
    assert m.band_of('no_band_here.fits') is None


def _band_dir(tmp_path, field='myfield', band='F182M'):
    pipe = tmp_path / field / band / 'pipeline'
    pipe.mkdir(parents=True)
    return pipe


def _age(path, days, now=None):
    """Write a placeholder and backdate it.

    Not a FITS file, so ``generation`` falls back to mtime -- which is the
    fallback path the script documents, exercised here on purpose.
    """
    path.write_bytes(b'x')
    now = now or time.time()
    os.utime(path, (now - days * 86400,) * 2)
    return path


def test_rename_stale_staleness_logic(tmp_path):
    """A pre-campaign realigned mosaic is renamed; a same-campaign one is kept.

    The stale file is a ``realigned-to-vvv`` one rather than a ``reproject``:
    this test is about rule 1's CLOCK, and the reproject family is decided by
    ``reproject_supersession`` instead (its own tests below), so a reproject
    name here would exercise the gate and never reach the clock.
    """
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = _band_dir(tmp_path)
    # rule 1 judges against the band's `*-merged_data_i2d.fits`, unchanged
    ref = pipe / 'jw01182-o001_t001_nircam_clear-f182m-merged_data_i2d.fits'
    stale = pipe / 'jw01182-o001_t001_nircam_clear-f182m-merged_realigned-to-vvv.fits'
    fresh = pipe / 'jw01182-o001_t001_nircam_clear-f182m-merged_realigned-to-refcat.fits'
    for p, age_days in ((ref, 0), (stale, 400), (fresh, 0.5)):
        _age(p, age_days)
    plan = m.rename_stale_for_field('myfield', execute=True)
    assert len(plan) == 1
    assert not stale.exists()
    # the suffix takes the file out of every `*.fits` glob, which is the
    # protection; `.bad` is the form issue #339 asked for
    assert (str(stale) + m.SUFFIX) == str(stale) + '.bad'
    assert not (str(stale) + m.SUFFIX).endswith('.fits')
    assert os.path.exists(str(stale) + m.SUFFIX)
    assert fresh.exists()


def test_a_canonically_named_orphan_is_caught_by_its_generation(tmp_path):
    """Issue #339: cloudc's 2023 F405N/F444W mosaic, in miniature.

    Its name is the ordinary level-3 form -- ``jw<prop>-o<obs>_t<NNN>_<instr>_
    <band>_i2d.fits`` -- so no name pattern distinguishes it from a live
    product, and every ``*_i2d.fits`` glob in the tree selects it.  Only its
    generation says it belongs to a superseded reduction.
    """
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = _band_dir(tmp_path, band='F405N')
    current = pipe / 'jw02221-o002_t001_nircam_clear-f405n-merged_i2d.fits'
    orphan = pipe / 'jw02221-o002_t001_nircam_f405n-f444w_i2d.fits'
    _age(current, 0)
    _age(orphan, 1100)                                   # 2023, three years back
    plan = m.rename_stale_for_field('myfield', execute=True)
    assert [os.path.basename(p[0]) for p in plan] == [orphan.name]
    assert not orphan.exists()
    assert os.path.exists(str(orphan) + m.SUFFIX)
    assert current.exists()


def test_rule_2_does_not_widen_rule_1(tmp_path):
    """Adding the generation rule must not change which NAMED files are taken.

    Rule 1 judges against the band's ``*-merged_data_i2d.fits`` and skips the
    band when there is none.  Rule 2's reference exists in far more band
    directories, so letting rule 1 borrow it would silently promote every
    previously-skipped rule-1 candidate into a rename -- measured archive-wide
    as 55 files becoming over 200.  That is a separate decision about a much
    larger set, and this test is what stops it happening by accident.
    """
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = _band_dir(tmp_path, band='F405N')
    # a current primary mosaic exists, but NO `*-merged_data_i2d.fits`
    _age(pipe / 'jw02221-o002_t001_nircam_clear-f405n-merged_i2d.fits', 0)
    named = _age(pipe / 'jw02221-o002_t001_nircam_clear-f405n-nrca'
                        '_realigned-to-vvv.fits', 400)
    assert m.rename_stale_for_field('myfield', execute=True) == []
    assert named.exists()


def test_a_second_pointing_in_the_same_band_is_not_superseded(tmp_path):
    """ngc6334 and sickle both keep several pointings in one band directory.

    ngc6334's ``F200W/pipeline`` holds proposals 6778 and 7213 side by side and
    sickle's ``F1130W/pipeline`` holds observations o001/o002/o003 of 3958.  A
    pointing reduced weeks before its neighbour is current, not superseded;
    against the archive this fired on nine live products under the SUPERSEDED
    design, which ranked a product against its band's newest.

    Under the family rule the pointing is no longer a safety axis: merging
    pointings makes a family look live MORE often, so dropping it can only cause
    false negatives, never false positives.  It is kept for completeness -- see
    the masking test below -- and this test is the regression guard on the nine.
    """
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = _band_dir(tmp_path, band='F200W')
    newer = _age(pipe / 'jw06778-o001_t001_nircam_clear-f200w-merged_i2d.fits', 0)
    older = _age(pipe / 'jw07213-o001_t001_nircam_clear-f200w-merged_i2d.fits', 60)
    assert m.rename_stale_for_field('myfield', execute=True) == []
    assert newer.exists() and older.exists()


def test_a_merged_mosaic_is_not_ranked_against_its_per_module_siblings(tmp_path):
    """wd1's real geometry, and the reason this script nearly deleted 5.1 GB.

    A merged drizzle routinely runs later in a campaign than the per-module
    ones, so the merged mosaic ends up the OLDEST primary product of its band
    while being the band's headline deliverable.  On wd1 F200W the gap is 18
    days (merged 2026-06-13, nrca/nrcb 2026-07-01), and an earlier version of
    this rule -- which ranked a product against every primary mosaic of its band
    and pointing -- put that 5.1 GB file three days from quarantine.

    WHAT PROTECTS IT NOW, stated precisely because an earlier version of this
    docstring credited the wrong mechanism: at 18 days the AGE GUARD covers it
    on its own, and this test still passes with the product token removed from
    the key.  The product token matters for a different reason -- see
    ``test_an_orphan_is_not_masked_by_a_current_product_of_the_same_pointing``,
    which fails without it.  This test pins wd1's real configuration, which is
    what matters operationally, not the mechanism that saves it.
    """
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = _band_dir(tmp_path, band='F200W')
    merged = _age(pipe / 'jw01905-o001_t001_nircam_clear-f200w-merged_i2d.fits', 18)
    nrca = _age(pipe / 'jw01905-o001_t001_nircam_clear-f200w-nrca_i2d.fits', 0)
    nrcb = _age(pipe / 'jw01905-o001_t001_nircam_clear-f200w-nrcb_i2d.fits', 0)
    # even with the campaign floor pulled right up under the merged product
    assert m.rename_stale_for_field('myfield', execute=True,
                                    campaign_days=3) == []
    for p in (merged, nrca, nrcb):
        assert p.exists(), p.name


def test_a_product_merely_older_than_the_campaign_is_not_an_orphan(tmp_path):
    """The age guard, on its own.

    A product of a family that is no longer written, but only weeks old, is a
    band that finished early -- not an orphan of a retired reduction.  Only the
    year-scale gap distinguishes the two, and the real margins are narrow:
    measured over the archive, 5.85x on the live side (the nearest miss is a
    sickle MIRI product 62.4 days behind its field's newest MIRI mosaic) and
    1.17x on the orphan side (w51's merged-reproject at 426.7 days).  An earlier
    version of this docstring said "~50x either way"; that was wrong on both
    sides.
    """
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = _band_dir(tmp_path, band='F405N')
    _age(pipe / 'jw02221-o002_t001_nircam_clear-f405n-merged_i2d.fits', 0)
    recent = _age(pipe / 'jw02221-o002_t001_nircam_f405n-f444w_i2d.fits', 60)
    assert m.rename_stale_for_field('myfield', execute=True,
                                    campaign_days=3) == []
    assert recent.exists()


def test_a_quarantined_file_is_never_overwritten(tmp_path):
    """`os.rename` replaces its destination silently.

    A product regenerated under a name that was already quarantined would, on
    the next run, destroy the quarantined bytes -- of a tool whose whole premise
    is that it is reversible and the file is kept.  Same guard and same
    reasoning as `quarantine_pre_obstoken_catalogs.py`.
    """
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = _band_dir(tmp_path, band='F405N')
    _age(pipe / 'jw02221-o002_t001_nircam_clear-f405n-merged_i2d.fits', 0)
    orphan = pipe / 'jw02221-o002_t001_nircam_f405n-f444w_i2d.fits'
    _age(orphan, 1100)
    orphan.write_bytes(b'GENERATION-1')
    os.utime(orphan, (time.time() - 1100 * 86400,) * 2)
    m.rename_stale_for_field('myfield', execute=True)
    assert (pipe / (orphan.name + m.SUFFIX)).read_bytes() == b'GENERATION-1'

    # the product comes back under the same name, and is quarantined again
    orphan.write_bytes(b'GENERATION-2')
    os.utime(orphan, (time.time() - 1100 * 86400,) * 2)
    m.rename_stale_for_field('myfield', execute=True)
    assert (pipe / (orphan.name + m.SUFFIX)).read_bytes() == b'GENERATION-1', \
        'the first quarantine was destroyed'
    assert orphan.exists(), 'the second copy should be left in place, not lost'


def test_a_fits_date_is_read_as_utc(tmp_path, monkeypatch):
    """FITS `DATE` is UTC; reading it as local time skews every comparison.

    Below the day-scale thresholds here, but it is a 4-5 hour systematic that is
    not even constant across a daylight-saving boundary, sitting between two
    generations measured on different clocks.

    `TZ` is forced to a non-UTC zone: under `TZ=UTC` the broken and the correct
    implementations agree, so the test would pass against `time.mktime` and pin
    nothing.
    """
    monkeypatch.setenv('TZ', 'America/New_York')
    time.tzset()
    pytest.importorskip('astropy')
    from astropy.io import fits
    m = _load('rename_stale_mosaics')
    p = tmp_path / 'x.fits'
    hdu = fits.PrimaryHDU()
    hdu.header['DATE'] = '2023-07-11T12:00:00'
    hdu.writeto(p)
    import calendar
    expect = calendar.timegm(time.strptime('2023-07-11T12:00:00',
                                           '%Y-%m-%dT%H:%M:%S'))
    assert m.date_header(str(p)) == expect
    assert m.generation(str(p)) == (expect, 'DATE')


def test_the_orphans_reason_is_recorded_beside_it(tmp_path):
    """A run log under the field root is easy to lose; the sidecar is not."""
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = _band_dir(tmp_path, band='F405N')
    _age(pipe / 'jw02221-o002_t001_nircam_clear-f405n-merged_i2d.fits', 0)
    orphan = _age(pipe / 'jw02221-o002_t001_nircam_f405n-f444w_i2d.fits', 1100)
    m.rename_stale_for_field('myfield', execute=True)
    note = str(orphan) + m.SUFFIX + '.why.txt'
    assert os.path.exists(note)
    text = open(note).read()
    assert 'retired product family' in text
    assert m.SUFFIX in text                              # says how to undo it


def test_per_exposure_intermediates_are_never_candidates(tmp_path):
    """The scope guard: hundreds of live intermediates share the band directory.

    ``jw02221002001_02201_00001_nrcalong_align_outlier_i2d.fits`` and its
    siblings are per-exposure outlier-detection products, and many are
    legitimately years old.  A generation rule that matched ``*_i2d.fits`` would
    quarantine the lot.

    Counts, measured over cloudc's eight band directories and stated against the
    right category, because two earlier versions of this docstring were not:

      * files matching ``*outlier*_i2d.fits``:   0-144 per band (two bands zero)
      * every ``*_i2d.fits`` that is NOT a primary mosaic:  20-184 per band
      * every ``*_i2d.fits``:                    21-188 per band

    "171-188" appeared here from the first commit and matches none of the three.
    The number that belongs in a SCOPE guard is the middle one -- 20-184 -- since
    that is everything the primary-mosaic pattern has to exclude.
    """
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = _band_dir(tmp_path, band='F405N')
    _age(pipe / 'jw02221-o002_t001_nircam_clear-f405n-merged_i2d.fits', 0)
    old_intermediates = [
        _age(pipe / 'jw02221002001_02201_00001_nrcalong_align_outlier_i2d.fits', 900),
        _age(pipe / 'jw02221002001_02201_00001_nrcalong_i2d.fits', 900),
    ]
    byproducts = [
        _age(pipe / 'jw02221-o002_t001_nircam_clear-f405n-nrca_data_i2d.fits', 900),
        _age(pipe / ('jw02221-o002_t001_nircam_clear-f405n-nrca_m2_daophot_basic'
                     '_mergedcat_residual_i2d.fits'), 900),
    ]
    assert m.rename_stale_for_field('myfield', execute=True) == []
    for p in old_intermediates + byproducts:
        assert p.exists(), p.name


def test_a_bands_only_mosaic_is_not_flagged_by_the_campaign_floor(tmp_path):
    """cloudc's MIRI F2550W case: 59 days below the field floor, and correct.

    A band that simply finished earlier than the rest of the campaign must not
    be quarantined for it.  There is no per-band condition in rule 2 -- what
    prevents this is the age guard, and removing MIN_ORPHAN_AGE_DAYS is what
    makes this test fail.
    """
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    nircam = _band_dir(tmp_path, band='F182M')
    miri = _band_dir(tmp_path, band='F2550W')
    _age(nircam / 'jw02221-o002_t001_nircam_clear-f182m-merged_i2d.fits', 0)
    lone = _age(miri / 'jw02221-o001_t001_miri_f2550w_i2d.fits', 59)
    assert m.rename_stale_for_field('myfield', execute=True) == []
    assert lone.exists()


def test_an_already_quarantined_file_is_not_renamed_again(tmp_path):
    """Idempotence, and the older suffix keeps being recognised as quarantined."""
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = _band_dir(tmp_path, band='F405N')
    _age(pipe / 'jw02221-o002_t001_nircam_clear-f405n-merged_i2d.fits', 0)
    orphan = _age(pipe / 'jw02221-o002_t001_nircam_f405n-f444w_i2d.fits', 1100)
    m.rename_stale_for_field('myfield', execute=True)
    assert m.rename_stale_for_field('myfield', execute=True) == []
    legacy = _age(pipe / ('jw02221-o002_t001_nircam_clear-f405n-nrca'
                          '_realigned-to-vvv.fits_badastrometry_stale'), 900)
    assert m.rename_stale_for_field('myfield', execute=True) == []
    assert legacy.exists()
    assert os.path.exists(str(orphan) + m.SUFFIX)


def test_purge_satstar_caches(tmp_path):
    m = _load('purge_satstar_caches')
    pipe = tmp_path / 'brick' / 'F182M' / 'pipeline'
    cats = tmp_path / 'brick' / 'catalogs'
    pipe.mkdir(parents=True)
    cats.mkdir(parents=True)
    a = pipe / 'exp1_m12_satstar_catalog.fits'
    b = cats / 'f182m_consolidated_satstar_catalog.fits'
    other = pipe / 'exp1_m12_daophot_basic.fits'
    for p in (a, b, other):
        p.write_bytes(b'x')
    # dry run: nothing moves
    n = m.purge(str(tmp_path), 'brick', ['F182M'], execute=False)
    assert n == 2 and a.exists() and b.exists()
    # execute: both cache levels sidelined, unrelated file untouched
    n = m.purge(str(tmp_path), 'brick', ['F182M'], execute=True)
    assert n == 2
    assert not a.exists() and not b.exists()
    assert os.path.exists(str(a) + m.SUFFIX) and os.path.exists(str(b) + m.SUFFIX)
    assert other.exists()
    # idempotent: second execute finds nothing
    assert m.purge(str(tmp_path), 'brick', ['F182M'], execute=True) == 0


# --- apply_m2_checkpoint_corrections: per-exposure extension ---------------

def _write_m2_record(records_dir, filt, visit_exposures):
    """visit_exposures: {visit_int: [exposure ints]} -> a minimal m2 record with
    the full exposure enumeration (one detector), no corrections needed."""
    visits = []
    for vnum, exps in visit_exposures.items():
        visits.append(dict(visit=str(vnum), filtername=filt, exposures=[
            dict(key=[str(vnum), e, 'nrca1', filt]) for e in exps]))
    rec = dict(stage='m2', filtername=filt, visits=visits, corrections=[])
    with open(os.path.join(records_dir, f'checkpoint_m2_{filt}_latest.json'), 'w') as fh:
        json.dump(rec, fh)


def test_exposure_universe_keyed_by_visit_and_filter(tmp_path):
    m = _load('apply_m2_checkpoint_corrections')
    rd = tmp_path / 'astrometry_checkpoints'
    rd.mkdir()
    # 1182-like: two visits sharing exposure numbers 1..12; plus a 2221-like
    # single-visit filter with 1..3
    _write_m2_record(str(rd), 'F200W', {1: list(range(1, 13)), 2: list(range(1, 13))})
    _write_m2_record(str(rd), 'F182M', {1: [1, 2, 3]})
    u = m.load_exposure_universe(str(rd))
    assert u[(1, 'F200W')] == list(range(1, 13))
    assert u[(2, 'F200W')] == list(range(1, 13))
    assert u[(1, 'F182M')] == [1, 2, 3]
    assert (2, 'F182M') not in u


def test_extend_covers_subfloor_exposure_no_phantom_rows(tmp_path):
    """The reviewer's gap: an exposure sub-floor in EVERY filter carries no
    correction but is a real frame -- it must still get a row (from the record
    universe), and a visit must never receive another visit's exposure numbers."""
    from astropy.table import Table
    m = _load('apply_m2_checkpoint_corrections')
    rd = tmp_path / 'astrometry_checkpoints'
    rd.mkdir()
    # F200W tiles two visits, exposures 1..3 each; NONE carry a correction here
    _write_m2_record(str(rd), 'F200W', {1: [1, 2, 3], 2: [1, 2, 3]})
    universe = m.load_exposure_universe(str(rd))

    # pristine per-visit table (no Exposure column): one row per (visit, filter)
    tbl = Table(dict(
        Visit=['jw01182004001', 'jw01182004002'],
        Filter=['F200W', 'F200W'],
        **{'dra (arcsec)': [0.0, 0.0], 'ddec (arcsec)': [0.0, 0.0]}))
    tp = tmp_path / 'offsets.csv'
    tbl.write(str(tp), overwrite=True)

    out, extended = m.extend_table_to_per_exposure(
        str(tp), universe, extend_filters={'F200W'})
    assert extended
    assert 'Exposure' in out.colnames
    # visit 1 gets exposures 1,2,3; visit 2 gets exposures 1,2,3 -- 6 rows total
    v1 = sorted(int(r['Exposure']) for r in out if m._table_visit_number(r['Visit']) == 1)
    v2 = sorted(int(r['Exposure']) for r in out if m._table_visit_number(r['Visit']) == 2)
    assert v1 == [1, 2, 3]      # incl. exp 3, which carried NO correction
    assert v2 == [1, 2, 3]      # no phantom exposure numbers from the other visit
    assert len(out) == 6


def test_extend_leaves_unextended_filter_as_single_visit_row(tmp_path):
    from astropy.table import Table
    m = _load('apply_m2_checkpoint_corrections')
    rd = tmp_path / 'astrometry_checkpoints'
    rd.mkdir()
    _write_m2_record(str(rd), 'F410M', {1: [1, 2, 3, 4]})
    universe = m.load_exposure_universe(str(rd))
    tbl = Table(dict(Visit=['jw02221001001'], Filter=['F410M'],
                     **{'dra (arcsec)': [0.0], 'ddec (arcsec)': [0.0]}))
    tp = tmp_path / 'o.csv'
    tbl.write(str(tp), overwrite=True)
    # filter NOT in extend_filters -> keeps one per-visit row, Exposure = -1
    out, extended = m.extend_table_to_per_exposure(
        str(tp), universe, extend_filters=set())
    assert extended is False


def test_extend_idempotent_when_exposure_column_present(tmp_path):
    from astropy.table import Table
    m = _load('apply_m2_checkpoint_corrections')
    tbl = Table(dict(Visit=['jw02221001001'], Filter=['F410M'], Exposure=[1],
                     **{'dra (arcsec)': [0.0], 'ddec (arcsec)': [0.0]}))
    tp = tmp_path / 'o.csv'
    tbl.write(str(tp), overwrite=True)
    out, extended = m.extend_table_to_per_exposure(str(tp), {}, {'F410M'})
    assert extended is False
    assert len(out) == 1


PERFRAME_SBATCH = os.path.join(
    SCRIPTS, 'submit_cataloging_perframe_phase.sbatch')


def _perframe_sbatch_text():
    """The script with line continuations folded away.

    A rename split across a backslash-continuation would otherwise be collected
    by nothing and pass the gate test vacuously.
    """
    with open(PERFRAME_SBATCH) as fh:
        return fh.read().replace('\\\n', ' ')


def test_perframe_runtime_rename_never_clobbers_a_submitted_name():
    """A submit-time job name must survive the per-frame phase script.

    The standing naming rule wants target+program+obsid+stage on the job at
    SUBMIT time, because a queued job shows only that.  The phase script used to
    rename itself unconditionally to "<target>-pf-<phase>-<mode>", which drops
    the program and the obsid -- brick and cloudc are both 2221, and gc2211 has
    five observations, so the degraded name is genuinely ambiguous.  Every
    `scontrol update ... JobName` here must therefore be gated on the rename
    guard, which only fires for a bare submission.
    """
    text = _perframe_sbatch_text()
    renames = [ln.strip() for ln in text.splitlines()
               if not ln.lstrip().startswith('#')
               and 'scontrol update' in ln and 'JobId=' in ln]
    assert renames, 'expected the phase script to still contain renames'
    for line in renames:
        assert line.startswith('_pf_rename_wanted &&'), (
            f'ungated runtime rename would clobber the submit-time name: {line}')


def test_perframe_shard_name_does_not_carry_the_array_index():
    """The shard index must not be baked into the job NAME.

    `scontrol update JobId=<task>` does not reliably address one element of an
    array: sgrb2 m12 fanout 38867646 came out `pf_sgrb2_m12_s15` on all 16
    tasks, and sgrc 38851171 had tasks 13 and 15 both reading
    `pf_sgrc_m12_s15` (2026-08-07).  The shard is already unambiguous in the
    array-task id, so the name must not try to carry it.
    """
    text = _perframe_sbatch_text()
    for line in text.splitlines():
        if line.lstrip().startswith('#'):
            continue
        if 'scontrol update' in line or 'JobName=' in line:
            assert 'SLURM_ARRAY_TASK_ID' not in line, (
                f'array index must not go into the job name: {line.strip()}')


RETIE_LOOP = os.path.join(SCRIPTS, 'run_field_retie_loop.sh')


def _run_reduce_gate(tmp_path, states, ntasks, jobid='9999', keep_errexit=False):
    """Drive reduce_fully_succeeded() for real, with a stub `sacct` on PATH.

    Returns (returncode, output).  Sourcing the loop needs its four required
    vars; RETIE_LOOP_SOURCE_ONLY makes it stop before the iteration loop.
    """
    stub = tmp_path / 'bin'
    stub.mkdir()
    (stub / 'sacct').write_text('#!/bin/bash\nprintf "%s\\n" $SACCT_STATES\n')
    (stub / 'sacct').chmod(0o755)
    relax = '' if keep_errexit else 'set +e +u +o pipefail'
    script = f"""
        export PATH="{stub}:$PATH"
        export PROPOSAL=4147 FIELD=012 TARGET=sgrc FILTERS="a b"
        export RETIE_LOOP_SOURCE_ONLY=1
        source "{RETIE_LOOP}" >/dev/null 2>&1
        {relax}
        export SACCT_STATES="{states}"
        reduce_fully_succeeded "{jobid}" {ntasks}
    """
    proc = subprocess.run(['bash', '-c', script], capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def test_reduce_gate_stops_on_the_sgrc_partial_failure(tmp_path):
    """The case this PR exists for: 4 COMPLETED + 4 FAILED must not be cataloged.

    A filter whose reduce failed keeps the PREVIOUS iteration's WCS, so the m12
    merge compares this iteration's frames for some bands against last
    iteration's for others, and the m2 checkpoint writes that mixture into the
    consensus table as a correction.  sgrc iteration 3 (38870453, 2026-08-07).
    """
    rc, out = _run_reduce_gate(
        tmp_path, 'COMPLETED COMPLETED COMPLETED COMPLETED FAILED FAILED FAILED FAILED', 8)
    assert rc == 1, f'a partially-failed reduce must stop the loop:\n{out}'
    assert 'STOPPING before cataloging' in out
    assert '4/8 completed' in out


def test_reduce_gate_proceeds_when_every_task_completed(tmp_path):
    """And the happy path must NOT stop -- including under `set -e`.

    `grep -c` exits 1 when the count is 0, so an unguarded count of the
    non-COMPLETED tasks would kill the loop on exactly the all-succeeded case.
    """
    rc, out = _run_reduce_gate(tmp_path, 'COMPLETED ' * 8, 8)
    assert rc == 0, f'a fully successful reduce must proceed:\n{out}'
    assert 'STOPPING' not in out
    assert '8/8 completed, 0 not' in out


def test_reduce_gate_stops_when_nothing_completed(tmp_path):
    """Zero COMPLETED is the other `grep -c` zero-count case."""
    rc, out = _run_reduce_gate(tmp_path, 'FAILED ' * 8, 8)
    assert rc == 1, f'a wholly failed reduce must stop the loop:\n{out}'
    assert '0/8 completed' in out


def test_reduce_gate_stops_when_a_requeued_task_double_counts(tmp_path):
    """n_done > ntasks fails in the safe direction: stop, never catalog."""
    rc, out = _run_reduce_gate(tmp_path, 'COMPLETED ' * 9, 8)
    assert rc == 1, f'an unexpected task count must stop the loop:\n{out}'


def test_reduce_gate_stops_when_no_job_id_was_parsed(tmp_path):
    """An unparseable sbatch must stop, not silently catalog."""
    rc, out = _run_reduce_gate(tmp_path, '', 8, jobid='')
    assert rc == 1, f'a missing job id must stop the loop:\n{out}'
    assert 'could not parse a job id' in out


def test_reduce_gate_survives_errexit_on_the_happy_path(tmp_path):
    """`grep -c` exits 1 on a zero count, and the loop runs under `set -euo`.

    Called OUTSIDE an if-condition (where bash would suspend errexit), an
    unguarded count of the non-COMPLETED tasks aborts the script on exactly the
    all-succeeded case -- so a fully successful reduce would kill the loop
    silently, which is worse than the bug the guard fixes.  Verified against the
    unguarded form: rc=1 with no output at all.
    """
    rc, out = _run_reduce_gate(tmp_path, 'COMPLETED ' * 8, 8, keep_errexit=True)
    assert rc == 0, f'errexit must not abort the all-completed case:\n{out}'
    assert '8/8 completed, 0 not' in out


def test_the_loop_actually_calls_the_gate_between_reduce_and_catalog():
    """The gate has to be WIRED, not just correct.

    Every other test here sources the script and calls
    `reduce_fully_succeeded` directly, so deleting the call site leaves the
    suite green while restoring #327 in full: catalog a mixed reduce, m12
    merges this iteration's frames for the filters that succeeded with last
    iteration's for the ones that did not, and m2 writes the mixture into the
    consensus table as a correction.
    """
    with open(RETIE_LOOP) as fh:
        text = fh.read()
    between = text[text.index('--- 1. reduce'):text.index('--- 2. catalog')]
    assert 'reduce_fully_succeeded' in between, (
        'the loop must call the gate between reduce and catalog')
    assert 'exit 1' in between, (
        'the loop must stop, not continue, when the reduce is incomplete')


# ---------------------------------------------------------------------------
# Runtime job renames, generalised over EVERY submitter that does one (#330).
#
# submit_cataloging_m7.sbatch had the same defect #326 fixed one script over,
# twice and neither guarded, so the second write won over the submit-time name.
# Parameterising rather than duplicating means the next script to grow a rename
# is covered without anyone remembering to add a test.
# ---------------------------------------------------------------------------

#: Every submitter that renames itself at runtime, with the placeholder its
#: `#SBATCH --job-name` carries.  The guard idiom is deliberately NOT pinned:
#: the repo uses two spellings (a `_*_rename_wanted` helper and a bare
#: `if [ "${SLURM_JOB_NAME:-x}" = "x" ]`), and an earlier version of this test
#: enforced one of them -- which excluded the correctly-guarded scripts written
#: in the other, and let a one-character `=` -> `!=` inversion through.  These
#: tests EXECUTE the guard instead.
RENAMING_SBATCH = {
    'submit_cataloging_perframe_phase.sbatch': 'pf',
    'submit_cataloging_m7.sbatch': 'catalog_m7',
    'submit_cataloging.sbatch': 'catalog',
}


def _folded(basename):
    """The script with line continuations folded away, so a rename split across
    a backslash is still one line to match against."""
    with open(os.path.join(SCRIPTS, basename)) as fh:
        return fh.read().replace('\\\n', ' ')


def _rename_lines(text):
    return [ln.strip() for ln in text.splitlines()
            if not ln.lstrip().startswith('#')
            and 'scontrol update' in ln and 'JobId=' in ln]


def _rename_attempts(script, placeholder, job_name):
    """Run the script's rename logic with a stub `scontrol` and report what it
    tried to set the name to.

    Behavioural, not textual: the previous version string-matched the guard's
    definition line, so inverting it (`=` -> `!=`) -- which reproduces the exact
    #330 defect -- left every test green.  It also only recognised the literal
    `JobId=`, so re-adding a rename with slurm's equally-valid lowercase
    `jobid=` was invisible.
    """
    import subprocess
    import tempfile
    text = _folded(script)
    lines = text.splitlines()
    keep, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith('#'):
            i += 1
            continue
        low = line.lower()
        st = line.lstrip()
        if st.startswith('echo ') or st.startswith('printf '):
            # a message that merely MENTIONS scontrol (perframe_phase:209
            # suggests re-pointing a dependency) is not a rename
            i += 1
            continue
        if 'rename_wanted()' in line:            # guard helper definition
            keep.append(line)
        elif line.lstrip().startswith('if ') and 'SLURM_JOB_NAME' in line:
            # a guard written as an if-block: keep the WHOLE block, or the
            # rename inside it runs unconditionally here and the test reports a
            # clobber that does not happen.
            block, depth = [line], 1
            j = i + 1
            while j < len(lines) and depth:
                block.append(lines[j])
                st = lines[j].strip()
                if st.startswith('if '):
                    depth += 1
                elif st == 'fi':
                    depth -= 1
                j += 1
            if any('scontrol update' in b.lower() for b in block):
                keep.extend(block)
            i = j
            continue
        elif 'scontrol update' in low and 'jobid=' in low:
            keep.append(line)
        i += 1
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'scontrol'), 'w') as fh:
            fh.write('#!/bin/bash\necho "$@"\n')
        os.chmod(os.path.join(td, 'scontrol'), 0o755)
        env = dict(os.environ)
        env.update(PATH=td + os.pathsep + env['PATH'],
                   SLURM_JOB_ID='12345', SLURM_JOB_NAME=job_name,
                   TARGET='brick', PROPOSAL='2221', FIELD='001',
                   FILT='F182M', PHASE='m12', MODE='fanout')
        out = subprocess.run(['bash', '-c', '\n'.join(keep)], env=env,
                             capture_output=True, text=True, timeout=60)
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


@pytest.mark.parametrize('script,placeholder', sorted(RENAMING_SBATCH.items()))
def test_a_submit_time_name_SURVIVES(script, placeholder):
    """The whole point of #330/#326: a name given at submit time must not be
    overwritten at runtime."""
    attempts = _rename_attempts(script, placeholder, 'brick2221-o001-cat')
    assert not attempts, (
        f'{script}: renamed itself over the submit-time name -> {attempts}')


@pytest.mark.parametrize('script,placeholder', sorted(RENAMING_SBATCH.items()))
def test_a_BARE_submission_still_gets_renamed(script, placeholder):
    """And the guard must actually fire when it should -- an inverted guard
    passes the test above vacuously."""
    attempts = _rename_attempts(script, placeholder, placeholder)
    assert attempts, (
        f'{script}: a bare submission (SLURM_JOB_NAME={placeholder!r}) was '
        f'never renamed; the guard cannot fire')


@pytest.mark.parametrize('script,placeholder', sorted(RENAMING_SBATCH.items()))
def test_the_bare_path_name_is_readable_by_the_monitor(script, placeholder):
    """`cat_brick_m7` returned None from parse_job_name -- invisible to the
    monitor, not merely ambiguous.  Whatever the bare path picks must resolve to
    a registered field AND name a stage.

    The obsid is NOT required here: #326 settled that the bare fallback keeps
    the `pf_<target>_<phase>` shape because that is what `_NAME_PF` reads, and
    it carries no obsid by construction.  The obsid requirement belongs on the
    SUBMIT-time names, which is where
    test_every_emitted_job_name_resolves_to_a_field_and_an_observation puts it.
    """
    from jwst_gc_pipeline.monitoring.jobs import parse_job_name
    checked = 0
    for line in _rename_attempts(script, placeholder, placeholder):
        for tok in line.split():
            low = tok.lower()
            if low.startswith('jobname=') or low.startswith('name='):
                name = tok.split('=', 1)[1]
                parsed = parse_job_name(name)
                assert parsed is not None, (
                    f'{script}: bare-path name {name!r} is unattributable -- '
                    f'the monitor cannot file it under any field')
                assert parsed['stage'], (
                    f'{script}: bare-path name {name!r} names no stage')
                checked += 1
    assert checked, f'{script}: no bare-path name was emitted to check'


def test_m7_renames_itself_only_once():
    """It used to do it twice, unconditionally, and the second write won."""
    assert len(_rename_lines(_folded('submit_cataloging_m7.sbatch'))) == 1


# ---------------------------------------------------------------------------
# The names themselves have to be readable by the monitor.  These go through
# parse_job_name rather than asserting a string, so a shape that looks fine but
# does not resolve to a field cannot pass.
# ---------------------------------------------------------------------------

def _emitted_job_names(text, env):
    """Every `--job-name=` / `JobName=` value in a script, with env expanded."""
    import re as _re
    out = []
    for m in _re.finditer(r'(?:--job-name=|JobName=)"([^"]+)"', text):
        name = m.group(1)
        for k, v in env.items():
            name = name.replace('${' + k + '}', v).replace('$' + k, v)
        out.append(name)
    return out


CHAIN_ENV = {'TARGET': 'brick', 'PROPOSAL': '2221', 'FIELD': '001',
             'ph': 'm12'}


@pytest.mark.parametrize('script', [
    'submit_cataloging_chain.sh',
    'submit_cataloging_m7.sbatch',
    'submit_cataloging_perframe.sh',
])
def test_every_emitted_job_name_resolves_to_a_field_and_an_observation(script):
    """`cat_brick_m7` -- the name m7 used to give itself -- returns None from
    parse_job_name: _NAME_PF only accepts a `pf_` head, so it matches no shape
    and _resolve_head finds no registered field inside it.  It was invisible to
    the monitor, not merely ambiguous.  A name that carries the obsid parses as
    `full` and is the only kind that can say WHICH observation is running --
    brick and cloudc are both 2221 and gc2211 has five observations.
    """
    from jwst_gc_pipeline.monitoring.jobs import parse_job_name
    names = _emitted_job_names(_folded(script), CHAIN_ENV)
    assert names, f'{script}: no job names found'
    for name in names:
        parsed = parse_job_name(name)
        assert parsed is not None, (
            f'{script}: job name {name!r} is unattributable -- the monitor '
            f'cannot file it under any field')
        assert parsed['obsid'], (
            f'{script}: job name {name!r} parsed as {parsed["name_kind"]} with '
            f'no obsid; it cannot identify which observation is running')


def test_the_underscore_form_this_replaced_really_was_unreadable():
    """Pins the measurement the fix rests on, so nobody reintroduces it."""
    from jwst_gc_pipeline.monitoring.jobs import parse_job_name
    assert parse_job_name('cat_brick_m7') is None
    assert parse_job_name('cat_gc2211_m7') is None
    # and the dashed form it alternated with parsed, but only loosely
    loose = parse_job_name('brick-catalog-m7')
    assert loose['name_kind'] == 'loose' and loose['obsid'] is None


def test_the_quarantine_suffix_is_recognised_by_the_release_gate():
    """`SUFFIX` and `release_freshness.QUARANTINE_GLOBS` must move together.

    A release reads a quarantined product as REPUDIATED only if it can find the
    twin; if the suffix changes and the glob does not, every quarantined product
    silently reclassifies from `quarantined` to `missing` in every field's
    listing -- which is a report that says the opposite of what happened.
    """
    m = _load('rename_stale_mosaics')
    freshness = os.path.join(os.path.dirname(__file__), '..', 'scripts',
                             'release', 'release_freshness.py')
    globs = open(freshness).read()
    assert "{src}" + m.SUFFIX in globs, (
        f"rename_stale_mosaics writes '{m.SUFFIX}' but release_freshness's "
        f"QUARANTINE_GLOBS does not match it")


def test_the_product_token_is_what_finds_the_orphan(tmp_path):
    """The product token's real job under the family rule.

    Without it, `f405n-f444w` and `clear-f405n-merged` of one pointing are the
    same "family", the current one makes that family look live, and the 2023
    orphan is never selected -- a false NEGATIVE, which is how this rule fails
    now that the age guard covers the false-positive side.
    """
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = _band_dir(tmp_path, band='F405N')
    _age(pipe / 'jw02221-o002_t001_nircam_clear-f405n-merged_i2d.fits', 0)
    orphan = _age(pipe / 'jw02221-o002_t001_nircam_f405n-f444w_i2d.fits', 1100)
    plan = m.rename_stale_for_field('myfield', execute=True)
    assert [os.path.basename(p[0]) for p in plan] == [orphan.name], (
        'the orphan shares its pointing with a current product; without the '
        'product token in the key that current product masks it')
    assert not orphan.exists()


def test_one_pointings_currency_does_not_mask_anothers_staleness(tmp_path):
    """Same masking failure, on the pointing axis."""
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = _band_dir(tmp_path, band='F1130W')
    _age(pipe / 'jw03958-o001_t001_miri_f1130w_i2d.fits', 0)
    stale = _age(pipe / 'jw03958-o003_t001_miri_f1130w_i2d.fits', 1100)
    plan = m.rename_stale_for_field('myfield', execute=True)
    assert [os.path.basename(p[0]) for p in plan] == [stale.name], (
        'without the pointing in the key, o001 being current makes o003 look '
        'live and a genuinely stale pointing is never reported')


def test_the_age_guard_audit_reports_the_NEAREST_MISS_not_the_safest(tmp_path):
    """The margin is set by the live file closest to being quarantined.

    Two revisions of `MIN_ORPHAN_AGE_DAYS`' comment quoted it from the wrong end
    of the distribution -- the smallest age gap among the held-back files, which
    is the SAFEST member of that set, not the nearest miss -- and so reported the
    guard as 4.8x safer than it is.  Both times the number came from a
    hand-written scan rather than from the rule, which is why `audit_age_guard`
    exists and why this pins the sort order.
    """
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = _band_dir(tmp_path, band='F405N')
    # one instrument throughout, so this test measures the sort order and
    # nothing else -- the per-instrument reference has its own test below
    _age(pipe / 'jw02221-o002_t001_nircam_clear-f405n-merged_i2d.fits', 0)
    for name, days in (('jw02221-o002_t001_nircam_f405n-f444w_i2d.fits', 300),
                       ('jw02221-o002_t001_nircam_f405n-f322w2_i2d.fits', 100),
                       ('jw02221-o002_t001_nircam_f405n-f356w_i2d.fits', 30)):
        _age(pipe / name, days)
    held, caught = m.audit_age_guard(['myfield'])
    assert [round(a) for a, _f, _b in held] == [300, 100, 30]
    assert caught == []
    # the margin is 365/300, not 365/30
    assert m.MIN_ORPHAN_AGE_DAYS / held[0][0] < 1.3


def test_the_audit_measures_through_the_rules_own_references(tmp_path):
    """An audit that recomputes the references can report a margin the rule
    does not have -- which is how both retracted numbers were produced.  Rule 2
    and the audit must agree on which files the constant alone is holding."""
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = _band_dir(tmp_path, band='F405N')
    _age(pipe / 'jw02221-o002_t001_nircam_clear-f405n-merged_i2d.fits', 0)
    young = _age(pipe / 'jw02221-o002_t001_nircam_f405n-f444w_i2d.fits', 300)
    old = _age(pipe / 'jw02221-o002_t001_nircam_f444w-f466n_i2d.fits', 500)

    held, caught = m.audit_age_guard(['myfield'])
    assert [os.path.basename(b) for _a, _f, b in held] == [young.name]
    assert [os.path.basename(b) for _a, _f, b in caught] == [old.name]

    # and rule 2 takes exactly the file the audit says it takes
    m.rename_stale_for_field('myfield', execute=True)
    assert young.exists(), 'the audit called this live and the rule quarantined it'
    assert not old.exists() and (pipe / (old.name + m.SUFFIX)).exists()


def test_nircam_activity_does_not_age_out_a_miri_mosaic(tmp_path):
    """MIRI and NIRCam are reduced on independent campaigns.

    Clocking every product against the field's newest primary mosaic of ANY
    instrument makes a field's NIRCam activity age out its MIRI products: a
    NIRCam-only re-drizzle moves a clock no MIRI product has any part in.  On
    the archive that was not hypothetical -- 22 of the 47 live mosaics the age
    guard was holding back were MIRI judged against a NIRCam newest, and NINE of
    those were the only primary mosaic in their band directory.  Sole copies
    would have been quarantined by the passage of time alone.

    Here: MIRI last ran 380 days ago, NIRCam ran today, and a MIRI mosaic of a
    product family MIRI no longer writes is 420 days old.  Measured against the
    field's newest of any instrument it is 420 days behind and rule 2 takes it;
    measured against MIRI's own campaign it is 40 days behind and is live.
    """
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    nir = _band_dir(tmp_path, band='F405N')
    mir = _band_dir(tmp_path, band='F770W')
    _age(nir / 'jw02221-o002_t001_nircam_clear-f405n-merged_i2d.fits', 0)
    _age(mir / 'jw02221-o002_t001_miri_f770w_i2d.fits', 380)
    retired = _age(mir / 'jw02221-o002_t001_miri_f770w-sub256_i2d.fits', 420)

    assert m.rename_stale_for_field('myfield', execute=True) == []
    assert retired.exists(), (
        'a MIRI mosaic was aged out by NIRCam reduction it has no part in')

    held, caught = m.audit_age_guard(['myfield'])
    assert caught == []
    assert [round(a) for a, _f, _b in held] == [40], (
        'the age is measured against MIRI, so it is 40 days and not 420')


def test_the_instrument_token_comes_from_the_primary_mosaic_pattern():
    """`instrument_of` must read the SAME match the rule selects on.

    Rule 2's reference is per instrument.  If `instrument_of` used a second,
    parallel regex and the two drifted, every name the second one did not
    recognise would return `'?'` and share one bucket -- so a MIRI mosaic and a
    NIRCam mosaic would once again be clocked against each other, silently, which
    is exactly what the per-instrument reference exists to prevent.
    """
    m = _load('rename_stale_mosaics')
    assert 'instrument' in m.PRIMARY_MOSAIC_RE.groupindex, (
        'instrument_of reads this group; without it the coupling is gone')
    for name, want in (
            ('jw02221-o002_t001_nircam_clear-f405n-merged_i2d.fits', 'nircam'),
            ('jw03958-o003_t001_miri_f770w_i2d.fits', 'miri'),
    ):
        assert m.instrument_of(name) == want
        assert m.PRIMARY_MOSAIC_RE.match(name).group('instrument') == want

    # anything the rule does not select on is not given an instrument either,
    # rather than being lumped in with a real one
    for name in ('jw02221002001_02201_00001_nrcalong_align_outlier_i2d.fits',
                 'jw02221-o002_t001_nircam_clear-f405n-nrca_data_i2d.fits',
                 'not_a_mosaic.fits'):
        assert m.instrument_of(name) == '?'
        assert m.PRIMARY_MOSAIC_RE.match(name) is None


def test_a_wholly_stale_instrument_is_a_known_blind_spot(tmp_path):
    """The price of the per-instrument reference, pinned as BEHAVIOUR.

    If a field's entire MIRI set is superseded, every MIRI mosaic is its own
    instrument's newest, both of rule 2's clauses degrade together, and nothing
    is selected.  The set is invisible, not merely held back -- so the audit
    reports it neither as held nor as caught.

    Pinned by running the rule, not by grepping the docstring: an earlier
    version of this test asserted only that the module said the words, which
    would have kept a FALSE limitation (the pointing one, below) locked in.

    NOTE the two members share an age.  That is the degenerate case, and it is
    the one this test covers; when the stale set SPANS more than
    MIN_ORPHAN_AGE_DAYS its older members are selected after all, which
    `test_an_older_member_of_a_wholly_stale_instrument_is_still_selected`
    pins.  An earlier version of this test used the shared age and the module
    then claimed total invisibility "at any --campaign-days", which is false.
    """
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    nir = _band_dir(tmp_path, band='F405N')
    mir = _band_dir(tmp_path, band='F770W')
    _age(nir / 'jw02221-o002_t001_nircam_clear-f405n-merged_i2d.fits', 0)
    stale = [_age(mir / 'jw02221-o002_t001_miri_f770w_i2d.fits', 1100),
             _age(mir / 'jw02221-o002_t001_miri_f770w-sub256_i2d.fits', 1100)]

    assert m.rename_stale_for_field('myfield', execute=True) == []
    for p in stale:
        assert p.exists(), 'the whole-instrument blind spot has closed'
    held, caught = m.audit_age_guard(['myfield'])
    assert held == [] and caught == [], 'invisible, not held back'

    assert 'known limitation' in m.__doc__.lower()


def test_a_family_retired_pointing_under_the_age_guard_is_still_kept(tmp_path):
    """Family-retired is not sufficient -- the 365-day guard still applies.

    sickle `jw03958-o003` is the live case, and it is the file the guard's own
    margin is measured from: all three of its MIRI primaries are family-retired
    while o001 and o002 are current, and none is selected, because they are 62
    days old rather than 365.  An earlier version of the module docstring said a
    stale pointing "IS caught whenever any other pointing of that instrument is
    current", which drops that condition.
    """
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = _band_dir(tmp_path, band='F770W')
    _age(pipe / 'jw03958-o001_t001_miri_f770w_i2d.fits', 0)
    young = [_age(pipe / 'jw03958-o003_t001_miri_f770w-sub128_i2d.fits', 62),
             _age(pipe / 'jw03958-o003_t001_miri_f770w-sub64_i2d.fits', 62)]
    assert m.rename_stale_for_field('myfield', execute=True) == []
    for p in young:
        assert p.exists(), 'held back by the age guard, not selected'
    held, caught = m.audit_age_guard(['myfield'])
    assert caught == [] and len(held) == 2


def test_an_older_member_of_a_wholly_stale_instrument_is_still_selected(tmp_path):
    """The whole-instrument blind spot is partial, not total.

    If the stale instrument's own products span more than MIN_ORPHAN_AGE_DAYS,
    the newest of them becomes the yardstick and the older ones ARE selected.
    Only the newest is protected.  The module claimed "nothing is selected at
    any --campaign-days"; that held only for a fixture whose members shared one
    age.
    """
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    nir = _band_dir(tmp_path, band='F405N')
    mir = _band_dir(tmp_path, band='F770W')
    _age(nir / 'jw02221-o002_t001_nircam_clear-f405n-merged_i2d.fits', 0)
    newest_miri = _age(mir / 'jw02221-o002_t001_miri_f770w_i2d.fits', 1100)
    older_miri = _age(mir / 'jw02221-o002_t001_miri_f770w-sub256_i2d.fits', 1600)

    plan = m.rename_stale_for_field('myfield', execute=True)
    assert [os.path.basename(p[0]) for p in plan] == [older_miri.name]
    assert newest_miri.exists(), 'the instrument-newest is what is protected'
    assert not older_miri.exists()


def test_a_wholly_stale_pointing_is_caught_when_another_pointing_is_current(tmp_path):
    """The pointing is NOT a second blind spot, and the module used to say it was.

    The pointing appears only in the family key -- it is not part of either
    reference -- so a stale pointing is judged against the instrument's newest
    like everything else.  brick is the live case: every NIRCam primary mosaic
    of `jw02221-o002` is a 2023 product and all three are selected today.

    The only invisible pointing is one that is its instrument's ONLY pointing,
    which is the instrument case, not a separate one.
    """
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = _band_dir(tmp_path, band='F405N')
    _age(pipe / 'jw02221-o002_t001_nircam_clear-f405n-merged_i2d.fits', 0)
    stale = [_age(pipe / 'jw02221-o009_t001_nircam_f405n-f444w_i2d.fits', 1100),
             _age(pipe / 'jw02221-o009_t001_nircam_clear-f410m_i2d.fits', 1100)]

    plan = m.rename_stale_for_field('myfield', execute=True)
    assert sorted(os.path.basename(p[0]) for p in plan) == \
        sorted(p.name for p in stale), 'a wholly stale pointing must be caught'
    for p in stale:
        assert not p.exists() and os.path.exists(str(p) + m.SUFFIX)


# --------------------------------------------------------------------------
# --only: execute the subset that has been measured (#339)
# --------------------------------------------------------------------------
def _two_orphans(tmp_path, m):
    """One band holding a current mosaic and TWO canonically-named orphans.

    Miniature of the live case: rule 2 selects both on generation, but only one
    of them has a measured arcsecond-scale offset, so only one of them clears
    the bar the quarantine instruction was given under.
    """
    m.BASE = str(tmp_path)
    pipe = _band_dir(tmp_path, band='F405N')
    _age(pipe / 'jw02221-o002_t001_nircam_clear-f405n-merged_i2d.fits', 0)
    measured = _age(pipe / 'jw02221-o002_t001_nircam_f405n-f444w_i2d.fits', 1100)
    unmeasured = _age(pipe / 'jw02221-o001_t001_nircam_f405n-f444w_i2d.fits', 1100)
    return pipe, measured, unmeasured


def test_only_quarantines_the_named_file_and_leaves_the_rest(tmp_path):
    """The gap this closes: --execute was all of a field's selections or none,
    so 'measure before quarantining' could not be carried out with this tool."""
    m = _load('rename_stale_mosaics')
    _pipe, measured, unmeasured = _two_orphans(tmp_path, m)
    plan = m.rename_stale_for_field('myfield', execute=True,
                                    only=[measured.name])
    assert [os.path.basename(p[0]) for p in plan] == [measured.name]
    assert not measured.exists()
    assert os.path.exists(str(measured) + m.SUFFIX)
    assert unmeasured.exists(), 'a file the filter withheld was renamed anyway'
    assert not os.path.exists(str(unmeasured) + m.SUFFIX)


def test_without_only_both_orphans_are_still_taken(tmp_path):
    """--only must be a restriction on the existing selection, not a new rule:
    absent the flag the behaviour is exactly what it was."""
    m = _load('rename_stale_mosaics')
    _pipe, measured, unmeasured = _two_orphans(tmp_path, m)
    plan = m.rename_stale_for_field('myfield', execute=True)
    assert sorted(os.path.basename(p[0]) for p in plan) == sorted(
        [measured.name, unmeasured.name])
    assert not measured.exists() and not unmeasured.exists()


def test_a_full_path_is_accepted_as_well_as_a_basename(tmp_path):
    """The dry run prints basenames but an operator may paste a path."""
    m = _load('rename_stale_mosaics')
    _pipe, measured, unmeasured = _two_orphans(tmp_path, m)
    plan = m.rename_stale_for_field('myfield', execute=True,
                                    only=[str(measured)])
    assert [os.path.basename(p[0]) for p in plan] == [measured.name]
    assert unmeasured.exists()


def test_an_only_name_that_matches_nothing_REFUSES(tmp_path, monkeypatch,
                                                   capsys):
    """A typo, or a file the rules stopped selecting, would otherwise rename
    nothing and exit 0 -- indistinguishable from a run that did its job."""
    m = _load('rename_stale_mosaics')
    _pipe, measured, unmeasured = _two_orphans(tmp_path, m)
    monkeypatch.setattr('sys.argv', ['rename_stale_mosaics.py',
                                     '--field', 'myfield', '--execute',
                                     '--only', 'jw02221-o009_t001_nircam_'
                                               'f405n-f444w_i2d.fits'])
    with pytest.raises(m.UnmatchedOnlyName) as excinfo:
        m.main()
    assert 'jw02221-o009' in str(excinfo.value)
    assert measured.name in str(excinfo.value)       # names what IS selectable
    # and nothing was renamed on the way to refusing
    assert measured.exists() and unmeasured.exists()


def test_the_refusal_happens_before_any_rename(tmp_path, monkeypatch):
    """One good name and one bad one: the good file must NOT be quarantined
    while the run reports a failure."""
    m = _load('rename_stale_mosaics')
    _pipe, measured, unmeasured = _two_orphans(tmp_path, m)
    monkeypatch.setattr('sys.argv', ['rename_stale_mosaics.py',
                                     '--field', 'myfield', '--execute',
                                     '--only', measured.name,
                                     '--only', 'not_a_real_product_i2d.fits'])
    with pytest.raises(m.UnmatchedOnlyName):
        m.main()
    assert measured.exists(), 'refused, but renamed the matching file anyway'
    assert unmeasured.exists()


def test_only_across_two_fields_does_not_refuse_on_the_other_fields_name(
        tmp_path, monkeypatch):
    """A name belongs to exactly one field, so a per-field refusal would fire
    on every other field named in the same command."""
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    a = _band_dir(tmp_path, field='fieldA', band='F405N')
    b = _band_dir(tmp_path, field='fieldB', band='F410M')
    _age(a / 'jw02221-o002_t001_nircam_clear-f405n-merged_i2d.fits', 0)
    _age(b / 'jw02221-o002_t001_nircam_clear-f410m-merged_i2d.fits', 0)
    oa = _age(a / 'jw02221-o002_t001_nircam_f405n-f444w_i2d.fits', 1100)
    ob = _age(b / 'jw02221-o002_t001_nircam_f444w-f410m_i2d.fits', 1100)
    monkeypatch.setattr('sys.argv', ['rename_stale_mosaics.py',
                                     '--field', 'fieldA', '--field', 'fieldB',
                                     '--execute',
                                     '--only', oa.name, '--only', ob.name])
    m.main()
    assert os.path.exists(str(oa) + m.SUFFIX)
    assert os.path.exists(str(ob) + m.SUFFIX)


def test_the_withheld_files_are_named_in_the_output(tmp_path, capsys):
    """A partial run must say so: the log is the record that the other
    selections were a deliberate choice rather than an oversight."""
    m = _load('rename_stale_mosaics')
    _pipe, measured, unmeasured = _two_orphans(tmp_path, m)
    m.rename_stale_for_field('myfield', execute=True, only=[measured.name])
    out = capsys.readouterr().out
    assert 'withheld' in out
    assert unmeasured.name in out




# ---------------------------------------------------------------------------
# The reproject precondition (#724)
#
# ``*-merged-reproject_i2d.fits`` sits beside a plain ``*-merged_i2d.fits`` on a
# frame up to ~2" away, and neither the name nor either mtime says which is
# current: the reproject is whatever ``align_to_catalogs`` last wrote, computed
# from a reference catalog and independent of the frames.  Where the frames
# carry a baked correction the plain mosaic is current; where they do not, the
# reproject holds the only tie the band has.  So the family is decided by state
# on disk: registration in ``ALIGNMENT_CONFIG``, a non-zero baked
# ``RAOFFSET``/``DEOFFSET`` on the mosaic's OWN pointing's frames, and -- the
# half that looks at the file taking over -- a plain sibling that EXISTS and
# POSTDATES those frames.
#
# THE FIXTURES ARE BUILT SO THAT DELETING THE GATE CHANGES EVERY ANSWER, and
# the two halves need OPPOSITE fixtures to manage it:
#
#   KEEP cases (the safety half -- the half whose failure removes a field's
#   astrometry) use ``rule1_clock=True``: the band gets a
#   ``*-merged_data_i2d.fits`` and the reproject is aged past rule 1's campaign
#   floor, so WITHOUT the gate rule 1 renames the file.  Three earlier versions
#   of these tests did not do this -- their bands had no ``data_i2d`` (rule 1
#   printed ``SKIP no-data_i2d``) and their fixtures were ten days old (rule 2's
#   365-day guard declined), so ``plan == []`` held for reasons that had nothing
#   to do with the precondition, and they passed with the feature deleted.
#
#   SELECT cases use ``rule1_clock=False``: no ``data_i2d`` and a young file, so
#   neither rule's own clock can reach it and the gate is the ONLY thing that
#   can select it.  Giving these a ``data_i2d`` too would have made rule 1
#   select them under the mutation and stopped THEM pinning anything.
#
# Mutation, `if False and REPROJECT_RE.search(base)` plus dropping the
# ``FOREIGN_QUARANTINE_MARKERS`` branches: 12 red, and the two that stay green
# are the control (which asserts what the mutation does) and the
# release-freshness measurement (which does not import this script's gate).
# ---------------------------------------------------------------------------
def _frame(pipe, name, raoffset=None, deoffset=None, days=30):
    """A 2-HDU FITS frame whose SCI header carries the baked shift, or none.

    Backdated, because the gate compares the plain mosaic's date against the
    NEWEST frame's: a fixture that writes both "now" decides that comparison on
    the order the fixture happened to write them in.
    """
    from astropy.io import fits
    sci = fits.ImageHDU(name='SCI')
    if raoffset is not None:
        sci.header['RAOFFSET'] = raoffset
    if deoffset is not None:
        sci.header['DEOFFSET'] = deoffset
    path = pipe / name
    fits.HDUList([fits.PrimaryHDU(), sci]).writeto(str(path))
    now = time.time()
    os.utime(str(path), (now - days * 86400,) * 2)
    return path


def _reproject_case(tmp_path, pointing, raoffset, deoffset, band='F150W',
                    rule1_clock=False, reproject_age=None, frame_age=30,
                    plain_age=0, plain=True, suffix='_i2d.fits'):
    """One band directory holding a plain mosaic, a reproject and frames.

    ``rule1_clock=True`` adds the band's ``*-merged_data_i2d.fits`` and ages the
    reproject past rule 1's campaign floor, so that rule 1 WOULD select the file
    and the gate is the only thing that can hold it back -- the shape a KEEP
    assertion needs to mean anything.  ``False`` leaves the band without that
    reference and the file young, so NEITHER rule's clock can reach it and the
    gate is the only thing that can select it -- the shape a SELECT assertion
    needs.
    """
    pipe = _band_dir(tmp_path, band=band)
    prop, obs = pointing[2:].split('-o')
    lo = band.lower()
    if reproject_age is None:
        reproject_age = 60 if rule1_clock else 10
    if rule1_clock:
        _age(pipe / f'{pointing}_t001_nircam_clear-{lo}-merged_data_i2d.fits', 0)
    if plain:
        _age(pipe / f'{pointing}_t001_nircam_clear-{lo}-merged_i2d.fits',
             plain_age)
    reproject = _age(
        pipe / f'{pointing}_t001_nircam_clear-{lo}-merged-reproject{suffix}',
        reproject_age)
    for exposure in (1, 2):
        _frame(pipe, f'jw{prop}{obs}001_02101_0000{exposure}_nrca1_align.fits',
               raoffset=raoffset, deoffset=deoffset, days=frame_age)
    return pipe, reproject


def test_the_fixture_would_be_renamed_without_the_gate(tmp_path):
    """The control for every KEEP test below: with the gate short-circuited,
    ``_reproject_case`` IS selected.  Without this, a KEEP assertion cannot tell
    "the precondition held the file" from "the rules never reached it"."""
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    _pipe, reproject = _reproject_case(tmp_path, 'jw09999-o001', -2.5, -0.71,
                                       rule1_clock=True)
    m.REPROJECT_RE = re.compile(r'(?!)')          # matches nothing
    plan = m.rename_stale_for_field('myfield', execute=False)
    assert [os.path.basename(p[0]) for p in plan] == [reproject.name]


def test_a_reproject_of_an_unregistered_pointing_is_kept(tmp_path):
    """No ALIGNMENT_CONFIG entry means untied frames, so the reproject is the
    only tie that band has -- renaming it would remove the field's astrometry."""
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    # 9999 is deliberately not a real program, so `resolve` returns None
    _pipe, reproject = _reproject_case(tmp_path, 'jw09999-o001', -2.5, -0.71,
                                       rule1_clock=True)
    assert m.rename_stale_for_field('myfield', execute=True) == []
    assert reproject.exists()


def test_a_registered_pointing_whose_frames_carry_no_shift_keeps_it(tmp_path):
    """Registration alone is not enough.  ngc6334 was registered on 2026-09-01
    and its frames still read RAOFFSET=0.0 because no reduction has run since,
    so its plain mosaics are untied and its reprojects are still the only tie."""
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    _pipe, reproject = _reproject_case(tmp_path, 'jw06778-o001', 0.0, 0.0,
                                       band='F187N', rule1_clock=True)
    assert m.rename_stale_for_field('myfield', execute=True) == []
    assert reproject.exists()


def test_a_reproject_with_no_plain_sibling_is_kept(tmp_path, capsys):
    """Registered, frames tied -- and the band holds NO plain mosaic, so there
    is nothing for the tie to have been inherited by.

    This state is on disk: ``ngc6334/F115W/pipeline`` holds
    ``jw07213-o001_..._merged-reproject_i2d_im0_badastrom.fits`` and no plain
    sibling.  The gate used to conclude "the plain mosaic is the current tie"
    from the frames alone, having never looked for a plain mosaic, which leaves
    the band with no mosaic at all.
    """
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    _pipe, reproject = _reproject_case(tmp_path, 'jw01334-o001', -2.54, -0.71,
                                       rule1_clock=True, plain=False)
    assert m.rename_stale_for_field('myfield', execute=True) == []
    assert reproject.exists()
    # The REASON is asserted, not just the outcome: with only the date
    # comparison, an absent sibling is kept as "cannot be dated against them",
    # which tells an operator the wrong thing about a band that has no mosaic.
    assert 'DOES NOT EXIST' in capsys.readouterr().out


def test_a_plain_sibling_older_than_the_correction_keeps_the_reproject(tmp_path):
    """``fix_alignment`` bakes the shift into the FRAMES; only the level-3
    re-drizzle after it puts that shift into the mosaic.  A plain mosaic
    drizzled before the frames were corrected has inherited nothing, so the
    reproject is still the band's only tie.

    This is the shape ngc6334 (13 reprojects) and wd1 (7) are one re-reduction
    away from, and the run that would take their ties is the one that happens if
    ``fix_alignment`` bakes the shift and the re-drizzle has not run or failed.
    """
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    _pipe, reproject = _reproject_case(tmp_path, 'jw01334-o001', -2.54, -0.71,
                                       rule1_clock=True, plain_age=400,
                                       frame_age=30)
    assert m.rename_stale_for_field('myfield', execute=True) == []
    assert reproject.exists()


def test_a_registered_pointing_with_a_baked_shift_loses_its_reproject(tmp_path):
    """The m92 case: frames carry the tie and the plain mosaic was drizzled
    after them, so the plain mosaic is the current product."""
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    _pipe, reproject = _reproject_case(tmp_path, 'jw01334-o001', -2.54, -0.71)
    plan = m.rename_stale_for_field('myfield', execute=True)
    assert [os.path.basename(p[0]) for p in plan] == [reproject.name]
    assert not reproject.exists()
    assert os.path.exists(str(reproject) + m.SUFFIX)


def test_the_rename_record_carries_the_frames_date_not_the_files_own(tmp_path):
    """The log line and the ``.why.txt`` name the date the file was judged
    against.  That slot used to hold the file's OWN mtime, so all three m92
    sidecars read "its generation is 2026-07-01; the frames it is judged against
    is 2026-07-01" -- the same date twice, and never the frames' 2026-09-02."""
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    _pipe, reproject = _reproject_case(tmp_path, 'jw01334-o001', -2.54, -0.71,
                                       reproject_age=10, frame_age=30)
    m.rename_stale_for_field('myfield', execute=True)
    sidecar = str(reproject) + m.SUFFIX + '.why.txt'
    text = open(sidecar).read()
    assert m.fmt(time.time() - 30 * 86400) in text        # the frames'
    own = m.fmt(time.time() - 10 * 86400)
    assert text.count(own) == 1                           # its own, once


def test_a_pure_declination_shift_counts_as_a_correction(tmp_path):
    """#724 says "non-zero RAOFFSET"; taken literally that reads a pointing
    corrected only in Dec as uncorrected and keeps a superseded mosaic."""
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    _pipe, reproject = _reproject_case(tmp_path, 'jw01334-o001', 0.0, -0.71)
    assert len(m.rename_stale_for_field('myfield', execute=True)) == 1
    assert not reproject.exists()


def test_the_precondition_also_governs_rule_2(tmp_path):
    """w51's F150W reproject is in BOTH rules' sets.  With the gate on rule 1
    alone, rule 1 would keep it (untied frames) while rule 2 quarantined the
    same file on age -- so the family's policy sits outside both rules."""
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    # 500 days behind the band's newest primary mosaic, and its own product
    # family is retired -- exactly what rule 2 selects on
    _pipe, reproject = _reproject_case(tmp_path, 'jw06151-o001', 0.0, 0.0,
                                       reproject_age=500)
    assert m.rename_stale_for_field('myfield', execute=True) == []
    assert reproject.exists()


def test_one_answer_and_one_line_for_a_file_in_both_rules(tmp_path, capsys):
    """A kept file used to be evaluated and printed by each rule that named it,
    so w51 reported "32 superseded, 2 current kept" for one file."""
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    _pipe, reproject = _reproject_case(tmp_path, 'jw06151-o001', 0.0, 0.0,
                                       rule1_clock=True, reproject_age=500)
    m.rename_stale_for_field('myfield', execute=False)
    out = capsys.readouterr().out
    assert out.count(f'KEEP [myfield] {reproject.name}') == 1
    assert '1 current kept' in out


def test_the_frames_read_are_the_mosaics_own_pointing(tmp_path):
    """One band directory routinely holds several independently-aligned
    pointings (ngc6334 keeps 6778 and 7213 in one F200W/pipeline; m4 keeps
    o002 and o003 in one F150W2/pipeline).  Reading a neighbour's frames would
    answer for the wrong data, in the direction that takes a live tie."""
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = _band_dir(tmp_path, band='F200W')
    for pointing, raoffset in (('jw06778-o001', -2.54), ('jw07213-o001', 0.0)):
        prop, obs = pointing[2:].split('-o')
        _age(pipe / f'{pointing}_t001_nircam_clear-f200w-merged_i2d.fits', 0)
        _age(pipe / (f'{pointing}_t001_nircam_clear-f200w-merged-reproject'
                     f'_i2d.fits'), 10)
        _frame(pipe, f'jw{prop}{obs}001_02105_00001_nrca1_align.fits',
               raoffset=raoffset, deoffset=0.0, days=30)
    plan = m.rename_stale_for_field('myfield', execute=True)
    assert [os.path.basename(p[0]) for p in plan] == [
        'jw06778-o001_t001_nircam_clear-f200w-merged-reproject_i2d.fits']
    assert (pipe / ('jw07213-o001_t001_nircam_clear-f200w-merged-reproject'
                    '_i2d.fits')).exists()


def test_the_newest_frame_lineage_is_read_not_the_first_suffix(tmp_path):
    """Four fields carry TWO reduction lineages in one directory (cloudc,
    cloudef, sickle, brick).  ``_align`` is searched before ``_destreak``, and
    on those fields the ``_align`` copy is a 2024 leftover with no RAOFFSET at
    all while the live ``_destreak`` carries the shift:

        sgrb2/F182M  jw05365001001_11101_00001_nrca1
            _align    2024-09-22  RAOFFSET absent
            _destreak 2026-08-24  RAOFFSET=-0.01154 DEOFFSET=-0.12106

    First-suffix-wins reports such a pointing as uncorrected, which keeps the
    file (the safe direction) but tells the operator something false about the
    field, and misdates the plain-sibling comparison.
    """
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    _pipe, reproject = _reproject_case(tmp_path, 'jw01334-o001',
                                       raoffset=None, deoffset=None,
                                       frame_age=700)               # _align
    _frame(_pipe, 'jw01334001001_02101_00001_nrca1_destreak.fits',
           raoffset=-2.54, deoffset=-0.71, days=30)
    plan = m.rename_stale_for_field('myfield', execute=False)
    assert [os.path.basename(p[0]) for p in plan] == [reproject.name]


def test_a_checkpoint_tagged_mosaic_is_not_tagged_again(tmp_path, capsys):
    """The m2 checkpoint stale-tags a mosaic to ``..._im0_badastrom.fits`` and
    writes the reason to a ``.why.json`` beside THAT name.  Renaming it again
    would double-tag it and orphan the first tag's explanation -- the state 32
    mosaics on six fields are in.  It is NAMED in the output, since withholding
    32 files silently is what made the size of that carve-out invisible."""
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    _pipe, tagged = _reproject_case(tmp_path, 'jw01979-o002', 0.11, -0.18,
                                    band='F150W2', rule1_clock=True,
                                    suffix='_i2d_im0_badastrom.fits')
    assert m.rename_stale_for_field('myfield', execute=True) == []
    assert tagged.exists()
    assert not os.path.exists(str(tagged) + m.SUFFIX)
    assert 'already tagged by another pass' in capsys.readouterr().out


def test_renaming_a_checkpoint_tag_would_hide_it_from_the_release_gate(tmp_path):
    """The measurement behind the carve-out, rather than an argument for it.

    ``release_freshness.QUARANTINE_GLOBS`` matches ``{stem}_im0_badastrom*.fits``
    against the LIVE product's stem, so the m2 checkpoint's tag is already
    recognised as a quarantine twin of that product.  Appending ``.bad`` to it
    matches neither that pattern (it no longer ends in ``.fits``) nor
    ``{src}.bad`` (whose ``src`` is the live name, not the tagged one) -- so the
    rename would REMOVE the recognition it is supposed to add.
    """
    rf = _load_release('release_freshness')
    pipe = _band_dir(tmp_path, band='F150W2')
    live = pipe / 'jw01979-o002_t001_nircam_clear-f150w2-merged-reproject_i2d.fits'
    tagged = pipe / (live.name.replace('_i2d.fits', '_i2d_im0_badastrom.fits'))
    tagged.write_bytes(b'x')
    assert rf.has_quarantine_twin(str(live))
    os.rename(str(tagged), str(tagged) + '.bad')
    assert not rf.has_quarantine_twin(str(live))
