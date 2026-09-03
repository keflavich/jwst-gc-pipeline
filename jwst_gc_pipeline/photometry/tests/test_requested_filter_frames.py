"""A requested filter that resolves to zero frames must stop the run (#592).

wd1's m12 pass was submitted for eleven filters.  F150W's 96 frames sit on disk
under the ``o001_crf`` lineage while the run globbed ``destreak_o001_crf``, so
its candidate list was empty before sharding; all 32 fan-out shards printed
``0 of 0 frames ... nothing to fit`` and exited 0, and only the finalize job --
after roughly ten hours of array time -- raised.  The pass was a
correction-floor measurement, so a floor computed from ten of eleven filters is
what it produced.

Both sides of the distinction are pinned here, because a guard that raises on
every empty filter is unusable: a caller sweeping one filter list across fields
legitimately names a band a given observation never took.
"""
import pytest

from jwst_gc_pipeline.photometry.requested_filters import (
    ALLOW_ABSENT_ENV,
    RequestedFilterHasNoFramesError,
    assert_requested_filters_have_frames,
    classify_requested_filter,
    frame_lineages_on_disk,
    requested_lineages,
)

WD1 = dict(target='wd1', proposal_id='1905', field='001',
           basepath='/orange/adamginsburg/jwst/wd1')


def _assert(counts, *, lineages=None, declared=None, suffix='destreak_o001_crf',
            **kw):
    kwargs = dict(WD1)
    kwargs.update(kw)
    return assert_requested_filters_have_frames(
        counts,
        each_suffix_for=lambda f: suffix,
        declared=declared if declared is not None else set(),
        lineages_for=(lambda f: dict((lineages or {}).get(f, {}))),
        **kwargs)


# --- the failure side ------------------------------------------------------

def test_wrong_lineage_raises_and_names_the_override():
    """wd1 F150W verbatim: frames on disk, under another suffix."""
    with pytest.raises(RequestedFilterHasNoFramesError) as excinfo:
        _assert({'F115W': 96, 'F150W': 0},
                lineages={'F150W': {'o001_crf': 96}},
                declared={'F115W', 'F150W'})
    msg = str(excinfo.value)
    assert 'F150W' in msg and '96' in msg
    assert 'destreak_o001_crf' in msg, 'the requested suffix must be named'
    assert '--each-suffix-overrides=F150W:o001_crf' in msg, (
        'the message must name the lineage that would have worked')
    assert 'F115W' not in msg, 'a filter with frames is not part of the failure'


def test_a_declared_filter_with_no_frames_at_all_raises():
    """The treasury shape: fields.yaml declares F480M for 10678, and the tile's
    F480M has not been reduced yet.  Nothing is on disk under any lineage, so
    the lineage check cannot see it; the registry is what makes it an error."""
    with pytest.raises(RequestedFilterHasNoFramesError) as excinfo:
        _assert({'F212N': 32, 'F480M': 0},
                lineages={},
                declared={'F212N', 'F480M', 'F770W'},
                target='gc-treasury', proposal_id='10678', field='088',
                suffix='o088_crf')
    msg = str(excinfo.value)
    assert 'F480M' in msg and 'fields.yaml declares it' in msg


def test_the_right_suffix_with_no_frames_in_scope_says_so():
    """A message that blamed the lineage here would send an operator to
    EACH_SUFFIX_OVERRIDES when the frames sit outside the run's --modules or
    visit range; it is still a failure, with its own wording."""
    with pytest.raises(RequestedFilterHasNoFramesError) as excinfo:
        _assert({'F150W': 0},
                lineages={'F150W': {'destreak_o001_crf': 96}},
                declared={'F150W'})
    msg = str(excinfo.value)
    assert '--modules' in msg and 'each-suffix-overrides' not in msg


def test_every_failing_filter_is_named_in_one_message():
    """One resubmission fixes all of them, rather than one run per filter."""
    with pytest.raises(RequestedFilterHasNoFramesError) as excinfo:
        _assert({'F150W': 0, 'F164N': 0, 'F115W': 96},
                lineages={'F150W': {'o001_crf': 96}},
                declared={'F115W', 'F150W', 'F164N'})
    msg = str(excinfo.value)
    assert 'F150W' in msg and 'F164N' in msg
    assert msg.count('resolved to 0 candidate frames') == 2


# --- the legitimate side ---------------------------------------------------

def test_a_band_the_observation_never_took_is_skipped_not_raised():
    """Not declared for this observation and nothing on disk under any lineage.
    A guard that raised here would refuse every run of a shared filter list."""
    skipped = _assert({'F212N': 48, 'F480M': 0},
                      lineages={},
                      declared={'F212N', 'F405N'})
    assert skipped == ['F480M']


def test_frames_present_never_raise():
    assert _assert({'F115W': 96, 'F150W': 96},
                   declared={'F115W', 'F150W'}) == []


def test_a_module_or_visit_empty_is_not_a_failure():
    """The caller sums over modules and visits before asking: `allow_empty` is
    for an absent visit, and a filter need not cover every module."""
    counts = {'F150W': 0 + 0 + 96}   # nrca empty, visit 2 empty, the rest full
    assert _assert(counts, declared={'F150W'}) == []


# --- the classifier's own verdicts ----------------------------------------

@pytest.mark.parametrize('n,lineages,declared,expect', [
    (96, {'o001_crf': 96}, {'F150W'}, 'ok'),
    (0, {'o001_crf': 96}, {'F150W'}, 'wrong-lineage'),
    (0, {}, {'F150W'}, 'declared-but-absent'),
    (0, {}, set(), 'not-observed'),
    # On disk under a lineage the registry does not know about: the disk is the
    # evidence, so an unregistered observation still fails.
    (0, {'o001_crf': 96}, set(), 'wrong-lineage'),
    # The requested lineage is the only one on disk, so the suffix is right and
    # the run's own modules / visit range are what missed the frames.
    (0, {'destreak_o001_crf': 96}, {'F150W'}, 'outside-this-run'),
])
def test_verdicts(n, lineages, declared, expect):
    verdict, _ = classify_requested_filter(
        'F150W', n, each_suffix='destreak_o001_crf', declared=declared,
        lineages=lineages, **WD1)
    assert verdict == expect


# --- which lineage the run actually asked for -----------------------------
#
# Both of these were wrong-lineage before: the classifier compared the raw
# `each_suffix` string against the keys on disk, so any key that was not that
# exact string counted as "somebody else's lineage".

def test_a_field_carrying_both_lineages_is_not_told_to_switch():
    """brick/2221 obs 001 F410M verbatim: {'o001_crf': 48,
    'destreak_o001_crf': 48}.  48 frames sit under the REQUESTED suffix, so the
    suffix is right and the run's modules/visits are what missed them.  The
    earlier order returned wrong-lineage and printed
    ``--each-suffix-overrides=F410M:o001_crf`` -- advice to read the other
    lineage while the requested one is on disk, which is the ~106 mas
    two-lineage mix, on exactly the four fields (brick, cloudc, cloudef,
    sickle) that carry both."""
    verdict, msg = classify_requested_filter(
        'F410M', 0, target='brick', proposal_id='2221', field='001',
        basepath='/orange/adamginsburg/jwst/brick',
        each_suffix='destreak_o001_crf', declared={'F410M'},
        lineages={'o001_crf': 48, 'destreak_o001_crf': 48})
    assert verdict == 'outside-this-run'
    assert 'each-suffix-overrides' not in msg, (
        'never recommend switching lineage on a field that has both')
    assert '--modules' in msg
    assert 'o001_crf (n=48)' in msg and 'MIX lineages' in msg, (
        'the other lineage is still named, as context rather than as the fix')


def test_the_joint_field_lineages_are_both_requested():
    """``get_filenames`` rewrites the obs token of ``each_suffix`` per subfield,
    so a joint field asks for one lineage per observation."""
    assert requested_lineages('o002_crf', '002-998') == {'o002_crf',
                                                         'o998_crf'}
    assert requested_lineages('destreak_o002_crf', '002-998') == {
        'destreak_o002_crf', 'destreak_o998_crf'}
    assert requested_lineages('destreak_o001_crf', '001') == {
        'destreak_o001_crf'}


def test_a_joint_field_is_not_wrong_lineage_on_every_call():
    """sickle 5365 MIRI ``002-998`` (and 3958 ``001-002``): both observations'
    frames ARE the requested lineage.  The raw-string comparison put the second
    one in `other` and diagnosed wrong-lineage every single time, recommending
    an override to a lineage the run was already globbing."""
    verdict, msg = classify_requested_filter(
        'F200W', 0, target='sickle', proposal_id='5365', field='002-998',
        basepath='/x', each_suffix='o002_crf', declared={'F200W'},
        lineages={'o002_crf': 10, 'o998_crf': 10})
    assert verdict == 'outside-this-run'
    assert 'each-suffix-overrides' not in msg


def test_a_joint_field_with_frames_in_only_one_observation():
    """sickle 3958 MIRI ``001-002`` with F770W frames written for obs 002 only.
    The run globs BOTH ``o001_crf`` and ``o002_crf``, so ``o002_crf`` is a
    requested lineage and the diagnosis is the run's modules/visits.  Comparing
    the raw ``each_suffix`` string instead made this wrong-lineage and printed
    ``--each-suffix-overrides=F770W:o002_crf`` -- an override to a lineage the
    run already globs, which changes nothing."""
    verdict, msg = classify_requested_filter(
        'F770W', 0, target='sickle', proposal_id='3958', field='001-002',
        basepath='/x', each_suffix='o001_crf', declared={'F770W'},
        lineages={'o002_crf': 12})
    assert verdict == 'outside-this-run'
    assert 'each-suffix-overrides' not in msg


def test_a_joint_field_can_still_be_wrong_lineage():
    """The other side: on the same joint field, a destreak request when only the
    bare lineages exist is a real lineage mismatch and must still say so."""
    verdict, msg = classify_requested_filter(
        'F200W', 0, target='sickle', proposal_id='5365', field='002-998',
        basepath='/x', each_suffix='destreak_o002_crf', declared={'F200W'},
        lineages={'o002_crf': 10, 'o998_crf': 10})
    assert verdict == 'wrong-lineage'
    assert '--each-suffix-overrides=F200W:' in msg


def test_requested_lineages_matches_what_the_globber_globs(tmp_path):
    """The invariant behind the two tests above, checked against the real
    globber rather than against a restatement of it: every frame
    ``get_filenames`` returns for a joint field carries a lineage
    ``requested_lineages`` calls requested.  If ``get_filenames``'s per-subfield
    rewrite ever changes, this fails instead of the diagnosis quietly drifting
    back."""
    import os

    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
        get_filenames)

    pipeline = tmp_path / 'F770W' / 'pipeline'
    pipeline.mkdir(parents=True)
    for obs, lineage in (('002', 'o002_crf'), ('998', 'o998_crf')):
        (pipeline / f'jw05365{obs}001_02101_00001_mirimage_{lineage}.fits').touch()

    found = get_filenames(str(tmp_path), 'F770W', '5365', '002-998',
                          each_suffix='o002_crf', module='mirimage',
                          visitid='001')
    assert len(found) == 2, 'the globber spans both observations'
    on_disk = {'_'.join(os.path.basename(f)[:-len('.fits')].split('_')[4:])
               for f in found}
    assert on_disk <= requested_lineages('o002_crf', '002-998')


# --- the registry side of the distinction ---------------------------------

def test_the_registry_declares_the_treasury_bands_for_a_wildcard_tile():
    """The declared-vs-not distinction is read from fields.yaml, and 10678
    registers its 139 tiles by wildcard (``obsids: {nircam: '*'}``) rather than
    by name, so the per-observation lookup resolves no instrument and returns
    ``[]``.  Without the proposal-scoped fallback every Treasury tile would read
    "this band was never observed" and skip an unreduced F480M -- the case this
    guard exists for on 2026-09-10."""
    from jwst_gc_pipeline.photometry.requested_filters import (
        declared_for_observation)
    assert declared_for_observation('gc-treasury', '10678', '088') == {
        'F212N', 'F480M', 'F770W'}


def test_the_registry_is_scoped_to_the_observation_not_the_field():
    """sickle 3958 obs 007 is NIRCam; the MIRI bands registered in the same
    field are not declared for it, so a NIRCam run naming F770W is a
    not-observed skip rather than a failure."""
    from jwst_gc_pipeline.photometry.requested_filters import (
        declared_for_observation)
    declared = declared_for_observation('sickle', '3958', '007')
    assert 'F187N' in declared and 'F770W' not in declared


def test_the_registry_fallback_leaves_niriss_out():
    """``fields.declared_filters`` defaults to NIRCam+MIRI and says why: NIRISS
    has its own band list and its own layout, so a NIRISS-only name can never
    have a ``{FILTER}/pipeline/`` directory.  sgrc/4147 declares F158M/F356W for
    NIRISS only; unioned in, a NIRCam sgrc run naming one would be refused as
    declared-but-absent with no reduction able to clear it, instead of skipped
    as a band this observation never took.  (sgrc/4147 reaches the fallback:
    ``filters_for_observation`` returns [] for its observations.)"""
    from jwst_gc_pipeline.photometry.requested_filters import (
        declared_for_observation)
    declared = declared_for_observation('sgrc', '4147', '001')
    assert 'F405N' in declared, 'the NIRCam/MIRI bands are still declared'
    assert 'F158M' not in declared and 'F356W' not in declared


def test_an_unregistered_proposal_declares_nothing():
    """The "cannot tell" answer: the on-disk lineage probe decides alone."""
    from jwst_gc_pipeline.photometry.requested_filters import (
        declared_for_observation)
    assert declared_for_observation('nosuchfield', '99999', '001') == set()


# --- the on-disk lineage probe --------------------------------------------

def test_lineage_probe_reads_the_suffix_off_real_filenames(tmp_path):
    """The probe answers with strings that can be pasted back into
    ``--each-suffix-overrides``, and counts frames only: the per-frame catalogs
    and satstar products that share the stem are not frames."""
    pipeline = tmp_path / 'F150W' / 'pipeline'
    pipeline.mkdir(parents=True)
    for exp in (1, 2):
        for det in ('nrca1', 'nrcb1'):
            (pipeline / f'jw01905001001_02101_{exp:05d}_{det}_o001_crf.fits').touch()
            (pipeline / f'jw01905001001_02101_{exp:05d}_{det}_o001_crf'
                        f'_m12_satstar_catalog.fits').touch()
    (pipeline / 'jw01905001001_02101_00001_nrca1_destreak_o001_crf.fits').touch()
    # another proposal's product in the same tree must not be counted
    (pipeline / 'jw02221001001_02101_00001_nrca1_o001_crf.fits').touch()

    found = frame_lineages_on_disk(str(tmp_path), 'F150W', '1905', '001',
                                   each_suffix='destreak_o001_crf')
    assert found == {'o001_crf': 4, 'destreak_o001_crf': 1}


def test_lineage_probe_spans_a_joint_field(tmp_path):
    pipeline = tmp_path / 'F200W' / 'pipeline'
    pipeline.mkdir(parents=True)
    (pipeline / 'jw05365002001_02101_00001_nrca1_o002_crf.fits').touch()
    (pipeline / 'jw05365998001_02101_00001_nrca1_o998_crf.fits').touch()
    found = frame_lineages_on_disk(str(tmp_path), 'F200W', '5365', '002-998')
    assert found == {'o002_crf': 1, 'o998_crf': 1}


# --- the call site ---------------------------------------------------------
#
# The helper above can be right while nothing calls it, which is the state the
# repository was already in: the finalize's `no {filt}/{module} frames produced
# output in phase {phase}` raise is correct and fires after the whole fan-out
# array has run.  These two exercise `run_manual_pipeline` itself against a
# tree on disk, so a preflight that is deleted or moved after the fitting fails
# here.

def _wd1_tree(tmp_path, lineage='o001_crf'):
    """wd1's F150W and F115W as they sit on disk: F150W bare, F115W destreaked."""
    for filt, suffix in (('F150W', lineage), ('F115W', 'destreak_o001_crf')):
        pipeline = tmp_path / filt / 'pipeline'
        pipeline.mkdir(parents=True)
        for exp in (1, 2):
            (pipeline / f'jw01905001001_02101_{exp:05d}_nrca1_{suffix}.fits').touch()
    return str(tmp_path)


def _options(**kw):
    import types

    from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

    class _Options(types.SimpleNamespace):
        def __getattr__(self, name):   # every un-set optparse dest is None
            return None

    opts = _Options(**dict(MANUAL_DEFAULTS))
    opts.each_suffix = 'destreak_o001_crf'
    opts.desaturated = opts.bgsub = opts.blur = False
    opts.target = 'wd1'
    opts.manual_iterations = True
    for key, value in kw.items():
        setattr(opts, key, value)
    return opts


def _tree(tmp_path, filters):
    """``{filter: lineage}`` -> two exposures each, under that lineage."""
    for filt, suffix in filters.items():
        pipeline = tmp_path / filt / 'pipeline'
        pipeline.mkdir(parents=True)
        for exp in (1, 2):
            (pipeline / f'jw01905001001_02101_{exp:05d}_nrca1_{suffix}.fits').touch()
    return str(tmp_path)


def _run(basepath, options, filternames=('F115W', 'F150W')):
    from jwst_gc_pipeline.photometry import cataloging

    return cataloging.run_manual_pipeline(
        options, ['nrca'], list(filternames), {'1905': {'wd1': 1}},
        '1905', 'wd1', '001', basepath, {}, {})


def test_run_manual_pipeline_refuses_the_wd1_lineage_split(tmp_path):
    """The frames are on disk; only the requested lineage misses them.  Before
    this guard the run reached the per-frame fan-out and every shard reported
    `0 of 0 frames ... nothing to fit` and exited 0."""
    basepath = _wd1_tree(tmp_path)
    with pytest.raises(RequestedFilterHasNoFramesError) as excinfo:
        _run(basepath, _options())
    msg = str(excinfo.value)
    assert 'F150W' in msg
    assert '--each-suffix-overrides=F150W:o001_crf' in msg
    assert 'F115W' not in msg


def test_the_override_gets_past_the_preflight(tmp_path):
    """The other side at the call site: with the per-filter override both
    filters resolve, so the preflight passes.  `--manual-start-phase` is given
    a name no phase has, whose raise is the FIRST thing after the preflight --
    reaching it is the proof that the preflight did not refuse the run, without
    running a phase."""
    basepath = _wd1_tree(tmp_path)
    options = _options(each_suffix_overrides='F150W:o001_crf',
                       manual_start_phase='not-a-phase')
    with pytest.raises(ValueError) as excinfo:
        _run(basepath, options)
    assert 'not-a-phase' in str(excinfo.value)
    assert not isinstance(excinfo.value, RequestedFilterHasNoFramesError)


# --- a skipped filter has to LEAVE the run --------------------------------
#
# The not-observed verdict is documented as "reported and skipped, not raised".
# Logging it and leaving it in `filternames` does not deliver that: the filter
# walks into the phase loop with zero frames and dies on
# `no {filt}/{module} frames produced output in phase {phase}`
# (cataloging.py), which is the raise this verdict exists to avoid.  The
# preflight therefore runs BEFORE `multifilter`/`phases`/`is_miri` are derived
# and drops the filter, and the "MANUAL PIPELINE:" banner -- printed after that
# point -- is where both halves are observable.

def test_a_not_observed_filter_is_dropped_from_the_run(tmp_path, capsys):
    """F480M is not declared for wd1/1905 and nothing is on disk for it, so the
    run continues on F115W alone.  It must not still be in `filternames`, and
    the m7 cross-band phase (multifilter, decided from that list) must be gone
    with it."""
    basepath = _tree(tmp_path, {'F115W': 'destreak_o001_crf'})
    options = _options(manual_start_phase='not-a-phase')
    with pytest.raises(ValueError) as excinfo:
        _run(basepath, options, filternames=('F115W', 'F480M'))
    assert 'not-a-phase' in str(excinfo.value), (
        'the preflight must not have refused the run')
    out = capsys.readouterr().out
    assert "dropping ['F480M']" in out
    banner = [ln for ln in out.splitlines() if ln.startswith('MANUAL PIPELINE:')]
    assert len(banner) == 1, out
    assert "filters=['F115W']" in banner[0], (
        'a dropped filter must not survive into the phase loop')
    assert "'m7'" not in banner[0], (
        'multifilter is decided AFTER the drop: one filter left means no m7')


def test_a_run_left_with_no_filters_refuses(tmp_path):
    """Dropping is not the same as succeeding.  If every requested band is one
    the observation never took, the run has nothing to catalog and must say so
    rather than walk through six phases and report success."""
    basepath = _tree(tmp_path, {'F115W': 'destreak_o001_crf'})
    with pytest.raises(RequestedFilterHasNoFramesError) as excinfo:
        _run(basepath, _options(), filternames=('F480M',))
    assert 'nothing to catalog' in str(excinfo.value)


# --- the fan-out consequence, and the one override ------------------------
#
# On origin/main a fan-out shard whose filter resolved to zero printed
# `nothing to fit` and exited 0, so the array still fitted the other bands and
# only the finalize died.  With the preflight the shard refuses before any
# phase, so a run naming one absent band now produces nothing for the good
# bands either.  That is the intended trade for #592 -- the wd1 run spent ~10 h
# of array time on a pass whose correction floor then came from ten of eleven
# filters -- so the refusal says it, and the one case an operator cannot fix by
# editing the command line (a band still being reduced) has a waiver.

def test_the_refusal_says_the_whole_run_stops(tmp_path):
    basepath = _wd1_tree(tmp_path)
    with pytest.raises(RequestedFilterHasNoFramesError) as excinfo:
        _run(basepath, _options())
    msg = str(excinfo.value)
    assert 'THE WHOLE RUN STOPS' in msg and 'fan-out shard' in msg


def test_a_declared_but_absent_band_is_waivable_with_a_reason(tmp_path,
                                                              capsys,
                                                              monkeypatch):
    """The Treasury shape on 2026-09-10: F444W is declared for wd1/1905 and its
    reduction has not written frames.  Unwaived the run stops and names the
    waiver; waived, the band is dropped and F115W still produces."""
    basepath = _tree(tmp_path, {'F115W': 'destreak_o001_crf'})
    with pytest.raises(RequestedFilterHasNoFramesError) as excinfo:
        _run(basepath, _options(), filternames=('F115W', 'F444W'))
    msg = str(excinfo.value)
    assert 'fields.yaml declares it' in msg
    assert ALLOW_ABSENT_ENV in msg, 'the refusal must name its own waiver'

    monkeypatch.setenv(ALLOW_ABSENT_ENV, '1')
    monkeypatch.setenv(f'{ALLOW_ABSENT_ENV}_REASON',
                       'F444W reduction still running')
    options = _options(manual_start_phase='not-a-phase')
    with pytest.raises(ValueError) as excinfo:
        _run(basepath, options, filternames=('F115W', 'F444W'))
    assert 'not-a-phase' in str(excinfo.value)
    out = capsys.readouterr().out
    assert f'WARNING (override {ALLOW_ABSENT_ENV}=1)' in out
    assert 'F444W reduction still running' in out
    assert "filters=['F115W']" in out


def test_the_waiver_does_not_cover_a_wrong_lineage(tmp_path, monkeypatch):
    """wd1's F150W is on disk under the other lineage: a mis-specified command
    line, fixed by one flag and a resubmission.  There is nothing to waive, so
    the waiver must not open that door."""
    monkeypatch.setenv(ALLOW_ABSENT_ENV, '1')
    monkeypatch.setenv(f'{ALLOW_ABSENT_ENV}_REASON', 'trying to get past this')
    basepath = _wd1_tree(tmp_path)
    with pytest.raises(RequestedFilterHasNoFramesError) as excinfo:
        _run(basepath, _options())
    assert '--each-suffix-overrides=F150W:o001_crf' in str(excinfo.value)


def test_a_waived_band_with_no_reason_is_said_loudly(tmp_path, capsys,
                                                     monkeypatch):
    """The flag without the justification is the state CLAUDE.md forbids for the
    astrometry gates; the same is said here rather than passing silently."""
    monkeypatch.setenv(ALLOW_ABSENT_ENV, '1')
    monkeypatch.delenv(f'{ALLOW_ABSENT_ENV}_REASON', raising=False)
    basepath = _tree(tmp_path, {'F115W': 'destreak_o001_crf'})
    options = _options(manual_start_phase='not-a-phase')
    with pytest.raises(ValueError):
        _run(basepath, options, filternames=('F115W', 'F444W'))
    assert 'NO JUSTIFICATION RECORDED' in capsys.readouterr().out
