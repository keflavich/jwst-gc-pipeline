"""Tests for the pipeline monitor.

The scanner is exercised against synthetic directory trees rather than the real
archive so the assertions are about the RULES (observation attribution, suffix
separation, ambiguity) and not about whatever happens to be on disk today.
"""
import json
import os
import re

import pytest

from jwst_gc_pipeline.monitoring import checks, jobs, render, scan


# --------------------------------------------------------------------------
# Job-name parsing
# --------------------------------------------------------------------------

TARGETS = ('sgrb2', 'sgrb', 'brick', 'gc2211', 'wd1', 'wd2', 'm4', 'm92', 'w51',
           'arches', 'quintuplet', 'ngc6334')


@pytest.mark.parametrize('name,target,proposal,obsid,stage', [
    # the concatenated head must split against the registry, not by regex:
    # 'sgrb25365' is sgrb2+5365, NOT sgrb+25365, and both are registered names.
    ('sgrb25365-o001-m12-finalize', 'sgrb2', '5365', '001', 'm12'),
    ('brick2221-o001-m12-fanout', 'brick', '2221', '001', 'm12'),
    ('gc22112211-o046-cat-F200W', 'gc2211', '2211', '046', 'cat'),
    ('wd11905-o001-cat', 'wd1', '1905', '001', 'cat'),
    ('m921334-o001-cat', 'm92', '1334', '001', 'cat'),
    ('arches-001-m12-fanout', 'arches', None, '001', 'm12'),
    ('pf_sgrb2_m12_s3', 'sgrb2', None, None, 'm12'),
    ('w51-catalog', 'w51', None, None, 'catalog'),
])
def test_parse_job_name(name, target, proposal, obsid, stage):
    got = jobs.parse_job_name(name, targets=sorted(TARGETS, key=len, reverse=True))
    assert got is not None, name
    assert (got['target'], got['proposal'], got['obsid'], got['stage']) == \
        (target, proposal, obsid, stage)


@pytest.mark.parametrize('name', ['interactive', 'data-qa-mast-download',
                                  'some-other-users-job'])
def test_unregistered_job_names_are_unattributed(name):
    """A job that is not a registered field must not inflate a field's count."""
    assert jobs.parse_job_name(name, targets=TARGETS) is None


def test_parse_job_name_filter_is_uppercased():
    got = jobs.parse_job_name('brick2221-o001-cat-f182m', targets=TARGETS)
    assert got['filter'] == 'F182M'


# --------------------------------------------------------------------------
# Log attribution
# --------------------------------------------------------------------------

def test_log_job_name():
    assert jobs.log_job_name(
        'catalog_brick2221-o001-cut5-F212N_38511678_4294967294.out') == \
        'brick2221-o001-cut5-F212N'
    assert jobs.log_job_name('reduce_w51-catalog_123.out') == 'w51-catalog'
    assert jobs.log_job_name('not-a-log.txt') is None


def test_log_belongs_to_respects_observation():
    """An o050 crash must not be reported against o023: they share only the field."""
    log = 'catalog_gc22112211-o050-m3-fanout_38263470_0.out'
    assert jobs.log_belongs_to(log, 'gc2211', '050')
    assert not jobs.log_belongs_to(log, 'gc2211', '023')
    assert not jobs.log_belongs_to(log, 'brick', None)
    # a name that carries no observation is field-level, shown for every obs
    assert jobs.log_belongs_to('catalog_w51-catalog_1.out', 'w51', '001')


def test_log_signatures_ignore_benign_zero_source_lines(tmp_path):
    """'Satstar summary: 0/0 sources accepted' is normal output, not a warning."""
    log = tmp_path / 'catalog_x_1_0.out'
    log.write_text('CATALOG start: brick\n'
                   + 'Satstar summary: 0/0 sources accepted, 0 rejected\n' * 50
                   + 'CATALOG done: filter=F212N rc=0\n')
    got = jobs.scan_log(str(log))
    assert got['worst'] == 'info'
    assert not any(sev in ('error', 'warn') for sev, _, _ in got['lines'])


def test_log_signatures_catch_psf_build_failure(tmp_path):
    """The F150W2 PSF failure that killed the m4/ngc6397 probes must be an error."""
    log = tmp_path / 'catalog_y_2_0.out'
    log.write_text('ValueError: Failed to download PSF after 11 attempts; last '
                   'error: RuntimeError: The requested wavelengths are too long '
                   'for NIRCam short wave channel.\n')
    got = jobs.scan_log(str(log))
    assert got['worst'] == 'error'
    assert 'psf-build' in got['hits']


# --------------------------------------------------------------------------
# Observation-safe counting
# --------------------------------------------------------------------------

def _touch(path, size=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(b'x' * size)


def test_count_matching_separates_observations(tmp_path):
    d = tmp_path / 'F212N' / 'pipeline'
    for exp in range(3):
        _touch(str(d / f'jw02221001001_05101_0000{exp}_nrcb1_destreak_o001_crf.fits'))
    for exp in range(2):
        _touch(str(d / f'jw02221002001_05101_0000{exp}_nrcb1_destreak_o002_crf.fits'))
    scan.clear_cache()
    entries = scan.listing(str(d))
    got = scan.count_matching(entries, lambda n: n.endswith('_crf.fits'), '2221', '001')
    assert got['n'] == 3
    assert got['scope'] == 'obs'


def test_count_matching_reports_ambiguity_not_a_guess(tmp_path):
    """A name with no observation token is reported ambiguous, never assumed."""
    d = tmp_path / 'catalogs'
    _touch(str(d / 'f212n_merged_indivexp_merged_resbgsub_m7_dao_basic.fits'))
    scan.clear_cache()
    got = scan.count_matching(scan.listing(str(d)), lambda n: n.endswith('.fits'),
                              '2221', '001')
    assert got['n'] == 1
    assert got['scope'] == 'ambiguous'


def test_crf_and_reduced_are_counted_separately(tmp_path):
    """`*_o001_crf.fits` also matches `*_destreak_o001_crf.fits`.

    Folding them together hides "reduction ran but destreaking never did" --
    the wd1 F150W failure the monitor exists to catch.
    """
    d = tmp_path / 'F150W' / 'pipeline'
    for exp in range(4):
        _touch(str(d / f'jw01905001001_02101_0000{exp}_nrca1_o001_crf.fits'))
    scan.clear_cache()
    rows = scan._reduction_stages(str(tmp_path), 'F150W', '1905', '001')
    assert rows['crf']['n'] == 4
    assert rows['reduced']['n'] == 0


def test_i2d_row_excludes_per_exposure_outlier_products(tmp_path):
    d = tmp_path / 'F212N' / 'pipeline'
    _touch(str(d / 'jw02221-o001_t001_nircam_clear-f212n-merged_i2d.fits'))
    _touch(str(d / 'jw02221001001_05101_00001_nrcb1_outlier_i2d.fits'))
    scan.clear_cache()
    rows = scan._reduction_stages(str(tmp_path), 'F212N', '2221', '001')
    assert rows['i2d']['n'] == 1


# --------------------------------------------------------------------------
# Registry-derived rules (these read the real fields.yaml on purpose: they are
# assertions about the registry's actual shape, which is what the monitor must
# handle)
# --------------------------------------------------------------------------

def test_shared_filters_expands_multiple_obsids_of_one_proposal():
    """gc2211's five observations sit under ONE registry entry sharing a filter
    list, so every one of its filters is ambiguous."""
    assert scan.shared_filters('gc2211') == {'F150W', 'F200W', 'F277W'}


def test_shared_filters_finds_the_ngc6334_cross_proposal_collision():
    assert scan.shared_filters('ngc6334') == {'F200W', 'F470N'}


def test_brick_filters_are_disjoint_so_nothing_is_ambiguous():
    """Brick is multi-observation but its two observations use different filters,
    so flagging all of them would bury the real cases."""
    assert scan.shared_filters('brick') == set()


def test_registered_filters_splits_miri_out_of_a_nircam_run():
    """fields.yaml keeps one flat filter list per proposal covering every
    instrument; a NIRCam run must not grow empty MIRI rows."""
    nircam = scan.registered_filters('w51', '6151', 'nircam')
    miri = scan.registered_filters('w51', '6151', 'miri')
    assert 'F770W' in miri and 'F770W' not in nircam
    assert 'F182M' in nircam and 'F182M' not in miri


def test_is_globbed_marks_registered_but_unread_observations():
    """wd1 registers o001 and o003 but the reduction globs only 001."""
    assert scan.is_globbed('wd1', '1905', '001')
    assert not scan.is_globbed('wd1', '1905', '003')
    # a '*' glob_obsid means every observation is read
    assert scan.is_globbed('sickle', '3958', '001')


def test_filter_dir_pattern_accepts_the_wide_globular_cluster_filters():
    """m4/ngc6397 use F150W2/F322W2; a pattern ending [WMN] misses both."""
    import re
    rx = re.compile(r'F\d{3,4}[A-Z]?\d*')
    for name in ('F212N', 'F182M', 'F150W2', 'F322W2', 'F2550W'):
        assert rx.fullmatch(name), name


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def _run(**kw):
    base = {'target': 't', 'proposal': '1', 'obsid': '001', 'basepath': '/x',
            'per_filter': {}, 'crossband': {}, 'astrometry': {}, 'provenance': {},
            'is_cutout': False, 'multi_obs': False, 'globbed': True,
            'filters_missing': []}
    base.update(kw)
    return base


def test_thresholds_are_imported_not_copied():
    """Every gate the monitor reports must come from the module that enforces it,
    so the page cannot drift away from the pipeline."""
    from jwst_gc_pipeline.photometry import astrometry_checkpoint, visit_consensus
    assert checks.EXPOSURE_CONSENSUS_TOL_MAS is visit_consensus.EXPOSURE_CONSENSUS_TOL_MAS
    assert checks.CROSSFILTER_TOL_MAS is astrometry_checkpoint.CROSSFILTER_TOL_MAS
    assert checks.LOCAL_CELL_TOL_MAS is astrometry_checkpoint.LOCAL_CELL_TOL_MAS


def test_misaligned_exposures_fail():
    run = _run(astrometry={'F212N': {'n_exposures': 48, 'n_misaligned': 3,
                                     'attributable': True, 'mtime': 100,
                                     'path': '/x/checkpoint_m2_F212N_latest.json',
                                     'visits': []}})
    verdicts = checks.check_astrometry(run)
    bad = [v for v in verdicts if v['name'] == 'astrometry-misaligned-F212N']
    assert bad and bad[0]['severity'] == 'fail'


def test_unattributable_checkpoint_is_a_warning_not_a_failure():
    """A per-filter checkpoint on a shared filter describes whichever observation
    ran last, so raising it as THIS observation's failure is a false alarm."""
    run = _run(astrometry={'F150W': {'n_exposures': 48, 'n_misaligned': 3,
                                     'attributable': False, 'mtime': 100,
                                     'path': '/x/c.json', 'visits': []}})
    bad = [v for v in checks.check_astrometry(run)
           if v['name'] == 'astrometry-misaligned-F150W']
    assert bad and bad[0]['severity'] == 'warn'
    assert 'unattributed' in bad[0]['summary']


def test_checkpoint_older_than_the_frames_is_downgraded():
    """At m2 a misalignment corrects the table and stops the run; if the frames
    were regenerated afterwards the record describes the state before the fix."""
    run = _run(
        astrometry={'F212N': {'n_exposures': 48, 'n_misaligned': 3,
                              'attributable': True, 'mtime': 100,
                              'path': '/x/c.json', 'visits': []}},
        per_filter={'F212N': {'reduced': {'n': 48, 'mtime': 200}}})
    bad = [v for v in checks.check_astrometry(run)
           if v['name'] == 'astrometry-misaligned-F212N']
    assert bad and bad[0]['severity'] == 'warn'


def test_mixed_provenance_tags_fail():
    run = _run(provenance={'m7': {'tags': {'a': 3, 'b': 2}, 'n_sidecars': 5,
                                  'n_distinct': 2, 'n_dirty': 0}})
    bad = [v for v in checks.check_provenance(run) if 'mixed' in v['name']]
    assert bad and bad[0]['severity'] == 'fail'


def test_single_provenance_tag_is_clean():
    run = _run(provenance={'m7': {'tags': {'a': 5}, 'n_sidecars': 5,
                                  'n_distinct': 1, 'n_dirty': 0}})
    assert all(v['severity'] == 'info' for v in checks.check_provenance(run))


def test_unreduced_filter_fails():
    run = _run(per_filter={'F150W': {'crf': {'n': 96}, 'reduced': {'n': 0}}})
    bad = [v for v in checks.check_products(run) if v['name'] == 'unreduced-F150W']
    assert bad and bad[0]['severity'] == 'fail'


def test_ungl0bbed_observation_reports_once_and_runs_nothing_else():
    """An observation the reduction never globs has no products by design."""
    run = _run(globbed=False, glob_obsid='001', obsid='003',
               astrometry={'F212N': {'n_exposures': 4, 'n_misaligned': 4,
                                     'attributable': True, 'mtime': 1,
                                     'path': '/x/c.json', 'visits': []}})
    verdicts = checks.run_checks(run)
    assert len(verdicts) == 1
    assert verdicts[0]['name'] == 'not-globbed'
    assert verdicts[0]['severity'] == 'info'


def test_array_job_log_failures_are_grouped():
    """One array job writes 16 near-identical logs; that is one finding."""
    scans = [{'path': f'/l/catalog_brick2221-o001-m3-fanout_1_{i}.out',
              'worst': 'error', 'hits': {'traceback': 1},
              'lines': [('error', 'traceback', 'Traceback (most recent call last)')]}
             for i in range(16)]
    verdicts = checks.check_jobs(_run(), [], scans)
    errors = [v for v in verdicts if v['name'] == 'log-error']
    assert len(errors) == 1
    assert '16 tasks' in errors[0]['summary']


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------

def _entry(**kw):
    run = _run(**kw.pop('run', {}))
    verdicts = kw.pop('verdicts', [])
    return {'run': run, 'verdicts': verdicts, 'tally': checks.tally(verdicts),
            'worst': checks.worst_severity(verdicts), 'jobs': [],
            'anchor': 'f-t', 'newest_mtime': None}


def test_page_is_self_contained():
    """No third-party assets, and nothing fetched before the reader asks.

    The page carries its own CSS and JS inline. The opt-in sky view is the one
    exception and is scoped: Aladin Lite is referenced by a RELATIVE path (a
    same-origin copy that publish() links in, not a CDN), and it is only
    fetched when the reader clicks -- so a page under a strict CSP renders
    completely and simply cannot expand that one panel.
    """
    html = render.render_page([_entry()], standalone=True)
    for forbidden in ('<link', '@import', '<script src', 'cdn.', 'unpkg',
                      'jsdelivr', 'aladin.cds.unistra.fr'):
        assert forbidden not in html, forbidden
    # any absolute URL that IS present must be our own host, and must be a HiPS
    # tile base -- image data for the opt-in view, never code
    import re as _re
    for url in set(_re.findall(r'https?://[^\s"\']+', html)):
        assert url.startswith('https://starformation.astro.ufl.edu/'), url
        assert not url.endswith('.js'), url


def test_sky_view_defaults_to_the_jwst_layers_only():
    """The page is a JWST pipeline monitor; Roman geometry is context."""
    from jwst_gc_pipeline.monitoring import skyview
    html = skyview.section({'program': '10678', 'title': 't', 'n_planned': 139,
                            'n_observed': 0, 'pa_v3': 87.0,
                            'pa_v3_range': [79.0, 95.0]})
    state = _re_search_state(html)
    assert state['nircam'] == 'true' and state['miri'] == 'true'
    # observed is on despite being empty, so the first executed visit shows up
    assert state['observed'] == 'true'
    assert state['spring'] == 'false' and state['autumn'] == 'false'
    assert state['target'] == 'false'


def _re_search_state(html):
    import re as _re
    m = _re.search(r'var on = \{(.*?)\}', html, _re.S)
    assert m, 'default layer state not found'
    return dict(_re.findall(r'(\w+):\s*(true|false)', m.group(1)))


def test_sky_view_keeps_the_observed_toggle_when_nothing_is_observed():
    """'nothing observed yet' is an answer; hiding the toggle is not."""
    from jwst_gc_pipeline.monitoring import skyview
    html = skyview.section({'program': '10678', 'title': 't', 'n_planned': 139,
                            'n_observed': 0, 'pa_v3': 87.0,
                            'pa_v3_range': [79.0, 95.0]})
    assert 'lyr-observed' in html
    assert 'none yet' in html


def test_sky_view_without_data_says_how_to_generate_it():
    from jwst_gc_pipeline.monitoring import skyview
    html = skyview.section(None)
    assert 'build_footprints.py' in html
    assert 'gcm-aladin' not in html


def test_page_defines_both_themes_with_the_toggle_winning():
    html = render.render_page([_entry()], standalone=True)
    assert 'prefers-color-scheme: dark' in html
    assert ':root[data-theme="dark"]' in html
    assert ':root[data-theme="light"]' in html


def test_page_escapes_content():
    entry = _entry(run={'target': '<script>alert(1)</script>'})
    html = render.render_page([entry], standalone=True)
    assert '<script>alert(1)</script>' not in html
    assert '&lt;script&gt;' in html


def test_filter_hue_runs_short_wavelength_cyan_to_long_amber():
    assert render.filter_hue('F115W') > render.filter_hue('F212N')
    assert render.filter_hue('F212N') > render.filter_hue('F480M')
    assert render.filter_micron('F212N') == pytest.approx(2.12)


def test_ladder_marks_partial_and_ambiguous_separately():
    run = _run(per_filter={'F1': {'m3': {'n': 1, 'scope': 'obs'}},
                           'F2': {'m3': {'n': 0, 'scope': 'none'}}},
               crossband={'m7': {'n': 1, 'scope': 'ambiguous'}})
    state = render._ladder_state(run)
    assert state['m3'] == 'part'
    assert state['m7'] == 'ambig'


# --------------------------------------------------------------------------
# Probe planning
# --------------------------------------------------------------------------

def test_probe_filter_is_chosen_after_the_suffix(tmp_path):
    """Picking the filter first and the suffix second is what produced the wd1
    F150W failure: a filter that was never destreaked, catalogued against the
    destreaked suffix, matching nothing."""
    from jwst_gc_pipeline.monitoring import probe
    _touch(str(tmp_path / 'F212N' / 'pipeline' /
               'jw01905001001_1_1_nrca1_o001_crf.fits'))
    _touch(str(tmp_path / 'F182M' / 'pipeline' /
               'jw01905001001_1_1_nrca1_destreak_o001_crf.fits'))
    filt, n = probe.choose_filter(str(tmp_path), 'destreak_o001_crf')
    assert filt == 'F182M' and n == 1


def test_probe_raises_when_no_filter_carries_the_suffix(tmp_path):
    from jwst_gc_pipeline.monitoring import probe
    _touch(str(tmp_path / 'F212N' / 'pipeline' /
               'jw01905001001_1_1_nrca1_o001_crf.fits'))
    with pytest.raises(probe.ProbeError):
        probe.choose_filter(str(tmp_path), 'destreak_o001_crf')


def test_probe_submission_carries_the_region_in_the_environment():
    """The cutout region contains commas, which sbatch --export treats as
    variable separators; it must travel in the exported environment instead."""
    from jwst_gc_pipeline.monitoring import probe
    plan = {'target': 'brick', 'proposal': '2221', 'obsid': '001',
            'filter': 'F212N', 'each_suffix': 'destreak_o001_crf',
            'cutout_region': '266.5,-28.7,5', 'label': 'monitor5as',
            'job_name': 'brick2221-o001-cut5-F212N'}
    env, argv = probe.submit_command(plan, repo_root='/repo')
    assert '--cutout-region=266.5,-28.7,5' in env['EXTRA_ARGS']
    assert not any('266.5' in a for a in argv)
    assert '--export=ALL' in argv
    assert f"--job-name={plan['job_name']}" in argv


def test_probe_frame_glob_is_pinned_to_the_proposal():
    """ngc6334's F182M holds frames from BOTH 6778 and 7213 under one suffix."""
    from jwst_gc_pipeline.monitoring import probe
    assert probe._frame_prefix('6778', '001') == 'jw06778001'
    assert probe._frame_prefix(None, None) == 'jw'


# --------------------------------------------------------------------------
# Non-finite handling
# --------------------------------------------------------------------------

def test_checkpoint_nan_is_dropped_not_propagated(tmp_path):
    """The checkpoint writer emits bare NaN; JSON.parse rejects it and NaN
    poisons min/max."""
    ckdir = tmp_path / 'astrometry_checkpoints'
    os.makedirs(ckdir)
    (ckdir / 'checkpoint_m2_F212N_latest.json').write_text(json.dumps({
        'stage': 'm2', 'date': '2026-01-01T00:00:00Z',
        'visits': [{'visit': '1',
                    'consensus': {'median_scatter_mas': float('nan'),
                                  'consensus_ok': True},
                    'exposures': [{'contrast': float('nan'), 'off': 5.0},
                                  {'contrast': 12.0, 'off': float('nan')}]}]}))
    got = scan.astrometry_checkpoints(str(tmp_path))
    rec = got['F212N']
    assert rec['min_contrast'] == 12.0
    assert rec['max_off_mas'] == 5.0
    assert rec['visits'][0]['scatter_mas'] is None
    json.dumps(rec, allow_nan=False)      # must not raise


# --------------------------------------------------------------------------
# Astrometry-paper verdicts
# --------------------------------------------------------------------------

def _paper_tree(tmp_path, generated='2026-07-19T08:14:38', problems=(),
                bands=None, catalog_mtime=None):
    """A minimal astrometry_paper checkout: config.py + one postrecat summary."""
    (tmp_path / 'config.py').write_text(
        'MIN_CATALOG_DATE = "2026-07-11"\nMIN_CONTRAST = 5.0\n'
        'SAME_FRAME_TOL_MAS = 5.0\nTIE_CLIP_MAS = 60.0\nJWST_QFIT_MAX = 0.4\n'
        'BANDS = {"2221": ["f212n"]}\nCATDIR = "/x"\n')
    cat = tmp_path / 'f212n_vetted.fits'
    cat.write_text('x')
    if catalog_mtime is not None:
        os.utime(cat, (catalog_mtime, catalog_mtime))
    od = tmp_path / 'outputs' / '2026-07-19_postrecat'
    os.makedirs(od)
    (od / 'summary.json').write_text(json.dumps({
        'generated': generated,
        '2221': bands if bands is not None else {
            'f212n': {'path': str(cat), 'mtime': '2026-07-12T00:00:00',
                      'vs_virac_p60': {'off': 27.4, 'contrast': 100.0},
                      'vs_virac_p90': {'off': 26.2, 'contrast': 90.0},
                      'mode_flip_mas': 1.3}},
        'certifiers': {'table': '/x/merged.fits'},
        'problems': list(problems)}))
    return od


def test_paper_reads_the_verdict_it_does_not_recompute_it(tmp_path):
    """The paper's own gates are applied in its script; the monitor reports them."""
    from jwst_gc_pipeline.monitoring import paper
    od = _paper_tree(tmp_path, problems=['2221/f212n: 44.0 mas vs anchor f212n (>30)'])
    summary = paper.summarize(paper.read_verdicts(str(tmp_path)))
    assert summary['problems'] == ['2221/f212n: 44.0 mas vs anchor f212n (>30)']
    assert summary['config']['min_catalog_date'] == '2026-07-11'
    assert 'postrecat' in os.path.basename(summary['postrecat_dir'])
    assert od.exists()


def test_paper_problems_are_scoped_to_the_run_programme(tmp_path):
    """The paper validates both brick programmes at once; a 1182 failure must not
    appear as a 2221 failure."""
    from jwst_gc_pipeline.monitoring import paper
    _paper_tree(tmp_path, problems=['1182/f115w: 92.7 mas vs anchor f200w (>30)'])
    summary = paper.summarize(paper.read_verdicts(str(tmp_path)))
    run = _run(target='brick', proposal='2221')
    verdicts = checks.check_paper(run, summary)
    assert not [v for v in verdicts if v['name'] == 'paper-problem']
    assert [v for v in verdicts if v['name'] == 'paper-problem-other']
    run1182 = _run(target='brick', proposal='1182')
    assert [v for v in checks.check_paper(run1182, summary)
            if v['name'] == 'paper-problem']


def test_paper_verdict_is_failed_when_the_catalog_was_rewritten_under_it(tmp_path):
    """A verdict certifying a product that has since been rewritten reads as a
    pass while describing a file that no longer exists."""
    import time
    from jwst_gc_pipeline.monitoring import paper
    od = _paper_tree(tmp_path)
    # make the summary old and the catalog new
    old = time.time() - 86400
    os.utime(od / 'summary.json', (old, old))
    summary = paper.summarize(paper.read_verdicts(str(tmp_path)))
    verdicts = checks.check_paper(_run(target='brick', proposal='2221'), summary)
    bad = [v for v in verdicts if v['name'] == 'paper-verdict-outdated']
    assert bad and bad[0]['severity'] == 'fail'


def test_paper_freshness_uses_the_current_mtime_not_the_recorded_one(tmp_path):
    """The paper's guard stats the file NOW, so the monitor must too."""
    import datetime
    from jwst_gc_pipeline.monitoring import paper
    stamp = datetime.datetime(2026, 7, 1).timestamp()      # before MIN_CATALOG_DATE
    _paper_tree(tmp_path, catalog_mtime=stamp)
    rows = paper.catalog_freshness(paper.read_verdicts(str(tmp_path)))
    assert rows and rows[0]['predates_min_catalog_date'] is True
    # the recorded mtime in the summary is 2026-07-12, i.e. AFTER the min date --
    # trusting it would have missed this.
    assert rows[0]['recorded_mtime'].startswith('2026-07-12')


def test_paper_checks_do_not_apply_to_other_fields_or_to_cutouts(tmp_path):
    from jwst_gc_pipeline.monitoring import paper
    _paper_tree(tmp_path, problems=['2221/f212n: bad'])
    summary = paper.summarize(paper.read_verdicts(str(tmp_path)))
    assert checks.check_paper(_run(target='w51'), summary) == []
    assert checks.check_paper(_run(target='brick', is_cutout=True), summary) == []


def test_paper_absent_is_a_skip_not_a_failure(tmp_path):
    from jwst_gc_pipeline.monitoring import paper
    got = paper.read_verdicts(str(tmp_path / 'nope'))
    assert got['present'] is False
    verdicts = checks.check_paper(_run(target='brick'), paper.summarize(got))
    assert all(v['severity'] in ('skip', 'info') for v in verdicts)


def test_paper_block_renders_and_escapes(tmp_path):
    from jwst_gc_pipeline.monitoring import paper
    _paper_tree(tmp_path, problems=['<script>x</script>'])
    summary = paper.summarize(paper.read_verdicts(str(tmp_path)))
    entry = _entry(run={'target': 'brick', 'proposal': '2221'})
    entry['paper'] = summary
    html = render.render_page([entry], standalone=True)
    assert 'post_recat_validation' in html
    assert '<script>x</script>' not in html


# --------------------------------------------------------------------------
# Per-tile map, satstar generations, header provenance
# --------------------------------------------------------------------------

def _visit(**kw):
    base = {'visit': '1', 'consensus_ok': True, 'scatter_mas': 1.0,
            'tiles_ok': 36, 'tiles_total': 36, 'tiles_clean': True,
            'worst_tile_mas': None, 'worst_tile_cell': '(0,0)',
            'min_tile_contrast': 100.0, 'tie_apply_ok': True,
            'tie_gross_ok': True}
    base.update(kw)
    return base


def _astrom(**kw):
    rec = {'n_exposures': 48, 'n_misaligned': 0, 'attributable': True,
           'mtime': 100, 'path': '/x/checkpoint_m2_F212N_latest.json',
           'visits': [_visit(**kw)]}
    return {'F212N': rec}


def test_all_tiles_ok_does_not_mean_within_tolerance():
    """measure_offset_grid runs with no max_off_mas, so astrometry_offsets sets
    off_ok=True unconditionally: `36/36 tiles ok` coexists with a 29 mas worst
    cell (measured on brick F182M). Only worst_off_mas reports the gate."""
    run = _run(astrometry=_astrom(worst_tile_mas=29.1, tiles_ok=36, tiles_total=36))
    bad = [v for v in checks.check_astrometry(run) if 'worst-tile' in v['name']]
    assert bad and bad[0]['severity'] == 'warn'
    assert bad[0]['threshold'] == checks.LOCAL_CELL_TOL_MAS
    assert '36/36' in bad[0]['detail']


def test_worst_tile_within_tolerance_is_silent():
    run = _run(astrometry=_astrom(worst_tile_mas=4.0))
    assert not [v for v in checks.check_astrometry(run) if 'worst-tile' in v['name']]


def test_weak_tile_contrast_fails():
    run = _run(astrometry=_astrom(min_tile_contrast=2.0))
    bad = [v for v in checks.check_astrometry(run) if 'tile-contrast' in v['name']]
    assert bad and bad[0]['severity'] == 'fail'


def test_unapplied_reference_tie_is_reported():
    run = _run(astrometry=_astrom(tie_apply_ok=False))
    assert [v for v in checks.check_astrometry(run) if 'tie-unapplied' in v['name']]


def test_all_satstar_rejected_is_a_failure_not_an_absence():
    """0 catalogs + N rejected still ships saturated photometry from an earlier
    stage; a present/absent count cannot see it."""
    run = _run(per_filter={'F405N': {'crf': {'n': 48}, 'reduced': {'n': 48},
                                     'satstar_frames': {'accepted': 0,
                                                        'rejected': 48}}})
    bad = [v for v in checks.check_products(run)
           if v['name'] == 'satstar-all-rejected-F405N']
    assert len(bad) == 1, f'{len(bad)} satstar verdicts, expected 1'
    assert bad[0]['severity'] == 'fail'


def test_satstar_present_is_silent():
    run = _run(per_filter={'F212N': {'crf': {'n': 48}, 'reduced': {'n': 48},
                                     'satstar_frames': {'accepted': 48,
                                                        'rejected': 48}}})
    assert not [v for v in checks.check_products(run) if 'satstar' in v['name']]


def test_lw_filteroffset_module_mismatch_fails():
    """A module-A LW frame carrying module B's filteroffset is displaced by up to
    ~26 mas, anti-symmetrically between modules."""
    run = _run(headers={'F410M': {
        'crds_ctx': {'jwst_1253.pmap': 4}, 'cal_ver': {'1.21.0': 4},
        'filteroffset_mismatch': [
            {'file': 'x.fits', 'module': 'A',
             'r_filoff': 'jwst_nircam_filteroffset_0008.asdf', 'expected': '0007'}]}})
    bad = [v for v in checks.check_headers(run)
           if v['name'] == 'filteroffset-module-mismatch']
    assert bad and bad[0]['severity'] == 'fail'


def test_matching_filteroffset_is_silent():
    run = _run(headers={'F410M': {'crds_ctx': {'jwst_1253.pmap': 4},
                                  'cal_ver': {'1.21.0': 4},
                                  'filteroffset_mismatch': []}})
    assert not [v for v in checks.check_headers(run)
                if v['name'] == 'filteroffset-module-mismatch']


def test_mixed_crds_context_is_reported():
    run = _run(headers={'F182M': {'crds_ctx': {'jwst_1253.pmap': 2,
                                               'jwst_1533.pmap': 2},
                                  'cal_ver': {'1.21.0': 4},
                                  'filteroffset_mismatch': []}})
    bad = [v for v in checks.check_headers(run) if v['name'] == 'crds-context-mixed']
    assert bad and bad[0]['severity'] == 'warn'


def test_absent_certifiers_are_unknown_not_passing(tmp_path):
    """A missing certifier key must not read as a pass — that is how a release
    gate gets skipped."""
    from jwst_gc_pipeline.monitoring import paper
    _paper_tree(tmp_path)          # certifiers = {'table': ...} only
    summary = paper.summarize(paper.read_verdicts(str(tmp_path)))
    bad = [v for v in checks.check_paper(_run(target='brick', proposal='2221'),
                                         summary)
           if v['name'] == 'paper-certifiers-absent']
    assert bad and bad[0]['severity'] == 'warn'


# --------------------------------------------------------------------------
# Publishing (hardlink into a web directory)
# --------------------------------------------------------------------------

def test_publish_hardlinks_and_tracks_rewrites(tmp_path):
    """A hardlink is the point: the renderer rewrites in place, so the published
    copy follows every regeneration with no second copy."""
    from jwst_gc_pipeline.monitoring import report
    out = tmp_path / 'out'
    os.makedirs(out / 'fields')
    (out / 'monitor.html').write_text('v1')
    (out / 'monitor.json').write_text('{}')
    (out / 'fields' / 'brick.html').write_text('brick v1')
    web = tmp_path / 'web'

    linked = report.publish(str(out), str(web))
    assert linked['monitor.html'] == 'hard'
    assert linked['fields/brick.html'] == 'hard'
    assert linked['index.html'] == 'hard'
    assert (web / 'index.html').read_text() == 'v1'
    assert os.stat(out / 'monitor.html').st_ino == os.stat(web / 'index.html').st_ino

    # rewrite in place, exactly as render.write_html does
    with open(out / 'monitor.html', 'w') as fh:
        fh.write('v2')
    assert (web / 'monitor.html').read_text() == 'v2'
    assert (web / 'index.html').read_text() == 'v2'


def test_publish_skips_the_body_fragments(tmp_path):
    """*_fragment.html has no doctype/charset/viewport — it is for the artifact
    publisher, not for serving."""
    from jwst_gc_pipeline.monitoring import report
    out = tmp_path / 'out'
    os.makedirs(out)
    (out / 'monitor.html').write_text('page')
    (out / 'monitor_fragment.html').write_text('fragment')
    linked = report.publish(str(out), str(tmp_path / 'web'))
    assert 'monitor.html' in linked
    assert 'monitor_fragment.html' not in linked
    assert not (tmp_path / 'web' / 'monitor_fragment.html').exists()


def test_publish_is_idempotent(tmp_path):
    """Re-running after every generation must be safe — that is what repairs the
    link if the writer ever switches to an atomic rename."""
    from jwst_gc_pipeline.monitoring import report
    out = tmp_path / 'out'
    os.makedirs(out)
    (out / 'monitor.html').write_text('page')
    web = tmp_path / 'web'
    first = report.publish(str(out), str(web))
    second = report.publish(str(out), str(web))
    # Not dict equality: a directory symlink correctly reports 'sym' when it is
    # created and 'same' when it is already correct. The property that matters
    # is that re-running covers the same entries and breaks nothing.
    assert set(first) == set(second)
    assert not [k for k, v in second.items() if v is None]
    assert (web / 'monitor.html').read_text() == 'page'


def test_publish_relinks_after_an_inode_replacing_write(tmp_path):
    """Simulate an atomic write (temp + rename): the old link goes stale, and
    re-publishing is what fixes it."""
    from jwst_gc_pipeline.monitoring import report
    out = tmp_path / 'out'
    os.makedirs(out)
    (out / 'monitor.html').write_text('v1')
    web = tmp_path / 'web'
    report.publish(str(out), str(web))

    tmp = out / 'monitor.html.tmp'
    tmp.write_text('v2')
    os.replace(tmp, out / 'monitor.html')          # new inode
    assert (web / 'monitor.html').read_text() == 'v1'   # stale, as expected

    report.publish(str(out), str(web))
    assert (web / 'monitor.html').read_text() == 'v2'


# --------------------------------------------------------------------------
# Evidence: drawn diagnostics and expandable detail
# --------------------------------------------------------------------------

def test_tile_map_reads_the_cell_key_the_checkpoint_actually_writes():
    """A cell records its offset as `off`; the worst_off_cell summary uses
    `off_mas`. Reading only one silently produces an empty map."""
    from jwst_gc_pipeline.monitoring import figures
    cells = [{'ix': i % 6, 'iy': i // 6, 'off_mas': float(i), 'contrast': 50.0}
             for i in range(36)]
    svg = figures.tile_map_svg(cells, tol_mas=15.0, worst_cell='(5,5)')
    assert svg.startswith('<svg') and svg.count('<rect') == 36
    assert '<circle' in svg              # the worst cell is marked


def test_scan_reads_off_and_off_mas_cell_spellings(tmp_path):
    ckdir = tmp_path / 'astrometry_checkpoints'
    os.makedirs(ckdir)
    (ckdir / 'checkpoint_m2_F212N_latest.json').write_text(json.dumps({
        'stage': 'm2', 'date': '2026-01-01T00:00:00Z',
        'visits': [{'visit': '1', 'consensus': {'consensus_ok': True},
                    'reference_tie': {'off_mas': 0.7, 'per_tile': {
                        'n_ok': 2, 'n_total': 2, 'worst_off_mas': 29.1,
                        'worst_off_cell': {'ix': 0, 'iy': 1, 'off_mas': 29.1},
                        'cells': [{'ix': 0, 'iy': 0, 'off': 3.0},
                                  {'ix': 0, 'iy': 1, 'off': 29.1}]}},
                    'exposures': []}]}))
    rec = scan.astrometry_checkpoints(str(tmp_path))['F212N']
    offs = [c['off_mas'] for c in rec['visits'][0]['cells']]
    assert offs == [3.0, 29.1]


def test_tile_map_empty_cells_does_not_claim_the_field_is_flat():
    """No recorded cells must not read as 'zero cells exceed tolerance'."""
    run = _run(astrometry={'F212N': {
        'n_exposures': 4, 'n_misaligned': 0, 'attributable': True, 'mtime': 1,
        'path': '/x/c.json',
        'visits': [_visit(worst_tile_mas=29.1, cells=[])]}})
    bad = [v for v in checks.check_astrometry(run) if 'worst-tile' in v['name']]
    assert bad
    assert 'not recorded' in bad[0]['cause']
    assert '0/0 cells exceed' not in bad[0]['cause']


def test_misaligned_evidence_names_the_affected_detectors():
    """'183 misaligned' is not actionable; which detectors is."""
    exposures = ([{'visit': '1', 'detector': 'nrca1', 'dra': 30.0, 'ddec': 5.0,
                   'off': 30.4, 'misaligned': True}] * 5
                 + [{'visit': '1', 'detector': 'nrcb2', 'dra': 0.2, 'ddec': 0.1,
                     'off': 0.2, 'misaligned': False}] * 5)
    run = _run(astrometry={'F212N': {
        'n_exposures': 10, 'n_misaligned': 5, 'attributable': True, 'mtime': 1,
        'path': '/x/c.json', 'visits': [_visit()],
        'all_exposures': exposures,
        'misaligned_exposures': [e for e in exposures if e['misaligned']]}})
    v = [x for x in checks.check_astrometry(run)
         if x['name'] == 'astrometry-misaligned-F212N'][0]
    assert 'nrca1' in v['cause']
    assert 'detector-local' in v['cause']
    assert v['evidence']['rows']['total'] == 5
    assert v['evidence']['quiver'].startswith('<svg')


def test_spread_misalignment_is_described_as_the_frame_moving():
    exposures = [{'visit': '1', 'detector': d, 'dra': 30.0, 'ddec': 5.0,
                  'off': 30.4, 'misaligned': True}
                 for d in ('nrca1', 'nrca2', 'nrcb1')]
    run = _run(astrometry={'F212N': {
        'n_exposures': 3, 'n_misaligned': 3, 'attributable': True, 'mtime': 1,
        'path': '/x/c.json', 'visits': [_visit()],
        'all_exposures': exposures, 'misaligned_exposures': exposures}})
    v = [x for x in checks.check_astrometry(run)
         if x['name'] == 'astrometry-misaligned-F212N'][0]
    assert 'frame as a whole moved' in v['cause']


def test_evidence_renders_as_a_disclosure_and_stays_self_contained():
    v = {'name': 'x', 'severity': 'fail', 'summary': 's', 'detail': 'd',
         'source': 'src', 'cause': 'because <b>reasons</b>',
         'evidence': {'tile_map': '<svg><rect/></svg>',
                      'rows': {'columns': ['a'], 'data': [[1]], 'total': 9},
                      'figures': [{'name': 'f.png', 'dir': 'audit_plots'}]}}
    entry = _entry(verdicts=[v])
    html = render.render_page([entry], standalone=True)
    assert '<details' in html and 'what is affected' in html
    assert 'showing 1 of 9' in html
    assert 'because &lt;b&gt;reasons&lt;/b&gt;' in html   # cause is escaped
    assert 'href="figures/f.png"' in html
    for forbidden in ('http://', 'https://', '<link', '@import'):
        assert forbidden not in html


def test_quiver_gives_vectors_only_to_flagged_exposures():
    """Aligned exposures sit within a couple of mas of the origin, so their
    vectors are sub-pixel stubs — but one line+circle+title each is what took a
    192-exposure filter to 92 kB of markup. They are drawn as bare dots."""
    from jwst_gc_pipeline.monitoring import figures
    svg = figures.quiver_svg([
        {'detector': 'nrca1', 'dra': 20.0, 'ddec': 0.0, 'misaligned': True},
        {'detector': 'nrcb1', 'dra': 0.5, 'ddec': 0.5, 'misaligned': False}])
    assert svg.startswith('<svg')
    assert svg.count('<title>') == 1            # only the flagged one
    assert '<path' in svg                       # the aligned dots
    assert 'mas' in svg                         # scale label


def test_quiver_stays_small_on_a_full_filter():
    """The size regression this guards against: 192 exposures must not produce
    tens of kB of markup."""
    from jwst_gc_pipeline.monitoring import figures
    exposures = [{'detector': f'nrc{"ab"[i % 2]}{i % 4 + 1}', 'visit': '1',
                  'dra': 0.4, 'ddec': -0.3, 'misaligned': False}
                 for i in range(192)]
    svg = figures.quiver_svg(exposures)
    assert len(svg) < 12000, len(svg)
    assert svg.count('<title>') == 0


def test_writeup_links_resolve_against_the_served_symlink_name(tmp_path):
    """The served copy carries a `diagnostics-<field>` symlink, so linking by
    that name resolves with no extra publishing step."""
    from jwst_gc_pipeline.monitoring import figures
    d = tmp_path / 'diagnostic_writeup' / 'figures'
    os.makedirs(d)
    (tmp_path / 'diagnostic_writeup' / 'main.pdf').write_text('pdf')
    for stem in ('D1_overview', 'D2_astrometry_internal', 'D3_astrometry_absolute'):
        (d / f'{stem}.pdf').write_text('x')
    wu = figures.writeup(str(tmp_path), 'diagnostics-brick')
    assert wu['main'] == 'diagnostics-brick/main.pdf'
    assert wu['figures']['D3']['href'] == \
        'diagnostics-brick/figures/D3_astrometry_absolute.pdf'
    assert 'D8' not in wu['figures']          # absent figures are not invented


def test_writeup_absent_is_none(tmp_path):
    from jwst_gc_pipeline.monitoring import figures
    assert figures.writeup(str(tmp_path), 'diagnostics-x') is None


@pytest.mark.parametrize('name,code', [
    ('astrometry-worst-tile-F115W-v1', 'D3'),   # longest key wins over 'astrometry'
    ('astrometry-misaligned-F115W', 'D2'),
    ('satstar-all-rejected-F405N', 'D8'),
    ('crds-context-mixed', 'D3'),
    ('unreduced-F150W', 'D1'),
    ('log-error', None),
])
def test_finding_maps_to_the_figure_that_shows_it(name, code):
    from jwst_gc_pipeline.monitoring import figures
    assert figures.figure_for_finding(name) == code


def test_writeup_link_renders_in_the_evidence_block():
    v = {'name': 'astrometry-misaligned-F115W', 'severity': 'fail',
         'summary': 's', 'detail': '', 'source': '', 'cause': '',
         'evidence': {'writeup': {
             'main': 'diagnostics-brick/main.pdf',
             'figure': {'href': 'diagnostics-brick/figures/D2_astrometry_internal.pdf',
                        'name': 'D2_astrometry_internal.pdf',
                        'label': 'internal astrometric repeatability'}}}}
    html = render.render_page([_entry(verdicts=[v])], standalone=True)
    assert 'diagnostics-brick/figures/D2_astrometry_internal.pdf' in html
    assert 'internal astrometric repeatability' in html
    assert 'full diagnostic writeup' in html


# --------------------------------------------------------------------------
# Review fixes
# --------------------------------------------------------------------------

def test_mid_size_log_is_not_blind_between_head_and_tail(tmp_path):
    """The band 8 kB < size <= 8 kB + 400 kB used to be scanned for its first
    8 kB only, with the tail branch skipped entirely -- 60% of the real log
    directory, and where the F150W2 PSF failure sits, so a dead run read green.
    """
    log = tmp_path / 'catalog_m41979-o002-cut5-F150W2_1_0.out'
    filler = 'manual [m12]: fitting frames with 8 parallel workers\n'
    body = (filler * 3000                                   # ~150 kB of noise
            + 'ValueError: Failed to download PSF after 11 attempts\n'
            + filler * 100)
    log.write_text(body)
    size = log.stat().st_size
    assert jobs.HEAD_BYTES < size <= jobs.HEAD_BYTES + jobs.TAIL_BYTES, size
    # the error sits well past the head window
    assert body.index('Failed to download PSF') > jobs.HEAD_BYTES

    got = jobs.scan_log(str(log))
    assert got['worst'] == 'error'
    assert 'psf-build' in got['hits']


def test_huge_log_still_reads_head_and_tail_only(tmp_path):
    """Above the band the file must NOT be slurped whole -- these reach GB.

    The guard is the MIDDLE signature being absent.  An earlier version of this
    test filled the middle with signature-free padding and asserted only that
    the head and tail markers were found, which a full slurp satisfies just as
    well -- so it passed when the branch was replaced with ``if True:``.  That is
    the same blind spot as the fixture that hid the head-only bug: a test that
    exercises the code without constraining the behaviour it is named for.
    """
    log = tmp_path / 'catalog_x_1_0.out'
    pad = 'filler line with no signature at all\n'
    # Each half must EXCEED tail_bytes, or the "middle" marker lands inside the
    # tail window and the test proves nothing.
    half = jobs.TAIL_BYTES // len(pad) + 5000
    log.write_text(
        'CATALOG start: brick\n'
        + pad * half
        + 'Traceback (most recent call last)\n'      # buried in the middle only
        + pad * half
        + 'CATALOG done: filter=F212N rc=0\n')
    size = log.stat().st_size
    assert size > jobs.HEAD_BYTES + jobs.TAIL_BYTES
    marker = len('CATALOG start: brick\n') + len(pad) * half
    assert jobs.HEAD_BYTES < marker < size - jobs.TAIL_BYTES, (
        'fixture does not place the marker in the unread middle')

    got = jobs.scan_log(str(log))
    assert 'start' in got['hits'] and 'done' in got['hits']
    # the middle was never read, so its signature must not appear
    assert 'traceback' not in got['hits'], (
        'the middle of an over-large log was read -- head+tail windowing is gone')


def test_scan_log_read_is_bounded_for_a_file_that_grows_mid_read(tmp_path):
    """The size comes from a prior stat; the read must not trust it forever.

    A log being appended to between the stat and the read would otherwise pull
    in however much arrived, unbounded.
    """
    log = tmp_path / 'catalog_x_2_0.out'
    log.write_text('CATALOG start: brick\n' + 'pad\n' * 100)
    real_getsize = os.path.getsize

    def small_lie(path):
        return 10 if str(path) == str(log) else real_getsize(path)

    import unittest.mock as _mock
    with _mock.patch.object(os.path, 'getsize', small_lie):
        got = jobs.scan_log(str(log))
    # It read at most head+tail regardless of what the file became.
    assert got is not None and got['size'] == 10


def test_logs_pin_on_proposal_not_just_observation():
    """ngc6334 is [('6778','001'), ('7213','001')] -- two proposals, one obsid,
    so pinning on obsid alone puts 6778's failures on 7213's card."""
    log = 'catalog_ngc63346778-o001-cut5-F182M_1_0.out'
    assert jobs.log_belongs_to(log, 'ngc6334', '001', '6778')
    assert not jobs.log_belongs_to(log, 'ngc6334', '001', '7213')


def test_unglobbed_observation_cannot_make_a_filter_ambiguous():
    """wd1 registers o001 and o003 with identical filter lists but globs only
    001; counting o003 flagged all 11 filters on the largest field."""
    assert scan.shared_filters('wd1') == set()
    # the genuinely shared cases still fire
    assert scan.shared_filters('ngc6334') == {'F200W', 'F470N'}
    assert scan.shared_filters('gc2211') == {'F150W', 'F200W', 'F277W'}


def test_unpinned_provenance_is_marked_ambiguous_not_asserted_as_fail():
    """Every other unpinned count is marked ambiguous; this one used to be the
    single unpinned number that still asserted a failure."""
    run = _run(multi_obs=True,
               provenance={'m7': {'tags': {'a': 3, 'b': 2}, 'n_sidecars': 5,
                                  'n_distinct': 2, 'n_dirty': 0,
                                  'scope': 'ambiguous'}})
    bad = [v for v in checks.check_provenance(run) if 'mixed' in v['name']]
    assert bad and bad[0]['severity'] == 'warn'
    assert 'unattributed' in bad[0]['summary']
    # a single-observation field still fails
    run = _run(provenance={'m7': {'tags': {'a': 3, 'b': 2}, 'n_sidecars': 5,
                                  'n_distinct': 2, 'n_dirty': 0, 'scope': 'obs'}})
    assert [v for v in checks.check_provenance(run)
            if 'mixed' in v['name']][0]['severity'] == 'fail'


def test_edge_attribution_uses_the_actual_grid_not_a_hard_coded_six():
    """On a non-6x6 grid a hard-coded edge inverts the edge-vs-interior
    attribution, which is what the evidence block is selling."""
    # 4x4 grid, the over-tolerance cells all on its real edge (ix or iy in 0,3)
    cells = [{'ix': i % 4, 'iy': i // 4, 'off_mas': 2.0} for i in range(16)]
    for c in cells:
        if c['ix'] in (0, 3) or c['iy'] in (0, 3):
            c['off_mas'] = 40.0
    run = _run(astrometry={'F212N': {
        'n_exposures': 4, 'n_misaligned': 0, 'attributable': True, 'mtime': 1,
        'path': '/x/c.json',
        'visits': [_visit(worst_tile_mas=40.0, cells=cells)]}})
    v = [x for x in checks.check_astrometry(run) if 'worst-tile' in x['name']][0]
    assert 'on the mosaic EDGE' in v['cause']


def test_publish_refuses_to_link_a_path_onto_itself(tmp_path):
    """publish(outdir, outdir) used to remove the page and symlink it to itself."""
    from jwst_gc_pipeline.monitoring import report
    out = tmp_path / 'out'
    os.makedirs(out)
    (out / 'monitor.html').write_text('page')
    linked = report.publish(str(out), str(out))
    assert linked['monitor.html'] == 'same'
    assert (out / 'monitor.html').read_text() == 'page'
    assert not os.path.islink(out / 'monitor.html')


def test_no_numeric_gate_literals_in_the_rendered_page_or_checks():
    """The page's footer claims it cannot drift from the gates it reports.

    test_thresholds_are_imported_not_copied asserts identity for three constants
    but does not scan for literals, which is how two inlined paper gates (`> 10`,
    `> 30`) passed CI while the footer said otherwise.
    """
    import io
    import re as _re
    import tokenize
    from jwst_gc_pipeline.monitoring import checks as _checks
    from jwst_gc_pipeline.monitoring import render as _render

    banned = {'10', '30', '15', '5', '0.10', '0.05', '2.0', '100'}
    offenders = []
    # checks.py too: it is where every check lives, and scanning only the
    # renderer let an inlined gate sit in the module that actually gates.
    #
    # Tokenize rather than scan raw lines: message text legitimately QUOTES gate
    # values ("vs anchor > 30 mas"), and a line-based grep cannot tell that from
    # an executable comparison. Dropping STRING and COMMENT tokens leaves only
    # code, which is the thing being constrained.
    for mod in (_render, _checks):
        with open(mod.__file__, 'rb') as fh:
            code = []
            for tok in tokenize.tokenize(fh.readline):
                if tok.type in (tokenize.STRING, tokenize.COMMENT,
                                tokenize.NL, tokenize.NEWLINE):
                    continue
                code.append(tok.string)
        joined = ' '.join(code)
        for m in _re.finditer(r'[<>]\s*=?\s*([\d.]+)\b', joined):
            if m.group(1) in banned:
                start = max(0, m.start() - 60)
                offenders.append(f'{mod.__name__}: …{joined[start:m.end() + 20]}…')
    assert not offenders, ('numeric gate literals in executable code: '
                           + '; '.join(offenders))


def test_paper_gates_are_read_from_the_paper_not_retyped(tmp_path):
    from jwst_gc_pipeline.monitoring import paper
    scripts = tmp_path / 'scripts'
    os.makedirs(scripts)
    (scripts / 'post_recat_validation.py').write_text(
        'MODE_FLIP_TOL_MAS = 7.5\n'
        'if rec["vs_anchor"]["off"] > 25:\n    pass\n')
    got = paper.gate_values(str(tmp_path))
    assert got['mode_flip_tol_mas'] == 7.5
    assert got['anchor_tol_mas'] == 25.0
    assert got['source'] == 'post_recat_validation.py'


def test_paper_gates_fall_back_and_say_so(tmp_path):
    """A page that silently invented a threshold would be worse than one quoting
    the documented value and admitting it."""
    from jwst_gc_pipeline.monitoring import paper
    got = paper.gate_values(str(tmp_path / 'nope'))
    assert got['source'] == 'defaults'
    assert got['mode_flip_tol_mas'] == 10.0


@pytest.mark.parametrize('present,expect', [
    (('m3', 'm4', 'm5', 'm6', 'm7'), ['m12']),
    (('m4', 'm5', 'm6', 'm7'), ['m12', 'm3']),
    (('m7',), ['m12', 'm3', 'm4', 'm5', 'm6']),
    (('m12', 'm4', 'm5', 'm6', 'm7'), ['m3']),
    (('m12', 'm3', 'm4', 'm5', 'm6', 'm7'), []),
])
def test_ladder_gap_sees_a_missing_first_phase(present, expect):
    """Anchoring on the FIRST present phase made the cleanest instance of this
    check -- m12 absent, everything above it present -- silent."""
    rows = {p: {'n': 1 if p in present else 0}
            for p in ('m12', 'm3', 'm4', 'm5', 'm6', 'm7')}
    rows['crf'] = {'n': 0}
    rows['reduced'] = {'n': 0}
    got = [v for v in checks.check_products(_run(per_filter={'F212N': rows}))
           if v['name'] == 'ladder-gap-F212N']
    if not expect:
        assert not got
    else:
        # len, not got[0]: asserting only the first verdict cannot see a second
        # check block left behind beside its replacement.
        assert len(got) == 1, f'{len(got)} ladder-gap verdicts, expected 1'
        assert all(p in got[0]['summary'] for p in expect)


def test_publish_creates_the_diagnostics_symlinks_the_pages_link_to(tmp_path,
                                                                    monkeypatch):
    """The pages emit ~395 `diagnostics-<field>/...` hrefs. They previously
    resolved only in a hand-prepared directory, so a fresh --publish-dir served
    dead links while the docs claimed otherwise."""
    from jwst_gc_pipeline.monitoring import report, scan
    field = tmp_path / 'brick'
    os.makedirs(field / 'diagnostic_writeup' / 'figures')
    (field / 'diagnostic_writeup' / 'main.pdf').write_text('pdf')
    monkeypatch.setattr(scan, 'all_targets', lambda: ['brick'])
    monkeypatch.setattr(scan, 'basepath',
                        lambda t, cutout_label=None: str(field))

    out = tmp_path / 'out'
    os.makedirs(out)
    (out / 'monitor.html').write_text('page')
    web = tmp_path / 'web'
    linked = report.publish(str(out), str(web))

    assert linked['diagnostics-brick'] == 'sym'
    assert os.path.isdir(web / 'diagnostics-brick')
    assert (web / 'diagnostics-brick' / 'main.pdf').read_text() == 'pdf'
    # idempotent: a second run recognises the existing link
    assert report.publish(str(out), str(web))['diagnostics-brick'] == 'same'


# --------------------------------------------------------------------------
# Sky view — defects found by adversarial verification
# --------------------------------------------------------------------------

def _fp(**kw):
    base = {'program': '10678', 'title': 't', 'n_planned': 139, 'n_observed': 0,
            'pa_v3': 87.0, 'pa_v3_range': [79.0, 95.0]}
    base.update(kw)
    return base


#: Two pointings, one of which has no MIRI parallel — enough geometry to give
#: the static map a real bounding box and to tell the two layers apart.
_POINTINGS = [
    {'target': 'GC_1', 'number': '1', 'ra': 266.16, 'dec': -29.47,
     'filters': {'ShortFilter': 'F212N', 'LongFilter': 'F480M'},
     'nircam': [[[266.14, -29.52], [266.14, -29.50],
                 [266.16, -29.50], [266.16, -29.52]]],
     'miri': [[[266.20, -29.40], [266.20, -29.38],
               [266.22, -29.38], [266.22, -29.40]]]},
    {'target': 'GC_2', 'number': '2', 'ra': 266.30, 'dec': -29.30,
     'nircam': [[[266.28, -29.32], [266.28, -29.30],
                 [266.30, -29.30], [266.30, -29.32]]]},
]


def test_hiding_beats_an_author_display():
    """`hidden` alone does not hide an element with an author `display`: origin
    beats specificity, so the old full-bleed overlay stayed over the map and
    swallowed every pointer and wheel event — the view looked loaded but could
    not be panned. The overlay is gone; what replaced it must still be able to
    win that fight, and nothing may reintroduce the overlay."""
    from jwst_gc_pipeline.monitoring import skyview
    assert '.gcm-sky-off { display: none !important; }' in skyview.CSS
    assert 'gcm-sky-msg' not in skyview.CSS
    assert 'gcm-sky-msg' not in skyview.section(_fp())


def test_static_map_needs_no_network():
    """The point of the static map: it renders where fetches are blocked (a
    published artifact, a file:// path, no network). Anything it has to fetch
    defeats it, so the SVG must reference no URL at all."""
    from jwst_gc_pipeline.monitoring import skyview
    svg, info = skyview.static_map(_fp(planned=_POINTINGS))
    assert svg.startswith('<svg')
    assert info['n_nircam'] == 2 and info['n_miri'] == 1
    for token in ('http', 'fetch(', '<script', 'src=', 'url('):
        assert token not in svg


def test_static_map_orientation_is_north_up_east_left():
    """Every other image of this field is shown north-up/east-left; a map that
    silently mirrors the sky would make a real astrometric offset look like the
    wrong sign."""
    from jwst_gc_pipeline.monitoring import skyview
    frame = skyview._frame_for(_fp(planned=_POINTINGS))
    here = frame.xy(frame.ra0, frame.dec0)
    east = frame.xy(frame.ra0 + 0.01, frame.dec0)
    north = frame.xy(frame.ra0, frame.dec0 + 0.01)
    assert east[0] < here[0]            # increasing RA moves left
    assert north[1] < here[1]           # increasing Dec moves up (SVG y down)


def test_static_map_survives_junk_geometry():
    """A footprint file with a truncated vertex or a null coordinate must lose
    that polygon, not the page."""
    from jwst_gc_pipeline.monitoring import skyview
    junk = [{'target': 'x', 'nircam': [[[266.1, -29.1], [266.2]],
                                       [[266.1, None], [266.2, -29.2],
                                        [266.3, -29.3]]],
             'miri': [[[266.1, -29.1], [266.2, -29.2], [266.3, -29.3],
                       [266.4, -29.4]]]}]
    svg, info = skyview.static_map({'planned': junk + _POINTINGS})
    assert info['n_nircam'] == 2            # the junk pointing contributes none
    assert '<svg' in svg


def test_static_map_absent_when_nothing_is_drawable():
    from jwst_gc_pipeline.monitoring import skyview
    assert skyview.static_map({'planned': []}) == ('', {})
    assert skyview.static_map({}) == ('', {})


def test_cross_filesystem_publish_copies_rather_than_symlinks(tmp_path,
                                                              monkeypatch):
    """A symlink whose target leaves the served tree is not a file a web server
    will hand out — Apache returns 403 for every one. That fails silently in the
    worst way: the pages are fine and only the figures they link to vanish."""
    from jwst_gc_pipeline.monitoring import report
    src = tmp_path / 'fig.png'
    src.write_bytes(b'\x89PNG-data')
    dst = tmp_path / 'out' / 'fig.png'
    dst.parent.mkdir()

    def no_hardlinks(a, b):
        raise OSError(18, 'Invalid cross-device link')
    monkeypatch.setattr(report.os, 'link', no_hardlinks)

    assert report._link(str(src), str(dst)) == 'copy'
    assert not dst.is_symlink()
    assert dst.read_bytes() == b'\x89PNG-data'


_DEPLOY_SH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))),
    'scripts', 'monitoring', 'deploy_monitor.sh')


def _run_deploy(tmp_path, dest, extra_env=None):
    """Run deploy_monitor.sh with ssh and rsync replaced by recording stubs."""
    import subprocess
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / 'calls.log'
    for tool in ('ssh', 'rsync'):
        stub = bin_dir / tool
        stub.write_text('#!/bin/bash\necho "%s $*" >> "%s"\nexit 0\n'
                        % (tool, log))
        stub.chmod(0o755)
    src = tmp_path / 'pub'
    src.mkdir(exist_ok=True)
    (src / 'index.html').write_text('<html>')
    env = dict(os.environ, PATH='%s:%s' % (bin_dir, os.environ['PATH']),
               MONITOR_WEB_DIR=dest, MONITOR_WEB_HOST='testhost')
    env.update(extra_env or {})
    done = subprocess.run(['bash', _DEPLOY_SH, str(src)], env=env,
                          capture_output=True, text=True)
    calls = log.read_text() if log.exists() else ''
    return done.returncode, calls, done.stderr


@pytest.mark.parametrize('dest', [
    '/h/x/htdocs/jwst-gc',            # the public data-release landing page
    '/h/x/htdocs',
    '/h/x/htdocs/jwst-gc/monitors',
])
def test_deploy_refuses_a_destination_that_is_not_the_monitor_dir(tmp_path, dest):
    """The guard is the only thing standing between an rsync --delete and the
    public data-release page, and it was pure string logic with no test."""
    rc, calls, err = _run_deploy(tmp_path, dest)
    assert rc == 2
    assert 'rsync' not in calls, 'refused, but rsync ran anyway'
    assert 'refusing' in err


def test_deploy_accepts_the_monitor_dir(tmp_path):
    rc, calls, _err = _run_deploy(tmp_path, '/h/x/htdocs/jwst-gc/monitor')
    assert rc == 0
    assert 'rsync' in calls and '--copy-links' in calls and '--delete' in calls


def test_deploy_reports_a_missing_page_set_above_the_findings_code(tmp_path):
    """refresh_monitor.sh reads 1 as 'the archive has failing runs'. A deploy
    failure is not that, so it must not land in the same bucket."""
    import subprocess
    empty = tmp_path / 'empty'
    empty.mkdir()
    done = subprocess.run(
        ['bash', _DEPLOY_SH, str(empty)],
        env=dict(os.environ, MONITOR_WEB_DIR='/h/x/htdocs/jwst-gc/monitor'),
        capture_output=True, text=True)
    assert done.returncode > 1
    assert 'index.html' in done.stderr


def _publish_a_page_set(tmp_path, monkeypatch, entry):
    """A written-out page set plus its publish directory, as a reader gets it."""
    from jwst_gc_pipeline.monitoring import report
    monkeypatch.setattr(report, 'build_entries', lambda *a, **kw: ([entry], []))
    monkeypatch.setattr(report, 'collect_cutouts', lambda *a, **kw: [])
    out = tmp_path / 'out'
    web = tmp_path / 'web'
    report.write_report(outdir=str(out), per_field=True)
    # The assets the pages link to, made real so a resolving href can be told
    # from a merely well-formed one.
    (out / 'figures').mkdir(exist_ok=True)
    (out / 'figures' / 'fig.png').write_bytes(b'PNG')
    wu = tmp_path / 'writeup'
    (wu / 'figures').mkdir(parents=True)
    (wu / 'main.pdf').write_bytes(b'%PDF')
    (wu / 'figures' / 'D2_astrometry_internal.pdf').write_bytes(b'%PDF')
    monkeypatch.setattr(report.scan, 'all_targets', lambda: ['brick'])
    monkeypatch.setattr(report.scan, 'basepath', lambda t: str(tmp_path))
    monkeypatch.setattr(report._figures, 'WRITEUP_DIR', 'writeup')
    report.publish(str(out), str(web))
    return web


def _dead_links(root):
    """Every relative href/src on every page, resolved against ITS OWN dir."""
    dead = []
    for base, _dirs, files in os.walk(root):
        for name in files:
            if not name.endswith('.html'):
                continue
            page = os.path.join(base, name)
            with open(page) as fh:
                body = fh.read()
            for href in set(re.findall(r'(?:href|src)="([^"]+)"', body)):
                if (href.startswith(('http://', 'https://', 'mailto:', 'data:',
                                     '#')) or not href.strip()):
                    continue
                target = os.path.normpath(
                    os.path.join(base, href.split('#')[0]))
                if not os.path.exists(target):
                    dead.append((os.path.relpath(page, root), href))
    return dead


def test_no_page_links_to_a_file_that_is_not_there(tmp_path, monkeypatch):
    """Walk every href on every published page and stat it **relative to that
    page's own directory**.

    This is the check that had already missed the same bug twice.  Both times
    the links were well-formed and pointed one directory too high: the per-field
    pages sit in ``fields/`` and emitted ``figures/...`` and
    ``diagnostics-<target>/...``, which are correct only at the publish root.
    Asserting that the *symlink exists* — which the older test did — cannot see
    it, because the symlink was fine and the href never reached it.
    """
    from jwst_gc_pipeline.monitoring import figures as _figures
    entry = _entry(verdicts=[{
        'name': 'astrometry.exposure_consensus', 'severity': 'fail',
        'summary': 's', 'detail': 'd', 'value': 1, 'threshold': 2,
        'source': 'x',
        'evidence': {
            'figures': [{'name': 'fig.png', 'dir': '/somewhere',
                         'path': '/somewhere/fig.png', 'filter': 'F200W'}],
            'writeup': {
                'main': 'diagnostics-brick/main.pdf',
                'figure': {'name': 'D2_astrometry_internal.pdf',
                           'label': 'astrometry internal',
                           'href': 'diagnostics-brick/figures/'
                                   'D2_astrometry_internal.pdf'}}}}])
    web = _publish_a_page_set(tmp_path, monkeypatch, entry)

    # The fixture has to actually exercise both kinds of link, or a green run
    # would only mean the page had nothing to get wrong.
    pages = [p for p in os.listdir(os.path.join(str(web), 'fields'))]
    assert pages, 'no per-field page was written'
    field_html = open(os.path.join(str(web), 'fields', pages[0])).read()
    assert 'figures/fig.png' in field_html
    assert 'diagnostics-brick/main.pdf' in field_html

    assert _dead_links(str(web)) == []


def test_fragment_stays_whole_while_the_page_set_splits(tmp_path, monkeypatch):
    """The fragment is published as a single-file artifact: `fields/x.html` is
    not a document that exists there, so a card linking to one would 404."""
    from jwst_gc_pipeline.monitoring import report
    monkeypatch.setattr(report, 'build_entries',
                        lambda *a, **kw: ([_entry()], []))
    monkeypatch.setattr(report, 'collect_cutouts', lambda *a, **kw: [])
    out = report.write_report(outdir=str(tmp_path), per_field=True)

    aggregate = open(out['aggregate']).read()
    fragment = open(out['fragment']).read()
    assert 'href="fields/' in aggregate and '<h2>Detail</h2>' not in aggregate
    assert 'href="fields/' not in fragment and '<h2>Detail</h2>' in fragment


def test_an_existing_symlink_is_replaced_by_a_copy(tmp_path, monkeypatch):
    """A symlink to src resolves to src, so the 'same file, nothing to do' guard
    matched it and left it in place — the switch from symlinking to copying then
    never converted the symlinks already on disk, and they kept 403-ing."""
    from jwst_gc_pipeline.monitoring import report
    src = tmp_path / 'fig.png'
    src.write_bytes(b'PNG')
    dst = tmp_path / 'out' / 'fig.png'
    dst.parent.mkdir()
    dst.symlink_to(src)
    monkeypatch.setattr(report.os, 'link',
                        lambda a, b: (_ for _ in ()).throw(OSError(18, 'xdev')))

    assert report._link(str(src), str(dst)) == 'copy'
    assert not dst.is_symlink()
    assert dst.read_bytes() == b'PNG'


def test_publishing_onto_itself_is_still_a_no_op(tmp_path):
    """The guard that rewrite protects: a one-character scrontab typo pointing
    --publish-dir at --outdir must not remove the report and link it to itself."""
    from jwst_gc_pipeline.monitoring import report
    src = tmp_path / 'monitor.html'
    src.write_text('page')
    assert report._link(str(src), str(src)) == 'same'
    assert src.read_text() == 'page'


class _Stat(object):
    def __init__(self, dev, size, mtime):
        self.st_dev, self.st_size, self.st_mtime = dev, size, mtime


def test_unchanged_copy_is_not_recopied():
    """~110 MB of figures re-copied on an hourly refresh would be most of the
    job's cost."""
    from jwst_gc_pipeline.monitoring import report
    same = _Stat(7, 100, 1000.0)
    assert report._copy_is_current(same, _Stat(9, 100, 1000.0), 9)


@pytest.mark.parametrize('dst_stat,dst_dev,why', [
    (_Stat(7, 100, 1000.0), 7, 'same filesystem — relink instead'),
    (_Stat(9, 101, 1000.0), 9, 'size changed'),
    (_Stat(9, 100, 1001.0), 9, 'mtime changed'),
])
def test_copy_is_refreshed_when_it_must_be(dst_stat, dst_dev, why):
    """The same-filesystem case is the important one: skipping the relink there
    is exactly the stale-page bug publish() exists to prevent."""
    from jwst_gc_pipeline.monitoring import report
    assert not report._copy_is_current(_Stat(7, 100, 1000.0), dst_stat, dst_dev), why


_ROMAN = {
    'tiles': {'Tile 1': {'spring': [[[266.5, -29.6], [266.6, -29.6],
                                     [266.6, -29.5], [266.5, -29.5]]],
                         'autumn': [[[266.7, -29.6], [266.8, -29.6],
                                     [266.8, -29.5], [266.7, -29.5]]]}},
    'target_area': [[266.4, -29.7], [266.9, -29.7], [266.9, -29.2]],
}


def test_roman_toggles_are_not_dead_without_aladin():
    """The Roman buttons used to map to empty static groups, so before anyone
    loaded 1.8 MB of Aladin a click toggled the button's own styling and nothing
    else — a live-looking control that did nothing."""
    from jwst_gc_pipeline.monitoring import skyview
    svg, info = skyview.static_map(_fp(planned=_POINTINGS), _ROMAN)
    assert info['n_roman'] == 3                    # 1 spring + 1 autumn + area
    for group in ('stat-spring', 'stat-autumn', 'stat-target'):
        assert 'id="%s"' % group in svg
    html = skyview.section(_fp(planned=_POINTINGS), _ROMAN)
    for group in ('stat-spring', 'stat-autumn', 'stat-target'):
        assert "'%s'" % group in html              # reachable from STATIC_GROUPS
    # and the buttons must not be inert
    spring = html[html.index('id="lyr-spring"'):html.index('id="lyr-autumn"')]
    assert 'disabled' not in spring


def test_the_emitted_javascript_parses():
    """~250 lines of JavaScript are emitted from a Python f-string, where every
    brace is doubled. A slip there produces a page that renders perfectly and
    whose every control is dead, which no assertion about the markup can see."""
    import shutil
    import subprocess
    node = shutil.which('node')
    if not node:
        pytest.skip('node not available')
    from jwst_gc_pipeline.monitoring import skyview
    html = skyview.section(_fp(planned=_POINTINGS), _ROMAN)
    script = re.search(r'<script>(.*)</script>', html, re.S).group(1)
    done = subprocess.run([node, '--check', '-'], input=script,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_the_view_opens_on_an_all_sky_background():
    """The survey's own CMZ HiPS declares hips_initial_fov = 0.077 deg and has
    no low-order tiles, so opening on it at the panel's 1.6 deg field drew a
    black rectangle — the imagery worked exactly as configured and showed
    nothing. Whatever leads this list has to cover the whole sky."""
    from jwst_gc_pipeline.monitoring import skyview
    first_url = skyview.SURVEYS[0][1]
    # A HiPS named by ID must be one that exists; `P/Spitzer/GLIMPSE360` matched
    # nothing at the CDS MOCServer, so that button selected a survey that has
    # never been served under that name and simply did nothing.
    assert all(len(entry) == 3 for entry in skyview.SURVEYS)
    assert 'P/Spitzer/GLIMPSE360' not in [u for _n, u, _t in skyview.SURVEYS]
    assert not first_url.startswith('http'), (
        'the opening background is a specific HiPS, not an all-sky survey')
    assert any(url.startswith('http') for _n, url, _note in skyview.SURVEYS), (
        'the survey imagery should still be reachable, just not as the default')


def test_background_buttons_work_before_the_viewer_exists():
    """They were disabled until someone loaded the interactive view, so the row
    every reader sees first was a row of dead controls — and the background they
    pick is exactly the reason to load that view."""
    from jwst_gc_pipeline.monitoring import skyview
    html = skyview.section(_fp(planned=_POINTINGS))
    row = html[html.index('Background'):html.index('Legend')]
    assert 'survey' in row
    assert 'disabled' not in row, 'background buttons are inert on arrival'
    # Clicking one before the viewer exists must remember the choice and load.
    assert 'function applySurvey(' in html
    assert 'wanted = target;' in html
    assert 'loadInteractive();' in html
    # ...and the load must honour it rather than opening on the default.
    assert 'survey: wanted ||' in html


def test_roman_layers_start_hidden():
    """Roman is context on a JWST monitor: drawn, but off until asked for."""
    from jwst_gc_pipeline.monitoring import skyview
    svg, _ = skyview.static_map(_fp(planned=_POINTINGS), _ROMAN)
    for group in ('stat-spring', 'stat-autumn', 'stat-target'):
        block = svg[svg.index('id="%s"' % group):]
        assert 'gcm-sky-off' in block[:block.index('>')]
    for group in ('stat-nircam', 'stat-miri'):
        block = svg[svg.index('id="%s"' % group):]
        assert 'gcm-sky-off' not in block[:block.index('>')]


def test_zoom_out_reaches_the_roman_tiles():
    """Roman's tiles lie outside the JWST bounding box that sets the home view,
    so a 3x zoom-out ceiling let you enable a layer you could never see."""
    from jwst_gc_pipeline.monitoring import skyview
    with_roman = skyview.section(_fp(planned=_POINTINGS), _ROMAN)
    without = skyview.section(_fp(planned=_POINTINGS))
    assert 'var ZOOM_OUT_MAX = 8;' in with_roman
    assert 'var ZOOM_OUT_MAX = 3;' in without


def test_map_is_drawn_in_galactic_coordinates():
    """A Galactic-centre survey read in RA/Dec runs diagonally across the grid;
    in l/b the tiled strip runs along it."""
    from jwst_gc_pipeline.monitoring import skyview
    pytest.importorskip('astropy')
    svg, info = skyview.static_map(_fp(planned=_POINTINGS))
    assert info['frame'] == 'galactic'
    assert 'l ' in svg and 'b ' in svg           # native graticule labels
    assert 'RA ' in svg and 'Dec ' in svg        # equatorial overlay labels
    assert 'Galactic longitude increasing left' in svg


def test_galactic_frame_reshapes_the_map():
    """The survey tiles along the plane, so the drawn aspect ratio is a check
    that the transform actually happened rather than silently no-opping."""
    from jwst_gc_pipeline.monitoring import skyview
    pytest.importorskip('astropy')
    fp = _fp(planned=_POINTINGS)
    gal = skyview.static_map(fp)[1]
    icrs = skyview.static_map(fp, frame_name='icrs')[1]
    assert icrs['frame'] == 'icrs'
    assert abs(gal['height'] - icrs['height']) > 1.0


def test_frame_falls_back_to_icrs_without_astropy(monkeypatch):
    """Losing astropy must cost the Galactic frame, not the map."""
    from jwst_gc_pipeline.monitoring import skyview
    monkeypatch.setattr(skyview, '_transform',
                        lambda pairs, src, dst: list(pairs) if src == dst else None)
    svg, info = skyview.static_map(_fp(planned=_POINTINGS), _ROMAN)
    assert info['frame'] == 'icrs'
    assert info['overlay_grid'] is False         # no second graticule
    assert info['n_nircam'] == 2                 # but the footprints are drawn
    assert '<svg' in svg


def test_front_page_leads_with_the_map_then_the_cards():
    """Requested layout: the visual overview first, then the status boxes."""
    from jwst_gc_pipeline.monitoring import render
    html = render.render_page([_entry()], footprints=_fp(planned=_POINTINGS),
                              include_detail=False,
                              detail_href=lambda e: 'fields/x.html')
    assert html.index('Sky view') < html.index('id="overview"')
    assert '<h2>Detail</h2>' not in html
    assert 'href="fields/x.html"' in html


def test_field_pages_carry_the_detail_and_not_the_map():
    """The map is ~100 kB of identical geometry; 18 copies of it is the cost of
    putting it on every field page."""
    from jwst_gc_pipeline.monitoring import render
    html = render.render_page([_entry()], footprints=_fp(planned=_POINTINGS),
                              include_skyview=False,
                              home_href='../monitor.html#overview')
    assert '<svg id="gcm-sky-static"' not in html
    assert '<h2>Detail</h2>' in html
    assert 'href="../monitor.html#overview"' in html


def test_layer_toggles_drive_both_views():
    """One set of buttons drives the static groups and, after the upgrade, the
    Aladin overlays. If the two used different ids the toggles would silently
    stop working the moment the interactive view loaded."""
    from jwst_gc_pipeline.monitoring import skyview
    html = skyview.section(_fp(planned=_POINTINGS))
    for button, group in (('lyr-nircam', 'stat-nircam'),
                          ('lyr-miri', 'stat-miri'),
                          ('lyr-observed', 'stat-obs-nircam')):
        assert 'id="%s"' % button in html
        assert 'id="%s"' % group in html
    assert 'STATIC_GROUPS' in html


def test_the_map_is_hidden_only_once_there_is_a_view_to_replace_it():
    """Hiding the map before `A.init` meant any failure in between left a black
    box. Worse, a synchronous throw there — `A` undefined, or `A.init` not a
    thenable because the script was truncated — happens while *evaluating*
    `A.init.then`, so it escapes the inner handler entirely; checking only the
    text after `function start(` could not see that."""
    from jwst_gc_pipeline.monitoring import skyview
    html = skyview.section(_fp(planned=_POINTINGS))
    start = html.index('function start(')
    body = html[start:]
    # The hide must sit after A.init resolves, not before it is even called.
    assert body.index("A.init") < body.index("svg.classList.add('gcm-sky-off')")

    # Both failure paths restore, and the outer one is *outside* start().
    assert html.count('function restore()') == 1
    assert body.count('restore();') >= 1                  # inner .catch
    outer = html[:start]
    assert '.then(start).catch(' in outer
    assert outer.index('.then(start).catch(') < outer.index('restore();') \
        if 'restore();' in outer else True
    assert 'restore();' in outer, 'the outer handler must restore too'


def test_wheel_zoom_does_not_hijack_page_scroll_unarmed():
    """A page-long report that eats the wheel whenever the cursor crosses a
    figure is worse than a map that needs one click first."""
    from jwst_gc_pipeline.monitoring import skyview
    html = skyview.section(_fp(planned=_POINTINGS))
    assert 'if (!armed) { return; }' in html
    assert html.index('if (!armed) { return; }') < html.index('ev.preventDefault()',
                                                              html.index('wheel'))


@pytest.mark.parametrize('bad', [
    {'pa_v3': None},
    {'pa_v3': '87'},
    {'pa_v3_range': [79.0]},
    {'pa_v3_range': None},
    {'n_planned': None},
    {'n_observed': 'lots'},
    {'pa_v3': float('nan')},
])
def test_schema_drift_cannot_take_the_page_down(bad):
    """render_page calls the sky view unguarded, so a formatting error there
    loses monitor.html — every pipeline check discarded for a decorative panel."""
    from jwst_gc_pipeline.monitoring import skyview
    html = skyview.section(_fp(**bad))
    assert 'gcm-sky' in html
    # and the whole page still renders
    page = render.render_page([_entry()], standalone=True, footprints=_fp(**bad))
    assert '</html>' in page


def test_roman_geometry_is_fetched_not_inlined():
    """All three Roman layers are off by default; inlining ~25 kB charges every
    reader for something almost nobody turns on."""
    from jwst_gc_pipeline.monitoring import skyview
    roman = {'tiles': {'Tile 1': {'spring': [[[1, 2]]] * 18, 'autumn': []}}}
    html = skyview.section(_fp(), roman=roman)
    assert 'roman_gbtds.json' in html
    assert '"Tile 1"' not in html          # not embedded


def test_embedded_json_cannot_close_the_script_element():
    """A '</script>' inside embedded data ends the block early: the whole IIFE
    never runs, so the button is inert with no message — the black-box outcome
    the module exists to avoid."""
    from jwst_gc_pipeline.monitoring import skyview
    html = skyview.section(_fp(title='a</script><h1>PWNED</h1>'))
    assert '<h1>PWNED</h1>' not in html
    # the script block that defines the loader must survive intact
    assert html.count('</script>') == 1


def test_aladin_start_failure_is_reported():
    """A.init rejects on a browser without WebGL2, and throws a BARE STRING, so
    e.message is undefined for the most likely real failure."""
    from jwst_gc_pipeline.monitoring import skyview
    html = skyview.section(_fp())
    assert '.catch(' in html
    assert 'describe(' in html
    assert "typeof e === 'string'" in html


def test_per_field_pages_reference_assets_one_level_up():
    """Per-field pages live in fields/; without the prefix they told the reader
    to generate a file that was already sitting beside them."""
    from jwst_gc_pipeline.monitoring import skyview
    html = skyview.section(_fp(), aladin_src='../aladin.js',
                           data_url='../footprints.json',
                           roman_url='../roman_gbtds.json')
    assert '"../footprints.json"' in html
    assert '"../aladin.js"' in html


def test_page_states_the_pa_uncertainty_and_the_dither():
    """'indicative' was too weak: across the allowed range the MIRI parallel
    moves ~125", and only the nominal dither position is drawn."""
    from jwst_gc_pipeline.monitoring import skyview
    html = skyview.section(_fp(dither={'PrimaryDitherType': ['FULLBOX'],
                                       'PrimaryDithers': ['6TIGHT']}))
    assert '125' in html and '180' in html
    assert 'FULLBOX' in html and 'nominal position is drawn' in html
