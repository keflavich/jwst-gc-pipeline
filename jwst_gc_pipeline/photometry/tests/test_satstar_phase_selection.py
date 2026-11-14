"""One cataloging phase's satstar catalog per exposure.

Every cataloging phase re-fits every exposure and leaves its
``..._m<N>_satstar_catalog.fits`` behind, so a re-cataloged tree holds one per
phase per exposure, each from whichever code last wrote that phase.
``load_satstar_catalog`` read all of them and the brightest-first dedup kept
the brightest.  On gc-treasury o111 F480M (an m7 re-run with the #972 fixes,
older m12..m6 catalogs left in place) 387 of 1120 consolidated rows kept an
old m12 fit, +2.5% median / +7.4% p90 in flux against a run with no old-phase
files.

These tests pin the fix: each exposure contributes the phase being merged (or
its latest earlier phase), never a later one and never several; with no phase
named, its latest; the consolidated cache is keyed on that selection; the
gate-rejected channel follows the accepted channel's choice; and every merge
entry point forwards its iteration label.  ``SATSTAR_POOL_PHASES=all``
restores the pooled read.
"""
import os
import time
import types

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry import merge_catalogs as MC
from jwst_gc_pipeline.photometry import satstar_phase_selection as SPS

PIXSCALE = 63.0 / 3600.0 / 1000.0   # NIRCam LW, deg/px
SHAPE = (128, 128)
RA0, DEC0 = 266.54, -28.71
STAR_A = (30.0, 40.0)
STAR_B = (90.0, 100.0)

#: Seconds before "now" at which each phase's files were written: the stale
#: m12 is oldest, as in the o111 re-run.
_AGE = {'_m12': 400, '_m4': 300, '_resbgsub_m5': 200, '_resbgsub_m6': 150,
        '_resbgsub_m7': 100}


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [64.5, 64.5]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.cdelt = [-PIXSCALE, PIXSCALE]
    return w


def _write_frame(path):
    hdu1 = fits.ImageHDU(np.zeros(SHAPE, dtype='float32'), name='SCI')
    hdu1.header.update(_wcs().to_header(relax=True))
    fits.HDUList([fits.PrimaryHDU(), hdu1]).writeto(path, overwrite=True)


def _write_product(frame, tail, stars, kind='catalog', extra=None):
    """One per-exposure satstar catalog (or rejected file) beside ``frame``.

    ``stars`` is a list of ``(x, y, flux)``; ``tail`` the run's file suffix
    (``'_m12'``, ``'_resbgsub_m7'``); ``extra`` further columns by name."""
    x = np.array([s[0] for s in stars], float)
    y = np.array([s[1] for s in stars], float)
    tbl = Table({'xcentroid': x, 'ycentroid': y,
                 'flux_fit': np.array([s[2] for s in stars], float),
                 'flux_err': np.full(len(x), 10.0),
                 'x_fit': x, 'y_fit': y,
                 'x_err': np.full(len(x), 0.01), 'y_err': np.full(len(x), 0.01),
                 'qfit': np.full(len(x), 0.1)})
    tbl['skycoord_fit'] = _wcs().pixel_to_world(x, y)
    for name, values in (extra or {}).items():
        tbl[name] = np.asarray(values, dtype=float)
    if kind == 'rejected':
        tbl['reject_reason'] = ['implied_peak_gate'] * len(x)
    path = str(frame).replace('.fits', f'{tail}_satstar_{kind}.fits')
    tbl.write(path, overwrite=True)
    stamp = time.time() - _AGE.get(tail, 50)
    os.utime(path, (stamp, stamp))
    return path


def _frame(pdir, expo):
    frame = pdir / f'jw02221001001_02101_0000{expo}_nrca1_o001_crf.fits'
    _write_frame(str(frame))
    return frame


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv(SPS.POOL_PHASES_ENV, raising=False)
    monkeypatch.setenv('SATSTAR_APERTURE_PHOT', '0')
    for var in ('SATSTAR_DEDUP_ARCSEC', 'SATSTAR_ENSEMBLE_POSITION',
                'SATSTAR_FP_USE_ANCHOR'):
        monkeypatch.delenv(var, raising=False)
    # these tests identify the selected phase by its flux; keep flux_fit on
    # the representative fit so the median over exposures (#1032) does not
    # blend the phases being told apart
    monkeypatch.setenv('SATSTAR_FLUX_STAT', 'brightest')


@pytest.fixture
def phase_tree(tmp_path, clean_env):
    """A re-cataloged single-observation tree (brick-shaped, unscoped).

    Exposure 1 was fit at m12 by OLD code (star A too bright, 9000), then at
    m4 (5500) and m7 (5000).  Exposure 2 has m4 and m5 only (A 4900, B 3000):
    an exposure whose latest phase is earlier than the others'.
    """
    pdir = tmp_path / 'F182M' / 'pipeline'
    pdir.mkdir(parents=True)
    (tmp_path / 'catalogs').mkdir()
    e1, e2 = _frame(pdir, 1), _frame(pdir, 2)
    _write_product(e1, '_m12', [(*STAR_A, 9000.0)])
    _write_product(e1, '_m4', [(*STAR_A, 5500.0)])
    _write_product(e1, '_resbgsub_m7', [(*STAR_A, 5000.0)])
    _write_product(e2, '_m4', [(*STAR_A, 4900.0), (*STAR_B, 3000.0)])
    _write_product(e2, '_resbgsub_m5', [(*STAR_A, 4900.0), (*STAR_B, 3000.0)])
    return tmp_path


def _row_at(tbl, xy):
    """The consolidated row at pixel ``xy`` (on the shared synthetic WCS)."""
    target = _wcs().pixel_to_world(*xy)
    sep = SkyCoord(tbl['skycoord_fit']).separation(target).to_value(u.arcsec)
    near = np.where(sep < 0.01)[0]
    return tbl[int(near[0])] if len(near) == 1 else None


def _load(base, **kw):
    return MC.load_satstar_catalog('f182m', target='brick',
                                   basepath=str(base) + '/', **kw)


def _cache(base):
    return base / 'catalogs' / 'f182m_consolidated_satstar_catalog.fits'


# --- the phase order -----------------------------------------------------

def test_phase_rank_follows_the_pipeline_order():
    order = ['m12', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8']
    ranks = [SPS.satstar_phase_rank(t) for t in order]
    assert ranks == sorted(ranks) and len(set(ranks)) == len(ranks)
    # the first phase writes m1 and m2 and names its satstar catalogs m12, so
    # a merge labelled m1 or m2 reads the m12 catalogs
    assert (SPS.satstar_phase_rank('m1') == SPS.satstar_phase_rank('m2')
            == SPS.satstar_phase_rank('m12'))
    # background-token variants and a leading underscore share the rank
    for label in ('_m7', 'resbgsub_m7', '_resbgsub_m7', 'bgsub_resbgsub_m7'):
        assert SPS.satstar_phase_rank(label) == SPS.satstar_phase_rank('m7')


@pytest.mark.parametrize('label', [None, '', 'iter2', 'iter3', 'final'])
def test_labels_that_name_no_phase(label):
    assert SPS.satstar_phase_rank(label) is None


def test_group_key_is_the_image_the_ensemble_counts():
    names = ['jw02221001001_02101_00001_nrca1_o001_crf_m12_satstar_catalog.fits',
             'jw02221001001_02101_00001_nrca1_o001_crf_resbgsub_m7_satstar_catalog.fits',
             'jw02221001001_02101_00001_nrca1_destreak_o001_crf_m4_satstar_catalog.fits',
             'jw02221001001_02101_00001_nrca1_align_o001_crf_m3_satstar_rejected.fits']
    keys = {SPS.satstar_phase_group_key(n) for n in names}
    assert keys == {MC.satstar_exposure_key(names[0])}
    assert (SPS.satstar_phase_group_key(
        'jw02221001001_02101_00002_nrca1_o001_crf_m7_satstar_catalog.fits')
        not in keys)
    # a non-observatory name is grouped by its frame stem, background and
    # phase tokens removed
    assert (SPS.satstar_phase_group_key('cut_crf_m3_satstar_catalog.fits')
            == SPS.satstar_phase_group_key('cut_crf_resbgsub_m6_satstar_catalog.fits')
            == 'cut_crf')
    assert (SPS.satstar_phase_group_key('cut_alpha_m7_satstar_catalog.fits')
            != SPS.satstar_phase_group_key('cut_beta_m7_satstar_catalog.fits'))


# --- the file selection --------------------------------------------------

def _touch(path, age):
    path.write_bytes(b'')
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))
    return str(path)


@pytest.fixture
def names(tmp_path, clean_env):
    """Empty per-exposure files: exposure 1 at every phase, exposure 2 at
    m12..m5 only."""
    out = {}
    for expo, tails in ((1, ('m12', 'm3', 'm4', 'resbgsub_m5', 'resbgsub_m6',
                             'resbgsub_m7')),
                        (2, ('m12', 'm3', 'm4', 'resbgsub_m5'))):
        for age, tail in enumerate(reversed(tails)):
            out[(expo, tail.split('_')[-1])] = _touch(
                tmp_path / (f'jw02221001001_02101_0000{expo}_nrca1_o001_crf'
                            f'_{tail}_satstar_catalog.fits'), 100 + 10 * age)
    return out


def test_latest_phase_per_exposure_when_no_phase_is_named(names):
    sel, rep = SPS.select_satstar_phase_files(names.values())
    assert sel == sorted([names[(1, 'm7')], names[(2, 'm5')]])
    assert dict(rep['used']) == {'m7': 1, 'm5': 1}
    assert sum(rep['superseded'].values()) == len(names) - 2
    assert not rep['ahead']


def test_explicit_phase_never_reads_a_later_phase(names):
    sel, rep = SPS.select_satstar_phase_files(names.values(), phase='m4')
    assert sel == sorted([names[(1, 'm4')], names[(2, 'm4')]])
    assert dict(rep['ahead']) == {'m5': 2, 'm6': 1, 'm7': 1}
    # the first phase's merge label is m2; its catalogs are m12
    sel, _ = SPS.select_satstar_phase_files(names.values(), phase='m2')
    assert sel == sorted([names[(1, 'm12')], names[(2, 'm12')]])


def test_exposure_without_the_merged_phase_uses_its_latest_earlier_one(names):
    sel, rep = SPS.select_satstar_phase_files(names.values(), phase='m7')
    assert sel == sorted([names[(1, 'm7')], names[(2, 'm5')]])
    assert dict(rep['used']) == {'m7': 1, 'm5': 1}


def test_a_label_that_names_no_phase_reads_the_latest(names):
    assert (SPS.select_satstar_phase_files(names.values(), phase='iter3')[0]
            == SPS.select_satstar_phase_files(names.values())[0])


@pytest.mark.parametrize('newer', ['align', 'destreak'])
def test_same_phase_tie_goes_to_the_newest_file(tmp_path, clean_env, newer):
    """Two frame variants (``_align_`` / ``_destreak_``) of one exposure, both
    fit at m6: the most recently written fit is the current run's.  Both age
    orders are checked, so the result cannot come from the name tie-break
    ('destreak' sorts after 'align') instead of the mtime."""
    paths = {
        'align': tmp_path / ('jw05365001001_09101_00001_nrcalong_align_o001_crf'
                             '_resbgsub_m6_satstar_catalog.fits'),
        'destreak': tmp_path / ('jw05365001001_09101_00001_nrcalong_destreak_o001'
                                '_crf_resbgsub_m6_satstar_catalog.fits'),
    }
    older = 'destreak' if newer == 'align' else 'align'
    new = _touch(paths[newer], 10)
    old = _touch(paths[older], 1000)
    sel, rep = SPS.select_satstar_phase_files([old, new], phase='m6')
    assert sel == [new]
    assert dict(rep['superseded']) == {'m6': 1}
    assert rep['tie_losers'] == [(os.path.basename(new), os.path.basename(old))]
    line = SPS.format_phase_report(rep)
    assert '1 same-phase tie(s) went to the newer file' in line
    assert f'{os.path.basename(new)} over {os.path.basename(old)}' in line


def test_a_superseded_earlier_phase_is_not_a_tie(names):
    """Only same-phase losers are named; an earlier phase is plain superseded."""
    _, rep = SPS.select_satstar_phase_files(names.values(), phase='m4')
    assert rep['tie_losers'] == []
    assert 'same-phase tie' not in SPS.format_phase_report(rep)


def test_pool_all_env_restores_the_pooled_read(names, monkeypatch):
    monkeypatch.setenv(SPS.POOL_PHASES_ENV, 'all')
    sel, rep = SPS.select_satstar_phase_files(names.values(), phase='m4')
    assert sel == sorted(names.values())
    assert rep['mode'] == 'all'
    assert 'SATSTAR_POOL_PHASES=all' in SPS.format_phase_report(rep)


def test_the_report_names_used_superseded_and_later_phases(names):
    _, rep = SPS.select_satstar_phase_files(names.values(), phase='m4')
    line = SPS.format_phase_report(rep)
    assert 'merging m4' in line
    assert 'using 2 satstar catalog file(s) [m4: 2]' in line
    assert 'excluded 4 superseded [m12: 2, m3: 2]' in line
    assert '4 from a later phase [m5: 2, m6: 1, m7: 1]' in line


# --- the consolidated catalog --------------------------------------------

def test_consolidated_keeps_the_current_fit_not_a_brighter_stale_one(phase_tree):
    """The defect: pooled, the stale m12 fit (9000) wins the brightest-first
    dedup over the m7 fit (5000) of the same exposure."""
    out = _load(phase_tree)
    star = _row_at(out, STAR_A)
    assert star['flux_fit'] == pytest.approx(5000.0)
    assert star[MC._SATSTAR_ITER_COL] == 'm7'
    # one measurement per exposure: exposure 1's m7, exposure 2's m5
    assert star['n_frames_fit'] == 2 and star['n_meas_fit'] == 2
    assert star['flux_med_fit'] == pytest.approx(4950.0)


def test_default_flux_is_the_median_of_the_current_fits(phase_tree,
                                                       monkeypatch):
    """Default SATSTAR_FLUX_STAT: the median over exposures of each
    exposure's current-phase fit (5000 and 4900), with the stale m12 9000
    excluded by the phase selection."""
    monkeypatch.delenv('SATSTAR_FLUX_STAT')
    star = _row_at(_load(phase_tree), STAR_A)
    assert star['flux_fit'] == pytest.approx(4950.0)
    assert star['flux_brightest_fit'] == pytest.approx(5000.0)


@pytest.fixture
def mixed_wingcal_tree(tmp_path, clean_env):
    """Three exposures whose per-frame wing self-calibration ran in some and
    was skipped in others (``wingcal_ratio`` 1.0), as on GC F410M.  The
    pooled C(r) is 1.25, the per-frame ratio where it ran is also 1.25, and
    every exposure measures the same star, so each star's calibrated flux is
    1000 in every exposure.

    Star A: the skipped exposure is the brightest raw fit (1250), so it
    became the representative.  Star B: a calibrated exposure (1300) is the
    representative and the two skipped ones read 1250 raw.
    """
    pdir = tmp_path / 'F182M' / 'pipeline'
    pdir.mkdir(parents=True)
    (tmp_path / 'catalogs').mkdir()
    rows = {1: [(*STAR_A, 1000.0, 1.25), (*STAR_B, 1300.0, 1.25)],
            2: [(*STAR_A, 1000.0, 1.25), (*STAR_B, 1250.0, 1.0)],
            3: [(*STAR_A, 1250.0, 1.0), (*STAR_B, 1250.0, 1.0)]}
    for expo, stars in rows.items():
        frame = _frame(pdir, expo)
        _write_product(frame, '_resbgsub_m7', [s[:3] for s in stars],
                       extra={'wingcal_ratio': [s[3] for s in stars],
                              'wingcal_rmask': [4.0] * len(stars)})
        calib = str(frame).replace('.fits',
                                   '_resbgsub_m7_wingcal_calibrators.fits')
        Table({'rmask_px': [4], 'ratio_median': [1.25], 'n_stars': [8],
               'ratio_madstd': [0.02]}).write(calib, overwrite=True)
    return tmp_path


def test_pooled_wingcal_reaches_every_exposure_before_the_median(
        mixed_wingcal_tree, monkeypatch):
    """The pooled C(r) is applied to each exposure's row before the dedup, so
    the median is over calibrated fluxes only.  Applied after the dedup it
    followed the representative: star A (skipped representative) was divided
    a second time (median 1000 -> 800) and star B (calibrated representative)
    kept its skipped members raw (median 1250)."""
    monkeypatch.delenv('SATSTAR_FLUX_STAT')
    cat = _load(mixed_wingcal_tree)
    for xy in (STAR_A, STAR_B):
        star = _row_at(cat, xy)
        assert star['flux_fit'] == pytest.approx(1000.0)
        assert star['flux_median_fit'] == pytest.approx(1000.0)
        assert star['n_frames_fit'] == 3
    assert _row_at(cat, STAR_B)['flux_brightest_fit'] == pytest.approx(1300.0)


def test_pooled_wingcal_before_the_dedup_with_the_brightest_statistic(
        mixed_wingcal_tree):
    """SATSTAR_FLUX_STAT=brightest picks the representative among calibrated
    fluxes, and a calibrated representative is not divided again."""
    cat = _load(mixed_wingcal_tree)
    assert _row_at(cat, STAR_A)['flux_fit'] == pytest.approx(1000.0)
    assert _row_at(cat, STAR_B)['flux_fit'] == pytest.approx(1300.0)


def test_median_flux_keeps_the_fractional_error(phase_tree, monkeypatch):
    """flux_err scales with the adopted median (10 at 5000 -> 9.9 at 4950)."""
    monkeypatch.delenv('SATSTAR_FLUX_STAT')
    star = _row_at(_load(phase_tree), STAR_A)
    assert star['flux_err'] == pytest.approx(10.0 * 4950.0 / 5000.0)


def test_pooled_read_reproduces_the_defect(phase_tree, monkeypatch):
    monkeypatch.setenv(SPS.POOL_PHASES_ENV, 'all')
    star = _row_at(_load(phase_tree), STAR_A)
    assert star['flux_fit'] == pytest.approx(9000.0)
    assert star[MC._SATSTAR_ITER_COL] == 'm12'
    assert star['n_meas_fit'] == 5


def test_exposure_with_only_an_earlier_phase_still_contributes(phase_tree):
    """Exposure 2 stops at m5; its m5 catalog is used (star B is only there)."""
    star_b = _row_at(_load(phase_tree, phase='m7'), STAR_B)
    assert star_b is not None
    assert star_b['flux_fit'] == pytest.approx(3000.0)
    assert star_b[MC._SATSTAR_ITER_COL] == 'm5'


def test_explicit_phase_is_respected(phase_tree):
    m4 = _load(phase_tree, phase='m4')
    assert _row_at(m4, STAR_A)['flux_fit'] == pytest.approx(5500.0)
    assert _row_at(m4, STAR_A)[MC._SATSTAR_ITER_COL] == 'm4'
    # the first phase's merge: exposure 1's m12 only; exposure 2 has no
    # catalog at or before m12, so star B is absent
    m2 = _load(phase_tree, phase='m2')
    assert _row_at(m2, STAR_A)['flux_fit'] == pytest.approx(9000.0)
    assert _row_at(m2, STAR_B) is None


def test_only_later_phases_on_disk_reads_as_no_catalog(tmp_path, clean_env,
                                                       capsys):
    pdir = tmp_path / 'F182M' / 'pipeline'
    pdir.mkdir(parents=True)
    _write_product(_frame(pdir, 1), '_resbgsub_m7', [(*STAR_A, 5000.0)])
    assert _load(tmp_path, phase='m3') is None
    assert 'left over from a previous run' in capsys.readouterr().out


def test_log_counts_files_per_phase(phase_tree, capsys):
    _load(phase_tree, phase='m7')
    out = capsys.readouterr().out
    assert ('using 2 satstar catalog file(s) [m5: 1, m7: 1]; excluded 3 '
            'superseded [m12: 1, m4: 2]') in out, out


# --- the cache key -------------------------------------------------------

def test_cache_built_from_the_pooled_set_is_rebuilt(phase_tree, monkeypatch,
                                                    capsys):
    """A consolidated cache written before this change pooled every phase and
    carries no SATPHSEL.  Make it match the new read on EVERY other key (count,
    frame state, radius, algorithm, mtime) so only the selection key can
    reject it."""
    monkeypatch.setenv(SPS.POOL_PHASES_ENV, 'all')
    _load(phase_tree)
    monkeypatch.delenv(SPS.POOL_PHASES_ENV)
    cache = _cache(phase_tree)
    legacy = Table.read(cache)
    assert _row_at(legacy, STAR_A)['flux_fit'] == pytest.approx(9000.0)
    pipe = str(phase_tree / 'F182M' / 'pipeline')
    chosen, _ = SPS.select_satstar_phase_files(
        [os.path.join(pipe, f) for f in os.listdir(pipe)
         if f.endswith('_satstar_catalog.fits')])
    del legacy.meta['SATPHSEL']
    legacy.meta['NSATSRC'] = len(chosen)
    legacy.meta['SATFRMSG'] = MC.satstar_frame_state_signature(chosen)
    legacy.write(cache, overwrite=True)
    capsys.readouterr()

    out = _load(phase_tree)
    log = capsys.readouterr().out
    assert 'phase selection' in log and 'every phase pooled' in log, log
    assert 'Building consolidated satstar catalog' in log
    assert _row_at(out, STAR_A)['flux_fit'] == pytest.approx(5000.0)
    assert Table.read(cache).meta['SATPHSEL'] == \
        SPS.satstar_phase_selection_signature(chosen)


def test_cache_follows_the_phase_being_merged(phase_tree, capsys):
    """m7 and m4 read the same two exposures (same count, same frames) and the
    cache is newer than every catalog, so only SATPHSEL tells them apart."""
    first = _load(phase_tree)
    assert _row_at(first, STAR_A)['flux_fit'] == pytest.approx(5000.0)
    capsys.readouterr()
    _load(phase_tree, phase='m7')
    assert 'Using consolidated satstar catalog' in capsys.readouterr().out

    m4 = _load(phase_tree, phase='m4')
    log = capsys.readouterr().out
    assert 'Rebuilding satstar cache' in log and 'phase selection' in log, log
    assert _row_at(m4, STAR_A)['flux_fit'] == pytest.approx(5500.0)


# --- the gate-rejected channel -------------------------------------------

def test_rejected_channel_follows_the_accepted_phase(tmp_path, clean_env):
    """Rejected files exist only where the gates rejected something.  Exposure
    1's stale m12 run rejected a candidate at C; its m7 run (the chosen one)
    rejected nothing, so C must not reach the merge.  Exposure 2 has rejects
    but no accepted catalog: its latest rejected phase at or before the merge
    is used."""
    pdir = tmp_path / 'F182M' / 'pipeline'
    pdir.mkdir(parents=True)
    star_c, star_d = (60.0, 60.0), (100.0, 20.0)
    e1, e2 = _frame(pdir, 1), _frame(pdir, 2)
    _write_product(e1, '_m12', [(*STAR_A, 9000.0)])
    _write_product(e1, '_m12', [(*star_c, 700.0)], kind='rejected')
    _write_product(e1, '_resbgsub_m7', [(*STAR_A, 5000.0)])
    _write_product(e2, '_m4', [(*star_d, 400.0)], kind='rejected')
    _write_product(e2, '_resbgsub_m6', [(*star_d, 450.0)], kind='rejected')
    base = str(tmp_path) + '/'

    rej = MC.load_rejected_satstar_catalog('f182m', basepath=base, phase='m7')
    assert _row_at(rej, star_c) is None
    assert _row_at(rej, star_d)['flux_fit'] == pytest.approx(450.0)
    assert len(rej) == 1
    # merging m4: exposure 1's chosen run is m12 (m7 is later), so its m12
    # rejects are that run's; exposure 2's m6 is later than m4
    rej4 = MC.load_rejected_satstar_catalog('f182m', basepath=base, phase='m4')
    assert _row_at(rej4, star_c)['flux_fit'] == pytest.approx(700.0)
    assert _row_at(rej4, star_d)['flux_fit'] == pytest.approx(400.0)


def test_rejected_channel_pools_under_the_escape_hatch(tmp_path, clean_env,
                                                       monkeypatch):
    pdir = tmp_path / 'F182M' / 'pipeline'
    pdir.mkdir(parents=True)
    e1 = _frame(pdir, 1)
    _write_product(e1, '_m12', [(60.0, 60.0, 700.0)], kind='rejected')
    _write_product(e1, '_resbgsub_m7', [(*STAR_A, 5000.0)])
    base = str(tmp_path) + '/'
    assert MC.load_rejected_satstar_catalog('f182m', basepath=base) is None
    monkeypatch.setenv(SPS.POOL_PHASES_ENV, 'all')
    assert len(MC.load_rejected_satstar_catalog('f182m', basepath=base)) == 1


# --- every merge entry point forwards its phase --------------------------

def _fake_svo(monkeypatch, filt='F182M'):
    jfilts = Table({'filterID': [f'JWST/NIRCam.{filt}'], 'ZeroPoint': [250.0]})
    monkeypatch.setattr(
        MC, 'SvoFps', types.SimpleNamespace(get_filter_list=lambda fac: jfilts))


def test_replace_saturated_substitutes_the_merged_phase_fit(phase_tree,
                                                            monkeypatch):
    """End to end through the append path, and the phase reaches the
    rejected-channel loader too."""
    _fake_svo(monkeypatch)
    seen = []
    real_rej = MC.load_rejected_satstar_catalog

    def _rej(*a, **k):
        seen.append(k.get('phase'))
        return real_rej(*a, **k)
    monkeypatch.setattr(MC, 'load_rejected_satstar_catalog', _rej)

    def _merged(**env):
        for key, val in env.items():
            monkeypatch.setenv(key, val)
        far = _wcs().pixel_to_world(120.0, 5.0)
        cat = Table({'skycoord': SkyCoord([far.ra.deg] * u.deg,
                                          [far.dec.deg] * u.deg),
                     'flux': [10.0], 'dflux': [1.0], 'x': [120.0], 'y': [5.0],
                     'dx': [0.01], 'dy': [0.01]})
        MC.replace_saturated(cat, 'f182m', target='brick',
                             basepath=str(phase_tree) + '/', phase='m7')
        added = cat[np.asarray(cat['replaced_saturated'], bool)]
        target = _wcs().pixel_to_world(*STAR_A)
        sep = SkyCoord(added['skycoord']).separation(target).to_value(u.arcsec)
        return float(np.asarray(added['flux'])[np.argmin(sep)])

    one_phase = _merged()
    pooled = _merged(**{SPS.POOL_PHASES_ENV: 'all'})
    assert one_phase / pooled == pytest.approx(5000.0 / 9000.0)
    assert seen and set(seen) == {'m7'}


@pytest.mark.parametrize('func', ['replace_saturated', 'flag_near_saturated'])
def test_satstar_entry_points_forward_the_phase(func, monkeypatch):
    seen = {}

    def _fake(filtername, target='brick', basepath='', proposal_id=None,
              field=None, phase=None, frame_variants=None):
        seen['phase'] = phase
        seen['frame_variants'] = frame_variants
        return None
    monkeypatch.setattr(MC, 'load_satstar_catalog', _fake)
    getattr(MC, func)(Table({'flux': [1.0]}), 'f480m', basepath='/nowhere/',
                      phase='m6', frame_variants={'destreak_o001_crf'})
    assert seen == {'phase': 'm6', 'frame_variants': {'destreak_o001_crf'}}


def test_per_filter_merge_forwards_its_iteration_label(tmp_path, monkeypatch):
    from .test_gc_treasury_obs_scoping import _stub_combine, _write_perframe
    (tmp_path / 'catalogs').mkdir()
    _write_perframe(tmp_path, '001')
    monkeypatch.setattr(MC, 'combine_singleframe', _stub_combine([]))
    seen = []
    monkeypatch.setattr(MC, 'replace_saturated',
                        lambda cat, **k: seen.append(k.get('phase')))
    MC.merge_individual_frames(
        module='nrcblong', filtername='f480m', progid='10678',
        method='dao', suffix='_basic', target='gc-treasury',
        basepath=str(tmp_path), iteration_label='m2', field='001',
        do_replace_saturated=True)
    assert seen == ['m2']


class _Stop(Exception):
    pass


def test_cross_band_merge_forwards_its_iteration_label(monkeypatch):
    jfilts = Table({'filterID': ['JWST/NIRCam.F212N', 'JWST/NIRCam.F480M'],
                    'ZeroPoint': [1.0, 1.0]})
    monkeypatch.setattr(
        MC, 'SvoFps', types.SimpleNamespace(get_filter_list=lambda fac: jfilts))
    seen = {}

    def _flag(tbl, **k):
        seen['flag'] = k.get('phase')

    def _replace(tbl, **k):
        seen['replace'] = k.get('phase')
        raise _Stop
    monkeypatch.setattr(MC, 'flag_near_saturated', _flag)
    monkeypatch.setattr(MC, 'replace_saturated', _replace)
    tbls = []
    for filt in ('f212n', 'f480m'):
        t = Table({'skycoord': SkyCoord([266.0] * u.deg, [-28.9] * u.deg)})
        t.meta['filter'] = filt
        tbls.append(t)
    with pytest.raises(_Stop):
        MC.merge_catalogs(tbls, catalog_type='basic', module='nrcblong',
                          ref_filter='f480m', iteration_label='m7',
                          basepath='/nowhere/')
    assert seen == {'flag': 'm7', 'replace': 'm7'}


# --- the frame variant (#1194) -------------------------------------------

EXPO = 'jw02221002001_06201_00001_nrca1'


def _variant_files(tmp_path, tail='_m12', newer='align'):
    """One exposure's satstar catalogs for both frame variants at one phase;
    ``newer`` is the variant written last."""
    out = {}
    for variant in ('align', 'destreak'):
        out[variant] = _touch(
            tmp_path / f'{EXPO}_{variant}_o002_crf{tail}_satstar_catalog.fits',
            10 if variant == newer else 1000)
    return out


def test_frame_variant_is_the_same_token_for_a_frame_and_its_products():
    want = 'destreak_o002_crf'
    assert SPS.satstar_frame_variant(f'{EXPO}_destreak_o002_crf.fits') == want
    assert SPS.satstar_frame_variant(
        f'/a/b/{EXPO}_destreak_o002_crf_m12_satstar_catalog.fits') == want
    assert SPS.satstar_frame_variant(
        f'{EXPO}_destreak_o002_crf_resbgsub_m6_satstar_rejected.fits') == want
    assert SPS.satstar_frame_variant(
        f'{EXPO}_destreak_o002_crf_bgsub_m3_wingcal_calibrators.fits') == want
    assert SPS.satstar_frame_variant(
        f'{EXPO}_align_o002_crf.fits') == 'align_o002_crf'
    # a variant the module does not list is still a token
    assert SPS.satstar_frame_variant(
        'jw01182004001_02101_00001_mirimage_align_o004_crf.fits') \
        == 'align_o004_crf'
    # nothing after the detector, or not an exposure name
    assert SPS.satstar_frame_variant(f'{EXPO}.fits') == ''
    assert SPS.satstar_frame_variant('cutout_crf_m3_satstar_catalog.fits') == ''


def test_frame_variant_falls_back_to_the_filename_card(tmp_path):
    path = tmp_path / f'{EXPO}_m12_satstar_catalog.fits'
    hdu = fits.ImageHDU(name='CAT')
    hdu.header['FILENAME'] = f'{EXPO}_destreak_o002_crf.fits'
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    assert SPS.satstar_frame_variant(str(path)) == ''
    assert SPS.satstar_frame_variant(str(path), use_header=True) \
        == 'destreak_o002_crf'


@pytest.mark.parametrize('newer', ['align', 'destreak'])
def test_selection_returns_the_variant_the_merge_reads(tmp_path, clean_env,
                                                       newer):
    files = _variant_files(tmp_path, newer=newer)
    sel, rep = SPS.select_satstar_phase_files(
        files.values(), phase='m2', frame_variants={'destreak_o002_crf'})
    assert sel == [files['destreak']]
    assert rep['tie_losers'] == []
    assert dict(rep['other_variant']) == {'m12': 1}
    assert not rep['no_variant']
    # the variant that was NOT written last is chosen just as readily
    sel, _ = SPS.select_satstar_phase_files(
        files.values(), phase='m2', frame_variants=['align_o002_crf'])
    assert sel == [files['align']]


@pytest.mark.parametrize('newer', ['align', 'destreak'])
def test_unfiltered_selection_keeps_the_file_time_rule(tmp_path, clean_env,
                                                       newer):
    files = _variant_files(tmp_path, newer=newer)
    sel, rep = SPS.select_satstar_phase_files(files.values(), phase='m2')
    assert sel == [files[newer]]
    assert not rep['frame_variants']


def test_read_variant_falls_back_to_its_own_earlier_phase(tmp_path, clean_env):
    """At the merged phase only the other variant has a file; the read
    variant's earlier phase is used, not the other variant's current one."""
    destreak_m4 = _touch(tmp_path / f'{EXPO}_destreak_o002_crf_m4_satstar_catalog.fits', 500)
    align_m7 = _touch(tmp_path / f'{EXPO}_align_o002_crf_m7_satstar_catalog.fits', 10)
    sel, rep = SPS.select_satstar_phase_files(
        [destreak_m4, align_m7], phase='m7', frame_variants={'destreak_o002_crf'})
    assert sel == [destreak_m4]
    assert dict(rep['used']) == {'m4': 1}


def test_exposure_without_the_read_variant_is_left_out_and_reported(
        tmp_path, clean_env):
    other = _touch(tmp_path / f'{EXPO}_align_o002_crf_m12_satstar_catalog.fits', 10)
    kept = _touch(tmp_path / ('jw02221002001_06201_00002_nrca1_destreak_o002_crf'
                              '_m12_satstar_catalog.fits'), 10)
    sel, rep = SPS.select_satstar_phase_files(
        [other, kept], phase='m12', frame_variants={'destreak_o002_crf'})
    assert sel == [kept]
    assert dict(rep['no_variant']) == {'align_o002_crf': 1}
    line = SPS.format_phase_report(rep)
    assert 'WARNING: 1 exposure(s) have no satstar catalog' in line
    assert 'align_o002_crf: 1' in line


def test_a_file_with_no_readable_variant_passes_through(tmp_path, clean_env):
    path = _touch(tmp_path / 'cutout_crf_m3_satstar_catalog.fits', 10)
    sel, rep = SPS.select_satstar_phase_files(
        [path], phase='m3', frame_variants={'destreak_o002_crf'})
    assert sel == [path]
    assert rep['unparsed_variant'] == 1


def test_background_tokens_of_one_variant_still_tie_by_file_time(
        tmp_path, clean_env):
    old = _touch(tmp_path / f'{EXPO}_destreak_o002_crf_m5_satstar_catalog.fits', 1000)
    new = _touch(tmp_path / f'{EXPO}_destreak_o002_crf_resbgsub_m5_satstar_catalog.fits', 10)
    align = _touch(tmp_path / f'{EXPO}_align_o002_crf_m5_satstar_catalog.fits', 1)
    sel, rep = SPS.select_satstar_phase_files(
        [old, new, align], phase='m5', frame_variants={'destreak_o002_crf'})
    assert sel == [new]
    assert rep['tie_losers'] == [(os.path.basename(new), os.path.basename(old))]
    assert 'WARNING' not in SPS.format_phase_report(rep)


def test_warning_only_when_a_tie_crosses_frame_variants(tmp_path, clean_env):
    files = _variant_files(tmp_path)
    _, rep = SPS.select_satstar_phase_files(files.values(), phase='m2')
    line = SPS.format_phase_report(rep)
    assert 'WARNING: 1 of them chose between frame variants' in line
    # same variant, two background tokens: a tie, but no warning
    a = _touch(tmp_path / f'{EXPO}_align_o002_crf_m6_satstar_catalog.fits', 100)
    b = _touch(tmp_path / f'{EXPO}_align_o002_crf_resbgsub_m6_satstar_catalog.fits', 10)
    _, rep = SPS.select_satstar_phase_files([a, b], phase='m6')
    assert len(rep['tie_losers']) == 1
    assert 'WARNING' not in SPS.format_phase_report(rep)


def test_more_than_one_read_variant_is_warned_about(tmp_path, clean_env):
    files = _variant_files(tmp_path)
    _, rep = SPS.select_satstar_phase_files(
        files.values(), phase='m2',
        frame_variants={'align_o002_crf', 'destreak_o002_crf'})
    assert 'WARNING: more than one frame variant is read' in \
        SPS.format_phase_report(rep)


def test_rejected_orphans_follow_the_read_variant(tmp_path, clean_env):
    rejected = {v: _touch(tmp_path / f'{EXPO}_{v}_o002_crf_m12_satstar_rejected.fits',
                          10 if v == 'align' else 1000)
                for v in ('align', 'destreak')}
    sel, _ = SPS.select_rejected_for_catalogs(
        rejected.values(), [], phase='m2', frame_variants={'destreak_o002_crf'})
    assert sel == [rejected['destreak']]
    sel, _ = SPS.select_rejected_for_catalogs(rejected.values(), [], phase='m2')
    assert sel == [rejected['align']]


def test_variant_selections_give_different_cache_signatures(tmp_path, clean_env):
    files = _variant_files(tmp_path)
    sigs = {v: SPS.satstar_phase_selection_signature(
        SPS.select_satstar_phase_files(files.values(), phase='m2',
                                       frame_variants={f'{v}_o002_crf'})[0])
        for v in ('align', 'destreak')}
    assert sigs['align'] and sigs['destreak']
    assert sigs['align'] != sigs['destreak']


@pytest.fixture
def variant_tree(tmp_path, clean_env):
    """One exposure fit on two frame variants at m12: the align fit puts star A
    5 px away from the destreak fit.  The align catalogue is written last."""
    pdir = tmp_path / 'F182M' / 'pipeline'
    pdir.mkdir(parents=True)
    (tmp_path / 'catalogs').mkdir()
    shifted = (STAR_A[0] + 5.0, STAR_A[1])
    for variant, xy, age in (('destreak', STAR_A, 300), ('align', shifted, 10)):
        frame = pdir / f'jw02221001001_02101_00001_nrca1_{variant}_o001_crf.fits'
        _write_frame(str(frame))
        path = _write_product(frame, '_m12', [(*xy, 5000.0)])
        os.utime(path, (time.time() - age, time.time() - age))
    return tmp_path, shifted


def test_load_follows_the_read_variant_and_rebuilds_the_cache(variant_tree,
                                                              capsys):
    base, shifted = variant_tree
    unfiltered = _load(base)
    assert _row_at(unfiltered, shifted) is not None     # newest file wins
    assert _row_at(unfiltered, STAR_A) is None
    capsys.readouterr()

    destreak = _load(base, frame_variants={'destreak_o001_crf'})
    log = capsys.readouterr().out
    assert 'Rebuilding satstar cache' in log and 'phase selection' in log, log
    assert _row_at(destreak, STAR_A) is not None
    assert _row_at(destreak, shifted) is None

    align = _load(base, frame_variants={'align_o001_crf'})
    assert _row_at(align, shifted) is not None
    assert _row_at(align, STAR_A) is None


def test_load_without_the_read_variant_contributes_nothing(variant_tree, capsys):
    base, _ = variant_tree
    assert _load(base, frame_variants={'other_o001_crf'}) is None
    log = capsys.readouterr().out
    assert 'WARNING: no f182m satstar catalog of frame variant(s)' in log, log


def test_replace_saturated_appends_the_read_variants_positions(variant_tree,
                                                               monkeypatch):
    _fake_svo(monkeypatch)
    base, shifted = variant_tree

    def _added(variants):
        far = _wcs().pixel_to_world(120.0, 5.0)
        cat = Table({'skycoord': SkyCoord([far.ra.deg] * u.deg,
                                          [far.dec.deg] * u.deg),
                     'flux': [10.0], 'dflux': [1.0], 'x': [120.0], 'y': [5.0],
                     'dx': [0.01], 'dy': [0.01]})
        MC.replace_saturated(cat, 'f182m', target='brick',
                             basepath=str(base) + '/', phase='m2',
                             frame_variants=variants)
        return SkyCoord(cat[np.asarray(cat['replaced_saturated'], bool)]['skycoord'])

    for variants, xy in (({'destreak_o001_crf'}, STAR_A),
                         ({'align_o001_crf'}, shifted)):
        added = _added(variants)
        target = _wcs().pixel_to_world(*xy)
        assert len(added) == 1
        assert added.separation(target).to_value(u.arcsec)[0] < 0.01


# --- stamping and forwarding FRMVAR --------------------------------------

def test_per_filter_merge_stamps_and_forwards_the_frame_variant(tmp_path,
                                                                monkeypatch):
    from .test_gc_treasury_obs_scoping import _stub_combine, _write_perframe
    (tmp_path / 'catalogs').mkdir()
    _write_perframe(tmp_path, '001')
    monkeypatch.setattr(MC, 'combine_singleframe', _stub_combine([]))
    seen = []
    monkeypatch.setattr(
        MC, 'replace_saturated',
        lambda cat, **k: seen.append((k.get('frame_variants'),
                                      cat.meta.get('FRMVAR'))))
    MC.merge_individual_frames(
        module='nrcblong', filtername='f480m', progid='10678',
        method='dao', suffix='_basic', target='gc-treasury',
        basepath=str(tmp_path), iteration_label='m2', field='001',
        do_replace_saturated=True)
    assert seen == [({'align_o001_crf'}, 'align_o001_crf')]


def test_recorded_frame_variants_reads_the_card(capsys):
    assert MC._recorded_frame_variants({'FRMVAR': 'a_o1_crf,b_o1_crf'}) \
        == frozenset({'a_o1_crf', 'b_o1_crf'})
    assert MC._recorded_frame_variants({'frmvar': MC.FRAME_VARIANT_NONE}) is None
    assert 'WARNING' not in capsys.readouterr().out
    assert MC._recorded_frame_variants({}, context='f212n') is None
    assert 'f212n has no FRMVAR card' in capsys.readouterr().out


def test_cross_band_merge_forwards_the_recorded_frame_variant(monkeypatch):
    jfilts = Table({'filterID': ['JWST/NIRCam.F212N', 'JWST/NIRCam.F480M'],
                    'ZeroPoint': [1.0, 1.0]})
    monkeypatch.setattr(
        MC, 'SvoFps', types.SimpleNamespace(get_filter_list=lambda fac: jfilts))
    seen = {}

    def _flag(tbl, **k):
        seen['flag'] = k.get('frame_variants')

    def _replace(tbl, **k):
        seen['replace'] = k.get('frame_variants')
        raise _Stop
    monkeypatch.setattr(MC, 'flag_near_saturated', _flag)
    monkeypatch.setattr(MC, 'replace_saturated', _replace)
    t = Table({'skycoord': SkyCoord([266.0] * u.deg, [-28.9] * u.deg)})
    t.meta['filter'] = 'f212n'
    t.meta['FRMVAR'] = 'destreak_o002_crf'
    with pytest.raises(_Stop):
        MC.merge_catalogs([t], catalog_type='basic', module='nrcblong',
                          ref_filter='f212n', iteration_label='m7',
                          basepath='/nowhere/')
    assert seen == {'flag': frozenset({'destreak_o002_crf'}),
                    'replace': frozenset({'destreak_o002_crf'})}
