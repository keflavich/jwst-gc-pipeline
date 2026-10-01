"""Faint-star reference fields: config integrity, metric unit tests, and the
reference tests themselves.

The metric tests are hermetic (synthetic arrays).  The reference tests score
real cutout runs (``reference_fields/run.py``) against each field's fixed
thresholds; they skip when that run's products are absent, so they only bite
on a machine with the data and a finished run.  Select the code under test
with ``JWST_GC_REFFIELD_VARIANT`` (default ``main``).
"""
import os
import subprocess

import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry import reference_fields as RF
from jwst_gc_pipeline.photometry.injection import flux_column
from jwst_gc_pipeline.photometry.reference_fields import evaluate as EV

_DEFAULTS, _FIELDS = RF.load_config()


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', sorted(_FIELDS))
def test_field_spec_complete(name):
    spec = _FIELDS[name]
    for key in ('environment', 'target', 'proposal', 'obsid', 'each_suffix',
                'filters', 'ra', 'dec', 'thresholds', 'seeds', 'size_arcsec'):
        assert key in spec, f'{name}: missing {key}'
    assert len(spec['filters']) >= 2, 'the m7 cross-band seed needs two filters'
    assert isinstance(spec['thresholds'], dict)
    assert 0 not in spec['seeds'], 'seed 0 is the clean run'
    assert spec['size_arcsec'] - 2 * spec['inner_margin_arcsec'] > 1.0


_THRESHOLD_KEYS = {'completeness_min', 'flux_bias_max_mag', 'residual_excess_max',
                   'oversubtracted_max', 'ring_ratio_max', 'emission_purity_min',
                   'labels_recovered_min', 'emission_labels_cataloged_max'}


@pytest.mark.parametrize('name', sorted(_FIELDS))
def test_thresholds_name_real_metrics(name):
    """check() skips a misspelt threshold key silently and fails every run on
    a misspelt S/N bin, so every key must be one check() reads and every
    completeness bin one of the field's snr_bins."""
    spec = _FIELDS[name]
    thr = spec['thresholds']
    assert set(thr) <= _THRESHOLD_KEYS, set(thr) - _THRESHOLD_KEYS
    bins = set(EV.completeness_by_bin(np.array([]), np.array([], bool), spec['snr_bins']))
    assert set(thr.get('completeness_min') or {}) <= bins
    if 'labels_recovered_min' in thr:
        assert spec.get('labels')
    if 'emission_labels_cataloged_max' in thr:
        assert spec.get('emission_labels')


@pytest.mark.parametrize('name', sorted(_FIELDS))
def test_injection_tables_frozen(name):
    """Every injection seed has its committed table, with a flux column per
    filter and positions inside the field's inner box."""
    spec = _FIELDS[name]
    for seed in spec['seeds']:
        path = RF.injection_table_path(name, seed)
        assert os.path.exists(path), path
        tbl = Table.read(path)
        assert len(tbl) == spec['n_inject']
        for f in spec['filters']:
            assert flux_column(f) in tbl.colnames
        prim = spec['filters'][0]
        snr = np.asarray(tbl[f'snr_true_{prim}'])
        assert snr.min() >= spec['snr_range'][0] and snr.max() <= spec['snr_range'][1]
        # the inner box is square on the mosaic's pixel grid, which may be
        # rotated on the sky: bound the radius by the box's half-diagonal
        half_as = spec['size_arcsec'] / 2 - spec['inner_margin_arcsec']
        cosd = np.cos(np.deg2rad(spec['dec']))
        r_as = 3600 * np.hypot((tbl['ra'] - spec['ra']) * cosd, tbl['dec'] - spec['dec'])
        assert np.all(r_as < half_as * np.sqrt(2) * 1.02)


def test_run_labels_distinct():
    labels = {RF.run_label(n, 'main', s) for n in _FIELDS
              for s in [0] + list(_FIELDS[n]['seeds'])}
    assert len(labels) == sum(1 + len(_FIELDS[n]['seeds']) for n in _FIELDS)
    assert all(lab.startswith('ref_') for lab in labels)


def test_sbatch_command_is_dev_run():
    # a reference run tests untagged code: without GC_ALLOW_DEV the cataloging
    # production guard stops it before the first stage
    from jwst_gc_pipeline.photometry.reference_fields.run import sbatch_command
    name = next(iter(_FIELDS))
    cmd = sbatch_command(_FIELDS[name], 'x', 0, '/wt', cpus=4, mem='24gb',
                         walltime='01:00:00', partition='hpg-dev')
    wrap = next(c for c in cmd if c.startswith('--wrap='))
    assert 'export GC_ALLOW_DEV=1;' in wrap
    assert '--partition=hpg-dev' in cmd
    assert not any(c.startswith('--partition') for c in
                   sbatch_command(_FIELDS[name], 'x', 0, '/wt', cpus=4,
                                  mem='24gb', walltime='01:00:00'))
    # the job records its provenance before the pipeline starts
    assert 'write_provenance' in wrap
    assert wrap.index('write_provenance') < wrap.index('crowdsource_catalogs_long ')


def _git(root, *args):
    subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)


def test_provenance_and_code_differences(tmp_path):
    """A run records the commit it ran; only pipeline-code changes (not
    scoring, thresholds or tests) make it stale."""
    from jwst_gc_pipeline.photometry.reference_fields import run as RUN
    repo = tmp_path / 'repo'
    rf = repo / 'jwst_gc_pipeline' / 'photometry' / 'reference_fields'
    rf.mkdir(parents=True)
    (repo / 'jwst_gc_pipeline' / 'photometry' / 'tests').mkdir()
    for f in ('cataloging.py', 'reference_fields/fields.yaml',
              'reference_fields/evaluate.py', 'tests/test_x.py'):
        (repo / 'jwst_gc_pipeline' / 'photometry' / f).write_text('a\n')
    _git(repo, 'init', '-q')
    _git(repo, 'add', '.')
    _git(repo, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'x')

    rdir = tmp_path / 'ref_x_v_s0'
    rec = RUN.write_provenance(str(rdir), str(repo), ['python', '-m', 'x'])
    assert RUN.read_provenance(str(rdir)) == rec
    assert rec['dirty'] is False and rec['argv'] == ['python', '-m', 'x']
    assert RUN.read_provenance(str(tmp_path / 'absent')) is None
    assert RUN.code_differences(rec['commit'], str(repo)) == []

    for f in ('reference_fields/fields.yaml', 'reference_fields/evaluate.py',
              'tests/test_x.py'):
        (repo / 'jwst_gc_pipeline' / 'photometry' / f).write_text('b\n')
    assert RUN.code_differences(rec['commit'], str(repo)) == []
    (repo / 'jwst_gc_pipeline' / 'photometry' / 'cataloging.py').write_text('b\n')
    assert RUN.code_differences(rec['commit'], str(repo)) == [
        'jwst_gc_pipeline/photometry/cataloging.py']
    assert RUN.write_provenance(str(rdir), str(repo), [])['dirty'] is True
    with pytest.raises(subprocess.CalledProcessError):
        RUN.code_differences('0' * 40, str(repo))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def test_match_one_to_one_is_one_to_one():
    # two truths compete for one catalog source; the closer one wins
    idx = EV.match_one_to_one(np.array([0.0, 0.6]), np.array([0.0, 0.0]),
                              np.array([1.0, 1.0]),
                              np.array([0.5]), np.array([0.0]), np.array([1.0]),
                              radius=1.0, max_dmag=0.5)
    assert list(idx) == [-1, 0]


def test_match_one_to_one_flux_window():
    idx = EV.match_one_to_one(np.array([0.0]), np.array([0.0]), np.array([1.0]),
                              np.array([0.1]), np.array([0.0]), np.array([2.0]),
                              radius=1.0, max_dmag=0.5)
    assert idx[0] == -1          # 0.75 mag off
    idx = EV.match_one_to_one(np.array([0.0]), np.array([0.0]), np.array([1.0]),
                              np.array([0.1]), np.array([0.0]), np.array([1.3]),
                              radius=1.0, max_dmag=0.5)
    assert idx[0] == 0


def test_completeness_by_bin():
    snr = np.array([6, 7, 12, 15, 30, 50])
    rec = np.array([0, 1, 1, 1, 1, 0], bool)
    res = EV.completeness_by_bin(snr, rec, [5, 10, 20, 40, 80])
    assert res['5-10'] == (2, 1, 0.5)
    assert res['10-20'] == (2, 2, 1.0)
    assert res['40-80'] == (1, 0, 0.0)


def _gauss(shape, x0, y0, fwhm, amp):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    s = fwhm / 2.3548
    return amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s ** 2))


def test_matched_filter_renormalised_and_detects():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, (200, 200))
    err = np.full(noise.shape, 3.0)       # ERR overstated 3x
    snr, scale = EV.matched_filter_snr(noise, err, 2.0)
    assert abs(np.std(snr[20:-20, 20:-20]) - 1) < 0.1
    img = noise + _gauss(noise.shape, 100, 100, 2.0, 10.0)
    snr, _ = EV.matched_filter_snr(img, err, 2.0)
    assert snr[100, 100] > 10


def test_residual_excess_counts_unsubtracted_not_symmetric():
    rng = np.random.default_rng(2)
    shape = (160, 160)
    base = rng.normal(0, 1, shape)
    err = np.ones(shape)
    inner = np.ones(shape, bool)
    stars = [(30, 30), (30, 90), (90, 30), (120, 120)]
    img = base.copy()
    for x, y in stars:
        img += _gauss(shape, x, y, 2.0, 15.0)
    snr, _ = EV.matched_filter_snr(img, err, 2.0, inner)
    exc, npos, nneg = EV.residual_excess(snr, inner, np.array([]), np.array([]),
                                         1.0, thresh=7, excl_pix=4.5)
    assert npos == 4 and nneg == 0 and exc == 4
    # the same stars, catalogued: excluded
    exc, npos, _ = EV.residual_excess(snr, inner, np.array([s[0] for s in stars], float),
                                      np.array([s[1] for s in stars], float),
                                      1.0, thresh=7, excl_pix=4.5)
    assert npos == 0 and exc == 0
    # a dipole pattern (over- and under-subtraction) cancels
    img = base + _gauss(shape, 40, 40, 2.0, 15.0) - _gauss(shape, 110, 110, 2.0, 15.0)
    snr, _ = EV.matched_filter_snr(img, err, 2.0, inner)
    exc, npos, nneg = EV.residual_excess(snr, inner, np.array([]), np.array([]),
                                         1.0, thresh=7, excl_pix=4.5)
    assert npos == 1 and nneg == 1 and exc == 0


def test_oversubtracted_counts_negative_cores_at_sources():
    rng = np.random.default_rng(5)
    shape = (120, 120)
    img = rng.normal(0, 1, shape)
    err = np.ones(shape)
    inner = np.ones(shape, bool)
    # two over-subtracted fits, one under-subtracted, one clean
    img -= _gauss(shape, 30, 30, 2.0, 15.0) + _gauss(shape, 90, 30, 2.0, 15.0)
    img += _gauss(shape, 30, 90, 2.0, 15.0)
    snr, _ = EV.matched_filter_snr(img, err, 2.0, inner)
    x = np.array([30.4, 89.6, 30.0, 90.0, 200.0])
    y = np.array([29.7, 30.3, 90.0, 90.0, 10.0])
    inside = np.array([True, True, True, True, False])
    per, n = EV.oversubtracted(snr, x, y, inside, 2.0, thresh=7)
    assert n == 2 and per == 1.0
    # the same negative cores, no source on them: not counted (residual_excess's job)
    per, n = EV.oversubtracted(snr, x[2:], y[2:], inside[2:], 2.0, thresh=7)
    assert n == 0
    # per-source mask: off-map and non-finite positions are False
    m = EV.oversubtracted_mask(snr, np.r_[x, np.nan], np.r_[y, 5.0], thresh=7)
    assert m.tolist() == [True, True, False, False, False, False]


def test_ring_ratio_uniform_vs_companions():
    rng = np.random.default_rng(3)
    n_b, n_f = 20, 400
    xb, yb = rng.uniform(0, 500, n_b), rng.uniform(0, 500, n_b)
    xf, yf = rng.uniform(0, 500, n_f), rng.uniform(0, 500, n_f)
    x = np.r_[xb, xf]
    y = np.r_[yb, yf]
    flux = np.r_[np.full(n_b, 1000.0), np.full(n_f, 1.0)]
    snr = np.r_[np.full(n_b, 500.0), np.full(n_f, 8.0)]
    inside = np.ones(len(x), bool)
    kw = dict(bright_snr=100, faint_ratio=0.05, ring=(1.5, 4.5), area_pix=500.0 ** 2)
    r_uniform, _, n_exp = EV.ring_ratio(x, y, flux, snr, inside, **kw)
    assert n_exp > 0 and r_uniform < 3
    # add one fake companion at 3 px from every bright star
    x2 = np.r_[x, xb + 3.0]
    y2 = np.r_[y, yb]
    flux2 = np.r_[flux, np.full(n_b, 1.0)]
    snr2 = np.r_[snr, np.full(n_b, 8.0)]
    r_fake, n_obs, _ = EV.ring_ratio(x2, y2, flux2, snr2, np.ones(len(x2), bool), **kw)
    assert n_obs >= n_b and r_fake > 10


def test_purity_estimate():
    assert EV.purity_estimate(0.8, 0.1, 0.8) == pytest.approx(1.0)
    assert EV.purity_estimate(0.45, 0.1, 0.8) == pytest.approx(0.5)
    assert np.isnan(EV.purity_estimate(0.3, 0.2, 0.3))      # contrast 0.1 < 0.2
    assert np.isnan(EV.purity_estimate(0.3, 0.2, float('nan')))


def test_n_within():
    x_lab, y_lab = np.array([10.0, 50.0, 90.0]), np.array([10.0, 50.0, 90.0])
    assert EV.n_within(x_lab, y_lab, np.array([11.0, 52.5]), np.array([10.0, 50.0]), 3.0) == 2
    assert EV.n_within(x_lab, y_lab, np.array([]), np.array([]), 3.0) == 0
    assert EV.n_within(np.array([]), np.array([]), np.array([1.0]), np.array([1.0]), 3.0) == 0


@pytest.mark.parametrize('name', sorted(_FIELDS))
def test_emission_labels_inside_inner_box(name):
    spec = _FIELDS[name]
    if not spec.get('emission_labels'):
        pytest.skip(f'{name}: no emission labels')
    ek = np.asarray(spec['emission_labels'], float)
    half_as = spec['size_arcsec'] / 2 - spec['inner_margin_arcsec']
    cosd = np.cos(np.deg2rad(spec['dec']))
    r_as = 3600 * np.hypot((ek[:, 0] - spec['ra']) * cosd, ek[:, 1] - spec['dec'])
    assert np.all(r_as < half_as)
    # labels far enough apart that one source cannot count twice
    sep = 3600 * np.hypot((ek[:, None, 0] - ek[None, :, 0]) * cosd, ek[:, None, 1] - ek[None, :, 1])
    assert np.all(sep[np.triu_indices(len(ek), 1)] > 2 * spec['emission_label_radius_arcsec'])


def test_check_thresholds():
    res = dict(completeness={'5-10': (10, 3, 0.3), '10-20': (10, 9, 0.9)},
               flux_bias_mag=-0.03,
               clean=dict(residual_excess=0.5, ring_ratio=1.1, labels_recovered=0.8),
               emission_purity=float('nan'))
    assert EV.check(res, {}) == []
    assert EV.check(res, {'completeness_min': {'10-20': 0.8},
                          'flux_bias_max_mag': 0.05, 'ring_ratio_max': 1.5}) == []
    fails = EV.check(res, {'completeness_min': {'5-10': 0.5},
                           'residual_excess_max': 0.2,
                           'emission_purity_min': 0.5})
    assert len(fails) == 3          # completeness, excess, NaN purity
    res['clean']['emission_labels_cataloged'] = 2
    assert EV.check(res, {'emission_labels_cataloged_max': 2}) == []
    assert len(EV.check(res, {'emission_labels_cataloged_max': 0})) == 1
    res['clean']['oversubtracted'] = 3.0
    assert EV.check(res, {'oversubtracted_max': 3.0}) == []
    assert len(EV.check(res, {'oversubtracted_max': 2.0})) == 1
    del res['clean']['emission_labels_cataloged']
    assert len(EV.check(res, {'emission_labels_cataloged_max': 0})) == 1   # missing


def test_figure_match_catalogs():
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from jwst_gc_pipeline.photometry.reference_fields import figures as FG
    ra0, dec0 = 266.5, -28.7
    off = np.array([0.0, 1.0, 2.0]) / 3600
    a = SkyCoord((ra0 + off) * u.deg, np.full(3, dec0) * u.deg)
    b = SkyCoord((ra0 + off[[0, 1]] + [0.01 / 3600, 0.2 / 3600]) * u.deg,
                 np.full(2, dec0) * u.deg)
    in_b, in_a = FG.match_catalogs(a, b, radius_as=0.05)
    assert list(in_b) == [True, False, False]
    assert list(in_a) == [True, False]


def test_figure_pick_zooms_ranks_and_separates():
    from jwst_gc_pipeline.photometry.reference_fields import figures as FG
    rng = np.random.default_rng(4)
    # a cluster of 10 differences at (30, 30), 3 at (120, 120), none elsewhere
    xn = np.r_[30 + rng.uniform(-3, 3, 10), 120 + rng.uniform(-3, 3, 3)]
    yn = np.r_[30 + rng.uniform(-3, 3, 10), 120 + rng.uniform(-3, 3, 3)]
    z = FG.pick_zooms(xn, yn, np.array([]), np.array([]), (0, 160, 0, 160), 40, 4)
    assert len(z) == 2

    def holds(c, xs, ys):
        return np.all((np.abs(xs - c[0]) < 20) & (np.abs(ys - c[1]) < 20))
    assert holds(z[0], xn[:10], yn[:10])        # the bigger cluster first
    assert holds(z[1], xn[10:], yn[10:])
    # each zoom is centred on the differences it holds, also at the box edge
    assert np.hypot(z[0][0] - xn[:10].mean(), z[0][1] - yn[:10].mean()) < 1e-6
    z = FG.pick_zooms(np.array([2.0]), np.array([158.0]), np.array([]), np.array([]),
                      (0, 160, 0, 160), 40, 4)
    assert z == [(2.0, 158.0)]
    assert FG.pick_zooms(np.array([]), np.array([]), np.array([]), np.array([]),
                         (0, 160, 0, 160), 40, 4) == [(80.0, 80.0)]


# ---------------------------------------------------------------------------
# the reference tests
# ---------------------------------------------------------------------------

#: Opt-in: the variant label of finished reference runs of THIS checkout's
#: code (``reference_fields.run --variant <label>``).  Unset = skipped.
_VARIANT = os.environ.get('JWST_GC_REFFIELD_VARIANT', '')


def _run_problems(spec, variant):
    """Why the runs of ``variant`` cannot be scored against this checkout
    (empty list = every run exists and ran this checkout's pipeline code)."""
    from jwst_gc_pipeline.photometry.reference_fields import run as RUN
    problems = []
    for seed in [0] + list(spec['seeds']):
        rdir = RF.run_dir(spec, variant, seed)
        prov = RUN.read_provenance(rdir)
        if prov is None:
            problems.append(f'seed {seed}: no run, or no {RUN.PROVENANCE}, at {rdir}')
            continue
        if prov['dirty']:
            problems.append(f'seed {seed}: ran with uncommitted pipeline changes')
        try:
            diff = RUN.code_differences(prov['commit'])
        except subprocess.CalledProcessError:
            problems.append(f"seed {seed}: run commit {prov['commit'][:10]} unknown here")
            continue
        if diff:
            problems.append(f"seed {seed}: ran {prov['commit'][:10]}, whose pipeline "
                            f"code differs from this checkout ({', '.join(diff[:4])})")
    return problems


@pytest.mark.skipif(not _VARIANT, reason='opt-in: set JWST_GC_REFFIELD_VARIANT to '
                    'the label of finished reference runs of this checkout')
@pytest.mark.parametrize('name', sorted(_FIELDS))
def test_reference_field_passes(name):
    """The runs of the code under test pass the field's fixed thresholds.

    Every run (clean + each seed) must exist and must have run this
    checkout's pipeline code; a missing or stale run fails the test."""
    spec = _FIELDS[name]
    assert spec['thresholds'], f'{name}: no thresholds'
    problems = _run_problems(spec, _VARIANT)
    assert not problems, problems
    res = EV.evaluate_field(spec, _VARIANT)
    assert 'clean_missing' not in res and 'injected_missing' not in res, (
        res.get('clean_missing'), res.get('injected_missing'))
    assert EV.check(res, spec['thresholds']) == []
