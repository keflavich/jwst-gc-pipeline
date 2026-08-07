"""The per-detector report's static-term test must actually discriminate.

The report is the standing justification for keeping the offsets table
module-keyed (#340/#342).  If its test cannot tell a static term from a
field-varying one, the justification is decoration.
"""
import importlib.util
import os

import numpy as np
import pytest

_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                     'scripts', 'analysis', 'per_detector_offset_report.py')


def _mod():
    spec = importlib.util.spec_from_file_location('pdor', os.path.abspath(_PATH))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _dev(pattern, nfields=6, nper=40, noise=0.4, seed=0):
    """Synthetic deviations: {det: [(dra, ddec, field), ...]}."""
    rng = np.random.default_rng(seed)
    dev = {}
    for det, per_field in pattern.items():
        rows = []
        for i in range(nfields):
            mu = per_field(i)
            for _ in range(nper):
                rows.append((rng.normal(0, noise), mu + rng.normal(0, noise),
                             f'field{i}'))
        dev[det] = rows
    return dev


def test_a_STATIC_term_is_detected():
    """Same offset in every field -- a real distortion term.  If this does not
    read STATIC the report can never justify applying a correction."""
    m = _mod()
    res = m.analyse(_dev({'nrca1': lambda i: 2.0}), min_per_field=10)
    assert res['nrca1']['static'] is True
    assert res['nrca1']['mean'] == pytest.approx(2.0, abs=0.2)
    assert res['nrca1']['between_field_sd'] < 0.5


def test_a_SIGN_FLIPPING_term_is_rejected():
    """The live case: significant within each field, reversing between them."""
    m = _mod()
    res = m.analyse(_dev({'nrca2': lambda i: 0.6 * (-1) ** i}), min_per_field=10)
    assert res['nrca2']['static'] is False
    assert res['nrca2']['between_field_sd'] > abs(res['nrca2']['mean'])


def test_pure_noise_is_rejected():
    m = _mod()
    res = m.analyse(_dev({'nrca3': lambda i: 0.0}), min_per_field=10)
    assert res['nrca3']['static'] is False


def test_a_field_varying_term_with_a_nonzero_MEAN_is_still_rejected():
    """The subtle one.  These average to ~1.0 mas -- clearly nonzero -- but they
    are not ONE number, so it is not a static term and the test must not be
    fooled by the mean alone."""
    m = _mod()
    vals = [0.2, 0.3, 2.5, 0.2, 2.8, 0.2]
    res = m.analyse(_dev({'nrca4': lambda i: vals[i]}), min_per_field=10)
    assert res['nrca4']['static'] is False


def test_the_module_mean_is_removed_not_the_median():
    """With n=4 a median averages the middle two and discards the extremes, so
    a detector that IS offset partly defines its own reference."""
    m = _mod()
    rows = []
    for e in range(30):
        for det, off in (('nrca1', 3.0), ('nrca2', 0.0),
                         ('nrca3', 0.0), ('nrca4', 0.0)):
            rows.append(dict(field='f', date=f'd{e}', filt='F212N', visit='1',
                             exp=e, vgroup='', module='nrca', det=det,
                             dra=0.0, ddec=off))
    dev, ngroups = m.deviations(rows)
    assert ngroups == 30
    # mean removal: nrca1 keeps 3 - 0.75 = 2.25; a median would leave 3.0
    assert np.mean([d[1] for d in dev['nrca1']]) == pytest.approx(2.25, abs=1e-6)
    assert np.mean([d[1] for d in dev['nrca2']]) == pytest.approx(-0.75, abs=1e-6)


def test_groups_with_too_few_detectors_are_skipped():
    """A 'module mean' from one detector is that detector, and its deviation is
    identically zero -- which would dilute every real signal toward zero."""
    m = _mod()
    rows = [dict(field='f', date='d', filt='F212N', visit='1', exp=1,
                 vgroup='', module='nrca', det='nrca1', dra=0.0, ddec=5.0)]
    dev, ngroups = m.deviations(rows)
    assert ngroups == 0 and not dev


def test_the_clip_survives_an_arcsecond_outlier():
    """Bulk-repair epochs put arcsecond values in the same array as the mas-scale
    term.  One of them must not carry the mean."""
    m = _mod()
    a = [0.5] * 60 + [9000.0]
    mean, sem, n = m._robust(a)
    assert mean == pytest.approx(0.5, abs=0.05), mean
    assert n == 60
