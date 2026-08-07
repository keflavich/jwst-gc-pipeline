"""`--skip-if-done` / `--list-missing-tasks` must match what the MANUAL
per-frame path actually writes (issue #333).

The fan-out is `afterok` and all-or-nothing: one shard hitting its wall clock
leaves the finalize permanently `DependencyNeverSatisfied` and discards every
other shard's completed work.  So a predictor that never matches does not merely
waste time, it makes a timed-out fan-out unresumable -- measured on the live
sgrb2 m12 fan-out (38867646), which reported all 96 tasks missing with all 96
on disk.

Two independent mismatches, both fixed here:

  * METHOD.  With `--daophot` unset the sentinel was `_crowdsource_nsky0`.  The
    manual path writes `_daophot_basic` REGARDLESS of that flag --
    `_save_manual_pass` (cataloging.py:2062) hardcodes
    `basic_or_iterative='basic'` and never reads `options.daophot`.
  * PHASE.  The real writer emits `..._exp00023_m2_daophot_basic.fits`.  The
    `iter_` slot in `_predict_tblfilename` can carry that, but nothing on the
    manual path populated it.

These tests write real files and ask `_expected_output_exists`, so a predictor
that is merely *spelled* right but does not match the writer fails.
"""
import os

import pytest

from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as C


class _Options:
    """Just the attributes the predictor reads."""
    def __init__(self, **kw):
        self.daophot = False
        self.basic_only = False
        self.desaturated = False
        self.bgsub = False
        self.use_iter3_residual_bg = False
        self.epsf = False
        self.blur = False
        self.group = False
        self.iteration_label = ''
        self.field = '001'
        self.proposal_id = '5365'
        self.manual_start_phase = ''
        self.manual_stop_after_phase = ''
        self.__dict__.update(kw)


# The exact shapes observed on disk in /orange/adamginsburg/jwst/sgrb2/F150W
# (24 per detector per phase).  Captured as literals so this is an external
# oracle rather than the predictor checking itself.
REAL_NAMES = [
    'f150w_nrca1_visit001_vgroup13101_exp00001_m1_daophot_basic.fits',
    'f150w_nrca1_visit001_vgroup13101_exp00001_m2_daophot_basic.fits',
    'f150w_nrca1_visit001_vgroup13101_exp00001_m3_daophot_basic.fits',
    'f150w_nrca1_visit001_vgroup13101_exp00001_m4_daophot_basic.fits',
    'f150w_nrca1_visit001_vgroup13101_exp00001_resbgsub_m5_daophot_basic.fits',
    'f150w_nrca1_visit001_vgroup13101_exp00001_resbgsub_m6_daophot_basic.fits',
]

TUPLE = dict(visit_id='001', vgroup_id='13101', exposure_id='00001')


def _touch(basepath, filtername, name):
    d = os.path.join(basepath, filtername)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    with open(path, 'w') as fh:
        fh.write('')
    return path


def _exists(basepath, options):
    return C._expected_output_exists(basepath, 'F150W', 'nrca1', options,
                                     **TUPLE)


# ---------------------------------------------------------------------------
# the sentinel label
# ---------------------------------------------------------------------------

def test_m12_writes_two_files_and_the_SECOND_is_the_sentinel():
    """`--manual-stop-after-phase=m12` runs BOTH m1 and m2 (cataloging.py:2271
    and :2291).  A job that died between them is not done, so the sentinel has
    to be the last write, not the first."""
    assert C._manual_sentinel_label(_Options(manual_stop_after_phase='m12')) == 'm2'


@pytest.mark.parametrize('phase', ['m3', 'm4', 'm5', 'm6'])
def test_every_other_phase_is_labelled_with_itself(phase):
    assert C._manual_sentinel_label(_Options(manual_stop_after_phase=phase)) == phase


def test_no_stop_phase_is_not_a_per_frame_sentinel():
    """A run that starts partway and goes to the END has no single per-frame
    output, so it must not claim one."""
    assert C._manual_sentinel_label(_Options()) is None
    assert C._manual_sentinel_label(_Options(manual_start_phase='m7')) is None


# ---------------------------------------------------------------------------
# behaviour: real files on disk
# ---------------------------------------------------------------------------

def test_a_completed_m12_frame_is_reported_DONE(tmp_path):
    """The #333 reproduction, inverted: with the work on disk, the predictor
    must say done.  On main this returned False for all 96 sgrb2 tasks."""
    bp = str(tmp_path)
    opts = _Options(manual_start_phase='m12', manual_stop_after_phase='m12')
    assert not _exists(bp, opts), 'nothing written yet'
    _touch(bp, 'F150W',
           'f150w_nrca1_visit001_vgroup13101_exp00001_m2_daophot_basic.fits')
    assert _exists(bp, opts), 'the m2 file is on disk and was not matched'


def test_an_m12_that_died_between_m1_and_m2_is_NOT_done(tmp_path):
    bp = str(tmp_path)
    opts = _Options(manual_start_phase='m12', manual_stop_after_phase='m12')
    _touch(bp, 'F150W',
           'f150w_nrca1_visit001_vgroup13101_exp00001_m1_daophot_basic.fits')
    assert not _exists(bp, opts), 'only the m1 pass finished; that is not done'


def test_the_manual_path_ignores_the_daophot_FLAG(tmp_path):
    """The first of the two defects.  `--daophot` is unset on the live fan-out,
    which used to select `_crowdsource_nsky0`; the manual writer emits
    `_daophot_basic` either way.  Both flag states must find the same file."""
    bp = str(tmp_path)
    _touch(bp, 'F150W',
           'f150w_nrca1_visit001_vgroup13101_exp00001_m2_daophot_basic.fits')
    for daophot in (False, True):
        opts = _Options(daophot=daophot, basic_only=False,
                        manual_stop_after_phase='m12')
        assert _exists(bp, opts), f'daophot={daophot} did not match the writer'


def test_basic_only_does_not_change_the_manual_sentinel(tmp_path):
    """`--basic-only` selects basic-vs-iterative on the NON-manual path.  The
    manual writer is always basic, so this flag must not move the prediction."""
    bp = str(tmp_path)
    _touch(bp, 'F150W',
           'f150w_nrca1_visit001_vgroup13101_exp00001_m2_daophot_basic.fits')
    for basic_only in (False, True):
        opts = _Options(daophot=True, basic_only=basic_only,
                        manual_stop_after_phase='m12')
        assert _exists(bp, opts)


@pytest.mark.parametrize('phase,name', [
    ('m3', 'f150w_nrca1_visit001_vgroup13101_exp00001_m3_daophot_basic.fits'),
    ('m4', 'f150w_nrca1_visit001_vgroup13101_exp00001_m4_daophot_basic.fits'),
])
def test_single_phase_frames_match_their_own_file(tmp_path, phase, name):
    bp = str(tmp_path)
    opts = _Options(manual_start_phase=phase, manual_stop_after_phase=phase)
    assert not _exists(bp, opts)
    _touch(bp, 'F150W', name)
    assert _exists(bp, opts)


def test_the_bgsub_token_still_participates(tmp_path):
    """m5/m6 run under `--use-iter3-residual-bg`, and the real files carry
    `_resbgsub` BEFORE the phase token.  If the manual override dropped the
    other tokens the prediction would match the wrong file -- or none."""
    bp = str(tmp_path)
    opts = _Options(use_iter3_residual_bg=True, manual_stop_after_phase='m5')
    _touch(bp, 'F150W',
           'f150w_nrca1_visit001_vgroup13101_exp00001_resbgsub_m5_daophot_basic.fits')
    assert _exists(bp, opts)
    # and the un-flagged run must NOT match it
    assert not _exists(bp, _Options(manual_stop_after_phase='m5'))


def test_predictions_reproduce_the_shapes_observed_on_disk(tmp_path):
    """Cross-check against filenames captured from the live sgrb2 tree, so the
    predictor is measured against the writer rather than against itself."""
    bp = str(tmp_path)
    cases = [
        (_Options(manual_stop_after_phase='m12'), REAL_NAMES[1]),
        (_Options(manual_stop_after_phase='m3'), REAL_NAMES[2]),
        (_Options(manual_stop_after_phase='m4'), REAL_NAMES[3]),
        (_Options(use_iter3_residual_bg=True, manual_stop_after_phase='m5'), REAL_NAMES[4]),
        (_Options(use_iter3_residual_bg=True, manual_stop_after_phase='m6'), REAL_NAMES[5]),
    ]
    for opts, name in cases:
        predicted = C._predict_tblfilename(
            bp, 'F150W', 'nrca1', opts, **TUPLE,
            iteration_label=C._manual_sentinel_label(opts),
            method='daophot', basic_or_iterative='basic')
        assert os.path.basename(predicted) == name, (
            f'predicted {os.path.basename(predicted)}, disk has {name}')


# ---------------------------------------------------------------------------
# the non-manual path must be untouched
# ---------------------------------------------------------------------------

def test_the_crowdsource_path_is_unchanged(tmp_path):
    bp = str(tmp_path)
    opts = _Options()          # no manual phase at all
    _touch(bp, 'F150W',
           'f150w_nrca1_visit001_vgroup13101_exp00001_crowdsource_nsky0.fits')
    assert _exists(bp, opts)


def test_the_daophot_iterative_path_is_unchanged(tmp_path):
    bp = str(tmp_path)
    opts = _Options(daophot=True)
    _touch(bp, 'F150W',
           'f150w_nrca1_visit001_vgroup13101_exp00001_daophot_iterative.fits')
    assert _exists(bp, opts)
