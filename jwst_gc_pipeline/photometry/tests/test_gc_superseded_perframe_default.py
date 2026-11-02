"""``manual_gc_superseded_perframe`` defaults ON.

Before this change the in-run per-frame cleanup
(``cataloging._gc_perframe_images``) was opt-in: a completed run left every
intermediate phase's raw and mergedcat per-frame residual/model images on
disk.  A consumer audit found that every path that crosses a
phase boundary -- restart/``--manual-start-phase``, the per-frame SLURM
fan-out + completion markers, the m7 cross-band seed, the m8 forced fill, and
release staging/registration/QA -- reads either a protected mosaic
(``_i2d.fits``), a catalog, or the CURRENT phase's own raw per-frame pair
(kept).  None of them reads a phase's raw pair once a LATER phase's mosaic
exists, so the selection is safe to turn on by default.

This module pins three things a future change must not quietly undo:

1. the ``MANUAL_DEFAULTS`` entry is ``True``;
2. the CLI still offers an explicit opt-out
   (``--no-manual-gc-superseded-perframe``);
3. the barrier sequence keeps exactly what a retry needs -- THIS phase's own
   raw pair survives its own barrier (a retry of its mergedcat build still
   needs it) and is only removed once the NEXT phase's barrier fires, while
   the FINAL phase's raw pair is never removed at all (there is no later
   barrier to retire it).
"""
import os
import types

from jwst_gc_pipeline import retention
from jwst_gc_pipeline.photometry import cataloging
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

PRE = 'jw02221-o001_t001_nircam_clear-f410m'
FRAME = 'nrcalong_visit001_vgroup11101_exp00001'
PROPOSAL_ID = '2221'
FIELD = '001'
FILTERNAME = 'F410M'
CRF = f'/data/brick/{FILTERNAME}/pipeline/jw02221001001_02101_00001_nrcalong_crf.fits'


def test_manual_gc_superseded_perframe_defaults_on():
    assert MANUAL_DEFAULTS['manual_gc_superseded_perframe'] is True


def test_cli_still_offers_an_opt_out():
    """Grep-guard: the default flip must not remove the escape hatch.

    This does not build the full ``optparse`` parser (``main()`` reads
    ``sys.argv`` directly and runs far more than argument parsing), so it
    checks the registered option text instead -- the same style as this
    repo's other guard tests.
    """
    import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as _csl
    src_path = _csl.__file__
    with open(src_path) as fh:
        src = fh.read()
    assert "'--no-manual-gc-superseded-perframe'" in src
    assert "dest='manual_gc_superseded_perframe'" in src
    # the opt-out must be store_false, paired with the store_true flag
    i = src.index("'--no-manual-gc-superseded-perframe'")
    tail = src[i:i + 400]
    assert "action='store_false'" in tail


def _perframe(label, what='residual', mergedcat=False, frame=FRAME):
    tag = '_mergedcat' if mergedcat else ''
    return f'{PRE}-{frame}_{label}_daophot_basic{tag}_{what}.fits'


def _mosaic(label, what='residual_i2d', tokens=''):
    return f'{PRE}-merged{tokens}_{label}_daophot_basic_mergedcat_{what}.fits'


def _write(directory, name, size=16):
    p = os.path.join(directory, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'wb') as fh:
        fh.write(b'\0' * size)
    return p


def _write_phase_products(pipeline_dir, label, frame=CRF, tokens=''):
    """Everything one phase's build leaves: raw pair, mergedcat pair, mosaic,
    and the ledger naming the pairs (what ``run_manual_pipeline`` writes from
    ``build_mergedcat_residuals(perframe_record=...)``)."""
    raw = [_write(pipeline_dir, _perframe(label, w))
           for w in ('residual', 'model')]
    ren = [_write(pipeline_dir, _perframe(label, w, mergedcat=True))
           for w in ('residual', 'model')]
    i2d = _write(pipeline_dir, _mosaic(label, tokens=tokens))
    retention.write_perframe_ledger(
        i2d, [{'frame': frame, 'kind': 'basic', 'raw': raw, 'rendered': ren}],
        phase=label)
    return i2d


def _exists(pipeline_dir, label, mergedcat=False):
    return all(os.path.exists(os.path.join(
        pipeline_dir, _perframe(label, w, mergedcat=mergedcat)))
        for w in ('residual', 'model'))


def test_barrier_keeps_current_phase_and_final_phase_raw_pair(tmp_path):
    pipeline_dir = str(tmp_path / FILTERNAME / 'pipeline')
    i2d = {label: _write_phase_products(pipeline_dir, label)
           for label in ('m3', 'm4', 'm5')}

    # -- m3 barrier: no previous mosaic, so no raw pair to retire.
    cataloging._gc_perframe_images(i2d['m3'], None, None, FILTERNAME, 'm3')
    assert _exists(pipeline_dir, 'm3'), \
        "m3's own raw pair must survive its own barrier (retry)"
    assert not _exists(pipeline_dir, 'm3', mergedcat=True), \
        "m3's spent mergedcat render should be gone"
    assert os.path.exists(i2d['m3'])

    # -- m4 barrier: retires m3's raw pair, keeps m4's own for its retry.
    cataloging._gc_perframe_images(i2d['m4'], i2d['m3'], None, FILTERNAME, 'm4')
    assert not _exists(pipeline_dir, 'm3'), \
        "m3's raw pair is unreachable once m4 exists"
    assert _exists(pipeline_dir, 'm4')
    assert not _exists(pipeline_dir, 'm4', mergedcat=True)

    # -- m5 barrier (final): retires m4's; m5's own is never retired.
    cataloging._gc_perframe_images(i2d['m5'], i2d['m4'], None, FILTERNAME, 'm5')
    assert not _exists(pipeline_dir, 'm4')
    assert _exists(pipeline_dir, 'm5'), \
        "the FINAL phase's raw pair is never removed"
    assert not _exists(pipeline_dir, 'm5', mergedcat=True)
    assert all(os.path.exists(p) for p in i2d.values()), \
        "mosaics are never touched"


def test_barrier_leaves_files_no_ledger_names(tmp_path):
    """Another observation, detector or variant in the same directory.

    The pre-ledger glob ``{prefix}*-{filt}-*_{label}_daophot_*`` matched all
    three, so a concurrent chain lost its pairs and cloudef's obs005 frames
    (``-o002_`` prefix, shared directory) went with obs002's cleanup.
    """
    pipeline_dir = str(tmp_path / FILTERNAME / 'pipeline')
    m3 = _write_phase_products(pipeline_dir, 'm3')
    m4 = _write_phase_products(pipeline_dir, 'm4')
    bystanders = [
        _write(pipeline_dir, _perframe('m3', frame=FRAME.replace('nrcalong',
                                                                'nrcblong'))),
        _write(pipeline_dir, _perframe('m3', mergedcat=True,
                                       frame=FRAME.replace('nrcalong',
                                                           'nrcblong'))),
        _write(pipeline_dir, _perframe('m3').replace('_m3_', '_resbgsub_m3_')),
        _write(pipeline_dir, _perframe('m4', mergedcat=True)
               .replace('_m4_', '_epsf_hybpsf_m4_')),
    ]
    cataloging._gc_perframe_images(m4, m3, None, FILTERNAME, 'm4')
    assert not _exists(pipeline_dir, 'm3')
    assert all(os.path.exists(p) for p in bystanders)


def test_barrier_without_ledger_removes_nothing(tmp_path):
    """A mosaic from before ledgers existed offers nothing to the cleanup."""
    pipeline_dir = str(tmp_path / FILTERNAME / 'pipeline')
    m3 = _write_phase_products(pipeline_dir, 'm3')
    m4 = _write_phase_products(pipeline_dir, 'm4')
    for i2d in (m3, m4):
        os.unlink(retention.perframe_ledger_path(i2d))
    cataloging._gc_perframe_images(m4, m3, None, FILTERNAME, 'm4')
    assert _exists(pipeline_dir, 'm3')
    assert _exists(pipeline_dir, 'm4', mergedcat=True)


def test_retiring_a_raw_pair_removes_its_completion_markers(tmp_path):
    """A --skip-if-done restart of a retired phase must refit, not resume.

    Without this, ``select_resumable_frames`` reads the m3 marker as done, the
    frame is skipped, and ``build_mergedcat_residuals`` raises on the m3 raw
    pair the m4 barrier deleted.
    """
    pipeline_dir = str(tmp_path / FILTERNAME / 'pipeline')
    marker_dir = str(tmp_path / 'catalogs' / '_perframe_markers')
    os.makedirs(marker_dir)
    crf = _write(pipeline_dir, os.path.basename(CRF))
    m3 = _write_phase_products(pipeline_dir, 'm3', frame=crf)
    m4 = _write_phase_products(pipeline_dir, 'm4', frame=crf)
    old = os.path.getmtime(crf) - 60
    os.utime(crf, (old, old))     # markers must be newer than the frame
    det = cataloging.perframe_detector_token(crf)
    markers = {}
    for phase in ('m3', 'm4'):
        markers[phase] = [
            cataloging.perframe_marker_path(marker_dir, crf, det, FILTERNAME,
                                            phase, 'ok', merge=m)
            for m in ('nrca', 'merged')]
        for p in markers[phase]:
            open(p, 'w').close()
    frame_args = [{'filename': crf}]
    todo, ok, _, _ = cataloging.select_resumable_frames(
        frame_args, marker_dir, FILTERNAME, 'm3', 'nrca')
    assert ok == [crf] and todo == []

    cataloging._gc_perframe_images(m4, m3, marker_dir, FILTERNAME, 'm4')

    assert not any(os.path.exists(p) for p in markers['m3'])
    assert all(os.path.exists(p) for p in markers['m4']), \
        "the current phase's markers stay: its raw pair stays too"
    todo, ok, _, _ = cataloging.select_resumable_frames(
        frame_args, marker_dir, FILTERNAME, 'm3', 'nrca')
    assert todo == frame_args and ok == []
    assert retention.read_perframe_ledger(m3)['raw_retired'] is True


def test_opt_out_suppresses_every_deletion(tmp_path):
    """``--no-manual-gc-superseded-perframe`` keeps everything."""
    pipeline_dir = str(tmp_path / FILTERNAME / 'pipeline')
    m3 = _write_phase_products(pipeline_dir, 'm3')
    m4 = _write_phase_products(pipeline_dir, 'm4')
    before = sorted(os.listdir(pipeline_dir))

    off = types.SimpleNamespace(manual_gc_superseded_perframe=False)
    cataloging._gc_perframe_after_barrier(off, m4, m3, None, FILTERNAME, 'm4')
    assert sorted(os.listdir(pipeline_dir)) == before

    on = types.SimpleNamespace(manual_gc_superseded_perframe=True)
    cataloging._gc_perframe_after_barrier(on, m4, m3, None, FILTERNAME, 'm4')
    assert not _exists(pipeline_dir, 'm3')
    assert not _exists(pipeline_dir, 'm4', mergedcat=True)


def test_variant_run_never_retires_production_pairs(tmp_path):
    """A hybrid/ePSF/blur run whose prev-mosaic rebuild lands on production's.

    Before #1111 the per-phase rebuild dropped ``_hybpsf``/``_epsf``/``_blur``,
    so a variant run's m4 barrier was handed production's m3 mosaic.  Its
    ledger must be refused: those raw pairs and markers are production's.
    """
    pipeline_dir = str(tmp_path / FILTERNAME / 'pipeline')
    marker_dir = str(tmp_path / 'catalogs' / '_perframe_markers')
    os.makedirs(marker_dir)
    prod_m3 = _write_phase_products(pipeline_dir, 'm3')
    hyb_m4 = _write_phase_products(pipeline_dir, 'm4', tokens='_hybpsf')
    marker = cataloging.perframe_marker_path(
        marker_dir, CRF, cataloging.perframe_detector_token(CRF), FILTERNAME,
        'm3', 'ok', merge='merged')
    open(marker, 'w').close()

    cataloging._gc_perframe_images(hyb_m4, prod_m3, marker_dir, FILTERNAME,
                                   'm4')
    assert _exists(pipeline_dir, 'm3'), "production's m3 raw pair must stay"
    assert os.path.exists(marker), "production's m3 marker must stay"
    assert retention.read_perframe_ledger(prod_m3)['raw_retired'] is False


def test_resbgsub_step_is_the_same_run(tmp_path):
    """m4 -> m5 adds ``_resbgsub``; that is still one run."""
    pipeline_dir = str(tmp_path / FILTERNAME / 'pipeline')
    m4 = _write_phase_products(pipeline_dir, 'm4', tokens='_hybpsf')
    m5 = _write_phase_products(pipeline_dir, 'm5',
                               tokens='_resbgsub_hybpsf')
    assert retention.same_run_mosaics(m5, m4)
    cataloging._gc_perframe_images(m5, m4, None, FILTERNAME, 'm5')
    assert not _exists(pipeline_dir, 'm4')


def test_ledger_writer_keeps_only_its_own_kind(tmp_path):
    rec = [{'frame': CRF, 'kind': 'basic', 'raw': ['a'], 'rendered': ['b']},
           {'frame': CRF, 'kind': 'iterative', 'raw': ['c'], 'rendered': ['d']}]
    i2d = str(tmp_path / _mosaic('m4'))
    cataloging._write_mergedcat_ledger(i2d, rec, 'm4')
    assert [r['kind'] for r in retention.read_perframe_ledger(i2d)['frames']] \
        == ['basic']
