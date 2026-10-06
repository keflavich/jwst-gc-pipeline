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

from jwst_gc_pipeline.photometry import cataloging
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

PRE = 'jw02221-o001_t001_nircam_clear-f410m'
FRAME = 'nrcalong_visit001_vgroup11101_exp00001'
PROPOSAL_ID = '2221'
FIELD = '001'
FILTERNAME = 'F410M'


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


def _perframe(label, what='residual', mergedcat=False):
    tag = '_mergedcat' if mergedcat else ''
    return f'{PRE}-{FRAME}_{label}_daophot_basic{tag}_{what}.fits'


def _mosaic(label, what='residual_i2d'):
    return f'{PRE}-merged_{label}_daophot_basic_mergedcat_{what}.fits'


def _write(directory, name, size=16):
    p = os.path.join(directory, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'wb') as fh:
        fh.write(b'\0' * size)
    return p


def _write_phase_products(pipeline_dir, label):
    """Everything one phase writes: raw pair, mergedcat pair, mosaic.

    Mirrors the real on-disk shape closely enough for the glob-based
    selectors in ``retention.py``: the mosaic must exist for
    ``spent_mergedcat_frames`` to fire, and the raw/mergedcat pairs must not
    collide with each other's regex (``PERFRAME_MERGEDCAT_RE`` is tried
    before ``PERFRAME_RAW_RE``).
    """
    for what in ('residual', 'model'):
        _write(pipeline_dir, _perframe(label, what))
        _write(pipeline_dir, _perframe(label, what, mergedcat=True))
    _write(pipeline_dir, _mosaic(label))


def test_barrier_keeps_current_phase_and_final_phase_raw_pair(tmp_path):
    cut_bp = str(tmp_path)
    pipeline_dir = os.path.join(cut_bp, FILTERNAME, 'pipeline')
    phases = ['m3', 'm4', 'm5']

    for label in phases:
        _write_phase_products(pipeline_dir, label)

    def raw_exists(label):
        return (os.path.exists(os.path.join(pipeline_dir, _perframe(label, 'residual')))
                and os.path.exists(os.path.join(pipeline_dir, _perframe(label, 'model'))))

    def mergedcat_exists(label):
        return (os.path.exists(os.path.join(pipeline_dir, _perframe(label, 'residual', mergedcat=True)))
                and os.path.exists(os.path.join(pipeline_dir, _perframe(label, 'model', mergedcat=True))))

    def mosaic_exists(label):
        return os.path.exists(os.path.join(pipeline_dir, _mosaic(label)))

    # -- m3 barrier: first phase in the list, so no PREVIOUS phase to retire.
    cataloging._gc_perframe_images(cut_bp, PROPOSAL_ID, FIELD, FILTERNAME, 'm3', phases)
    assert raw_exists('m3'), "m3's own raw pair must survive its own barrier (retry)"
    assert not mergedcat_exists('m3'), "m3's spent mergedcat render should be gone"
    assert mosaic_exists('m3')

    # -- m4 barrier: retires m3's raw pair, keeps m4's own for its retry.
    cataloging._gc_perframe_images(cut_bp, PROPOSAL_ID, FIELD, FILTERNAME, 'm4', phases)
    assert not raw_exists('m3'), "m3's raw pair is unreachable once m4 exists"
    assert raw_exists('m4'), "m4's own raw pair must survive its own barrier (retry)"
    assert not mergedcat_exists('m4')
    assert mosaic_exists('m3') and mosaic_exists('m4'), "mosaics are never touched"

    # -- m5 barrier (final phase in this run): retires m4's raw pair; m5's own
    # raw pair is never retired because there is no later phase.
    cataloging._gc_perframe_images(cut_bp, PROPOSAL_ID, FIELD, FILTERNAME, 'm5', phases)
    assert not raw_exists('m4'), "m4's raw pair is unreachable once m5 exists"
    assert raw_exists('m5'), "the FINAL phase's raw pair is never removed"
    assert not mergedcat_exists('m5')
    assert mosaic_exists('m3') and mosaic_exists('m4') and mosaic_exists('m5')
