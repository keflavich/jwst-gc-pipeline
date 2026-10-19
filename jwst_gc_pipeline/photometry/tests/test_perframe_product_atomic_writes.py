"""Per-frame products written by two fan-out tasks at once must stay readable.

With ``MODULES=nrca,nrcb,merged`` the module pass and the ``merged`` pass both
fit every NIRCam frame (#840), and the fan-out spreads their labels over
different shards, so two tasks can fit the same frame at the same moment.  Both
write the same detector-keyed products: the satstar catalog / rejected / flags
/ model / residual, the per-frame daophot catalog, and the per-frame residual
and model images.

Written in place, a reader in the other task can open one of those files
between the writer's create and close.  10678 o098 m5 (2026-10-18): the two
tasks fitting nrcblong exp5 both wrote ``*_m5_satstar_rejected.fits`` at 18:45,
the file ended at zero bytes, both tasks logged ``gate-reject hand-off skipped:
unreadable`` and exited 0, and the m5 finalize died four hours later with
``IORegistryError: Format could not be identified``.  o087 m5 repeated it.

These tests pin both halves of the fix:

* every write goes to a hidden temporary and is ``os.replace``-d into
  position, so a reader sees the old file or the new one (the mechanism is
  shown deterministically: the reader runs at a fixed point inside the write);
* the hand-off reader raises on an existing file it cannot parse, so a corrupt
  product fails the fan-out task in minutes instead of the finalize in hours.
"""
import ast
import os
import pathlib
import shutil

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.photometry import cataloging as C
from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as L
from jwst_gc_pipeline.reduction import saturated_star_finding as SSF

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _table(n):
    return Table({'xcentroid': np.arange(n, dtype=float),
                  'ycentroid': np.arange(n, dtype=float),
                  'reject_reason': ['implied_peak_gate'] * n})


def _no_debris(directory):
    """Nothing but finished products left behind -- no temporaries, hidden or not."""
    return sorted(p.name for p in pathlib.Path(directory).iterdir()
                  if '.tmp' in p.name or p.name.startswith('.building'))


# --- the satstar products ----------------------------------------------------

def test_a_reader_mid_write_sees_the_old_satstar_file_whole(tmp_path):
    path = str(tmp_path / 'f_m5_satstar_rejected.fits')
    _table(3).write(path)
    seen = []

    def write(p):
        assert p != path
        assert os.path.basename(p).startswith('.')
        # The new table is fully on disk under the temporary name, and the
        # published path still holds the previous table, whole.
        _table(7).write(p, overwrite=True)
        seen.append(len(Table.read(path)))

    SSF._publish_satstar_product(path, write, overwrite=True)
    assert seen == [3]
    assert len(Table.read(path)) == 7
    assert _no_debris(tmp_path) == []


def test_an_in_place_write_is_what_the_reader_used_to_catch(tmp_path):
    """The failure the helper prevents, reproduced: a zero-byte file in place."""
    path = tmp_path / 'f_m5_satstar_rejected.fits'
    path.write_bytes(b'')        # a writer that has created the file, not filled it
    with pytest.raises(Exception, match='identified|empty|SIMPLE'):
        Table.read(str(path))


def test_overwrite_false_still_refuses_an_existing_product(tmp_path):
    path = str(tmp_path / 'f_m5_satstar_catalog.fits')
    _table(2).write(path)
    with pytest.raises(OSError, match='already exists'):
        SSF._publish_satstar_product(
            path, lambda p: _table(5).write(p, overwrite=True), overwrite=False)
    assert len(Table.read(path)) == 2


def test_a_failed_write_leaves_the_old_product_and_no_temporary(tmp_path):
    path = str(tmp_path / 'f_m5_satstar_model.fits')
    fits.PrimaryHDU(np.ones((4, 4), dtype='float32')).writeto(path)

    def write(p):
        with open(p, 'wb') as fh:
            fh.write(b'SIMPLE  =')       # a partial header, then the writer dies
        raise OSError('disk went away')

    with pytest.raises(OSError, match='disk went away'):
        SSF._publish_satstar_product(path, write, overwrite=True)
    assert fits.getdata(path).shape == (4, 4)
    assert _no_debris(tmp_path) == []


def _calls_publish(node):
    return (isinstance(node, ast.Call)
            and getattr(node.func, 'id', None) == '_publish_satstar_product')


def test_every_write_in_remove_saturated_stars_is_published():
    """Structural: no ``.write``/``.writeto`` in the satstar writer outside the helper.

    The race needs two concurrent shards fitting one frame, which CI does not
    have, so this is what stops a revert.  An AST walk rather than a grep: the
    call has to sit inside a lambda handed to ``_publish_satstar_product``, not
    merely near one.
    """
    src = (ROOT / 'jwst_gc_pipeline/reduction/saturated_star_finding.py').read_text()
    tree = ast.parse(src)
    func = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == 'remove_saturated_stars')
    published = set()
    for call in ast.walk(func):
        if _calls_publish(call):
            for arg in call.args:
                if isinstance(arg, ast.Lambda):
                    published.update(id(n) for n in ast.walk(arg))
    writes = [n for n in ast.walk(func)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr in ('write', 'writeto')]
    assert len(writes) >= 7          # catalog, rejected, model, residual, flags, wingcal, severity
    bare = [n.lineno for n in writes if id(n) not in published]
    assert bare == [], f'saturated_star_finding.py lines {bare} write a product in place'


# --- the per-frame daophot catalog -------------------------------------------

@pytest.mark.parametrize('scratch', [False, True])
def test_the_perframe_catalog_is_published_atomically(tmp_path, monkeypatch, scratch):
    out_dir = tmp_path / 'pipeline'
    out_dir.mkdir()
    final = str(out_dir / 'f212n_nrca1_o098_exp00005_m5_daophot_basic.fits')
    _table(3).write(final)
    if scratch:
        local = tmp_path / 'local'
        local.mkdir()
        monkeypatch.setenv('SLURM_TMPDIR', str(local))
    else:
        monkeypatch.delenv('SLURM_TMPDIR', raising=False)
    seen = []

    def write_fn(p):
        assert p != final
        _table(9).write(p, overwrite=True)
        seen.append(len(Table.read(final)))

    L.write_via_local_scratch(final, write_fn)
    assert seen == [3]
    assert len(Table.read(final)) == 9
    assert _no_debris(out_dir) == []


def test_the_scratch_copy_does_not_land_in_place(tmp_path, monkeypatch):
    """The copy from node-local scratch goes to a temporary, never to ``final``."""
    out_dir = tmp_path / 'pipeline'
    out_dir.mkdir()
    local = tmp_path / 'local'
    local.mkdir()
    monkeypatch.setenv('SLURM_TMPDIR', str(local))
    final = str(out_dir / 'cat.fits')
    targets = []
    real_copy = shutil.copyfile

    def spy(src, dst, *a, **k):
        targets.append(dst)
        return real_copy(src, dst, *a, **k)

    monkeypatch.setattr(shutil, 'copyfile', spy)
    L.write_via_local_scratch(final, lambda p: _table(4).write(p, overwrite=True))
    assert len(targets) == 1 and targets[0] != final
    assert os.path.basename(targets[0]).startswith('.')
    assert len(Table.read(final)) == 4


# --- the per-frame residual / model images -----------------------------------

def test_residual_datamodel_is_published_under_its_real_name(tmp_path):
    """``ImageModel.save`` stamps the basename it is given into ``meta.filename``,
    so the residual is staged under its real name in a private directory."""
    from astropy.modeling import models
    from gwcs import WCS as GWCS
    from stdatamodels.jwst.datamodels import ImageModel
    src = str(tmp_path / 'jw_nrca1_crf.fits')
    with ImageModel(np.zeros((8, 8), dtype='float32')) as m:
        m.meta.wcs = GWCS(models.Shift(0) & models.Shift(0),
                          input_frame='detector', output_frame='world')
        m.save(src)
    out_dir = tmp_path / 'pipeline'
    out_dir.mkdir()
    out = str(out_dir / 'f212n_nrca1_m5_daophot_basic_residual.fits')
    L.save_residual_datamodel(src, out, np.full((8, 8), 2.0, dtype='float32'))
    with ImageModel(out) as m:
        assert m.meta.filename == os.path.basename(out)
        assert float(m.data[0, 0]) == 2.0
    assert _no_debris(out_dir) == []
    assert sorted(os.listdir(out_dir)) == [os.path.basename(out)]


# --- the reader ---------------------------------------------------------------

@pytest.mark.parametrize('content', [b'', b' ' * 2880])
def test_gate_reject_handoff_raises_on_a_corrupt_rejected_file(tmp_path, content):
    path = tmp_path / 'f_m5_satstar_rejected.fits'
    path.write_bytes(content)
    with pytest.raises(RuntimeError, match='unreadable.*rerun the fan-out task'):
        C._gate_reject_handoff_xy(str(path), None, 2.0, label='t')


def test_severity_drop_handoff_raises_on_a_corrupt_file(tmp_path):
    path = tmp_path / 'f_m5_satstar_severity_dropped.fits'
    path.write_bytes(b'')
    with pytest.raises(RuntimeError, match='severity-drop hand-off'):
        C._severity_drop_handoff_xy(str(path), None, 2.0, label='t')


def test_a_missing_handoff_file_is_still_no_handoff(tmp_path):
    missing = str(tmp_path / 'absent.fits')
    assert C._gate_reject_handoff_xy(missing, None, 2.0) == (None, 0.0)
    assert C._severity_drop_handoff_xy(missing, None, 2.0) == (None, 0.0)
