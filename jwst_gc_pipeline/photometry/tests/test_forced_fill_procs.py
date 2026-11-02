"""m8 forced fill with --forced-fill-procs: frames prepared and fitted in
forked processes give the serial fill's catalog, bit for bit (#1230).

Each frame carries its own noise and is offset by a sub-pixel shift, so the
per-frame fluxes differ and the inverse-variance sums depend on the order in
which they are added; a fill that summed the frames in another order would
not reproduce the serial result exactly.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry import cataloging
from jwst_gc_pipeline.photometry import forced_fill as ff
from jwst_gc_pipeline.photometry.tests.test_forced_fill import _MockPSF, _build

pytestmark = pytest.mark.filterwarnings(
    'ignore:This process .* is multi-threaded:DeprecationWarning')

FRAMES = [f'jw01234001001_02101_{k:05d}_nrcalong_crf.fits' for k in range(1, 8)]


def _frames_prepare(pids_dir=None, fail=None):
    """A prepare_frame giving each frame its own noise and WCS shift, built
    around ``_build``'s injected sources.  Writes ``<pid>_<frame index>`` under
    ``pids_dir`` for every frame it prepares; raises ``fail[name]`` for the
    frames named there."""
    _, prepare0, *_ = _build()
    base = prepare0()
    ny, nx = base.nan_replaced_data.shape
    # _build's noise is the first draw from default_rng(0); remove it
    clean = base.nan_replaced_data - np.random.default_rng(0).normal(
        0, 1.0, (ny, nx)).astype('float32')

    def prepare_frame(name):
        if fail is not None and name in fail:
            raise fail[name]
        k = FRAMES.index(name)
        if pids_dir is not None:
            (pids_dir / f'{os.getpid()}_{k}').write_text('')
        ww = WCS(base.ww.to_header())
        ww.wcs.crpix = [ww.wcs.crpix[0] + 0.1 * k, ww.wcs.crpix[1] - 0.07 * k]
        noise = np.random.default_rng(100 + k).normal(0, 1.0, (ny, nx))
        return SimpleNamespace(nan_replaced_data=(clean + noise).astype('float32'),
                               err=np.ones((ny, nx), 'float32'),
                               mask=np.zeros((ny, nx), bool), ww=ww,
                               dao_psf_model=_MockPSF())
    return prepare_frame


def _fill(n_procs, **kw):
    tbl, *_ = _build()
    prepare_frame = _frames_prepare(**kw)
    nrec = ff.forced_fill_band(tbl, 'f405n', FRAMES, prepare_frame=prepare_frame,
                               frame_arg_builder=lambda fn: {'name': fn},
                               nsigma=3.0, fit_shape=(5, 5), verbose=True,
                               n_procs=n_procs)
    return nrec, tbl


def _assert_tables_identical(a, b):
    assert a.colnames == b.colnames
    for col in a.colnames:
        if col == 'skycoord_ref':
            continue
        x, y = np.asarray(a[col]), np.asarray(b[col])
        assert x.dtype == y.dtype, col
        if x.dtype.kind == 'f':
            assert np.array_equal(x, y, equal_nan=True), col
        else:
            assert np.array_equal(x, y), col


def test_forked_fill_equals_the_serial_fill(tmp_path, capfd):
    pids = tmp_path / 'pids'
    pids.mkdir()
    nrec1, serial = _fill(1)
    nrec3, forked = _fill(3, pids_dir=pids)
    assert nrec1 == nrec3 == 2
    _assert_tables_identical(serial, forked)
    assert bool(serial['forced_filled_f405n'].any())
    # every frame was prepared in a child, none here
    seen = [p.name.split('_') for p in pids.iterdir()]
    assert sorted(int(k) for _, k in seen) == list(range(len(FRAMES)))
    assert str(os.getpid()) not in {pid for pid, _ in seen}
    assert '7 frames across 3 processes (--forced-fill-procs)' in capfd.readouterr().out


def test_the_frame_order_matters_to_the_sums():
    """Guard on the test above: with these frames, summing in another order
    changes the result, so the equality there is not automatic."""
    tbl, *_ = _build()
    prepare_frame = _frames_prepare()
    ref = tbl['skycoord_ref']
    tgt = np.where(np.asarray(tbl['mask_f405n'], bool))[0]
    ra, dec = ref.ra.deg[tgt], ref.dec.deg[tgt]
    fits_ = [ff._forced_fit_frame(fn, 'f405n', ra, dec,
                                  prepare_frame=prepare_frame,
                                  frame_arg_builder=lambda fn: {'name': fn},
                                  fit_shape=(5, 5)) for fn in FRAMES]

    def _sum(order):
        wfsum = np.zeros(tgt.size)
        for i in order:
            loc, f, var = fits_[i]
            wfsum[loc] += f / var
        return wfsum

    forward = _sum(range(len(FRAMES)))
    assert any(not np.array_equal(forward, _sum(order)) for order in
               (range(len(FRAMES) - 1, -1, -1), [3, 0, 6, 1, 5, 2, 4]))


def test_a_failed_frame_prep_in_a_child_is_printed_and_skipped(capfd):
    fail = {FRAMES[2]: FileNotFoundError('no such frame product')}
    _, serial = _fill(1, fail=fail)
    capfd.readouterr()
    _, forked = _fill(3, fail=fail)
    _assert_tables_identical(serial, forked)
    out = capfd.readouterr().out
    assert f'prep failed {FRAMES[2]}: no such frame product' in out
    assert 'Traceback' in out


def test_an_unexpected_exception_in_a_child_propagates():
    with pytest.raises(RuntimeError, match='must not be swallowed'):
        _fill(3, fail={FRAMES[4]: RuntimeError('must not be swallowed')})


def test_the_runner_passes_n_procs_through(tmp_path, monkeypatch):
    seen = []
    _real = ff.forced_fill_band

    def _spy(*args, **kwargs):
        seen.append(kwargs.get('n_procs'))
        return _real(*args, **kwargs)

    monkeypatch.setattr(ff, 'forced_fill_band', _spy)
    tbl, *_ = _build()
    merged = str(tmp_path / 'merged_m7.fits')
    tbl.write(merged)
    outs = []
    for n_procs in (1, 3):
        out = str(tmp_path / f'merged_m8_{n_procs}.fits')
        ff.run_forced_crossband_fill(
            merged, out, filternames=['F405N'], module='merged',
            frames_for=lambda filt: FRAMES,
            prepare_frame=_frames_prepare(),
            frame_arg_builder_for=lambda filt: (lambda fn: {'name': fn}),
            verbose=False, n_procs=n_procs)
        outs.append(Table.read(out))
    assert seen == [1, 3]
    _assert_tables_identical(outs[0], outs[1])


def test_both_m8_call_sites_read_the_option():
    import inspect
    src = ' '.join(inspect.getsource(cataloging.run_manual_pipeline).split())
    assert src.count("n_procs=int(getattr(options, 'forced_fill_procs', 1) or 1))") == 2
