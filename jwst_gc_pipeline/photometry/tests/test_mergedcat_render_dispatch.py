"""Serial-vs-parallel equivalence for the merged-catalog residual render
dispatch (build_mergedcat_residuals' per-frame fan-out).

Repo rule: a parallel refactor must be verified bit-identical to the serial
path.  The per-frame render body is a pure function of (frame, shared read-only
state), so equivalence of the whole build reduces to equivalence of the DISPATCH
that maps it over frames and collects results -- that dispatch is _run_frame_renders,
exercised here.  We assert the collected results are identical, in identical
order, for n_threads=1 vs >1, and that exceptions propagate (no silent drop).
"""
import threading

import pytest

from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
    _run_frame_renders,
)


def _render(orig):
    # mimic _render_frame's return shape: {kind: (out_resid, out_model)}
    return {'basic': (f'{orig}_mergedcat_residual.fits',
                      f'{orig}_mergedcat_model.fits')}


FRAMES = [f'jw01182004001_02101_{i:05d}_nrca{(i % 4) + 1}_crf.fits'
          for i in range(1, 13)]


def test_dispatch_order_identical_serial_vs_threaded():
    serial = _run_frame_renders(FRAMES, _render, 1)
    threaded = _run_frame_renders(FRAMES, _render, 8)
    # ThreadPoolExecutor.map preserves input order -> results must match exactly,
    # so the caller's frame-order accumulation into written[]/written_model[] is
    # byte-identical regardless of thread count.
    assert serial == threaded
    assert serial == [_render(f) for f in FRAMES]


def test_dispatch_processes_every_frame():
    for nt in (1, 2, 4, 16):
        out = _run_frame_renders(FRAMES, _render, nt)
        assert len(out) == len(FRAMES)
        # every frame's product appears exactly once, keyed by its own name
        resids = [v['basic'][0] for v in out]
        assert sorted(resids) == sorted(f'{f}_mergedcat_residual.fits'
                                        for f in FRAMES)


def test_single_frame_stays_serial():
    # the >1-frame guard: a lone frame never spins up a pool
    assert _run_frame_renders(FRAMES[:1], _render, 8) == [_render(FRAMES[0])]


def test_exception_propagates_no_silent_drop():
    def _boom(orig):
        if orig == FRAMES[5]:
            raise FileNotFoundError(f'missing raw products for {orig}')
        return _render(orig)

    for nt in (1, 8):
        with pytest.raises(FileNotFoundError):
            _run_frame_renders(FRAMES, _boom, nt)


def test_threaded_actually_uses_multiple_threads():
    seen = set()
    lock = threading.Lock()
    barrier = threading.Barrier(4, timeout=10)

    def _record(orig):
        # block until 4 threads are concurrently inside -> proves real parallelism
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        with lock:
            seen.add(threading.get_ident())
        return _render(orig)

    _run_frame_renders(FRAMES, _record, 4)
    assert len(seen) >= 2
