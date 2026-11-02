"""Map an independent per-frame step over forked worker processes.

Two steps of the m7 finalize loop over every frame of a band with nothing
carried from one frame to the next: the merged-catalog residual render
(``crowdsource_catalogs_long._run_frame_renders``) and the m8 forced fill
(``forced_fill.forced_fill_band``).  Threads give them little.  Both evaluate
the PSF through photutils' ``RectBivariateSpline``, and scipy serializes every
FITPACK call behind one process-wide lock
(``scipy.interpolate._fitpack2.FITPACK_LOCK``); the rest of each loop holds the
GIL.  On sgrb2 (job 46648154, 2026-10) the two loops took ~22.5 h and ~11 h of
a 72 h finalize that timed out, one frame at a time (#1230).

``map_frames`` runs such a loop in forked processes.  A forked child starts
with a copy of the parent's memory, so the per-frame function can be a closure
over the PSF grids, the merged catalog and the options: nothing is pickled on
the way in except the frame's index, and only the function's return value
comes back.  Results are returned in input order, so a caller that consumes
them in that order computes what its serial loop computed, bit for bit.
"""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# (function, items) for the forked children.  Set only for the duration of a
# map_frames call; a child reads it from the memory it was forked with.
_TASK = None


def _run_one(index):
    fn, items = _TASK
    return fn(items[index])


def map_frames(fn, items, n_procs=1):
    """Return ``[fn(item) for item in items]``, computed in ``n_procs`` forked
    processes.

    ``n_procs <= 1`` or a single item runs the plain serial loop in this
    process.  The parallel path keeps the serial loop's contract:

    - results come back in ``items`` order;
    - an exception raised by ``fn`` in a child is raised here, with the child's
      traceback attached as its ``__cause__``, and the items not yet started
      are cancelled;
    - a child that dies outright (an out-of-memory kill) raises
      ``concurrent.futures.process.BrokenProcessPool`` here.  A
      ``multiprocessing.Pool`` would wait for the lost result until the job's
      wall time ran out.

    A child's changes to module state, caches, or the objects ``fn`` closed
    over stay in the child.  Everything the caller needs must be in the return
    value, which has to pickle, or in a file the child writes.
    """
    global _TASK
    items = list(items)
    n = min(int(n_procs or 1), len(items))
    if n <= 1:
        return [fn(item) for item in items]
    if _TASK is not None:
        raise RuntimeError("map_frames is already running in this process; a "
                           "per-frame function cannot start a nested pool")
    _TASK = (fn, items)
    try:
        # Fork, so the children inherit fn and its closure.  ProcessPoolExecutor
        # forks every worker before it starts its own management thread.
        with ProcessPoolExecutor(
                max_workers=n,
                mp_context=multiprocessing.get_context('fork')) as ex:
            return list(ex.map(_run_one, range(len(items))))
    finally:
        _TASK = None
