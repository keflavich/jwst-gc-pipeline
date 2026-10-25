"""frame_pool.map_frames returns what the serial loop returns.

The m7 finalize's merged-catalog render and m8 forced fill hand their
per-frame step to map_frames (#1230).  Their output equals the serial loop's
only if the map returns every item's result, in item order, from a function
that may close over objects that do not pickle, and raises what the step
raises -- including when a child dies outright, which a multiprocessing.Pool
would leave waiting until the job's wall time.
"""
import multiprocessing
import os
import signal
import threading
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pytest

from jwst_gc_pipeline.photometry import frame_pool

# Python 3.12+ warns on any fork from a process with more than one OS thread
# (numpy's BLAS pool counts); the production callers fork the same way.
pytestmark = pytest.mark.filterwarnings(
    'ignore:This process .* is multi-threaded:DeprecationWarning')


def _square(x):
    return x * x


@pytest.mark.parametrize('n_procs', [1, 2, 4, 64])
def test_results_come_back_in_item_order(n_procs):
    items = list(range(23))
    assert frame_pool.map_frames(_square, items, n_procs) == [x * x for x in items]


def test_a_closure_over_unpicklable_state_runs_in_the_children():
    """The production steps close over PSF grids, catalogs and locks."""
    big = np.random.default_rng(3).normal(size=(40, 4096))
    lock = threading.Lock()

    def _step(i):
        with lock:
            row = big[i]
        return np.sum(row ** 2) / np.sum(np.abs(row)), os.getpid()

    serial = frame_pool.map_frames(_step, range(len(big)), 1)
    forked = frame_pool.map_frames(_step, range(len(big)), 3)
    # bit-identical floats, in order
    assert [v for v, _ in forked] == [v for v, _ in serial]
    assert {pid for _, pid in serial} == {os.getpid()}
    assert os.getpid() not in {pid for _, pid in forked}


def test_the_children_run_concurrently():
    """Two items that each wait for the other finish only if two processes
    run at once."""
    barrier = multiprocessing.get_context('fork').Barrier(2, timeout=60)

    def _meet(i):
        barrier.wait()
        return i

    assert frame_pool.map_frames(_meet, [0, 1], 2) == [0, 1]


def test_one_process_or_one_item_stays_in_this_process():
    assert frame_pool.map_frames(lambda _: os.getpid(), range(5), 1) == [os.getpid()] * 5
    assert frame_pool.map_frames(lambda _: os.getpid(), range(5), 0) == [os.getpid()] * 5
    assert frame_pool.map_frames(lambda _: os.getpid(), [7], 8) == [os.getpid()]
    assert frame_pool.map_frames(_square, [], 8) == []


def test_an_exception_in_a_child_is_raised_here():
    def _step(i):
        if i == 5:
            raise FileNotFoundError(f'missing raw products for frame {i}')
        return i

    with pytest.raises(FileNotFoundError, match='frame 5'):
        frame_pool.map_frames(_step, range(12), 3)
    assert frame_pool._TASK is None


def test_a_child_killed_outright_raises_instead_of_hanging():
    def _step(i):
        if i == 3:
            os.kill(os.getpid(), signal.SIGKILL)
        return i

    with pytest.raises(BrokenProcessPool):
        frame_pool.map_frames(_step, range(8), 2)
    assert frame_pool._TASK is None


def test_a_nested_pool_is_refused():
    def _step(i):
        return frame_pool.map_frames(_square, [1, 2], 2)

    with pytest.raises(RuntimeError, match='nested pool'):
        frame_pool.map_frames(_step, range(3), 2)


def test_changes_a_child_makes_stay_in_the_child():
    """The contract the callers rely on: only the return value comes back."""
    seen = []

    def _step(i):
        seen.append(i)
        return i

    assert frame_pool.map_frames(_step, range(6), 3) == list(range(6))
    assert seen == []
    assert frame_pool.map_frames(_step, range(6), 1) == list(range(6))
    assert seen == list(range(6))
