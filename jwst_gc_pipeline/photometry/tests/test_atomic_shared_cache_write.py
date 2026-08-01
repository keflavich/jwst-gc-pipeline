"""Concurrent writers of a shared per-phase cache must not kill each other.

The per-frame fan-out runs 16 shards per phase and several caches are keyed by
(filter, module) rather than by shard, so every shard rebuilds and writes the
SAME path.  `overwrite=True` alone is racy: astropy's FITS table writer unlinks
the file and then calls writeto() WITHOUT the flag, so a writer that loses the
gap between unlink and open dies with "already exists ... use overwrite=True".
"""
import multiprocessing as mp
import os

import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry.cataloging import write_table_atomic

N_WRITERS = 8
N_ROUNDS = 12


def _table(seed):
    rng = np.random.default_rng(seed)
    return Table({'a': rng.random(500), 'b': np.arange(500)})


def _hammer_atomic(path, seed, q):
    try:
        for i in range(N_ROUNDS):
            write_table_atomic(_table(seed + i), path)
        q.put(None)
    except BaseException as ex:                      # noqa: BLE001 - reported, not swallowed
        q.put(f'{type(ex).__name__}: {ex}')


def _hammer_plain(path, seed, q):
    try:
        for i in range(N_ROUNDS):
            _table(seed + i).write(path, overwrite=True)
        q.put(None)
    except BaseException as ex:                      # noqa: BLE001
        q.put(f'{type(ex).__name__}: {ex}')


def _run(target, path):
    ctx = mp.get_context('fork')
    q = ctx.Queue()
    procs = [ctx.Process(target=target, args=(path, 100 * k, q))
             for k in range(N_WRITERS)]
    for p in procs:
        p.start()
    errs = [q.get() for _ in procs]
    for p in procs:
        p.join(120)
    return [e for e in errs if e]


def test_atomic_write_survives_concurrent_writers(tmp_path):
    path = str(tmp_path / 'crossband_seed_manual.fits')
    errs = _run(_hammer_atomic, path)
    assert not errs, f'atomic writers collided: {errs[:3]}'
    # and the survivor is a COMPLETE table, never a truncated one
    assert len(Table.read(path)) == 500


def test_atomic_write_leaves_no_temp_files(tmp_path):
    path = str(tmp_path / 'x_i2dseed.fits')
    _run(_hammer_atomic, path)
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith('.')]
    assert not leftovers, f'temp files left behind: {leftovers}'


def test_plain_overwrite_is_the_race_being_fixed(tmp_path):
    """Characterise the bug: the same hammer on plain overwrite=True fails.

    xfail(strict=False) -- it is a race, so it is not guaranteed to trip on
    every machine.  If it stops tripping the guard above still holds; this test
    exists to document WHY write_table_atomic is needed, not to gate CI.
    """
    path = str(tmp_path / 'plain.fits')
    errs = _run(_hammer_plain, path)
    if not errs:
        pytest.skip('race did not trip on this run (timing-dependent)')
    assert any('exist' in e or 'Format could not be identified' in e
               for e in errs), errs
