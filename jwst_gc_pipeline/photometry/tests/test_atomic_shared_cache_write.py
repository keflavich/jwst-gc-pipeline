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


def test_temp_name_is_unique_per_writer_not_per_path(tmp_path, monkeypatch):
    """A SHARED staging name is worse than the race it replaces.

    Two writers using one temp file interleave inside it, and os.replace then
    publishes the mixture as a well-formed FITS file with wrong contents --
    silent, where the unlink race at least raises.  So the temp name must vary
    per writer even when no SLURM variable is set (interactive rerun, manual
    finalize), since a PID repeats freely across nodes on a shared filesystem.
    """
    monkeypatch.delenv('SLURM_ARRAY_TASK_ID', raising=False)
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    # capture on os.replace, NOT by patching Table.write -- `write` is astropy's
    # registry connector descriptor, and replacing it breaks format sniffing
    # (the read/write registry never sees the extension).
    seen = []
    real_replace = os.replace

    def _capture(src, dst, *a, **kw):
        seen.append(os.path.basename(str(src)))
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(os, 'replace', _capture)
    for _ in range(6):
        write_table_atomic(_table(0), str(tmp_path / 'shared.fits'))
    assert len(set(seen)) == len(seen), f'temp name reused: {seen}'
    assert all(n.endswith('.fits') for n in seen), seen


def test_partial_write_does_not_replace_the_good_file(tmp_path):
    """A failed write must leave the previous file intact, not a stub."""
    path = str(tmp_path / 'seed.fits')
    write_table_atomic(_table(0), path)
    good = Table.read(path)

    class _Boom(Table):
        def write(self, *a, **kw):
            raise ValueError('simulated writer failure')

    with pytest.raises(ValueError, match='simulated writer failure'):
        write_table_atomic(_Boom(), path)
    assert len(Table.read(path)) == len(good)
    assert not [f for f in os.listdir(tmp_path) if f.startswith('.')]
