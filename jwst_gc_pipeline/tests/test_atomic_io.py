"""Writing a file two SLURM tasks can reach.

See ``docs/RACE_CONDITIONS.md``.  These tests are about what a *reader* sees
while a write is in progress, which is the part that has bitten this pipeline:
a reader that finds no offsets table aligns its frame at (0, 0) and says nothing.
"""
import multiprocessing
import os
import time

import pytest

from jwst_gc_pipeline.atomic_io import (LockTimeout, atomic_write, keep_a_copy,
                                        locked)


def test_a_reader_never_sees_a_partial_file(tmp_path):
    path = tmp_path / 'table.csv'
    path.write_text('old\n')
    with atomic_write(str(path)) as tmp:
        with open(tmp, 'w') as fh:
            fh.write('new\n')
        # mid-write: the reader still gets the whole previous file
        assert path.read_text() == 'old\n'
    assert path.read_text() == 'new\n'


def test_a_failed_write_leaves_the_old_file_and_no_debris(tmp_path):
    path = tmp_path / 'table.csv'
    path.write_text('old\n')
    with pytest.raises(ValueError):
        with atomic_write(str(path)) as tmp:
            open(tmp, 'w').write('half')
            raise ValueError('validation failed')
    assert path.read_text() == 'old\n'
    assert list(tmp_path.iterdir()) == [path]


def test_keeping_a_backup_leaves_the_original_in_place(tmp_path):
    """Moving it aside is what opened the window this module closes."""
    path = tmp_path / 'offsets.csv'
    path.write_text('rows\n')
    backup = keep_a_copy(str(path), str(tmp_path / 'offsets.csv.pre_m2'))
    assert path.exists() and path.read_text() == 'rows\n'
    assert open(backup).read() == 'rows\n'


def test_a_second_holder_waits(tmp_path):
    path = str(tmp_path / 'table.csv')
    with locked(path):
        assert os.path.exists(f'{path}.lock')
        with pytest.raises(LockTimeout, match='stale'):
            with locked(path, timeout=0.3, poll=0.05):
                pass
    assert not os.path.exists(f'{path}.lock')


def test_the_lock_says_who_holds_it(tmp_path):
    path = str(tmp_path / 'table.csv')
    with locked(path):
        assert str(os.getpid()) in open(f'{path}.lock').read()


def test_the_lock_is_released_when_the_body_raises(tmp_path):
    path = str(tmp_path / 'table.csv')
    with pytest.raises(ValueError):
        with locked(path):
            raise ValueError
    with locked(path):          # would time out if the first left it behind
        pass


def _increment_under_lock(path):
    """Read, add one, write back -- the shape update_offsets_table has."""
    for _ in range(20):
        with locked(path, timeout=60):
            value = int(open(path).read())
            time.sleep(0.001)                  # widen the window
            with atomic_write(path) as tmp:
                open(tmp, 'w').write(str(value + 1))


def test_concurrent_read_modify_writes_do_not_lose_updates(tmp_path):
    """Two filters' checkpoints correcting one shared table.

    Without the lock each process reads the same value and the last write wins,
    so the count comes out short.
    """
    path = str(tmp_path / 'counter')
    open(path, 'w').write('0')
    workers = [multiprocessing.Process(target=_increment_under_lock,
                                       args=(path,)) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=120)
    assert all(worker.exitcode == 0 for worker in workers)
    assert int(open(path).read()) == 80


@pytest.mark.parametrize('name', ['offsets.csv', 'catalog.fits', 'table.ecsv'])
def test_an_astropy_table_round_trips(tmp_path, name):
    """Table.write infers its format from the file name.

    A temporary called `offsets.csv.tmp1234` has no recognised extension, and
    astropy raises IORegistryError rather than writing -- so the temporary keeps
    the suffix and carries the pid before it.
    """
    from astropy.table import Table
    path = tmp_path / name
    table = Table({'visit': [1, 2], 'dra': [0.1, 0.2]})
    with atomic_write(str(path)) as tmp:
        assert tmp.endswith(os.path.splitext(name)[1])
        table.write(tmp)
    assert list(Table.read(str(path))['visit']) == [1, 2]


def test_temp_name_is_unique_per_writer_not_just_per_pid(tmp_path, monkeypatch):
    """A pid-only temporary is not unique on a shared filesystem.

    These paths are written by many SLURM nodes at once and pids repeat freely
    across nodes, so two writers could land on the SAME temporary, interleave
    inside it, and have os.replace publish the mixture -- a well-formed file
    with wrong contents, worse than the collision it replaces because nothing
    raises.  Pin that the token varies even with the pid held fixed.
    """
    monkeypatch.setattr(os, 'getpid', lambda: 4242)
    seen = []
    for _ in range(6):
        with atomic_write(str(tmp_path / 'offsets.csv')) as tmp:
            seen.append(os.path.basename(tmp))
            open(tmp, 'w').write('a,b\n1,2\n')
    assert len(set(seen)) == len(seen), f'temporary name reused: {seen}'
    assert all(n.endswith('.csv') for n in seen), seen
