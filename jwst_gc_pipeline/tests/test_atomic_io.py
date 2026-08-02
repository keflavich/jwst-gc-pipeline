"""Writing a file two SLURM tasks can reach.

See ``docs/RACE_CONDITIONS.md``.  These tests are about what a *reader* sees
while a write is in progress, which is the part that has bitten this pipeline:
a reader that finds no offsets table aligns its frame at (0, 0) and says nothing.
"""
import multiprocessing
import os
import pathlib
import time

import pytest

from jwst_gc_pipeline.atomic_io import (LockTimeout, atomic_write, keep_a_copy,
                                        locked, publish_into)


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


def test_two_writers_never_share_a_temporary(tmp_path):
    """Pids repeat across nodes and over time, so the pid alone is not a tag.

    Two writers on one temporary interleave inside it and `os.replace` publishes
    the mixture -- a well-formed file with wrong contents, and nothing raises.
    """
    path = str(tmp_path / 'offsets.csv')
    seen = set()
    for _ in range(50):
        with atomic_write(path) as tmp:      # same pid every time
            open(tmp, 'w').write('rows\n')
        seen.add(tmp)
    assert len(seen) == 50
    assert all(name.endswith('.csv') for name in seen)
# --- a writer that names its own files --------------------------------------

def test_files_appear_in_the_cache_only_when_finished(tmp_path):
    """psf_grid(save=True) names its own output, so it cannot be handed a
    temporary path; it gets a private directory instead."""
    cache = tmp_path / 'psfs'
    cache.mkdir()
    with publish_into(str(cache)) as build_dir:
        (pathlib.Path(build_dir) / 'nrca1_f405n_grid.fits').write_text('half')
        # mid-build: nothing a reader checks for is there yet
        assert list(cache.glob('*.fits')) == []
    assert [p.name for p in cache.glob('*.fits')] == ['nrca1_f405n_grid.fits']
    assert list(cache.glob('.building-*')) == []


def test_an_abandoned_build_leaves_the_cache_alone(tmp_path):
    cache = tmp_path / 'psfs'
    cache.mkdir()
    (cache / 'existing.fits').write_text('good')
    with pytest.raises(RuntimeError):
        with publish_into(str(cache)) as build_dir:
            (pathlib.Path(build_dir) / 'partial.fits').write_text('half')
            raise RuntimeError('MAST timed out')
    assert [p.name for p in cache.iterdir()] == ['existing.fits']


def test_a_writer_tag_is_more_than_a_pid():
    """Shared storage across nodes, and pid reuse over time on one."""
    from jwst_gc_pipeline.atomic_io import _writer_tag
    tags = {_writer_tag() for _ in range(50)}      # same pid throughout
    assert len(tags) == 50
    assert all(tag.startswith(str(os.getpid())) for tag in tags)


def test_the_lock_file_names_a_person_readable_holder(tmp_path):
    """The lock is read by someone deciding whether it is stale, so it carries
    host/pid/job rather than the uniqueness token."""
    import socket
    path = str(tmp_path / 'offsets.csv')
    with locked(path):
        held = open(f'{path}.lock').read()
    assert socket.gethostname() in held and str(os.getpid()) in held


def test_the_temporary_and_the_build_dir_both_carry_it(tmp_path):
    with atomic_write(str(tmp_path / 'offsets.csv')) as tmp:
        assert os.path.basename(tmp).startswith('offsets.tmp')
        open(tmp, 'w').write('rows\n')
    cache = tmp_path / 'psfs'
    cache.mkdir()
    with publish_into(str(cache)) as build_dir:
        assert os.path.basename(build_dir).startswith('.building-')


def test_writing_nothing_says_so(tmp_path):
    """os.replace on a temporary that was never written raises FileNotFoundError
    naming two paths and no cause."""
    with pytest.raises(FileNotFoundError, match='nothing was written'):
        with atomic_write(str(tmp_path / 'offsets.csv')):
            pass
