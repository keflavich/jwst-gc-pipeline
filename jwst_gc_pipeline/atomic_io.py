"""Writing files that more than one SLURM task can reach.

The tools for the shapes described in ``docs/RACE_CONDITIONS.md``:

* :func:`atomic_write` — build the new file beside the old one and ``os.replace``
  it into position, so a concurrent reader sees the old file or the new one and
  never a partial one, and never nothing.
* :func:`write_table_atomic` — the same for an astropy Table, which is most of
  what this pipeline writes.
* :func:`locked` — serialise a read-modify-write, so two tasks correcting the
  same table do not each read the original and drop one another's change.
* :func:`publish_into` — for a writer that names its own output files and so
  cannot be handed a temporary path.

The lock is a file created with ``O_EXCL``, which is atomic on Lustre (where the
survey lives) and needs no ``flock`` mount option.  It records who holds it, so a
lock left behind by a killed job names the job that left it.
"""
import contextlib
import os
import uuid
import shutil
import socket
import time

#: How long to wait for a lock before giving up.  Long enough for a table write
#: (well under a second) with room for a loaded filesystem; short enough that a
#: stuck job is noticed rather than waited on for the length of a run.
DEFAULT_TIMEOUT = 120.0


def _writer_tag():
    """Something no two writers share, for a temporary name.

    A pid is not enough twice over: this storage is shared between nodes, which
    can hold the same pid at once, and a pid is reused over time on one node.
    ``uuid4`` settles both.  (:func:`locked` records host, pid and job instead —
    that name is read by a person diagnosing a stale lock, not compared.)
    """
    return f'{os.getpid()}{uuid.uuid4().hex[:8]}'


class LockTimeout(RuntimeError):
    """Someone else has held the lock for longer than the timeout allows."""


@contextlib.contextmanager
def locked(path, timeout=DEFAULT_TIMEOUT, poll=0.2):
    """Hold an exclusive lock on ``path`` for the body of the ``with``.

    The lock is ``<path>.lock``.  It is removed on the way out, including when
    the body raises.
    """
    lock_path = f'{path}.lock'
    holder = (f'{socket.gethostname()} pid {os.getpid()} '
              f'job {os.environ.get("SLURM_JOB_ID", "-")}')
    deadline = time.monotonic() + timeout
    while True:
        try:
            handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                try:
                    with open(lock_path) as fh:
                        held_by = fh.read().strip()
                except OSError:
                    held_by = 'unknown'
                raise LockTimeout(
                    f'waited {timeout:g}s for {lock_path}, held by {held_by}.  '
                    f'If that job is gone, the lock is stale: delete the file.')
            time.sleep(poll)
            continue
        else:
            break
    try:
        os.write(handle, holder.encode())
        os.close(handle)
        yield
    finally:
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass


@contextlib.contextmanager
def atomic_write(path):
    """Yield a temporary path to write, and move it onto ``path`` on success.

    The temporary keeps the original **suffix** and carries a per-writer tag
    before it: ``offsets.csv`` is written as ``offsets.tmp1234a1b2c3d4.csv``.

    The suffix matters because ``Table.write`` infers its format from the file
    name, and a name ending in ``.tmp1234`` raises ``IORegistryError`` instead
    of writing a CSV.  The tag matters because two writers sharing one temporary
    interleave inside it, and ``os.replace`` then publishes the mixture -- a
    well-formed file with wrong contents, which nothing raises on.  ``uuid4``
    and not the pid alone: these paths are on shared storage written from many
    nodes, and pids repeat both across nodes and over time on one.

    A body that raises leaves ``path`` untouched and removes the temporary.
    """
    root, suffix = os.path.splitext(path)
    tmp = f'{root}.tmp{_writer_tag()}{suffix}'
    try:
        yield tmp
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    if not os.path.exists(tmp):
        raise FileNotFoundError(
            f'nothing was written to {tmp}, so there is nothing to move onto '
            f'{path}.  The body of `with atomic_write(...) as tmp` has to write '
            f'to tmp, not to the original path.')
    os.replace(tmp, path)


def write_table_atomic(table, path, **kwargs):
    """Write a table so concurrent writers of the same path cannot collide.

    The per-frame fan-out runs NSHARDS (16) tasks per phase, and several caches
    are keyed by (filter, module) rather than by shard -- so every shard
    independently rebuilds and writes the SAME path.  ``overwrite=True`` is not
    enough: astropy's FITS table writer implements overwrite by unlinking the
    file and then calling ``writeto`` WITHOUT the flag, so between the unlink
    and the open another shard can recreate the file and the loser dies with

        OSError: File ...crossband_seed_manual.fits already exists.
                 If you mean to replace it then use the argument "overwrite=True"

    -- an error message that names the flag the caller already passed.  A reader
    can equally catch the file mid-write and get
    ``IORegistryError: Format could not be identified``.

    Observed 2026-08-01: quintuplet m7 lost shards 2/4/7 to the OSError (a
    dropped exposure aborts the phase, and the afterok finalize then never
    runs), and arches m3 shard 13 lost a frame to the truncated-read form on
    ``*_i2dseed.fits``.  Both cost a full phase.

    Write to a UNIQUE temp file in the destination directory, then
    ``os.replace`` -- atomic on POSIX within one filesystem, so a reader sees
    either the old file or the new one and never a partial one, and N
    concurrent writers of equivalent content all succeed.

    The temp name must be unique per WRITER, not merely per path.  A shared
    staging name is strictly worse than the race it replaces: two writers then
    interleave inside one temp file and ``os.replace`` publishes the mixture as
    a well-formed FITS file with wrong contents -- silent, where the unlink race
    at least raises.  ``uuid4`` rather than the PID alone, because a PID repeats
    freely across nodes on a shared filesystem and the helper must be safe
    outside an array job too (interactive rerun, manual finalize, two people
    debugging one field).
    """
    directory = os.path.dirname(path) or '.'
    # KEEP THE EXTENSION: astropy sniffs the format from it, so a temp name
    # ending in '.tmp12345' fails with "Format could not be identified" -- the
    # same error this helper exists to prevent (caught by the concurrency test).
    root, ext = os.path.splitext(os.path.basename(path))
    tmp = os.path.join(directory, f'.{root}.tmp{os.getpid()}{uuid.uuid4().hex[:8]}{ext}')
    published = False
    try:
        table.write(tmp, overwrite=True, **kwargs)
        os.replace(tmp, path)
        published = True
    finally:
        # try/finally, not `except BaseException` -- same cleanup without
        # catching KeyboardInterrupt/SystemExit (repo rule: specific exceptions
        # only).  os.replace consumed the temp file on the success path.
        if not published and os.path.exists(tmp):
            os.remove(tmp)
    return path


@contextlib.contextmanager
def publish_into(directory):
    """Yield a private directory whose files land in ``directory`` at the end.

    For a writer that names its own output files — ``psf_grid(save=True,
    outdir=...)`` — so it cannot be handed a temporary path.  It writes into a
    private subdirectory instead, and each finished file is moved into place
    with ``os.replace``, so a concurrent reader's ``os.path.exists`` check is
    never true for a file that is still being written.

    Publication is atomic **per file**, not per set: a reader can see some of a
    build's grids and not the rest.  That is sufficient here because every
    consumer resolves one ``(filter, detector)`` grid and loads exactly that
    path, and the one place that wants a whole channel rebuilds when any of its
    grids is missing.  A consumer that enumerated the cache and assumed a
    complete set would need a directory rename instead.

    A body that raises leaves ``directory`` untouched; the private directory and
    whatever is in it are removed.  A killed process leaves its
    ``.building-<host>-<pid>/`` behind — debris rather than corruption, since
    readers only ever see published names, but PSF grids are large and it is
    safe to delete.
    """
    building = os.path.join(directory, f'.building-{_writer_tag()}')
    os.makedirs(building, exist_ok=True)
    try:
        yield building
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    for name in sorted(os.listdir(building)):
        os.replace(os.path.join(building, name),
                   os.path.join(directory, name))
    shutil.rmtree(building, ignore_errors=True)


def keep_a_copy(path, backup_path):
    """Copy ``path`` to ``backup_path``, leaving ``path`` where it is.

    Moving it aside and rebuilding it leaves a window with no file at all, and
    a reader in that window takes the this-table-does-not-exist branch — which,
    for the offsets tables, means aligning a frame at (0, 0).  A hard link is
    the cheap version of the copy; it falls back when the filesystem refuses
    one.
    """
    with contextlib.suppress(FileNotFoundError):
        os.unlink(backup_path)
    try:
        os.link(path, backup_path)
    except OSError:
        shutil.copy2(path, backup_path)
    return backup_path
