"""Writing files that more than one SLURM task can reach.

Two tools, for the two shapes described in ``docs/RACE_CONDITIONS.md``:

* :func:`atomic_write` — build the new file beside the old one and ``os.replace``
  it into position, so a concurrent reader sees the old file or the new one and
  never a partial one, and never nothing.
* :func:`locked` — serialise a read-modify-write, so two tasks correcting the
  same table do not each read the original and drop one another's change.

The lock is a file created with ``O_EXCL``, which is atomic on Lustre (where the
survey lives) and needs no ``flock`` mount option.  It records who holds it, so a
lock left behind by a killed job names the job that left it.
"""
import contextlib
import os
import shutil
import socket
import time

#: How long to wait for a lock before giving up.  Long enough for a table write
#: (well under a second) with room for a loaded filesystem; short enough that a
#: stuck job is noticed rather than waited on for the length of a run.
DEFAULT_TIMEOUT = 120.0


class LockTimeout(RuntimeError):
    """Someone else has held the lock for longer than the timeout allows."""


@contextlib.contextmanager
def locked(path, timeout=DEFAULT_TIMEOUT, poll=0.2):
    """Hold an exclusive lock on ``path`` for the body of the ``with``.

    The lock is ``<path>.lock``.  It is removed on the way out, including when
    the body raises.
    """
    lock_path = f'{path}.lock'
    holder = f'{socket.gethostname()} pid {os.getpid()} ' \
             f'job {os.environ.get("SLURM_JOB_ID", "-")}'
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

    The temporary keeps the original **suffix** and carries the pid before it:
    ``offsets.csv`` is written as ``offsets.tmp1234.csv``.  The suffix matters
    because ``Table.write`` infers its format from the file name, and a name
    ending in ``.tmp1234`` raises ``IORegistryError`` instead of writing a CSV.
    The pid matters so two writers do not collide in the temporary either.

    A body that raises leaves ``path`` untouched and removes the temporary.
    """
    root, suffix = os.path.splitext(path)
    tmp = f'{root}.tmp{os.getpid()}{suffix}'
    try:
        yield tmp
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    os.replace(tmp, path)


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
