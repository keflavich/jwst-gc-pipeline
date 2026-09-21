#!/usr/bin/env python
"""Distribute already-built HiPS layers to the two hosts that serve them.

This does NOT build anything -- `gc_treasury_rgb_images.py` (jwst_scripts) and
`publish_hips.py` do that.  It takes a layer that exists in a build directory
and makes the two public copies match it, when and only when they do not.

WHY IT EXISTS
-------------
Building is automatic and hourly; publishing was a hand-run rsync, so the
served NIRCam layer sat ~5 h and ~3700 tiles behind its build (measured
2026-09-13), and data.rc's copies were a day stale with nothing scheduled to
touch them at all.

THREE THINGS IT DOES DELIBERATELY
---------------------------------
1. **An explicit layer list, never a glob.**  The hand-run publish is
   ``rsync ... avm_images/jwst_* ...``, which sweeps any new ``jwst_*``
   directory to the web the next time anyone runs it.  That is how
   ``jwst_gc_treasury_miri_bgmatch_hips`` would have become a public layer as a
   side effect of appearing on disk rather than as a decision.  Adding a layer
   here is a diff.

2. **Gated on the source being newer.**  The gate reads `hips_release_date`
   from each `properties`, not filesystem mtime -- a copy rewrites mtime, so an
   mtime gate republishes forever.  Nothing to do is the common case and costs
   two small reads.

3. **Staged, per layer.**  rsync into ``<name>.new``, verify it independently
   (properties parses, Norder3 present, tile count matches the source), then
   swap: old aside, new in, old deleted.  A partial tree is never reachable and
   a failed transfer leaves the live layer untouched.  There is no ``--delete``
   anywhere outside a single layer's own directory: the docroot holds other
   people's layers.

BOTH DESTINATIONS
-----------------
`data.rc` serves the docroot directly; starformation is a separate copy behind
ssh.  Publishing one and not the other is what left them a day apart, so a
layer is only "published" when both agree.

    python scripts/monitoring/publish_hips_layers.py --dry-run
    python scripts/monitoring/publish_hips_layers.py --layer jwst_gc_treasury_hips
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import time
import contextlib

#: Layers to distribute, and where each is BUILT.  Explicit on purpose -- see 1.
BUILD_ROOT = '/orange/adamginsburg/jwst/gc-treasury/pngs'
#: Where the RGB mosaic HiPS are built -- a different tree from the per-filter
#: quicklook pngs, and reprojected by astropy rather than by the RGB builder.
MOSAIC_BUILD = '/orange/adamginsburg/jwst/gc-treasury/mosaics'
#: Where the CMZ overview coadds are built, which is the docroot itself.
DOCROOT_BUILD = '/orange/adamginsburg/web/public/avm_images'
LAYERS = {
    'jwst_gc_treasury_hips': f'{BUILD_ROOT}/jwst_gc_treasury_hips',
    'jwst_gc_treasury_miri_hips': f'{BUILD_ROOT}/jwst_gc_treasury_miri_hips',
    'jwst_gc_treasury_miri_bgmatch_hips':
        f'{BUILD_ROOT}/jwst_gc_treasury_miri_bgmatch_hips',
    # Three stretches of the SAME NIRCam data, all published: `vminmax` is
    # fixed asinh cuts (-0.5..100 MJy/sr) and is the default everywhere;
    # the unsuffixed one stretches each field on its own percentiles; `log`
    # runs -0.5..500 and keeps structure in cluster cores that vminmax
    # saturates, at the cost of the faint end.  Registered together because
    # the gate is per layer: a flavour mid-rebuild is refused on its own
    # (`verify` reads the staged properties) without holding up the others.
    'jwst_gc_treasury_vminmax_hips':
        f'{BUILD_ROOT}/jwst_gc_treasury_vminmax_hips',
    'jwst_gc_treasury_log_hips': f'{BUILD_ROOT}/jwst_gc_treasury_log_hips',
    # The three-band colour composite: F770W red, F480M green, F212N blue, so
    # it covers only where MIRI and NIRCam both observed. Published off by
    # default in the viewer -- it is a different projection (galactic frame,
    # astropy/reproject) from the single-band layers and covers less sky, so
    # it is a thing to turn on rather than a background to work against.
    'gctreasury_mosaic_RGB_770-480-212_hips':
        f'{MOSAIC_BUILD}/gctreasury_mosaic_RGB_770-480-212_hips',
    # The CMZ overview coadds are rebuilt IN PLACE in the docroot by
    # `rebuild_jwst_cmz_hips.py`, so for these the docroot is the build
    # location and the local step is a no-op by construction: `needs_publish`
    # compares a properties file against itself, reports "up to date", and
    # nothing is copied or removed. What they need is the second destination.
    # Without them here, starformation had no scheduled path to these layers
    # at all -- its jwst_nir_hips copy was 14 months behind the docroot's
    # (2025-07-05 against 2026-09-15) and nothing reported it.
    'jwst_nir_hips': f'{DOCROOT_BUILD}/jwst_nir_hips',
    'jwst_miri_hips': f'{DOCROOT_BUILD}/jwst_miri_hips',
}

#: Served directly by data.rc.
DOCROOT = '/orange/adamginsburg/web/public/avm_images'
#: Served by starformation, reachable over ssh.
WEB_HOST = 'starformation'
WEB_DIR = ('/h/cnswww-starformation.astro/starformation.astro.ufl.edu'
           '/htdocs/avm_images')

_DATE_RE = re.compile(r'^hips_release_date\s*=\s*(.+)$', re.M)


def release_date(properties_text):
    """``hips_release_date`` from a properties file, or ``None``.

    ISO-8601 UTC as HiPS writes it, so string comparison orders correctly and
    no parsing is needed.  Returning None means "unknown", which the gate
    treats as "publish" rather than as "up to date": an unreadable destination
    is not evidence it matches.
    """
    if not properties_text:
        return None
    m = _DATE_RE.search(properties_text)
    return m.group(1).strip() if m else None


def needs_publish(src_props, dst_props, force=False):
    """Is the source strictly newer than the destination?

    Unknown on EITHER side means yes.  The failure this avoids is a silent
    no-op: a destination whose properties cannot be read looks "same" to any
    equality test, and the layer then never updates again.

    ``force`` exists because the date is the BUILDER's claim about itself, and
    a builder can rewrite every tile without advancing it.  That happened on
    2026-09-15: the MIRI treasury coadd was rebuilt at 05:59 with the corrected
    AVM in all 34 fields, and its ``hips_release_date`` still read the 04:19
    value already published, so this returned False and the corrected layer was
    unpublishable by its own publisher.  Fresh content under a stale date is
    the mirror of the case the checklist already warns about, and the date gate
    cannot see either.

    The gate stays on by default.  It is what keeps the hourly path from
    re-copying 10,000 tiles for nothing, and that is worth more than making the
    rare case automatic.
    """
    if force:
        return True
    src, dst = release_date(src_props), release_date(dst_props)
    if src is None or dst is None:
        return True
    return src > dst


def count_tiles(walker, root):
    """Number of tile files under a HiPS root.  `walker` is os.walk or a stub."""
    return sum(1 for _d, _sub, files in walker(root)
               for f in files if f.endswith(('.png', '.jpg', '.fits')))


def verify(properties_text, has_norder3, tiles, expect_tiles):
    """Is a staged copy fit to swap in?  Returns ``None`` or a reason string.

    Deliberately checks the STAGED tree rather than trusting rsync's exit code:
    a truncated transfer can exit 0 on a killed connection, and the thing that
    would then be renamed over a good layer is a partial pyramid.
    """
    if not release_date(properties_text):
        return 'no readable properties'
    if not has_norder3:
        return 'no Norder3'
    if tiles != expect_tiles:
        return f'tile count {tiles} != source {expect_tiles}'
    return None


def _run(cmd, dry):
    print(('  would run: ' if dry else '  ') + ' '.join(shlex.quote(c) for c in cmd),
          flush=True)
    if dry:
        return 0
    return subprocess.call(cmd)


def _read_local(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return None


def _read_remote(host, path):
    done = subprocess.run(['ssh', host, f'cat {shlex.quote(path)}'],
                          capture_output=True, text=True)
    return done.stdout if done.returncode == 0 else None


# --- the build lock ----------------------------------------------------------
# `gc_treasury_rgb_images.py` (keflavich/jwst_scripts) rebuilds these coadds
# hourly and holds `<BUILD_ROOT>/.auto.lock` while it writes.  A publish that
# ignores it copies a tree being rewritten underneath it: on 2026-09-16 an
# rsync of 12,707 vminmax tiles died with `Stale file handle (116)` partway
# through, because the cron replaced the files it was reading.
#
# So the publisher takes the same lock.  Know what that costs the builder,
# because the two sides behave differently when blocked: `coadd_lock` (the
# manual --coadd path) WAITS, but `cmd_auto` -- the hourly job -- prints
# "another run holds the lock" and EXITS (gc_treasury_rgb_images.py:1066).  So
# a publish holding the lock does not delay a rebuild by the length of a
# transfer; it makes that tick do nothing, and the rebuild happens the next
# hour.  At hourly cadence against a ~30 min copy the practical difference is
# small, but "skipped" and "late" are not the same claim and the next person
# reasoning about contention needs the accurate one.
#
# The trade is still the right way round: the tick it costs is one that would
# otherwise have raced the copy, and a rebuild is idempotent -- nothing is lost
# by doing it an hour later.  What is lost by racing is a published tree that
# changed while it was being read.
#
# The protocol is COPIED from that script rather than invented, because a lock
# only works if both sides implement it the same way: same path, O_CREAT|O_EXCL
# (not exists-then-create, which lets two arrivals both proceed), the same
# `pid timestamp what` line, the same 6 h staleness takeover, and release in a
# `finally` so a crash does not starve the schedule.
LOCK_FILE = '.auto.lock'
#: A publish is minutes-to-an-hour; a rebuild is hours.  Waiting longer than
#: this means the next scheduled publish will do the job anyway.  It is the
#: budget for the whole RUN, not for each layer: see `WaitBudget`.
LOCK_WAIT_S = int(os.environ.get('HIPS_PUBLISH_LOCK_WAIT_S', 2 * 3600))
#: Matches the builder's own takeover threshold.  A shorter one here would let
#: the publisher steal a lock from a rebuild that is merely slow.
LOCK_STALE_S = 6 * 3600


class WaitBudget:
    """How long one RUN may spend waiting on the build lock, in total.

    Per layer, the wait multiplies by the layer count.  Measured 2026-09-21
    with eight registered layers and a 30 min per-layer wait against a rebuild
    that held the lock for 4 h: the run waited 30 min on each of the first four
    layers and published none of them, and because the cron line serialises
    with `flock -n`, every hourly fire inside that window was a no-op.  A run
    that gives up after its budget leaves the next hour free to try again,
    which is the property the schedule is built on.

    Time spent HOLDING the lock (the copy itself) is productive and is not
    charged; only time spent waiting for someone else is.
    """

    def __init__(self, seconds):
        self.remaining = max(0, int(seconds))

    def spend(self, seconds):
        self.remaining = max(0, self.remaining - int(seconds))


def lock_path(src):
    """The build lock guarding ``src``, or None if nothing builds it.

    Only the trees built by the hourly job are guarded. The CMZ overview
    coadds are rebuilt in the docroot by a different script that does not take
    this lock, so claiming it for them would block the treasury rebuild while
    protecting nothing.
    """
    if os.path.dirname(os.path.normpath(src)) != os.path.normpath(BUILD_ROOT):
        return None
    return os.path.join(BUILD_ROOT, LOCK_FILE)


@contextlib.contextmanager
def build_lock(src, what, wait_s=None, budget=None, poll=30, dry=False):
    """Hold the build lock for the duration, or wait for whoever has it.

    `budget`, when given, is a `WaitBudget` shared by every layer in the run:
    the wait here is capped by what is left of it, and whatever this call
    waits is deducted, so a contended run stops after one budget rather than
    one budget per layer.
    """
    path = lock_path(src)
    if path is None or dry:
        yield True
        return
    if wait_s is None:
        wait_s = LOCK_WAIT_S if budget is None else budget.remaining
    waited = 0
    while os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age > LOCK_STALE_S:
            print(f'  stale build lock ({age / 3600:.1f} h); taking it')
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            break
        if waited == 0:
            try:
                holder = open(path).read().strip()
            except OSError:
                holder = 'unreadable'
            print(f'  waiting for the build lock ({age / 60:.0f} min old; '
                  f'{holder}) before {what}', flush=True)
        if waited >= wait_s:
            if budget is not None:
                budget.spend(waited)
            print(f'  build lock still held after {waited // 60} min; '
                  f'skipping {what} -- the next run will publish it',
                  file=sys.stderr)
            yield False
            return
        time.sleep(poll)
        waited += poll
    if budget is not None:
        budget.spend(waited)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            os.write(fd, f"{os.getpid()} "
                         f"{time.strftime('%Y-%m-%dT%H:%M:%S')} "
                         f"{what}\n".encode())
        finally:
            os.close(fd)
    except FileExistsError:
        # taken between the last look and now
        print(f'  build lock was taken while we waited; skipping {what}',
              file=sys.stderr)
        yield False
        return
    try:
        yield True
    finally:
        # In a finally, like the builder's: the other half of this lock's
        # history is runs that left the file behind and starved the schedule.
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def publish_local(name, src, dry=False, force=False):
    """Docroot copy (what data.rc serves), staged and swapped."""
    dst = os.path.join(DOCROOT, name)
    src_props = _read_local(os.path.join(src, 'properties'))
    if not needs_publish(src_props, _read_local(os.path.join(dst, 'properties')),
                         force=force):
        print(f'  {name} docroot: up to date')
        return 0
    stage = dst + '.new'
    expect = count_tiles(os.walk, src)
    rc = _run(['rm', '-rf', stage], dry)
    if rc and not dry:
        print(f'  {name} docroot: could not clear the staging path rc={rc} -- '
              f'nothing transferred, live layer untouched', file=sys.stderr)
        return rc
    rc = _run(['rsync', '-a', src + '/', stage + '/'], dry)
    if rc and not dry:
        subprocess.call(['rm', '-rf', stage])
        print(f'  {name} docroot: TRANSFER FAILED rc={rc} -- live layer '
              f'untouched, staging removed', file=sys.stderr)
        return rc
    if dry:
        return rc
    why = verify(_read_local(os.path.join(stage, 'properties')),
                 os.path.isdir(os.path.join(stage, 'Norder3')),
                 count_tiles(os.walk, stage), expect)
    if why:
        subprocess.call(['rm', '-rf', stage])
        print(f'  {name} docroot: REFUSED -- {why}', file=sys.stderr)
        return 1
    old = dst + '.old'
    subprocess.call(['rm', '-rf', old])
    if os.path.isdir(dst):
        os.rename(dst, old)
    os.rename(stage, dst)
    subprocess.call(['rm', '-rf', old])
    print(f'  {name} docroot: published ({expect} tiles)')
    return 0


def publish_remote(name, src, dry=False, host=WEB_HOST, web_dir=WEB_DIR,
                   force=False):
    """starformation copy, staged and verified REMOTELY before the swap."""
    dst = f'{web_dir}/{name}'
    src_props = _read_local(os.path.join(src, 'properties'))
    if not needs_publish(src_props, _read_remote(host, f'{dst}/properties'),
                         force=force):
        print(f'  {name} {host}: up to date')
        return 0
    stage = dst + '.new'
    expect = count_tiles(os.walk, src)
    rc = _run(['ssh', host, f'rm -rf {shlex.quote(stage)}'], dry)
    if rc and not dry:
        print(f'  {name} {host}: could not clear the staging path rc={rc} -- '
              f'nothing transferred, live layer untouched', file=sys.stderr)
        return rc
    rc = _run(['rsync', '-az', src + '/', f'{host}:{stage}/'], dry)
    if rc and not dry:
        # A failed transfer used to return here having printed NOTHING, so a
        # publish that moved no bytes was indistinguishable from one that had
        # nothing to do. That happened on 2026-09-15: an rsync of 11,430 tiles
        # was killed by a caller's `timeout`, and the run reported only the
        # docroot line while starformation stayed eight hours behind, with a
        # 1.2 GB partial `.new` left on the far side. The staging tree is
        # removed rather than left to be mistaken for progress.
        subprocess.call(['ssh', host, f'rm -rf {shlex.quote(stage)}'])
        print(f'  {name} {host}: TRANSFER FAILED rc={rc} -- live layer '
              f'untouched, staging removed', file=sys.stderr)
        return rc
    if dry:
        return rc
    # Verify on the far side: counting locally would check the wrong tree.
    probe = subprocess.run(
        ['ssh', host,
         f'test -d {shlex.quote(stage)}/Norder3 && echo yes || echo no; '
         f'find {shlex.quote(stage)} -type f \\( -name "*.png" -o -name "*.jpg" '
         f'-o -name "*.fits" \\) | wc -l; cat {shlex.quote(stage)}/properties'],
        capture_output=True, text=True)
    lines = probe.stdout.split('\n')
    has_n3 = lines[0].strip() == 'yes' if lines else False
    tiles = int(lines[1]) if len(lines) > 1 and lines[1].strip().isdigit() else -1
    why = verify('\n'.join(lines[2:]), has_n3, tiles, expect)
    if why:
        subprocess.call(['ssh', host, f'rm -rf {shlex.quote(stage)}'])
        print(f'  {name} {host}: REFUSED -- {why}', file=sys.stderr)
        return 1
    rc = subprocess.call([
        'ssh', host,
        f'rm -rf {shlex.quote(dst + ".old")}; '
        f'if [ -d {shlex.quote(dst)} ]; then mv {shlex.quote(dst)} '
        f'{shlex.quote(dst + ".old")}; fi; '
        f'mv {shlex.quote(stage)} {shlex.quote(dst)}; '
        f'rm -rf {shlex.quote(dst + ".old")}'])
    print(f'  {name} {host}: published ({expect} tiles)' if rc == 0
          else f'  {name} {host}: swap failed rc={rc}', flush=True)
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--layer', action='append',
                    help='publish only this layer (repeatable); default all')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-wait', action='store_true',
                    help='do not wait for the build lock: skip any layer that '
                         'is being rebuilt right now, and say so')
    ap.add_argument('--skip-remote', action='store_true',
                    help='docroot only (no ssh; for a host without web access)')
    ap.add_argument('--force', action='store_true',
                    help='publish even when hips_release_date says up to date '
                         '-- for a rebuild that changed tiles without advancing '
                         'the date. Still staged and verified before the swap.')
    args = ap.parse_args(argv)

    names = args.layer or sorted(LAYERS)
    bad = [n for n in names if n not in LAYERS]
    if bad:
        # An unknown name is a typo or a layer someone forgot to register; both
        # are worth stopping for, since the alternative is publishing nothing
        # and reporting success.
        ap.error(f'unknown layer(s): {", ".join(bad)}; known: {", ".join(sorted(LAYERS))}')

    rc = 0
    waiting = WaitBudget(LOCK_WAIT_S)
    for name in names:
        src = LAYERS[name]
        print(f'{name}:', flush=True)
        if not os.path.isdir(src):
            print(f'  source missing: {src}', file=sys.stderr)
            rc = rc or 1
            continue
        # The lock is taken per LAYER -- a whole-run hold would block rebuilds
        # for hours, and a layer rebuilt while a LATER one is being copied is
        # simply newer next time.  The WAIT, though, is one budget for the run
        # (`waiting`): per layer it multiplied by the layer count and ran past
        # the next scheduled fire.
        with build_lock(src, f'publishing {name}', dry=args.dry_run,
                        budget=waiting,
                        wait_s=0 if args.no_wait else None) as held:
            if not held:
                rc = rc or 1
                continue
            rc = publish_local(name, src, dry=args.dry_run,
                               force=args.force) or rc
            if not args.skip_remote:
                rc = publish_remote(name, src, dry=args.dry_run,
                                    force=args.force) or rc
    return rc


if __name__ == '__main__':
    sys.exit(main())
