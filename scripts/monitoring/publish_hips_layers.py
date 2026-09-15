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

#: Layers to distribute, and where each is BUILT.  Explicit on purpose -- see 1.
BUILD_ROOT = '/orange/adamginsburg/jwst/gc-treasury/pngs'
#: Where the CMZ overview coadds are built, which is the docroot itself.
DOCROOT_BUILD = '/orange/adamginsburg/web/public/avm_images'
LAYERS = {
    'jwst_gc_treasury_hips': f'{BUILD_ROOT}/jwst_gc_treasury_hips',
    'jwst_gc_treasury_miri_hips': f'{BUILD_ROOT}/jwst_gc_treasury_miri_hips',
    'jwst_gc_treasury_miri_bgmatch_hips':
        f'{BUILD_ROOT}/jwst_gc_treasury_miri_bgmatch_hips',
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


def needs_publish(src_props, dst_props):
    """Is the source strictly newer than the destination?

    Unknown on EITHER side means yes.  The failure this avoids is a silent
    no-op: a destination whose properties cannot be read looks "same" to any
    equality test, and the layer then never updates again.
    """
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


def publish_local(name, src, dry=False):
    """Docroot copy (what data.rc serves), staged and swapped."""
    dst = os.path.join(DOCROOT, name)
    src_props = _read_local(os.path.join(src, 'properties'))
    if not needs_publish(src_props, _read_local(os.path.join(dst, 'properties'))):
        print(f'  {name} docroot: up to date')
        return 0
    stage = dst + '.new'
    expect = count_tiles(os.walk, src)
    rc = _run(['rm', '-rf', stage], dry) or \
        _run(['rsync', '-a', src + '/', stage + '/'], dry)
    if rc or dry:
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


def publish_remote(name, src, dry=False, host=WEB_HOST, web_dir=WEB_DIR):
    """starformation copy, staged and verified REMOTELY before the swap."""
    dst = f'{web_dir}/{name}'
    src_props = _read_local(os.path.join(src, 'properties'))
    if not needs_publish(src_props, _read_remote(host, f'{dst}/properties')):
        print(f'  {name} {host}: up to date')
        return 0
    stage = dst + '.new'
    expect = count_tiles(os.walk, src)
    rc = _run(['ssh', host, f'rm -rf {shlex.quote(stage)}'], dry) or \
        _run(['rsync', '-az', src + '/', f'{host}:{stage}/'], dry)
    if rc or dry:
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
    ap.add_argument('--skip-remote', action='store_true',
                    help='docroot only (no ssh; for a host without web access)')
    args = ap.parse_args(argv)

    names = args.layer or sorted(LAYERS)
    bad = [n for n in names if n not in LAYERS]
    if bad:
        # An unknown name is a typo or a layer someone forgot to register; both
        # are worth stopping for, since the alternative is publishing nothing
        # and reporting success.
        ap.error(f'unknown layer(s): {", ".join(bad)}; known: {", ".join(sorted(LAYERS))}')

    rc = 0
    for name in names:
        src = LAYERS[name]
        print(f'{name}:', flush=True)
        if not os.path.isdir(src):
            print(f'  source missing: {src}', file=sys.stderr)
            rc = rc or 1
            continue
        rc = publish_local(name, src, dry=args.dry_run) or rc
        if not args.skip_remote:
            rc = publish_remote(name, src, dry=args.dry_run) or rc
    return rc


if __name__ == '__main__':
    sys.exit(main())
