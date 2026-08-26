"""Fill a new field's webbpsf grid cache from the grids other fields already have.

Gridded PSFs are set by the physics -- instrument, detector, filter,
oversampling -- so one grid serves every field.  The production cataloging path
does not act on that.  ``cataloging.py`` always passes
``psf_cache_dir={field_basepath}/psfs``, and ``get_psf_model`` uses that
directly (``_psf_outdir = psf_cache_dir or central_psf_dir(jwst_root)``), so
the shared ``psfs_shared/`` store is never consulted on that branch.  A field
with an empty ``psfs/`` therefore rebuilds grids that already exist one
directory over: the code prices a per-detector build at ~17-20 min and ~300 GB
peak, and the merged/all-detectors rebuild at ~7-8 h per phase, each behind a
MAST login from a compute node.

For ``gc-treasury`` that is eight SW builds for F212N plus the LW and MIRI
bands, on the critical path of tile 1 of 139, with an external network
dependency -- and every one of the grids exists already (#420).

This script links them in.  It reads only the donors and writes only into the
destination ``psfs/``; it is a dry run unless ``--apply`` is passed::

    python scripts/reduction/seed_psf_cache.py --field gc-treasury \\
        --filters F212N F480M F770W --apply

It does NOT decide the psfs_shared-versus-per-field question.  That is a
maintainer call (one store on /blue serving orange-rooted targets too, which is
what ``photometry/psf_paths.py`` and the ``basepath='/blue/.../jwst/'`` literal
recorded in ``tests/test_driver_basepath_from_registry.py`` intend), and this
seeds the directory the production path actually reads either way.

Link mechanics, because the roots straddle two filesystems: ``/blue/adamginsburg``
is ``/blue2/hpg`` and ``/orange/adamginsburg`` is ``/orange/hpg``, so
``os.link`` across them is ``EXDEV``.  A same-filesystem donor is hard-linked
(no extra bytes, no dangling reference if the donor tree is later moved), a
cross-filesystem donor is symlinked -- which is already the established pattern
here: ``brick/psfs`` and ``cloudc/psfs`` both reach F770W by a symlink to
``sgrb2``'s tree, and ``os.path.exists`` follows symlinks, so
``to_griddedpsfmodel`` loads them the same way.  ``--copy`` takes real bytes
instead (5.2 MB at samp2, 20.9 MB at samp4).
"""
import argparse
import glob
import os
import shutil
import sys

from jwst_gc_pipeline import fields

#: The name ``get_psf_model`` probes for, with the detector and oversampling
#: left open.  Pinned to the two f-strings in
#: ``photometry/crowdsource_catalogs_long.py`` by
#: ``tests/test_seed_psf_cache.py`` -- seeding a name the reader does not look
#: for is silent, and costs exactly the rebuild this exists to avoid.
GRID_GLOB = '*_{filt}_fovp101_samp*_npsf16.fits'

#: Where every field keeps its grids, relative to the field basepath.
PSFS_DIRNAME = 'psfs'


def donor_dirs(roots=None, skip=()):
    """Every ``{root}/{target}/psfs`` on disk, deduplicated by real path.

    Deduplicated because several targets under the /orange root are symlinks
    into /blue (``/orange/adamginsburg/jwst/brick`` is one), so a plain listing
    of both roots offers the same directory twice and the second copy looks
    like an independent donor.
    """
    roots = list((roots if roots is not None else fields.ROOTS.values()))
    seen, out = set(), []
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, '*', PSFS_DIRNAME))):
            if os.path.basename(os.path.dirname(path)) in skip:
                continue
            real = os.path.realpath(path)
            if real in seen or not os.path.isdir(path):
                continue
            seen.add(real)
            out.append(path)
    return out


def donor_grids(filtername, dirs):
    """``{grid filename: donor path}`` for one filter, first donor wins.

    Keyed by filename because that is what the reader looks up: the same
    ``nircam_nrca1_f212n_fovp101_samp2_npsf16.fits`` in brick and in cloudc is
    one grid to seed, not two.
    """
    found = {}
    pattern = GRID_GLOB.format(filt=filtername.lower())
    for directory in dirs:
        for path in sorted(glob.glob(os.path.join(directory, pattern))):
            found.setdefault(os.path.basename(path), path)
    return found


def _device(path, stat=os.stat):
    """``st_dev`` of ``path``'s nearest existing ancestor, or None.

    ``stat`` is injectable: a test cannot monkeypatch ``os.stat`` itself
    without breaking pytest's own tmpdir handling.
    """
    probe = os.path.realpath(path)
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return None
        probe = parent
    return stat(probe).st_dev


def prefer_same_filesystem(dirs, dest, stat=os.stat):
    """``dirs`` reordered so donors on the destination's filesystem come first.

    Several grids exist in both roots -- the eight F212N SW grids are in brick
    and cloudc (/blue) and in arches, quintuplet, sgra and sgrb2 (/orange).
    Taking the /blue copy for a /blue-rooted field turns a symlink into a hard
    link: no extra bytes either way, but a hard link survives the donor tree
    being moved or repathed, which a symlink into another target's directory
    does not.  Order within each group is unchanged, so the choice stays
    deterministic.
    """
    dev = _device(dest, stat=stat)
    return ([d for d in dirs if _device(d, stat=stat) == dev]
            + [d for d in dirs if _device(d, stat=stat) != dev])


def link_kind(src, dst_dir, copy=False, stat=os.stat):
    """'copy', 'hardlink' or 'symlink' for this donor and destination.

    ``os.link`` across filesystems raises ``EXDEV``, and the two roots are two
    filesystems, so the choice is made from ``st_dev`` rather than attempted
    and rescued.
    """
    if copy:
        return 'copy'
    dst_dev = _device(dst_dir, stat=stat)
    if dst_dev is None:
        return 'copy'
    return 'hardlink' if stat(os.path.realpath(src)).st_dev == dst_dev \
        else 'symlink'


def place(src, dst, kind):
    """Put ``src`` at ``dst`` by ``kind``.  Never overwrites."""
    if os.path.exists(dst):
        raise FileExistsError(dst)
    if kind == 'hardlink':
        os.link(os.path.realpath(src), dst)
    elif kind == 'symlink':
        os.symlink(os.path.realpath(src), dst)
    elif kind == 'copy':
        # Copy to a temporary name in the destination directory and rename, so
        # a killed job cannot leave a half-written grid that `os.path.exists`
        # reports as a cache hit -- the same reason the writer in
        # `get_psf_model` publishes atomically.
        tmp = f'{dst}.partial'
        shutil.copyfile(os.path.realpath(src), tmp)
        os.replace(tmp, dst)
    else:
        raise ValueError(f'unknown link kind {kind!r}')


def plan(field, filters, roots=None, dest=None, stat=os.stat):
    """``[(name, donor, dest_path, kind, action)]`` for one field.

    ``action`` is 'seed', 'present' (the destination already has it) or
    'missing' (no donor anywhere -- the field will rebuild that grid).
    """
    dest = dest or os.path.join(fields.fields_basepath(field).rstrip('/'),
                                PSFS_DIRNAME)
    dirs = prefer_same_filesystem(donor_dirs(roots=roots, skip=(field,)), dest,
                                  stat=stat)
    rows = []
    for filtername in filters:
        grids = donor_grids(filtername, dirs)
        if not grids:
            rows.append((f'(any {filtername.lower()} grid)', None, dest,
                         None, 'missing'))
            continue
        for name, src in sorted(grids.items()):
            dst = os.path.join(dest, name)
            if os.path.exists(dst):
                rows.append((name, src, dst, None, 'present'))
            else:
                rows.append((name, src, dst,
                             link_kind(src, dest, stat=stat), 'seed'))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--field', required=True,
                    help="registry field name, e.g. gc-treasury.  The "
                         "destination is that field's own psfs/, which is what "
                         "cataloging passes as psf_cache_dir; psfs_shared/ is "
                         "not on the production read path (#420).")
    ap.add_argument('--filters', required=True, nargs='+',
                    help='e.g. F212N F480M F770W')
    ap.add_argument('--copy', action='store_true',
                    help='take real bytes instead of linking')
    ap.add_argument('--apply', action='store_true',
                    help='actually create the links (default: report only)')
    args = ap.parse_args(argv)

    rows = plan(args.field, args.filters)
    if args.copy:
        rows = [(n, s, d, ('copy' if k else k), a) for n, s, d, k, a in rows]
    dest = os.path.join(fields.fields_basepath(args.field).rstrip('/'),
                        PSFS_DIRNAME)
    print(f'destination: {dest}')
    for name, src, dst, kind, action in rows:
        print(f'  {action:8s} {name}'
              + (f'  <- {src} ({kind})' if action == 'seed' else ''))

    to_seed = [r for r in rows if r[4] == 'seed']
    missing = [r for r in rows if r[4] == 'missing']
    if not args.apply:
        print(f'\ndry run: {len(to_seed)} grid(s) would be seeded.  '
              f'Re-run with --apply.')
        return 1 if missing else 0

    os.makedirs(dest, exist_ok=True)
    for name, src, dst, kind, _ in to_seed:
        place(src, dst, kind)
        print(f'seeded {dst} ({kind})')
    print(f'\n{len(to_seed)} grid(s) seeded.')
    if missing:
        for name, _, _, _, _ in missing:
            print(f'NO DONOR  {name} -- this filter will rebuild from '
                  f'MAST/Poppy on first use')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
