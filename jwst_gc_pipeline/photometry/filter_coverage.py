"""Coverage audit: filters that were REDUCED but are not registered for cataloging.

The cataloging filter list for a field is derived from
``merge_catalogs.obs_filters``.  A filter that has ``*_crf.fits`` frames on disk
(i.e. the reduction ran and produced cataloging-ready input) but is absent from
that map is never *asked for*: no job is submitted, nothing errors, and the gap
only shows up later as a missing catalog.

That is exactly how issue #160 happened.  W51 F444W was reduced onto the same
``align_o001_crf`` path as F335M/F405N/F480M, but ``obs_filters['w51']['6151']``
listed ten filters and F444W was not one of them, so every W51 cataloging
submission ran on the other ten and F444W ended up the only W51 long-wavelength
filter with no catalog at all.  A *failed* run leaves a traceback in the logs; an
*omitted* one leaves nothing, which is why this audit exists.

Usage::

    python -m jwst_gc_pipeline.photometry.filter_coverage --target w51
    python -m jwst_gc_pipeline.photometry.filter_coverage --all

Exit code is 1 when any gap is found, so it can gate a submission script.
"""
import argparse
import glob
import os
import re
import sys

# A filter directory is the uppercase filter name at the top of a field's
# basepath: F444W, F1280W, F150W2, F322W2.  Anything else (catalogs/, pipeline/,
# region files, ...) is not a filter.
FILTER_DIR_RE = re.compile(r'^F\d{3,4}[A-Z]{1,2}\d?$')

DEFAULT_BASEPATH_ROOT = '/orange/adamginsburg/jwst'


def _known_filters(target, obs_filters_map=None):
    """Lowercase set of filters registered for ``target`` (union over proposals).

    ``obs_filters_map`` overrides the production map (used by the tests).
    """
    if obs_filters_map is None:
        from jwst_gc_pipeline.photometry.merge_catalogs import obs_filters
        obs_filters_map = obs_filters
    if target not in obs_filters_map:
        raise KeyError(f'target {target!r} is not in obs_filters; '
                       f'known targets: {sorted(obs_filters_map)}')
    return {filn.lower()
            for filts in obs_filters_map[target].values()
            for filn in filts}


def reduced_filters_on_disk(basepath, crf_glob='*_crf.fits'):
    """``{lowercase filter: n_frames}`` for every filter dir under ``basepath``
    that holds reduced per-exposure frames matching ``crf_glob``.

    Filters whose directory exists but holds no matching frame are omitted: an
    empty ``F200W/pipeline`` means the reduction produced nothing to catalog, so
    it is not a cataloging gap.
    """
    counts = {}
    for entry in sorted(os.listdir(basepath)):
        if not FILTER_DIR_RE.match(entry):
            continue
        frames = glob.glob(os.path.join(basepath, entry, 'pipeline', crf_glob))
        if frames:
            counts[entry.lower()] = len(frames)
    return counts


def uncataloged_filters(basepath, target, obs_filters_map=None,
                        crf_glob='*_crf.fits', min_frames=1):
    """Filters reduced on disk under ``basepath`` but missing from ``obs_filters``.

    Returns a sorted list of ``(filter, n_frames)``.  A non-empty result means
    the next cataloging pass for ``target`` will silently skip those filters.
    """
    known = _known_filters(target, obs_filters_map=obs_filters_map)
    on_disk = reduced_filters_on_disk(basepath, crf_glob=crf_glob)
    return sorted((filn, n) for filn, n in on_disk.items()
                  if filn not in known and n >= min_frames)


def _report(target, basepath, crf_glob, min_frames, obs_filters_map=None):
    """Print one target's audit; return the number of gaps found."""
    if not os.path.isdir(basepath):
        print(f'{target}: SKIP (no basepath {basepath})')
        return 0
    gaps = uncataloged_filters(basepath, target,
                               obs_filters_map=obs_filters_map,
                               crf_glob=crf_glob, min_frames=min_frames)
    if not gaps:
        print(f'{target}: OK (every reduced filter under {basepath} is registered)')
        return 0
    print(f'{target}: {len(gaps)} REDUCED-BUT-UNREGISTERED filter(s) under {basepath}:')
    for filn, nframes in gaps:
        print(f'    {filn.upper():8s} {nframes:4d} frame(s) matching {crf_glob} '
              f'-- absent from obs_filters[{target!r}], so cataloging skips it')
    return len(gaps)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target', default=None,
                        help='field name as keyed in obs_filters (e.g. w51)')
    parser.add_argument('--all', action='store_true',
                        help='audit every target in obs_filters')
    parser.add_argument('--basepath', default=None,
                        help='field data root (default <root>/<target>)')
    parser.add_argument('--basepath-root', default=DEFAULT_BASEPATH_ROOT,
                        help=f'parent of the per-field dirs (default {DEFAULT_BASEPATH_ROOT})')
    parser.add_argument('--crf-glob', default='*_crf.fits',
                        help='per-exposure frame glob counted as "reduced" '
                             '(default *_crf.fits; e.g. *align_o001_crf.fits)')
    parser.add_argument('--min-frames', type=int, default=1,
                        help='ignore filters with fewer than this many frames')
    args = parser.parse_args(argv)

    if not args.target and not args.all:
        parser.error('give --target or --all')
    if args.basepath and args.all:
        parser.error('--basepath is for a single --target')

    from jwst_gc_pipeline.photometry.merge_catalogs import obs_filters
    targets = sorted(obs_filters) if args.all else [args.target]

    ngaps = 0
    for target in targets:
        basepath = args.basepath or os.path.join(args.basepath_root, target)
        ngaps += _report(target, basepath, args.crf_glob, args.min_frames)
    if ngaps:
        print(f'\n{ngaps} gap(s).  Add the filter(s) to obs_filters in '
              f'photometry/merge_catalogs.py, then catalog them.')
    return 1 if ngaps else 0


if __name__ == '__main__':
    sys.exit(main())
