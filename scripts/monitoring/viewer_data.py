#!/usr/bin/env python
"""Shared pieces of the sky viewer's derived data files, and their publish check.

Two files in the web docroot are built from inputs that keep changing: the
focal plane (`build_focal_plane.py`) and the guide stars
(`build_guidestars.py`).  `refresh_viewer_data.sh` rebuilds both on a schedule;
this module holds what the two builders and that script have in common.

    python scripts/monitoring/viewer_data.py check focal NEW [PREVIOUS]
    python scripts/monitoring/viewer_data.py check guide NEW [PREVIOUS] --allow-shrink
    python scripts/monitoring/viewer_data.py web-target
"""
import argparse
import datetime
import json
import os
import re
import sys

#: Where the data releases live; each holds `gc-treasury/exposures`.
RELEASES_ROOT = '/orange/adamginsburg/jwst/releases'
#: The directory both viewer files are served from (data.rc serves it directly,
#: starformation holds a copy).  Kept in step with `publish_hips_layers.DOCROOT`
#: by a test rather than by an import, so a builder does not load the publisher.
DOCROOT = '/orange/adamginsburg/web/public/avm_images'

_RELEASE_RE = re.compile(r'^v(\d+)\.(\d+)-')

#: The counts that must not shrink, per document kind.  The focal plane's
#: pointing count tracks the plan and is already complete when the as-flown
#: attitudes are stale, so `n_as_flown` is the quantity that catches a rebuild
#: that lost the delivered frames.
_COUNTS = {
    'focal': (('pointings', lambda d: len(d.get('pointings') or [])),
              ('n_as_flown', lambda d: int(d.get('n_as_flown') or 0))),
    'guide': (('n', lambda d: int(d.get('n') or 0)),
              ('sources', lambda d: len(d.get('sources') or []))),
}


def release_key(name):
    """Sort key of a release directory name such as ``v1.9-2026.10``, or None."""
    match = _RELEASE_RE.match(name)
    return (int(match.group(1)), int(match.group(2))) if match else None


def release_name(path):
    """The release directory a path lies under, or None."""
    for part in os.path.normpath(path or '').split(os.sep):
        if release_key(part) is not None:
            return part
    return None


def newest_release_exposures(root=RELEASES_ROOT):
    """The delivered-exposures directory of the newest release, or None.

    Release directories without `gc-treasury/exposures` are passed over, so a
    release that holds only another survey does not hide the Treasury frames.
    """
    try:
        names = os.listdir(root)
    except OSError:
        return None
    for name in sorted((n for n in names if release_key(n)),
                       key=release_key, reverse=True):
        candidate = os.path.join(root, name, 'gc-treasury', 'exposures')
        if os.path.isdir(candidate):
            return candidate
    return None


def resolve_exposures(explicit=None, root=RELEASES_ROOT):
    """``explicit`` if given (an empty string is kept), else ``$EXPOSURES``,
    else the newest release's exposures directory.

    Pinning one release is what froze the as-flown set at v1.8 while v1.9 was
    already delivered.  A newer release that is only partly delivered shows up
    as a shrunken `n_as_flown`, which the publish check refuses.
    """
    if explicit is not None:
        return explicit
    env = os.environ.get('EXPOSURES')
    if env:
        return env
    found = newest_release_exposures(root)
    if found is None:
        raise SystemExit(f'no release under {root} has gc-treasury/exposures; '
                         f'pass --exposures or set EXPOSURES')
    return found


def utc_stamp():
    """ISO-8601 UTC, to the second, for a document's ``generated`` field."""
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        '%Y-%m-%dT%H:%M:%SZ')


def _load(path):
    with open(path) as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f'{path} is not a JSON object')
    return doc


def counts(kind, doc):
    return {name: fn(doc) for name, fn in _COUNTS[kind]}


def check(kind, new_path, previous_path=None, allow_shrink=False):
    """Problems that make ``new_path`` unfit to replace ``previous_path``.

    An empty list means it may be published.  A document with no entries is
    never publishable; a smaller one than the published copy is refused unless
    ``allow_shrink``.  A previous copy that is absent or unreadable imposes no
    floor.
    """
    if not os.path.isfile(new_path) or os.path.getsize(new_path) == 0:
        return [f'{new_path} is missing or empty']
    try:
        new = counts(kind, _load(new_path))
    except (OSError, ValueError) as err:
        return [f'{new_path} does not parse: {err}']
    problems = []
    entries = new['pointings'] if kind == 'focal' else new['sources']
    if entries == 0:
        problems.append(f'{new_path} holds no '
                        f"{'pointings' if kind == 'focal' else 'guide stars'}")
    if previous_path and os.path.isfile(previous_path) and not allow_shrink:
        try:
            old = counts(kind, _load(previous_path))
        except (OSError, ValueError):
            old = {}
        for name, value in new.items():
            if value < old.get(name, 0):
                problems.append(f'{name} would shrink from {old[name]} to '
                                f'{value}; set ALLOW_SHRINK=1 to publish it')
    return problems


def web_target():
    """``(host, directory)`` of the second copy, as `publish_hips_layers` has it."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import publish_hips_layers as ph
    return ph.WEB_HOST, ph.WEB_DIR


def summary(focal_path, guide_path):
    """One log line: pointings, as-flown pointings, guide stars, unflown."""
    focal, guide = _load(focal_path), _load(guide_path)
    return (f"{len(focal.get('pointings') or [])} pointings "
            f"({int(focal.get('n_as_flown') or 0)} as-flown), "
            f"{len(guide.get('sources') or [])} guide stars, "
            f"{len(guide.get('unflown') or [])} unflown observations")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    chk = sub.add_parser('check', help='validate a staged file')
    chk.add_argument('kind', choices=sorted(_COUNTS))
    chk.add_argument('new')
    chk.add_argument('previous', nargs='?')
    chk.add_argument('--allow-shrink', action='store_true')
    sub.add_parser('web-target', help='print the starformation host and docroot')
    summ = sub.add_parser('summary', help='one log line for the two files')
    summ.add_argument('focal')
    summ.add_argument('guide')
    args = ap.parse_args(argv)

    if args.cmd == 'check':
        problems = check(args.kind, args.new, args.previous, args.allow_shrink)
        for line in problems:
            print(f'REFUSED: {line}', file=sys.stderr)
        return 1 if problems else 0
    if args.cmd == 'web-target':
        print('\n'.join(web_target()))
        return 0
    print(summary(args.focal, args.guide))
    return 0


if __name__ == '__main__':
    sys.exit(main())
