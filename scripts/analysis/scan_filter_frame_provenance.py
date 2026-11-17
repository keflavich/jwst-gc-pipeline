#!/usr/bin/env python
"""List which F162M/F164N frames carry the filter-frame correction (issue #1133).

``fix_alignment`` has applied the per-detector F162M/F164N shift by default
since PR #1261.  Frames reduced before that, or regenerated with
``FILTER_FRAME_CORRECTION=0``, carry the ~10 mas per-detector term and no
marker says so on the catalogues built from them.  This script reads the
``FFRAMCOR`` / ``FFRAMPND`` / ``FFRAMANC`` cards of the SCI headers and turns
"which products are stale" into a list.

Frames are read from ``<basepath>/<FILTER>/pipeline/`` for each field
directory given, over the ``_align``, ``_destreak`` and ``_crf`` lineages (the
``_crf`` copy carries the working copy's header cards; the ``align`` and
``destreak`` crf copies are grouped separately).  Explicit FITS paths
are accepted too.  The selection follows ``FILTER_FRAME_CORRECTION`` exactly
as ``fix_alignment`` reads it, unless ``--mode`` overrides it.

Groups (one band of one observation of one field) that hold both corrected
and uncorrected frames are flagged ``MIXED``.  The exit status is 1 when a
selected frame is stale, pending or on the wrong anchor and ``--check`` is
given, so the scan can gate a rerun; ``--lineage`` restricts the scan (and the
gate) to some lineages, e.g. ``--lineage destreak_crf`` once a field has been
regenerated through ``destreak`` and the old ``align`` copies are to be ignored.
Other exit statuses: 2 no frame found (the field directory must hold
``<FILTER>/pipeline/jw*_nrc[ab][1-4]*_{align,destreak,crf}.fits``, so pass the
field directory, not ``<FILTER>/pipeline``), 3 frames found but none selected
under the mode (``FILTER_FRAME_CORRECTION=0`` or ``--mode off`` select nothing,
which is not the same as "all corrected"), 4 an invalid
``FILTER_FRAME_CORRECTION`` value; 0 otherwise.  Groups are keyed on the field
directory too, so two trees of one program are never pooled.

Regenerate stale frames from ``_cal``; do not re-apply the shift on top of an
old working copy.  Read-only: no header is written.

Usage:
    scan_filter_frame_provenance.py /blue/.../jwst/cloudef
    scan_filter_frame_provenance.py --filters F162M F164N --list /blue/.../jwst/wd2
    scan_filter_frame_provenance.py /blue/.../jwst/sgrc/F162M/pipeline/*_crf.fits
"""
import argparse
import collections
import glob
import os
import sys

from astropy.io import fits

from jwst_gc_pipeline.reduction import filter_frame_correction as ffc

FRAME_SUFFIXES = ('_align.fits', '_destreak.fits', '_crf.fits')
DEFAULT_FILTERS = ffc.DEFAULT_ON_FILTERS
LINEAGES = ('align', 'destreak', 'align_crf', 'destreak_crf')
EXIT_NONE_FOUND, EXIT_NONE_SELECTED, EXIT_BAD_MODE = 2, 3, 4
BAD = (ffc.STATUS_STALE, ffc.STATUS_PENDING, ffc.STATUS_WRONG_ANCHOR)


def find_frames(path, filters):
    """FITS frames under ``path``: the file itself, or a field directory."""
    if os.path.isfile(path):
        return [path]
    found = []
    for filt in filters:
        for suffix in FRAME_SUFFIXES:
            found += glob.glob(os.path.join(path, filt, 'pipeline',
                                            f'jw*_nrc[ab][1-4]*{suffix}'))
    return sorted(set(found))


def lineage_of(fn):
    """``align``, ``destreak``, ``align_crf`` or ``destreak_crf``.

    The two crf lineages are separate products (cataloguing reads either one
    through ``--each-suffix``), so a field regenerated through ``destreak`` is
    not a mixed group because older ``align_o*_crf`` copies sit beside it.
    """
    base = os.path.basename(fn)
    stem = 'destreak' if '_destreak' in base else 'align'
    return stem + ('_crf' if base.endswith('_crf.fits') else '')


def field_of(fn):
    """The field directory of ``<field>/<FILTER>/pipeline/<frame>``."""
    pipeline_dir = os.path.dirname(os.path.abspath(fn))
    return os.path.dirname(os.path.dirname(pipeline_dir))


def read_provenance(fn, mode):
    """``(group_key, record)`` for one frame; the key is (field dir, program,
    observation, filter, lineage)."""
    hdr0 = fits.getheader(fn, ext=0)
    hdr = fits.getheader(fn, ext=('SCI', 1))
    rec = ffc.frame_provenance(hdr0, hdr, mode=mode)
    rec['file'] = fn
    key = (field_of(fn), str(hdr0.get('PROGRAM', '')), str(hdr0.get('OBSERVTN', '')),
           rec['filter'], lineage_of(fn))
    return key, rec


def scan(paths, filters, mode, lineages=LINEAGES):
    records = []
    for path in paths:
        for fn in find_frames(path, filters):
            if lineage_of(fn) not in lineages:
                continue
            try:
                records.append(read_provenance(fn, mode))
            except (OSError, KeyError, ValueError) as exc:
                print(f"WARNING: cannot read {fn}: {exc}", file=sys.stderr)
    return records


def report(records, list_files=False):
    counts = collections.defaultdict(collections.Counter)
    for key, rec in records:
        counts[key][rec['status']] += 1
    mixed = set(ffc.mixed_groups(records))
    print(f"{'field':>10} {'program':>8} {'obs':>4} {'filter':>7} {'lineage':>12}  "
          f"{'corrected':>9} {'stale':>5} {'pending':>7} {'wrong-anc':>9}  flag")
    for key in sorted(counts):
        c = counts[key]
        if set(c) == {ffc.STATUS_NOT_SELECTED}:
            continue
        flag = 'MIXED' if key in mixed else ('STALE' if any(c[s] for s in BAD) else 'ok')
        print(f"{os.path.basename(key[0]):>10} {key[1]:>8} {key[2]:>4} "
              f"{key[3]:>7} {key[4]:>12}  "
              f"{c[ffc.STATUS_CORRECTED]:>9} {c[ffc.STATUS_STALE]:>5} "
              f"{c[ffc.STATUS_PENDING]:>7} {c[ffc.STATUS_WRONG_ANCHOR]:>9}  {flag}")
    if list_files:
        for _, rec in records:
            if rec['status'] in BAD:
                print(f"{rec['status']:>12}  {rec['file']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('paths', nargs='+',
                    help='field directories (<path>/<FILTER>/pipeline/...) or FITS files')
    ap.add_argument('--filters', nargs='+', default=list(DEFAULT_FILTERS),
                    help='filter subdirectories to scan in a field directory '
                         f'(default {" ".join(DEFAULT_FILTERS)})')
    ap.add_argument('--mode', choices=('default', 'all', 'off'), default=None,
                    help='which frames count as selected; default reads '
                         'FILTER_FRAME_CORRECTION as fix_alignment does')
    ap.add_argument('--lineage', nargs='+', choices=LINEAGES, default=list(LINEAGES),
                    help='lineages to scan and to gate on (default all four)')
    ap.add_argument('--list', action='store_true', dest='list_files',
                    help='list every stale, pending or wrong-anchor frame')
    ap.add_argument('--check', action='store_true',
                    help='exit 1 when any selected frame is not corrected')
    args = ap.parse_args(argv)
    filters = [f.upper() for f in args.filters]
    try:
        mode = args.mode or ffc.correction_mode()
    except ffc.FilterFrameError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_BAD_MODE
    records = scan(args.paths, filters, mode, lineages=args.lineage)
    if not records:
        print("no frames found: pass a field directory holding "
              "<FILTER>/pipeline/jw*_nrc[ab][1-4]*_{align,destreak,crf}.fits "
              "(not <FILTER>/pipeline itself) or FITS files", file=sys.stderr)
        return EXIT_NONE_FOUND
    if all(rec['status'] == ffc.STATUS_NOT_SELECTED for _, rec in records):
        print(f"no frame selected under mode {mode!r}: nothing was checked "
              "(FILTER_FRAME_CORRECTION=0 / --mode off select no frame)",
              file=sys.stderr)
        return EXIT_NONE_SELECTED if args.check else 0
    report(records, list_files=args.list_files)
    bad = sum(rec['status'] in BAD for _, rec in records)
    return 1 if (args.check and bad) else 0


if __name__ == '__main__':
    sys.exit(main())
