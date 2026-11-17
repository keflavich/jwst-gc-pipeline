#!/usr/bin/env python
"""List the per-visit bulk differential of every ``RECORDED_BULK`` field (#857).

Reads ``alignment_config.ALIGNMENT_CONFIG`` only; no data product and no
compute.  A visit whose filters carry different recorded bulks is held apart
by that difference once ``fix_alignment`` has run, and the m7 cross-filter gate
then reports it.  ``tie_through_reference_filter`` and ``reference_filter``
are shown beside each entry, since an opted-in entry is expected to show a
zero differential.

Usage
-----
    python scripts/release/audit_recorded_bulk_consistency.py [--tol 5.0]

Exit status is 1 if any visit exceeds ``--tol`` mas.
"""
import argparse
import sys

from jwst_gc_pipeline.reduction import alignment_config as ac
from jwst_gc_pipeline.reduction.recorded_bulk_audit import bulk_spread_by_visit


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--tol', type=float, default=5.0,
                    help='per-visit differential (mas) above which a visit is '
                         'flagged (default: the m7 cross-filter gate, 5 mas)')
    args = ap.parse_args(argv)

    n_bad = 0
    for cfg in ac.ALIGNMENT_CONFIG:
        if cfg.source != ac.RECORDED_BULK or not cfg.recorded_bulk:
            continue
        print(f"proposal {cfg.proposal}  reference_filter={cfg.reference_filter}  "
              f"tie_through_reference_filter={cfg.tie_through_reference_filter}")
        for visit, row in sorted(bulk_spread_by_visit(cfg).items()):
            flag = 'FLAG' if row['max_differential_mas'] > args.tol else 'ok'
            print(f"  {visit}  anchor {row['anchor']}  max differential "
                  f"{row['max_differential_mas']:.1f} mas  [{flag}]")
            for filt, (dra, ddec) in row['differential_mas'].items():
                bra, bdec = row['filters'][filt]
                print(f"    {filt:7s} bulk ({bra:+9.1f}, {bdec:+9.1f}) mas  "
                      f"minus anchor ({dra:+8.1f}, {ddec:+8.1f}) mas")
            n_bad += flag == 'FLAG'
    print(f"{n_bad} visit(s) above {args.tol:g} mas")
    return 1 if n_bad else 0


if __name__ == '__main__':
    sys.exit(main())
