#!/usr/bin/env python
"""Write a date-tagged astrometry-audit report + evidence plot into each
field's parent folder, in one consistent layout.

For every field with an audit JSON (produced by ``astrometry_audit.py``):

  <basepath>/<field>/ASTROMETRY_AUDIT_<date>.md      -- summary doc
  <basepath>/<field>/audit_plots/astrometry_offsets_<date>.png

The doc records: audit method + thresholds, per-filter offsets for each class
(inter-module / per-filter / absolute), flags, the verdict, any stale-mosaic
renames logged for the field, and pointers to additional (e.g. photometry)
evidence plots placed in the same ``audit_plots/`` folder by other QA steps.

Usage:
  python make_astrometry_audit_report.py [--results DIR] [--basepath DIR]
      [--field NAME ...]        # default: every field with a JSON
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THRESH = dict(intermodule=15.0, perfilter=30.0, absolute=75.0)
COLOR = dict(intermodule='#2563eb', perfilter='#ea580c', absolute='#7c3aed')


def plot_field(r, out_png):
    classes = [c for c in ('intermodule', 'perfilter', 'absolute') if r.get(c)]
    if not classes:
        return False
    filters = sorted({f for c in classes for f in r[c]})
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(filters) + 2.2))
    y = np.arange(len(filters))
    h = 0.8 / max(len(classes), 1)
    for k, cls in enumerate(classes):
        vals = [r[cls].get(f, {}).get('off', np.nan) for f in filters]
        ax.barh(y + (k - (len(classes) - 1) / 2) * h, vals, height=h * 0.9,
                color=COLOR[cls], label=f'{cls} (thresh {THRESH[cls]:.0f})')
        ax.axvline(THRESH[cls], color=COLOR[cls], lw=1.1, ls='--', alpha=0.7)
    ax.set(yticks=y, yticklabels=filters, xlabel='offset  [mas]',
           title=f"{r['field']} astrometry audit -- worst offsets per filter")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return True


def write_report(r, field_dir, date):
    os.makedirs(f'{field_dir}/audit_plots', exist_ok=True)
    png = f'{field_dir}/audit_plots/astrometry_offsets_{date}.png'
    have_plot = plot_field(r, png)

    lines = [
        f"# Astrometry audit -- {r['field']} -- {date}",
        "",
        "Method: offset-histogram stacking (pair-histogram peak; crowding-proof,",
        "NOT nearest-neighbour) against the VIRAC2+Gaia reference catalog, plus a",
        "reference-free NRCA-vs-NRCB inter-module check.  Tool:",
        "`scripts/reduction/astrometry_audit.py` (jwst-gc-pipeline);",
        f"thresholds: inter-module {THRESH['intermodule']:.0f} mas, per-filter "
        f"{THRESH['perfilter']:.0f} mas, absolute {THRESH['absolute']:.0f} mas.",
        "",
    ]
    if r.get('epoch'):
        lines.append(f"Mosaic epoch: {r['epoch']:.3f}")
    if r.get('anchor'):
        lines.append(f"Per-filter anchors: {r['anchor']}")
    lines.append("")

    for cls, title in (('intermodule', 'Inter-module (NRCA vs NRCB, reference-free)'),
                       ('perfilter', 'Per-filter vs channel anchor (internal)'),
                       ('absolute', 'Absolute vs VIRAC2/Gaia reference')):
        d = r.get(cls) or {}
        if not d:
            continue
        lines += [f"## {title}", "", "| filter | offset (mas) | detail |", "|---|---|---|"]
        for f in sorted(d):
            v = d[f]
            extra = []
            if 'npairs' in v:
                extra.append(f"n={v['npairs']}")
            if 'nshared' in v:
                extra.append(f"n={v['nshared']}")
            if 'peak_ratio' in v:
                extra.append(f"peak/bg={v['peak_ratio']:.0f}")
            mark = ' **FLAG**' if v['off'] > THRESH[cls] else ''
            lines.append(f"| {f} | {v['off']:.1f}{mark} | {', '.join(extra)} |")
        lines.append("")

    flags = r.get('flags') or []
    lines += ["## Verdict", ""]
    if flags:
        lines += ["**FLAGGED** -- this field's astrometry needs remediation:", ""]
        lines += [f"- {fl}" for fl in flags]
    else:
        lines += ["**CLEAN** -- no offsets above threshold."]
    lines.append("")

    renames = sorted(glob.glob(f'{field_dir}/_stale_rename_*.log'))
    if renames:
        lines += ["## Stale bad-astrometry mosaics",
                  "",
                  "Superseded realigned/reprojected mosaics were renamed to",
                  "`*_badastrometry_stale` / `*_stale` so they cannot be read by",
                  "accident.  Rename logs:", ""]
        lines += [f"- `{os.path.basename(p)}`" for p in renames]
        lines.append("")

    lines += ["## Evidence plots (audit_plots/)", ""]
    if have_plot:
        lines.append(f"- `astrometry_offsets_{date}.png` -- per-filter offsets vs thresholds")
    for p in sorted(glob.glob(f'{field_dir}/audit_plots/*.png')):
        b = os.path.basename(p)
        if b != f'astrometry_offsets_{date}.png':
            lines.append(f"- `{b}`")
    lines.append("")

    out = f'{field_dir}/ASTROMETRY_AUDIT_{date}.md'
    with open(out, 'w') as fh:
        fh.write('\n'.join(lines))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results', default='/orange/adamginsburg/jwst/astrometry_audit_results')
    ap.add_argument('--basepath', default='/orange/adamginsburg/jwst')
    ap.add_argument('--field', action='append', default=None)
    ap.add_argument('--date', default=time.strftime('%Y-%m-%d'))
    args = ap.parse_args()

    jsons = sorted(glob.glob(f'{args.results}/*.json'))
    for j in jsons:
        r = json.load(open(j))
        field = r.get('field') or os.path.basename(j)[:-5]
        if args.field and field not in args.field:
            continue
        if r.get('error'):
            print(f"[{field}] audit errored ({r['error']}); skipping")
            continue
        fdir = f'{args.basepath}/{field}'
        if not os.path.isdir(fdir):
            print(f"[{field}] no field dir {fdir}; skipping")
            continue
        out = write_report(r, fdir, args.date)
        print(f"[{field}] wrote {out}")


if __name__ == '__main__':
    main()
