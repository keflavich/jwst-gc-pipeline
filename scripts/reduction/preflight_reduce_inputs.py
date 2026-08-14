"""Does this field's reduce have inputs, before it waits 20 h to find out?

The reduce array fails a task with ::

    ValueError: Mismatch: Did not find any NIRCam asn files for module nrca
    for field 001 in /orange/adamginsburg/jwst/sgra/F115W/pipeline/

and ``run_field_retie_loop.sh`` then refuses to catalog a partially-failed
reduce -- correct, and it costs the whole iteration.  On a saturated cluster the
iteration is mostly queue: on 2026-08-14 the campaign's reduce arrays were
estimated to start ~20 h after submission, so a wrong field spec costs a day and
learns nothing.

**This is not a hypothetical.**  It caught ``run_sgra_4147_o001.sh``, which had
driven Sgr A* as proposal 4147 for the whole campaign.  There is no 4147 data in
the sgra tree at all -- every ``_cal`` frame is ``jw01939001001_...``, 216 of
them, and ``jw04147-o001_*_asn.json`` matches nothing in any of its three
filters.  Nothing already in the loop could see it: ``alignment_config``
resolves 4147/001 to a real offsets table, and that table exists.  It is simply
the wrong field's table, and every check the loop makes passes.

Checks, per (field, filter, module):

* an ``_asn.json`` exists for that proposal and observation;
* ``_cal`` frames exist for it;
* the detectors in those frames cover every requested module family.

Reports and returns nonzero if anything is missing.  Changes nothing.
"""
import argparse
import glob
import os
import re
import sys

#: ``jw01939001001_02101_00001_nrca1_cal.fits``
DETECTOR_RE = re.compile(r'_(nrc[ab](?:long|[1-4]))_')


def module_family(detector):
    """``nrca1`` / ``nrcalong`` -> ``nrca``.  The token a module spec names."""
    return detector[:4]


def check(root, target, proposal, obsid, filters, modules):
    """``[(filter, n_asn, n_cal, families, missing)]`` for one field."""
    # `merged` is a product of the two module reductions, not an input to look
    # for; a field with no A/B overlap drops it and nothing is missing.
    wanted = [m for m in modules if m != 'merged']
    rows = []
    for filt in filters:
        d = os.path.join(root, target, filt, 'pipeline')
        if not os.path.isdir(d):
            rows.append((filt, 0, 0, [], list(wanted), f'no directory {d}'))
            continue
        asns = glob.glob(os.path.join(
            d, f'jw{int(proposal):05d}-o{obsid}_*_asn.json'))
        cals = glob.glob(os.path.join(
            d, f'jw{int(proposal):05d}{obsid}*_cal.fits'))
        families = sorted({module_family(m.group(1)) for f in cals
                           for m in [DETECTOR_RE.search(os.path.basename(f))]
                           if m})
        missing = [m for m in wanted if m not in families]
        why = ''
        if not asns:
            why = f'no asn for jw{int(proposal):05d}-o{obsid}'
        elif not cals:
            why = f'no _cal for jw{int(proposal):05d}{obsid}'
        elif missing:
            why = f'no _cal frames for module(s) {missing}'
        rows.append((filt, len(asns), len(cals), families, missing, why))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--target', required=True)
    ap.add_argument('--proposal', required=True)
    ap.add_argument('--obsid', required=True)
    ap.add_argument('--filters', required=True,
                    help='space-separated, e.g. "F115W F212N F405N"')
    ap.add_argument('--modules', default='nrca,nrcb,merged')
    ap.add_argument('--root', default='/orange/adamginsburg/jwst')
    args = ap.parse_args(argv)

    rows = check(args.root, args.target, args.proposal, args.obsid,
                 args.filters.split(), args.modules.split(','))
    bad = 0
    for filt, n_asn, n_cal, families, missing, why in rows:
        ok = not why
        bad += 0 if ok else 1
        print(f'{"OK   " if ok else "MISSING"}  {args.target} '
              f'{args.proposal}/o{args.obsid} {filt:6s} '
              f'asn={n_asn:<4d} cal={n_cal:<4d} modules={families or "-"}'
              + (f'  -- {why}' if why else ''))
    if bad:
        print(f'\n{bad} filter(s) have no usable input.  The reduce would fail '
              f'those tasks and the loop would refuse to catalog the rest -- '
              f'check the PROPOSAL and OBSID against what is on disk before '
              f'spending the queue.')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
