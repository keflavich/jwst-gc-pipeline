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
them, and ``jw04147-o001_*_image3_*_asn.json`` matches nothing in any of its
three filters.

Nothing already in the loop could see it.  ``alignment_config`` builds a path
for 4147/001 without asking whether that observation exists, and the loop's own
preconditions only require that path to be non-empty -- so the loop would have
watched a file that is not there, and the m2 checkpoint would have CREATED an
sgra offsets table named for Sgr C's proposal.  (Sgr C's real table lives in
sgrc's own directory; the two do not collide on disk, but the name asserts a
relationship that is not real.)  Worse, when no association file is present the
NIRCam reduce queries MAST for the proposal it was given and downloads what it
finds, so a 4147 sgra run would have written Sgr C's products into the sgra tree
before failing.

Two checks, cheapest first:

1. **against the registry** -- ``fields.yaml`` already records which target owns
   each (proposal, observation).  This needs no filesystem access, catches the
   sgra case outright, and also catches the case the on-disk scan cannot: the
   right proposal against the wrong target.
2. **against the frames** -- for each (filter, module): an ``image3`` association
   file exists for that proposal and observation, ``_cal`` frames exist for it,
   and the detectors in those frames cover every requested module family.

Reports and returns nonzero if anything is missing.  Changes nothing.
"""
import argparse
import glob
import os
import re
import sys

#: The association glob the reduce ITSELF uses -- copied verbatim from
#: ``reduction/PipelineRerunNIRCAM-LONG.py`` and ``reduction/PipelineMIRI.py``,
#: and pinned to them by ``test_preflight_reduce_inputs.py``.
#:
#: It has to be this pattern and not a looser one.  A field's ``pipeline``
#: directory holds three kinds of association file: the ``image3`` ones the
#: reduce consumes, the ``image2`` ones from the earlier stage, and the
#: pipeline's OWN catalog outputs (``..._mergedcat_model_asn.json``).  Counting
#: all of them made this check self-confirming from a previous run's products --
#: sgra/F115W reads 113 association files of which exactly ONE is an image3, and
#: brick/F115W 547 of which 3 are.
ASN_GLOB = 'jw0{proposal}-o{obsid}*_image3_*0[0-9][0-9]_asn.json'

#: ``jw01939001001_02101_00001_nrca1_cal.fits`` / ``..._mirimage_cal.fits``
DETECTOR_RE = re.compile(r'_(nrc[ab](?:long|[1-4])|mirimage|nis)_')

#: Detector tokens that carry no module distinction.  MIRI has one imager and
#: NIRISS one detector, so "does this module have frames" is not a question
#: their data can answer, and asking it of them reported every complete MIRI and
#: NIRISS field as missing both NIRCam modules.
SINGLE_DETECTOR = {'mirimage', 'nis'}


def module_family(token):
    """``nrca1`` / ``nrcalong`` / ``nrca`` -> ``nrca``.

    Applied to BOTH sides.  Applied only to the detector, a spec written as
    ``nrcalong,nrcblong`` -- which is how the long-wavelength submitters spell
    it -- compares ``nrcalong`` against the family ``nrca`` and reports both
    modules missing on a complete field.
    """
    token = str(token).lower()
    if token in SINGLE_DETECTOR:
        return token
    return token[:4]


class Row(object):
    """One (filter) verdict.  Attributes, not a tuple: the tuple had six
    elements and a docstring claiming five, and the tests indexed ``[5]``."""

    def __init__(self, filtername, n_asn, n_cal, families, missing, why):
        self.filtername = filtername
        self.n_asn = n_asn
        self.n_cal = n_cal
        self.families = families
        self.missing = missing
        self.why = why

    @property
    def ok(self):
        return not self.why

    def __repr__(self):
        return (f'Row({self.filtername!r}, asn={self.n_asn}, cal={self.n_cal}, '
                f'families={self.families!r}, why={self.why!r})')


def normalize_obsid(obsid):
    """``1`` / ``01`` / ``001`` -> ``001``; anything else is refused.

    Used raw, ``--obsid 1`` silently matches nothing and reports a real field
    as missing, and ``--obsid '*'`` pools every observation so two DIFFERENT
    observations can satisfy the two modules between them and read as OK.
    """
    s = str(obsid).strip()
    if not s.isdigit():
        raise ValueError(f'observation id {obsid!r} is not a number '
                         f'(a wildcard would pool several observations and '
                         f'let them satisfy the module check between them)')
    return f'{int(s):03d}'


def registry_verdict(target, proposal, obsid, instrument='nircam'):
    """``(ok, message)`` from ``fields.yaml`` alone -- no filesystem access.

    The registry already records which target owns each (proposal,
    observation).  Asking it first is cheaper than any scan and answers a
    question the scan cannot: whether the proposal belongs to a DIFFERENT
    target, which is exactly the sgra/Sgr C case.
    """
    from jwst_gc_pipeline import fields
    try:
        owner = fields.target_for_obsid(proposal, obsid, instrument=instrument)
    except KeyError as exc:
        return False, f'{exc}'
    # A target may be spelled with an instrument suffix on the command line
    # (`sgrc/niriss`), which is a directory, not a registry name.
    named = str(target).split('/')[0]
    if owner != named:
        return False, (f'proposal {proposal} observation {obsid} belongs to '
                       f'target {owner!r}, not {named!r}')
    return True, f'{owner} owns {proposal}/o{obsid}'


def check(root, target, proposal, obsid, filters, modules):
    """A :class:`Row` per requested filter."""
    if not filters:
        raise ValueError('no filters requested -- an empty filter list checks '
                         'nothing and would report success')
    obsid = normalize_obsid(obsid)
    # `merged` is a product of the two module reductions, not an input.  It is
    # not simply dropped: a merged product needs BOTH modules present, so
    # asking for it asks for both, and `--modules merged` alone used to switch
    # the module check off entirely.
    wanted = {module_family(m) for m in modules if m != 'merged'}
    if any(m == 'merged' for m in modules):
        wanted |= {'nrca', 'nrcb'}
    rows = []
    for filt in filters:
        d = os.path.join(root, target, filt, 'pipeline')
        if not os.path.isdir(d):
            rows.append(Row(filt, 0, 0, [], sorted(wanted), f'no directory {d}'))
            continue
        pat = ASN_GLOB.format(proposal=int(proposal), obsid=obsid)
        asns = glob.glob(os.path.join(d, pat))
        cals = glob.glob(os.path.join(
            d, f'jw{int(proposal):05d}{obsid}*_cal.fits'))
        families = sorted({module_family(m.group(1)) for f in cals
                           for m in [DETECTOR_RE.search(os.path.basename(f))]
                           if m})
        # A single-detector instrument answers "are there frames", not "which
        # module"; requiring NIRCam module families of it fails every MIRI and
        # NIRISS field that is entirely fine.
        single = bool(families) and set(families) <= SINGLE_DETECTOR
        missing = [] if single else sorted(w for w in wanted if w not in families)
        why = ''
        if not asns:
            why = f'no image3 association matching {pat}'
        elif not cals:
            why = f'no _cal for jw{int(proposal):05d}{obsid}'
        elif missing:
            why = f'no _cal frames for module(s) {missing}'
        rows.append(Row(filt, len(asns), len(cals), families, missing, why))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--target', required=True)
    ap.add_argument('--proposal', required=True)
    ap.add_argument('--obsid', required=True)
    ap.add_argument('--filters', required=True,
                    help='space-separated, e.g. "F115W F212N F405N"')
    ap.add_argument('--modules', default='nrca,nrcb,merged')
    ap.add_argument('--instrument', default='nircam',
                    help='which registry table to check the spec against')
    ap.add_argument('--root', default='/orange/adamginsburg/jwst')
    ap.add_argument('--skip-registry', action='store_true',
                    help='do not cross-check the spec against fields.yaml '
                         '(for a field that is deliberately not registered)')
    args = ap.parse_args(argv)

    filters = args.filters.split()
    if not filters:
        ap.error('--filters is empty; an empty list checks nothing')
    try:
        obsid = normalize_obsid(args.obsid)
    except ValueError as exc:
        ap.error(str(exc))
    if not str(args.proposal).strip().isdigit():
        ap.error(f'--proposal {args.proposal!r} is not a number')

    bad = 0
    if not args.skip_registry:
        ok, msg = registry_verdict(args.target, args.proposal, obsid,
                                   instrument=args.instrument)
        print(f'{"OK     " if ok else "MISMATCH"}  registry: {msg}')
        if not ok:
            bad += 1

    rows = check(args.root, args.target, args.proposal, obsid,
                 filters, args.modules.split(','))
    for r in rows:
        bad += not r.ok
        print(f'{"OK" if r.ok else "MISSING":<7s}  {args.target} '
              f'{args.proposal}/o{obsid} '
              f'{r.filtername:6s} '
              f'asn={r.n_asn:<4d} cal={r.n_cal:<4d} '
              f'modules={r.families or "-"}'
              + (f'  -- {r.why}' if r.why else ''))
    if bad:
        print(f'\n{bad} problem(s).  The reduce would fail those tasks and the '
              f'loop would refuse to catalog the rest -- check the PROPOSAL and '
              f'OBSID against what is on disk before spending the queue.')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
