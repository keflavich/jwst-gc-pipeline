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
   the reduce would actually use exists for that proposal and observation,
   ``_cal`` frames exist for it, and that association's MEMBERS cover every
   requested module family.  Coverage comes from the members rather than from a
   listing of the directory, because the reduce narrows the association to one
   module's members and raises when none remain -- a module present on disk but
   absent from the association is a failure a listing cannot see.

Reports and returns nonzero if anything is missing.  Changes nothing.
"""
import argparse
import glob
import json
import os
import re
import sys

# `jw_prefix` is imported inside `check()`.  This module's own imports are
# stdlib-only on purpose -- it PARSES the reduce driver (see
# `reduce_module_policy`) and defers `fields` for the same reason -- so `--help`
# and a parse-only run stay off the numpy/astropy import that executing the
# package `__init__` performs.

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
#: ``{jw}`` is filled with ``jw_prefix(proposal)``, the 5-digit-padded MAST
#: prefix (issue #414).
ASN_GLOB = '{jw}-o{obsid}*_image3_*0[0-9][0-9]_asn.json'

#: ``jw01939001001_02101_00001_nrca1_cal.fits`` / ``..._mirimage_cal.fits``
DETECTOR_RE = re.compile(r'_(nrc[ab](?:long|[1-4])|mirimage|nis)_')

#: Detector tokens that carry no module distinction.
SINGLE_DETECTOR = {'mirimage', 'nis'}

#: Instruments with one detector, for which "which module has frames" has no
#: answer.  Asking it of them reported every complete MIRI and NIRISS field as
#: missing both NIRCam modules.  Keyed on the INSTRUMENT, since keying it on the
#: detector tokens found on disk let a NIRCam spec aimed at a NIRISS directory
#: report OK.
_SINGLE_DETECTOR_INSTRUMENTS = {'miri', 'niriss'}

#: The member-exposure token the reduce requires of an association it will use.
#: One observation can produce associations for several instruments under the
#: same proposal and observation number, and the reduce keeps only its own.
_MEMBER_TOKEN = {'nircam': 'nrc', 'miri': 'mirimage', 'niriss': '_nis_'}

#: Module tokens a `--modules` spec may name.  A token outside this set is a
#: typo, and a typo used to be truncated to four characters and reported as a
#: module genuinely missing from the data.
_MODULE_TOKENS = ({'nrca', 'nrcb', 'merged', 'nrcalong', 'nrcblong'}
                  | {f'nrc{ab}{n}' for ab in 'ab' for n in '1234'})


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


#: Where the reduce declares that an observation uses only some modules.
_REDUCE_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'jwst_gc_pipeline', 'reduction', 'PipelineRerunNIRCAM-LONG.py')


def reduce_module_policy(script=None):
    """``MODULES_BY_PROPOSAL_FIELD_FILTER`` as the reduce declares it.

    Read by PARSING rather than importing: that module pulls in the whole JWST
    stack, and this check exists to be run in ten seconds before submitting.
    Parsing also means a policy the reduce cannot express cannot be invented
    here.

    Without it, a correct field reads as broken: sickle 3958/007 is restricted
    to module B by this policy, so the default module spec reported both its
    filters MISSING on data that reduces fine -- and the README documented that
    invocation.  A check whose documented use produces false alarms is a check
    operators learn to ignore.
    """
    import ast
    path = script or _REDUCE_SCRIPT
    try:
        tree = ast.parse(open(path).read(), filename=path)
    except (OSError, SyntaxError):
        return {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id == 'MODULES_BY_PROPOSAL_FIELD_FILTER':
                try:
                    return ast.literal_eval(node.value)
                except ValueError:
                    return {}
    return {}


def allowed_modules(proposal, obsid, filtername, requested, policy=None,
                    as_written=None):
    """``requested``, narrowed to what the reduce will actually run.

    Returns the families the reduce would produce for this (proposal,
    observation, filter).  A field with no policy entry is unrestricted, which
    is every field but one.
    """
    policy = reduce_module_policy() if policy is None else policy
    entry = (policy.get(str(proposal), {}).get(str(obsid), {})
             .get(str(filtername).upper()))
    if not entry:
        return set(requested)
    allowed = {module_family(m) for m in entry} & set(requested)
    # The reduce matches the modules AS WRITTEN, so `merged` against a policy
    # listing only detectors is an empty intersection there and raises.  Here
    # `merged` had already been expanded to both families, which made the
    # intersection non-empty and hid the failure -- on the one field the policy
    # covers.
    if as_written is not None:
        groups = {module_family(m) for m in entry}
        if not {m if m == 'merged' else module_family(m)
                for m in as_written} & groups:
            allowed = set()
    if not allowed:
        # The reduce RAISES here -- `No requested modules are allowed for
        # proposal_id=... field=... filtername=...` -- before it does any work.
        # Narrowing to an empty set and reporting OK is the opposite verdict,
        # and it fired on the one field this narrowing exists for: `--modules
        # nrca` and `--modules merged` against sickle 3958/007, which is module
        # B only, both read OK and both make the reduce stop.
        raise NoAllowedModules(
            f'the reduce allows only {sorted({module_family(m) for m in entry})} '
            f'for {proposal}/o{obsid} {filtername}, and none of the requested '
            f'{sorted(requested)} is among them -- it would raise "No requested '
            f'modules are allowed" before doing any work')
    return allowed


def association_members(path):
    """The member exposure names in an association file.

    Raises ``UnreadableAssociation`` rather than returning empty: an
    association the reduce cannot parse is a failure there, so reading it as
    "no members" here would turn a stop into a pass.  ``OSError`` is included
    because an unreadable-by-permissions file is the same problem as a
    malformed one from this side.
    """
    try:
        with open(path) as fh:
            data = json.load(fh)
        return [m.get('expname', '') for m
                in (data.get('products') or [{}])[0].get('members') or []]
    except (OSError, ValueError, KeyError, IndexError, TypeError,
            AttributeError) as exc:
        raise UnreadableAssociation(f'{os.path.basename(path)}: {exc}')


class UnreadableAssociation(Exception):
    """An association file the reduce would refuse to parse."""


class NoAllowedModules(Exception):
    """Every requested module is excluded by the reduce's own policy."""


def usable_associations(paths, instrument):
    """The associations the reduce would actually keep, and why the rest went.

    The reduce does not use every association matching its glob: for NIRCam it
    keeps only those with a member whose exposure name contains ``nrc``
    (`PipelineRerunNIRCAM-LONG.py`), because one observation can produce NIRISS
    and MIRI associations under the same proposal and observation.  23% of the
    665 image3 associations on disk have no NIRCam member, and sgrc/F480M holds
    a NIRCam one beside a NIRISS one right now.

    Counting the glob alone reported those as usable input, so a directory
    holding only another instrument's associations read as OK and then failed
    the reduce.
    """
    token = _MEMBER_TOKEN.get(str(instrument).lower())
    keep, dropped = [], []
    for path in sorted(paths):
        members = association_members(path)
        if not members:
            dropped.append((path, 'no members'))
        elif token and not any(token in m for m in members):
            dropped.append((path, f'no {token} member'))
        else:
            keep.append(path)
    return keep, dropped


def _families_from_members(members):
    """Module families present among an association's member exposures."""
    out = set()
    for name in members:
        m = DETECTOR_RE.search(os.path.basename(name))
        if m:
            out.add(module_family(m.group(1)))
    return sorted(out)


def check(root, target, proposal, obsid, filters, modules, instrument='nircam'):
    """A :class:`Row` per requested filter."""
    from jwst_gc_pipeline.naming import jw_prefix
    if not filters:
        raise ValueError('no filters requested -- an empty filter list checks '
                         'nothing and would report success')
    obsid = normalize_obsid(obsid)
    # `merged` names a product built from the two module reductions, so asking
    # for it asks for both modules.  Dropping it instead left `--modules merged`
    # with nothing to check.
    wanted = {module_family(m) for m in modules if m != 'merged'}
    if any(m == 'merged' for m in modules):
        wanted |= {'nrca', 'nrcb'}
    # A single-detector instrument can answer only whether frames exist.  Keyed
    # off the INSTRUMENT rather than off what the directory happens to contain:
    # deciding it from the data let a NIRCam spec pointed at a NIRISS directory
    # report OK because every frame there was `nis`.
    single_detector_instrument = str(instrument).lower() in _SINGLE_DETECTOR_INSTRUMENTS
    policy = reduce_module_policy()
    rows = []
    for filt in filters:
        d = os.path.join(root, target, filt, 'pipeline')
        if not os.path.isdir(d):
            rows.append(Row(filt, 0, 0, [], sorted(wanted), f'no directory {d}'))
            continue
        pat = ASN_GLOB.format(jw=jw_prefix(proposal), obsid=obsid)
        candidates = glob.glob(os.path.join(d, pat))
        cals = glob.glob(os.path.join(
            d, f'{jw_prefix(proposal)}{obsid}*_cal.fits'))
        try:
            asns, dropped = usable_associations(candidates, instrument)
        except UnreadableAssociation as exc:
            rows.append(Row(filt, len(candidates), len(cals), [], sorted(wanted),
                            f'association the reduce cannot parse -- {exc}'))
            continue
        # Module coverage comes from the MEMBERS of the association the reduce
        # would use, not from a listing of the directory.  The reduce narrows
        # the association to one module's members and raises when none remain,
        # so a module present on disk but absent from the association is a
        # failure the directory listing cannot see.
        members = []
        for path in asns:
            members.extend(association_members(path))
        families = _families_from_members(members)
        # Narrow to what the reduce will actually run for this filter: one
        # observation is declared module-B-only in the reduce's own policy, and
        # asking it for module A is a false alarm rather than a finding.
        try:
            want_here = allowed_modules(proposal, obsid, filt, wanted,
                                        policy=policy, as_written=modules)
        except NoAllowedModules as exc:
            rows.append(Row(filt, len(asns), len(cals), families,
                            sorted(wanted), str(exc)))
            continue
        missing = ([] if single_detector_instrument
                   else sorted(w for w in want_here if w not in families))
        why = ''
        if not candidates:
            why = f'no image3 association matching {pat}'
        elif not asns:
            why = ('no image3 association the reduce would use: '
                   + '; '.join(f'{os.path.basename(p)} ({r})' for p, r in dropped))
        elif not cals:
            why = f'no _cal for {jw_prefix(proposal)}{obsid}'
        elif missing:
            why = (f'module(s) {missing} have no members in the association '
                   f'the reduce would use')
        rows.append(Row(filt, len(asns), len(cals), families, missing, why))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--target', required=True)
    ap.add_argument('--proposal', required=True)
    ap.add_argument('--obsid', required=True)
    ap.add_argument('--filters', required=True,
                    help='space-separated, e.g. "F115W F212N F405N"')
    ap.add_argument('--modules', default='nrca,nrcb,merged',
                    help='comma- or space-separated NIRCam modules the reduce '
                         'will be asked for, e.g. "nrca,nrcb,merged". Must '
                         'match the runner\'s MODULES; a module the '
                         'observation does not have is a real failure.')
    ap.add_argument('--instrument', default='nircam',
                    choices=('nircam', 'miri', 'niriss'),
                    help='which registry table to check the spec against, and '
                         'whose associations to accept; an unknown value used '
                         'to disable the association filter silently')
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

    # `--filters` is space-separated and `--modules` was comma-only, so mixing
    # them is the natural mistake -- and `--modules "nrcb nrca"` used to parse
    # as one token, truncate to `nrcb`, and exit 0 on a field with no module A.
    mods = [m for m in re.split(r'[,\s]+', args.modules) if m]
    unknown = sorted(set(mods) - _MODULE_TOKENS)
    if unknown:
        ap.error(f'--modules names {unknown}, which are not module tokens; '
                 f'expected some of {sorted(_MODULE_TOKENS)}')
    rows = check(args.root, args.target, args.proposal, obsid,
                 filters, mods, instrument=args.instrument)
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
