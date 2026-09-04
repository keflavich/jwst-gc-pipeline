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

Three checks, cheapest first:

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
3. **against the disk the reduce WRITES to** (#421) -- free space on the
   filesystem holding ``--root``, against ``--min-free-tb``.  Nothing else in
   this repo looks at free space (``git grep -E 'statvfs|disk_usage|min_free'``
   over the package: no hits).  data-qa's monitor has a ``--min-free-tb`` floor,
   but it is evaluated at its ``--download-dir`` on /orange while the treasury's
   ``root: blue`` sends every reduction product to /blue -- so the monitor can
   keep downloading into a healthy /orange and keep triggering reductions into a
   full /blue, and the products die on ENOSPC after the queue wait with no gate
   having looked.  The measured unit: brick F212N holds 1.42 TB for one filter
   of one field, of which 1.23 GB per detector-exposure is the part that scales
   one-per-frame.

Reports and returns nonzero if anything is missing.  Changes nothing.
"""
import argparse
import glob
import json
import os
import re
import sys


def jw_prefix(proposal):
    """The MAST filename prefix: the proposal zero-padded to five digits.

    Inlined rather than imported.  This script's whole point is to answer in
    ten seconds a question the reduce takes 20 h of queue to answer, so it
    stays runnable with no package on the path: its imports are stdlib-only,
    it PARSES the reduce driver instead of importing it (see
    `reduce_module_policy`), and it defers `fields` to the registry check that
    needs it.  Importing `jwst_gc_pipeline.mast_names` for a one-line pad would
    put the package `__init__` -- numpy, astropy, and the provenance
    `HDUList.writeto` hook -- on every functional path, including
    `--skip-registry`, which needs no package at all.

    `jwst_gc_pipeline.mast_names.jw_prefix` is the canonical helper (issue #414);
    `test_preflight_reduce_inputs.py` asserts this agrees with it over 4- and
    5-digit proposals, so the two cannot drift.
    """
    text = proposal if isinstance(proposal, str) else str(proposal)
    if not re.fullmatch(r'[0-9]{1,5}', text) or int(text) == 0:
        raise ValueError(f'proposal {proposal!r} is not a JWST proposal '
                         f'number: expected one to five decimal digits')
    return f'jw{int(text):05d}'


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


#: Free TB the reduction root must have before a reduce is worth submitting.
#:
#: One filter of one field, measured on ``brick/F212N/pipeline`` (2026-08-24):
#: 1.42 TB all-in over 192 detector-exposures, of which 236 GB (1.23 GB per
#: detector-exposure) is the per-frame product chain -- uncal, ramp, rate, cal,
#: destreak, tweakreg, crf, bgsub, unsatstar -- and the rest is cataloging
#: byproducts and mosaics.  2 TB is one such pass plus margin: enough that a
#: reduce which clears the gate can finish, small enough that it does not
#: refuse work on a filesystem that is merely busy.  Raise it with
#: ``--min-free-tb`` for a deep field; ``--min-free-tb 0`` turns the check off.
DEFAULT_MIN_FREE_TB = 2.0


def _registry():
    """The field registry, or None when the package is not importable.

    This script is deliberately runnable with no package on the path -- its
    module-level imports are stdlib only -- so every registry read is lazy and
    has to tolerate the registry not being there at all.
    """
    try:
        from jwst_gc_pipeline import fields
    except ImportError:
        return None
    return fields


def write_root(root, target, registry=None):
    """Where the reduce will actually write for ``target``.

    ``{root}/{target}`` is right only when it happens to coincide with the
    registry's answer.  It does for brick -- but only by accident, because
    ``/orange/adamginsburg/jwst/brick`` is a symlink whose target ``statvfs``
    follows.  It does NOT for gc-treasury: the registry puts it on
    ``/blue/.../jwst/gc-treasury`` while an /orange root reports /orange's free
    space, and gc-treasury has no compensating symlink because the directory
    does not exist yet -- which is precisely the pre-tile-1 state this gate is
    for.  Measured at the time: 67.5 TB reported for a filesystem the reduce
    never touches, against 18.4 TB where tile 1 actually lands.

    Resolving through the registry also drops the gate's dependence on
    ``--root`` matching the registry's ``root:`` key, which is the thing that
    differed between the two fields.  An unregistered target keeps the old
    ``{root}/{target}``: there is nothing better to consult, and inventing a
    /blue path for a field nobody has declared would be a worse guess than the
    one the operator just typed.
    """
    reg = registry if registry is not None else _registry()
    known = None if reg is None else getattr(reg, 'BY_NAME', {}).get(target)
    if known is not None and getattr(known, 'basepath', None):
        return str(known.basepath).rstrip('/')
    return os.path.join(root, target)


def free_tb(path, statvfs=os.statvfs):
    """Free TB on the filesystem holding ``path``, or None if it cannot tell.

    Walks up to the nearest existing ancestor, because the field tree is
    normally created BY the reduce: ``/blue/.../jwst/gc-treasury`` does not
    exist before tile 1, and a check that gave up there would be silent
    exactly on the first run, which is the run this exists for.

    ``f_bavail`` (available to a non-root user), not ``f_bfree`` -- the
    reserved blocks are not usable and counting them overstates the headroom.
    TB here is 1e12 bytes, matching ``df -h``'s ``T`` closely enough for a
    threshold and matching the units the issue and the monitor are stated in.
    """
    probe = os.path.abspath(path)
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        st = statvfs(probe)
    except OSError:
        # An unmounted or unreadable path: report "cannot tell", which the
        # verdict turns into a refusal.  Guessing would be the #421 failure.
        return None
    return st.f_bavail * st.f_frsize / 1e12


def headroom_verdict(root, min_free_tb=DEFAULT_MIN_FREE_TB, free=None):
    """``(ok, message)`` for free space on the reduction root (#421).

    ``min_free_tb <= 0`` disables the check and says so, rather than passing
    silently -- a gate that reports "not checked" is a different thing from a
    gate that reports OK, and the whole point of #421 is that a filesystem
    nobody looked at read as fine.

    ``free`` is injectable for testing; None looks up the module-level
    ``free_tb`` at call time, so a test can replace that instead.
    """
    if min_free_tb is None or min_free_tb <= 0:
        return True, f'free-space check disabled (--min-free-tb {min_free_tb})'
    available = (free_tb if free is None else free)(root)
    if available is None:
        return False, (f'cannot determine free space for {root} -- statvfs '
                       f'failed on it and on every existing ancestor, so '
                       f'nothing here knows where the reduce would write')
    ok = available >= min_free_tb
    return ok, (f'{available:.1f} TB free on the filesystem holding {root} '
                f'(floor {min_free_tb:.1f} TB)'
                + ('' if ok else ' -- the reduce would write into this and the '
                                 'products die on ENOSPC after the queue wait'))


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


def registry_obs_key(field):
    """``field`` spelled the way the module registry keys an observation.

    The reduce's own ``registry_obs_key`` (in ``PipelineRerunNIRCAM-LONG.py``);
    kept in step so the preflight answers what the reduce will do for `7` as
    well as for `007`.  A number is padded to three digits, anything else
    (a wildcard obsid `'*'`) is handed back untouched.  Issue #438.
    """
    text = str(field).strip()
    return f'{int(text):03d}' if text.isdigit() else text


def allowed_modules(proposal, obsid, filtername, requested, policy=None,
                    as_written=None):
    """``requested``, narrowed to what the reduce will actually run.

    Returns the families the reduce would produce for this (proposal,
    observation, filter).  A field with no policy entry is unrestricted, which
    is every field but one.
    """
    policy = reduce_module_policy() if policy is None else policy
    entry = (policy.get(str(proposal), {}).get(registry_obs_key(obsid), {})
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
    ap.add_argument('--min-free-tb', type=float, default=DEFAULT_MIN_FREE_TB,
                    help='refuse when the filesystem holding --root has less '
                         'than this many TB free (#421; 0 disables). The '
                         'reduce writes there, and nothing else in this repo '
                         'checks it -- data-qa\'s floor watches its download '
                         'staging, which is a different filesystem whenever a '
                         'field\'s root: differs from it.')
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
    # The FIELD TREE the REGISTRY names, not `{--root}/{target}`.  Several
    # targets under an /orange root actually write to /blue; brick gets the
    # right answer from the joined path only because
    # `/orange/adamginsburg/jwst/brick` is a symlink that `statvfs` follows,
    # and gc-treasury -- the field whose 139 tiles are the reason this gate
    # exists -- has no such symlink, because before tile 1 the directory does
    # not exist.  Joined, it reported 67.5 TB for a filesystem the reduce never
    # touches while 18.4 TB was free where the tile lands.  `free_tb` still
    # walks up to the nearest existing ancestor, so the pre-tile-1 case reads
    # the filesystem the reduce will create the tree on.
    _dest = write_root(args.root, args.target)
    ok, msg = headroom_verdict(_dest, args.min_free_tb)
    print(f'{"OK     " if ok else "NO SPACE"}  disk: {msg}')
    if not ok:
        bad += 1
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
              f'OBSID against what is on disk, and the free space on the '
              f'filesystem the field tree lives on, before spending the queue.')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
