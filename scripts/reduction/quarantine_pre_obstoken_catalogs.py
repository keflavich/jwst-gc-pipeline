#!/usr/bin/env python
"""Rename pre-obs-token per-frame catalogs out of the consensus glob.

A multi-observation proposal writes one per-frame catalog per (observation,
visit, vgroup, exposure, detector).  Before the obs token existed those names
carried no observation:

    f150w_nrca1_visit001_vgroup02201_exp00001_m2_daophot_basic.fits       2026-06-22
    f150w_nrca1_o028_visit001_vgroup02201_exp00001_m2_daophot_basic.fits  2026-08-08

Both describe the SAME (visit, vgroup, exposure, detector, filter) identity, so
the m2 visit-consensus ingests each one twice and refuses to build:

    error_kind: duplicate_exposure
    192 exposure identity/ies ingested more than once

Every gc2211 observation hit this.  The consensus then contained ZERO exposures,
the m12 finalize still exited 0, the retie loop printed "m2 checkpoint PASSED --
converged", and the frozen-stage m3 check failed for want of a baseline that was
never written (#350).  So the field's astrometry gate was skipped silently.

The m2 CORRECTION path already excludes untokened files -- o023's log reads
``excluded 560 of 592 foreign-observation per-frame catalog(s) (['<untokened>',
'o046', 'o049', 'o050'])`` -- so the filter exists; the consensus build does not
use it.  Fixing that is #350 and is the durable half.  This script is the data
half: get the superseded files out of the glob so the current generation can be
measured.

## Safety

Renames, never deletes: ``<name>_pre_obstoken_stale``, the same shape
``rename_stale_mosaics.py`` uses.  Reversible with a rename back.

Refuses to touch a file unless BOTH hold:

  * it carries no observation token, AND
  * a tokened file exists for the same identity

so a field that simply predates the token (and has no replacement) is left
alone.  The second condition is what makes this safe to run anywhere rather than
only where someone has already checked the dates by hand.

## MERGED catalogs

The same rename lands on a second family of names.  ``catalogs/`` holds the
per-filter MERGED products::

    f150w2_nrca_indivexp_merged_m2_dao_basic.fits        (pre-token)
    f150w2_nrca_o003_indivexp_merged_m2_dao_basic.fits   (post-token)

and the token sits in the MODULE slot, so a post-token run asks for the second
name while disk holds only the first.  Every reader of a merged name --
``cataloging.merged_catalog_path``, the m7 seed glob in ``merge_catalogs``,
``merge_daophot``'s input glob -- spells ``merged_catalog_module_token``, so the
pre-token file is not read, not overwritten, and not reported: it is simply
unreachable.  m4 (1979/002+003) had 59 of them when this was written (23 with a
newer tokened replacement already on disk, 36 still waiting on one); brick had
the same shape in #620/#625.

Ownership of a pre-token MERGED catalog is not recoverable.  It pooled per-frame
tables from BOTH observations (that pooling is the bug), and its HDU-1
``FILENAME`` names one arbitrary member of that pool, so no filename or header
evidence says which observation it belongs to.  This tool therefore never
RE-tokens a merged catalog; it only renames it out of the way, and only when

  * a tokened merged catalog of the same (filter, module, stage) identity
    exists, AND
  * the untokened file is OLDER than that replacement.

The mtime condition is the one that matters after a rollback: a fresh untokened
merge (someone ran an old checkout) is not stale, and must not be renamed away
because an older tokened file happens to sit beside it.

A merged catalog that is unreachable but has NO replacement yet is REPORTED and
left alone -- it becomes quarantinable once the re-fit reaches that stage.  So
the ordered landing step for a field like m4 is: archive the ambiguous per-frame
tables, re-fit from ``_cal`` under the new token, then run this tool again to
clear the merged leftovers stage by stage.

"Unreachable" is a claim about the FIELD BEING SCANNED, not about the name, and
it is asked per (filter, module) through ``merged_token_expected``.  Most fields
have an EMPTY ``merged_catalog_module_token``, and for those the untokened name
is the one their own writer and readers spell -- the current product, with no
re-fit owed.  Reported blind, this pass called w51's 1259 and ngc6397's 107
merged catalogs unreachable (measured 2026-09-09), and cloudc is both at once:
its NIRCam half (2221 obs 002) takes no token while its MIRI half (2221 obs 001)
does, so a field-level answer is not enough either.  Where the registry has no
answer -- an unregistered field name such as ``gc2211`` (split into
``gc2211_o0NN``), or a filter no registered observation declares -- the report
says so instead, and the disk-evidence-only behaviour is unchanged.

Usage::

    python quarantine_pre_obstoken_catalogs.py --field gc2211            # dry run
    python quarantine_pre_obstoken_catalogs.py --field gc2211 --execute
    python quarantine_pre_obstoken_catalogs.py --field m4 --no-merged    # per-frame only
"""
import argparse
import collections
import os
import re
import sys

BASE = os.environ.get("GC_BASEPATH_OVERRIDE",
                      os.environ.get("JWST_BASE", "/orange/adamginsburg/jwst"))

SUFFIX = "_pre_obstoken_stale"

#: <filter>_<detector>[_<token>]_visit<NNN>_vgroup<...>_exp<NNNNN>_<rest>
#:
#: The token is a FAMILY, not just ``o###``.  gc2211 disambiguates by observation
#: (``_o028``); ngc6334 shares a directory, filter list and obsid 001 between
#: proposals 6778 and 7213, so its disambiguator is the PROPOSAL (``_j6778``).
#: Matching only ``o\d{3}`` made ngc6334's 1680 F200W tokened catalogs invisible
#: as replacements, and the tool printed a confident "nothing superseded" for the
#: one other field with exactly this collision.
#:
#: Filters are 3 OR 4 digits: NIRCam F200W, MIRI F1000W.
_NAME = re.compile(
    r"^(?P<filt>f\d{3,4}[a-z]\d?)_"
    r"(?P<det>nrc[ab](?:\d|long)|mirim|nis)_"
    r"(?:(?P<token>o\d{3}|j\d{4})_)?"
    r"(?P<rest>visit\d+_.*\.fits)$",
    re.IGNORECASE)


def identity(name):
    """``(filter, detector, rest)`` -- the same frame regardless of obs token."""
    m = _NAME.match(name)
    if not m:
        return None
    return (m.group("filt").lower(), m.group("det").lower(), m.group("rest"))


def token_of(name):
    """The disambiguating token (``o028``, ``j6778``) or None."""
    m = _NAME.match(name)
    return m.group("token").lower() if (m and m.group("token")) else None


def has_obs_token(name):
    return token_of(name) is not None


def source_token(path):
    """The token the catalog's OWN source frame carries, from HDU-1 FILENAME.

    The replacement guard must ask whether THIS file's observation has a
    tokened twin, not whether *some* observation does.  On gc2211 the identity
    is shared across observations -- that collision is why the token exists --
    so "a tokened file shares the identity" is satisfied by a DIFFERENT
    observation's catalog.  It happens to hold for every file on disk today
    (audited 502/502) only because every gc2211 observation has been re-reduced;
    an observation that had not been would have its only catalog renamed away,
    which is exactly what the second condition exists to prevent.

    Returns None when the header cannot be read, and a caller that gets None
    must DECLINE the file rather than assume.
    """
    try:
        from astropy.io import fits
        with fits.open(path, memmap=False) as hdul:
            for hdu in hdul[1:]:
                fn = hdu.header.get("FILENAME")
                if fn:
                    m = re.search(r"_(o\d{3}|j\d{4})_", os.path.basename(str(fn)),
                                  re.IGNORECASE)
                    return m.group(1).lower() if m else None
    except (OSError, ValueError, IndexError, ImportError):
        return None
    return None


def plan_for_dir(directory, require_own_token=True):
    """Untokened files in ``directory`` that a tokened file supersedes."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    tokened = collections.defaultdict(list)
    untokened = {}
    for n in names:
        if n.endswith(SUFFIX) or not n.endswith(".fits"):
            continue
        ident = identity(n)
        if ident is None:
            continue
        if has_obs_token(n):
            tokened[ident].append(n)
        else:
            untokened[ident] = n
    plan = []
    for i in sorted(untokened):
        if i not in tokened:
            continue
        name = untokened[i]
        if not require_own_token:
            plan.append((name, sorted(tokened[i])))
            continue
        own = source_token(os.path.join(directory, name))
        if own is None:
            # Unreadable header, or a source frame with no token: cannot
            # establish that THIS file's observation was replaced.  Decline.
            continue
        twins = [t for t in tokened[i] if token_of(t) == own]
        if twins:
            plan.append((name, sorted(twins)))
    return plan


# --- MERGED catalogs -------------------------------------------------------
#: ``<filter>_<module>[_<token>]_indivexp_merged<...>_<stage>_dao_basic<...>``
#:
#: The module slot is where the token lives for a merged name (the per-frame
#: token sits between detector and visit instead), which is why this needs its
#: own pattern rather than a widened ``_NAME``.  ``merged`` is itself a module
#: spelling, so ``f150w2_merged_indivexp_merged_m2_...`` has to parse as
#: module=``merged``, token=absent.
_MERGED_NAME = re.compile(
    r"^(?P<filt>f\d{3,4}[a-z]\d?)_"
    r"(?P<mod>merged|nrc[ab](?:long|\d)?|mirimage|mirim|nis(?:image)?|niriss)"
    r"(?:_(?P<token>o\d{3}(?:-\d{3})*|j\d{4}))?"
    r"_(?P<rest>indivexp_merged.*\.fits)$",
    re.IGNORECASE)


def merged_identity(name):
    """``(filter, module, rest)`` -- the same merged product regardless of token.

    ``rest`` carries the stage (``m2``/``resbgsub_m5``) and the variant
    (``_vetted``, ``_allcols``, ``_i2dseed``), so an ``_o003`` m2 catalog is NOT
    a replacement for an untokened m5 one.  That distinction is the whole reason
    a field mid-re-fit reports leftovers instead of losing them.
    """
    m = _MERGED_NAME.match(name)
    if not m:
        return None
    return (m.group("filt").lower(), m.group("mod").lower(), m.group("rest").lower())


def merged_token_of(name):
    m = _MERGED_NAME.match(name)
    return m.group("token").lower() if (m and m.group("token")) else None


def source_observation(path):
    """``(proposal, observation)`` named by HDU-1 ``FILENAME``, or ``None``.

    INFORMATIONAL ONLY.  A pre-token merged catalog pooled frames from more than
    one observation, and this header names one arbitrary member of that pool, so
    it does not establish ownership and is never used as a guard -- it is
    printed so a reader of the dry run can see what the blend was made of.
    """
    try:
        from astropy.io import fits
        with fits.open(path, memmap=False) as hdul:
            for hdu in hdul[1:]:
                fn = hdu.header.get("FILENAME")
                if fn:
                    m = re.search(r"jw(\d{5})(\d{3})\d{3}_",
                                  os.path.basename(str(fn)))
                    return (m.group(1).lstrip("0"), m.group(2)) if m else None
    except (OSError, ValueError, IndexError, ImportError):
        return None
    return None


# --- does this field's merged names carry a module-slot token at all? -------
#
# The orphan report says a file is UNREACHABLE.  That is only true where the
# field's own writer stamps a module-slot token: `merged_catalog_module_token`
# is EMPTY for most fields, and for those the untokened name is the name every
# reader spells -- it is the CURRENT product, not a leftover.  Reported blind,
# w51's 1259 and ngc6397's 107 untokened merged catalogs read as a re-fit owed
# on a field where nothing is owed at all (PR #778 review).
#
# The question is per (filter, module), not per field.  One field can hold both
# spellings: cloudc's NIRCam half is 2221 observation 002, which takes NO token,
# while its MIRI half is 2221 observation 001, a `PER_OBS_MERGED_FIELDS` entry
# that DOES -- so a field-level "does any observation carry a token" answer
# would report cloudc's untokened NIRCam merges, which are its live products.

#: Instrument of a merged catalog's module slot.  ``merged`` is the NIRCam
#: nrca+nrcb product and names no instrument, so it falls back to the filter.
_MODULE_INSTRUMENT = (
    ('mirim', 'miri'),
    ('nis', 'niriss'),
    ('nrc', 'nircam'),
)


def _registry():
    """``(fields, naming)`` from the installed package, or ``(None, None)``.

    Imported lazily and failure-tolerantly: the per-frame pass needs neither,
    and a checkout without the package on ``sys.path`` must still be able to
    quarantine per-frame duplicates.
    """
    try:
        from jwst_gc_pipeline import fields
        from jwst_gc_pipeline.photometry import naming
    except ImportError:
        return None, None
    return fields, naming


def merged_token_expected(field, filtername, module):
    """Is an untokened merged name UNREACHABLE for this field/filter/module?

    ``True``   every observation that could have written it stamps a module-slot
               token, so nothing reads the untokened spelling.
    ``False``  some observation writes exactly this untokened name -- the file
               is that observation's current product, not a leftover.
    ``None``   no answer: the package is unavailable, the field is not in
               ``fields.yaml`` (``gc2211`` is split into ``gc2211_o0NN``), or no
               registered observation declares this filter.  Callers must fall
               back to the evidence on disk rather than assume either way.
    """
    fields_mod, naming = _registry()
    if fields_mod is None:
        return None
    fobj = fields_mod.BY_NAME.get(field)
    if fobj is None:
        return None
    mod = str(module).lower()
    instrument = None
    for prefix, inst in _MODULE_INSTRUMENT:
        if mod.startswith(prefix):
            instrument = inst
            break
    if instrument is None:      # module == 'merged'
        instrument = naming._instrument_from_filter(filtername).lower()
    tokened = []
    for obs in fobj.observations:
        declared = (obs.niriss_filters if instrument == 'niriss'
                    else obs.filters) or ()
        if declared and str(filtername).lower() not in {str(f).lower()
                                                        for f in declared}:
            continue
        obsids = list(obs.obsids.get(instrument, ()))
        obsids += list((obs.joint_obsids or {}).get(instrument, ()))
        for obsid in obsids:
            if str(obsid) == fields_mod.WILDCARD_OBSID:
                # A wildcard claims every observation of the proposal, so the
                # token depends on the proposal alone.  Asked with a made-up
                # obsid instead, a `PER_OBS_MERGED_FIELDS` entry for that number
                # under a different field would answer for this one.
                tokened.append(
                    str(obs.proposal) in naming.PER_OBS_MERGED_PROPOSALS
                    or str(obs.proposal) in naming.SHARED_TREE_PROPOSALS)
                continue
            try:
                tok = naming.merged_catalog_module_token(obs.proposal, obsid)
            except naming.ObservationFieldError:
                continue
            tokened.append(bool(tok))
    if not tokened:
        return None
    return all(tokened)


def plan_for_merged_dir(directory, field=None):
    """``(plan, orphans)`` for the ``catalogs/`` directory.

    ``plan``    -- untokened merged catalogs a tokened, NEWER twin supersedes.
    ``orphans`` -- untokened merged catalogs that the module-slot token has made
                   unreachable but that nothing has replaced yet.  Reported, not
                   touched: renaming one would take the field from "reads a
                   stale catalog" to "has no catalog at all" for that stage.

    ``field`` names the field being scanned, which is what says whether the
    untokened spelling is unreachable at all: where ``merged_token_expected``
    answers False the untokened name is the one this field's own writer and
    readers use, so the file is neither an orphan nor renameable, whatever else
    sits beside it.  A tokened neighbour there is the foreign one.  ``None``
    (the default, and every unregistered field) keeps the disk-evidence-only
    behaviour, so the per-frame use on ``gc2211`` is unchanged.
    """
    try:
        names = os.listdir(directory)
    except OSError:
        return [], []
    tokened = collections.defaultdict(list)
    untokened = {}
    for n in names:
        if n.endswith(SUFFIX) or not n.endswith(".fits"):
            continue
        ident = merged_identity(n)
        if ident is None:
            continue
        if merged_token_of(n):
            tokened[ident].append(n)
        else:
            untokened[ident] = n

    def _mtime(n):
        try:
            return os.path.getmtime(os.path.join(directory, n))
        except OSError:
            return None

    plan, orphans = [], []
    for i in sorted(untokened):
        name = untokened[i]
        filt, module, _ = i
        if merged_token_expected(field, filt, module) is False:
            # This field spells the merged name for this filter WITHOUT a
            # token, so the untokened file is the current product.  Calling it
            # unreachable sends an operator re-fitting a field that owes
            # nothing, and renaming it would delete the only catalog a reader
            # can find.
            continue
        twins = sorted(tokened.get(i, ()))
        if not twins:
            orphans.append(name)
            continue
        own = _mtime(name)
        # A fresh untokened merge (an old checkout was re-run) is not stale, and
        # an older tokened file beside it does not make it so.
        newer = [t for t in twins
                 if own is not None and (_mtime(t) or 0) > own]
        if not newer:
            orphans.append(name)
            continue
        plan.append((name, newer))
    return plan, orphans


def sidecars(directory, name):
    """Files that must move WITH ``name`` -- today only ``.prov.json``.

    Leaving the sidecar behind orphans a provenance record pointing at a path
    that no longer exists, and the provenance reader treats a missing target as
    a corrupt record rather than an absent one.
    """
    return [n for n in (name + ".prov.json",)
            if os.path.exists(os.path.join(directory, n))]


def _show(names, limit=5):
    """Print the first ``limit`` names and how many more there are."""
    for n in names[:limit]:
        print(f"    {n}")
    if len(names) > limit:
        print(f"    ... and {len(names) - limit} more")


def field_dirs(field):
    root = os.path.join(BASE, field)
    if not os.path.isdir(root):
        return []
    # 3 OR 4 digits: NIRCam F200W, MIRI F1000W.  The name pattern accepts both
    # too; matching only 3 made every MIRI directory invisible, which reads as
    # "nothing superseded" rather than "not examined".
    return sorted(os.path.join(root, d) for d in os.listdir(root)
                  if re.fullmatch(r"[Ff]\d{3,4}[A-Za-z]\d?", d)
                  and os.path.isdir(os.path.join(root, d)))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True, help="field dir, e.g. gc2211")
    ap.add_argument("--execute", action="store_true",
                    help="rename (default is a dry run that changes nothing)")
    ap.add_argument("--no-merged", dest="merged", action="store_false",
                    help="skip the catalogs/ merged pass (per-frame only)")
    args = ap.parse_args(argv)

    dirs = field_dirs(args.field)
    catdir = os.path.join(BASE, args.field, "catalogs")
    if not dirs and not (args.merged and os.path.isdir(catdir)):
        print(f"no per-filter directories under {BASE}/{args.field}",
              file=sys.stderr)
        return 2

    total = renamed = skipped = 0
    for d in dirs:
        plan = plan_for_dir(d)
        if not plan:
            print(f"{os.path.basename(d)}: nothing superseded")
            continue
        print(f"{os.path.basename(d)}: {len(plan)} untokened file(s) superseded "
              f"by a tokened one")
        for old, replacements in plan[:2]:
            print(f"    {old}")
            print(f"      superseded by: {', '.join(replacements[:3])}"
                  + (" ..." if len(replacements) > 3 else ""))
        if len(plan) > 2:
            print(f"    ... and {len(plan) - 2} more")
        total += len(plan)
        if args.execute:
            for old, _ in plan:
                src = os.path.join(d, old)
                dst = src + SUFFIX
                if os.path.exists(dst):
                    # os.rename would silently replace an earlier quarantine,
                    # so a second run would destroy the file the first run
                    # preserved -- the opposite of reversible.
                    print(f"    SKIP {old}: {os.path.basename(dst)} already "
                          f"exists; not overwriting an earlier quarantine")
                    skipped += 1
                    continue
                os.rename(src, dst)
                renamed += 1

    if args.merged:
        mplan, orphans = plan_for_merged_dir(catdir, field=args.field)
        if mplan:
            print(f"catalogs/: {len(mplan)} untokened MERGED catalog(s) "
                  f"superseded by a newer tokened one")
        for old_name, replacements in mplan:
            print(f"    {old_name}")
            print(f"      superseded by: {', '.join(replacements[:3])}"
                  + (" ..." if len(replacements) > 3 else ""))
            obs = source_observation(os.path.join(catdir, old_name))
            if obs:
                print(f"      (HDU-1 FILENAME names {obs[0]}/{obs[1]}; a pooled "
                      f"merge is NOT attributable, so this is not ownership)")
        total += len(mplan)
        # An orphan is only UNREACHABLE where this field stamps a module-slot
        # token for that filter.  Where the answer is unknown -- an unregistered
        # field, or a filter no registered observation declares -- say that
        # rather than assert a re-fit is owed.
        unreachable, undetermined = [], []
        for o in orphans:
            filt, module, _ = merged_identity(o)
            (unreachable if merged_token_expected(args.field, filt, module)
             else undetermined).append(o)
        if unreachable:
            print(f"catalogs/: {len(unreachable)} untokened MERGED catalog(s) "
                  f"are UNREACHABLE under the module-slot token and have no "
                  f"replacement yet -- left alone; re-run once the re-fit "
                  f"reaches that stage:")
            _show(unreachable)
        if undetermined:
            print(f"catalogs/: {len(undetermined)} untokened MERGED catalog(s) "
                  f"have no tokened twin, and the registry does not say whether "
                  f"a module-slot token applies to them ({args.field!r} "
                  f"unregistered, or the filter undeclared) -- left alone:")
            _show(undetermined)
        if args.execute:
            for old_name, _ in mplan:
                for n in [old_name] + sidecars(catdir, old_name):
                    src_p = os.path.join(catdir, n)
                    dst_p = src_p + SUFFIX
                    if os.path.exists(dst_p):
                        print(f"    SKIP {n}: {os.path.basename(dst_p)} already "
                              f"exists; not overwriting an earlier quarantine")
                        skipped += 1
                        continue
                    os.rename(src_p, dst_p)
                    renamed += 1

    if args.execute:
        print(f"\nrenamed {renamed} file(s) -> *{SUFFIX}"
              + (f"; SKIPPED {skipped} that already had a quarantine"
                 if skipped else ""))
        print("Reverse with: for f in *"+SUFFIX+"; do mv \"$f\" \"${f%"
              + SUFFIX + "}\"; done")
    else:
        print(f"\ndry run: {total} file(s) would be renamed; "
              f"re-run with --execute")
    return 0


if __name__ == "__main__":
    sys.exit(main())
