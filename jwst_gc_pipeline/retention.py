"""Which products the pipeline may delete, and which it must never touch.

The archive holds several TB of per-frame image intermediates from merge phases
that have since been superseded, plus hand-made backup directories and renamed
quarantines that nothing expires.  This module is the single place that decides
what is dead, so the decision is stated once and testable, rather than re-derived
by hand every time someone needs space.

WHAT MAKES A PRODUCT DEAD
-------------------------
The phase chain carries very little state forward.  ``cataloging.py``
(``_reconstruct_smoothed_bg_path``) calls the smoothed-background mosaic "the
only cross-phase state", and ``_reconstruct_resid_i2d_path`` exists so a
restarted per-frame worker can rebuild the next phase's detection image from
disk.  Everything else a phase writes is consumed inside that phase:

* ``*_{label}_daophot_{kind}_{residual,model}.fits`` -- the per-frame fit
  products.  Read by ``build_mergedcat_residuals`` for the SAME label (the
  hard-crash guard in ``crowdsource_catalogs_long.py``: a missing one would
  punch a hole in the mosaic) and by the iter2 qfilt-bg path.  Nothing reads a
  label's pair once the next label has run.
* ``*_{label}_daophot_{kind}_mergedcat_{residual,model}.fits`` -- rendered only
  so they can be resampled into ``*_mergedcat_residual_i2d.fits``.  Once that
  i2d exists they support nothing.

So a label's per-frame images are dead once the NEXT label's mosaic exists, and
a label's mergedcat frames are dead once that label's own i2d exists.  This is
the same argument the code already makes for the intermediate model i2d
(``--manual-keep-intermediate-model-i2d``: "display-only ... read by no pipeline
logic"), extended one product class further.

Catalogs are deliberately NOT in scope.  Every per-stage merged catalog for
brick totals ~41 GB against 4+ TB of that field's stage images: they are the
scientific record, they are cheap, and this module never selects one.  The two
exceptions are explicit, opt-in derivative rules (``_allcols`` supersets and
duplicate table formats), both off unless the caller asks for them.

WHAT PROTECTS A PRODUCT
-----------------------
Selection alone never deletes.  :class:`Guard` holds the four facts that veto a
candidate, and :func:`plan` applies it to every match:

1. anything a published release points at.  ``releases/v1.3-*/brick/exposures``
   is 1,200 SYMLINKS into the live tree, so deleting a live exposure silently
   breaks a published download.  Targets are resolved, not assumed.
2. anything belonging to a field with a queued or running SLURM chain.  A phase
   that restarts into missing inputs either trips the mergedcat guard or, worse,
   resumes from a partial marker set.
3. anything younger than ``min_age_days``.
4. anything under a caller-supplied protect glob.

Nothing in this module deletes unless :func:`apply` is called with
``dry_run=False``, and that refuses to run without a manifest path.
"""
import datetime
import fnmatch
import getpass
import glob
import json
import os
import re
import subprocess
from dataclasses import dataclass, field as _dcfield

__all__ = ['Rule', 'Candidate', 'Guard', 'POLICY', 'DEFAULT_RULES',
           'classify', 'plan', 'apply', 'release_symlink_targets',
           'busy_targets', 'superseded_perframe_products',
           'spent_mergedcat_frames', 'RetentionError']


class RetentionError(RuntimeError):
    """A retention action was asked for in a way that cannot be made safe."""


# --------------------------------------------------------------------------
# Never-touch patterns.  A file matching any of these is out of scope for every
# rule, no matter what else it looks like.  These are the products other code
# globs for: the mosaics (including every phase's residual/model i2d and the
# smoothed-bg the next phase reads), the exposure-level science products the
# release ships, the astrometry sidecars, and the PSF/satstar caches.
# --------------------------------------------------------------------------
PROTECTED_SUFFIXES = (
    '_i2d.fits',        # every mosaic, incl. *_mergedcat_residual_i2d.fits and
                        # *_mergedcat_residual_smoothed_bg_i2d.fits
    '_crf.fits', '_cal.fits', '_rate.fits', '_rateints.fits', '_uncal.fits',
    '_destreak.fits', '_align.fits', '_asn.json',
    '_consensus.fits', '_satstar_reconciled_m12.fits',
    '.asdf', '.ecsv.gz',
)

PROTECTED_SUBSTRINGS = (
    '/releases/',
    '/_perframe_markers/',
    '/astrometry_checkpoints/',
    '/offsets/',
    '/psfs/',
)


def _is_protected_name(path):
    p = str(path)
    if any(s in p for s in PROTECTED_SUBSTRINGS):
        return True
    return any(p.endswith(s) for s in PROTECTED_SUFFIXES)


# --------------------------------------------------------------------------
# Filename grammar.  The per-frame stem is
#   jw<prop>-o<field>_t001_<inst>_<pupil>-<filt>-<detector><visit><vgroup>
#   <exp><desat><bgsub><epsf><blur><group>_<label>_daophot_<kind>[_mergedcat]
#   _<residual|model>.fits
# so the label always sits immediately before ``_daophot_``.  ``[a-z]+`` cannot
# span the underscore in ``basic_mergedcat``, which is what keeps the raw and
# mergedcat patterns from matching each other; they are still ordered
# mergedcat-first so a future kind name cannot blur them.
# --------------------------------------------------------------------------
_LABEL = r'(?P<label>m\d+|iter\d+)'
PERFRAME_MERGEDCAT_RE = re.compile(
    _LABEL + r'_daophot_[a-z]+_mergedcat_(residual|model)\.fits$')
PERFRAME_RAW_RE = re.compile(
    _LABEL + r'_daophot_[a-z]+_(residual|model)\.fits$')

CORE_DUMP_RE = re.compile(r'(^core\.[^/]*$|\.core$)')

# The rename-based quarantines the repo already writes (rename_stale_mosaics.py,
# quarantine_pre_obstoken_catalogs.py and friends).  A rename moves a file out
# of ``*.fits`` so no glob can pick it up -- which is the whole point -- but
# nothing has ever expired one, so they are permanent storage with a confusing
# extension.
QUARANTINE_RENAME_RE = re.compile(
    r'(\.fits_[a-z_]*stale$|\.fits_badastrometry_stale$|\.STALE[^/]*$'
    r'|_stale$|\.bak[^/]*$|\.fits_stale$)')

# Directory names that say outright that the contents are superseded.  Matched
# against the directory's own basename, not the path, so a live product under a
# field called e.g. ``cloudef_controlfield`` is never caught by ``field``.
QUARANTINE_DIR_RE = re.compile(
    r'^(_?(pre|v\d+)[-_].*backup.*|.*_backup(_\d+)?|backup[s]?(_.*)?|'
    r'obsolete|old|old_.*|.*_old|_?stale.*|.*_stale.*|_?.*quarantine.*|'
    r'.*_pre_.*|_broken.*|failed_.*|.*_failed.*|junk.*)$',
    re.IGNORECASE)

ALLCOLS_RE = re.compile(r'_allcols\.fits$')


@dataclass(frozen=True)
class Rule:
    """One reason a product may be deleted.

    ``matches`` is called with ``(path, ctx)`` where ctx is a small dict the
    walker fills in (``siblings``, ``final_labels``); returning a string means
    "selected, and here is the reason to record in the manifest".  Returning
    None means the rule does not apply.
    """
    name: str
    description: str
    matches: callable
    min_age_days: float = 30.0
    default_on: bool = True


def _sibling_names(ctx):
    return ctx.get('siblings') or frozenset()


# ---- rule implementations -------------------------------------------------

def _rule_core_dump(path, ctx):
    if CORE_DUMP_RE.search(os.path.basename(path)):
        return 'process core dump; never an input to anything'
    return None


def _rule_quarantine_rename(path, ctx):
    if QUARANTINE_RENAME_RE.search(os.path.basename(path)):
        return ('renamed out of *.fits by a quarantine script, so already '
                'declared superseded and already invisible to every glob')
    return None


def _rule_spent_mergedcat_frame(path, ctx):
    """A per-frame mergedcat render whose mosaic has been written.

    The i2d it fed is the evidence: it sits in the same directory and carries
    the same label.  Without that i2d on disk the render is still needed, so
    this rule stays silent -- an interrupted phase is not a cleanup target.
    """
    base = os.path.basename(path)
    m = PERFRAME_MERGEDCAT_RE.search(base)
    if m is None:
        return None
    if m.group('label') not in ctx.get('mosaic_labels', ()):  # no i2d yet
        return None
    return (f"per-frame mergedcat render for {m.group('label')}; its "
            f"*_mergedcat_residual_i2d.fits exists, and nothing else reads it")


def _rule_superseded_perframe(path, ctx):
    """Per-frame residual/model from a label a later label has superseded.

    "Later" is read off the mosaics actually on disk, not off a hard-coded phase
    list: the final label for this directory is whichever label has the highest
    ordinal among the ``*_mergedcat_residual_i2d.fits`` present.  A field that
    stopped at m5 therefore keeps m5's pair and loses m12..m4's.
    """
    base = os.path.basename(path)
    m = PERFRAME_RAW_RE.search(base)
    if m is None:
        return None
    label = m.group('label')
    final = ctx.get('final_label')
    if final is None or label == final:
        return None
    if _label_ordinal(label) >= _label_ordinal(final):
        return None
    return (f"per-frame fit products for {label}, superseded by {final}; read "
            f"only by {label}'s own mergedcat build, which has completed")


def _rule_allcols(path, ctx):
    if ALLCOLS_RE.search(os.path.basename(path)):
        return ('_allcols superset table; the minimal table beside it is '
                'derived from it and is what downstream reads')
    return None


def _rule_duplicate_table_format(path, ctx):
    """An ECSV that has a byte-for-byte equivalent FITS twin, or vice versa.

    Off by default: which format is canonical is a project decision (the write
    site prefers ECSV for mixin/mask fidelity; the release ships FITS), and this
    rule must not make it silently.
    """
    base = os.path.basename(path)
    sibs = _sibling_names(ctx)
    if base.endswith('.ecsv') and (base[:-5] + '.fits') in sibs:
        return 'ECSV duplicate of the FITS table beside it'
    return None


DEFAULT_RULES = (
    Rule('core_dump', 'process core dumps', _rule_core_dump, min_age_days=7.0),
    Rule('quarantine_rename', 'files a quarantine script renamed out of *.fits',
         _rule_quarantine_rename, min_age_days=90.0),
    Rule('spent_mergedcat_frame',
         'per-frame mergedcat renders whose i2d mosaic exists',
         _rule_spent_mergedcat_frame, min_age_days=7.0),
    Rule('superseded_perframe',
         'per-frame residual/model below the final phase on disk',
         _rule_superseded_perframe, min_age_days=7.0),
    Rule('allcols', '_allcols superset tables', _rule_allcols,
         min_age_days=30.0, default_on=False),
    Rule('duplicate_table_format', 'ECSV/FITS twins of one table',
         _rule_duplicate_table_format, min_age_days=30.0, default_on=False),
)

POLICY = {r.name: r for r in DEFAULT_RULES}


_LABEL_ORDER_RE = re.compile(r'^(?:m|iter)(\d+)$')


def _label_ordinal(label):
    """Sort key for a phase label.

    ``m12`` is iter1+iter2 fused and runs FIRST, before m3 -- the digits are a
    concatenation of the two iterations it replaces, not the number twelve.
    Anything else sorts by its digits, and ``iterN`` sorts below every ``mN`` so
    a stale pre-rename product never looks newer than a phase.
    """
    if label == 'm12':
        return 2
    m = _LABEL_ORDER_RE.match(label or '')
    if not m:
        return -1
    n = int(m.group(1))
    return n if label.startswith('m') else -n


def classify(path, ctx=None, rules=DEFAULT_RULES, enabled=None):
    """First rule that selects ``path``, or ``(None, None)``.

    ``enabled`` is an optional set of rule names; rules outside it are skipped
    even when ``default_on``.
    """
    if _is_protected_name(path):
        return None, None
    ctx = ctx or {}
    for rule in rules:
        if enabled is None:
            if not rule.default_on:
                continue
        elif rule.name not in enabled:
            continue
        why = rule.matches(path, ctx)
        if why:
            return rule, why
    return None, None


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

@dataclass
class Guard:
    """Everything that vetoes a candidate.

    Empty defaults are deliberately unsafe-looking: a caller that builds a Guard
    by hand and forgets the release targets gets no protection, so
    :func:`guard_for` is the way to make one for real trees.
    """
    release_targets: frozenset = _dcfield(default_factory=frozenset)
    busy_fields: frozenset = _dcfield(default_factory=frozenset)
    min_age_days: float = 30.0
    protect_globs: tuple = ()
    now: float = None

    def veto(self, path, st, rule):
        """Reason this candidate must not be deleted, or None."""
        real = os.path.realpath(path)
        if real in self.release_targets:
            return 'a published release symlinks to this file'
        for pat in self.protect_globs:
            if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(real, pat):
                return f'matches protect glob {pat!r}'
        for fieldname in self.busy_fields:
            if f'/{fieldname}/' in path or f'/{fieldname}/' in real:
                return (f'field {fieldname} has a queued or running SLURM '
                        f'chain')
        now = self.now if self.now is not None else _now()
        age_days = (now - st.st_mtime) / 86400.0
        floor = max(self.min_age_days, rule.min_age_days if rule else 0.0)
        if age_days < floor:
            return f'only {age_days:.1f} d old (floor {floor:.0f} d)'
        return None


def _now():
    return datetime.datetime.now().timestamp()


def release_symlink_targets(releases_root):
    """Resolved targets of every symlink under a release tree.

    v1.3 publishes brick exposures as symlinks into
    ``brick/mastDownload/JWST/F*/pipeline``; those live files are load-bearing
    for a public download even though nothing in the release directory holds
    their bytes.
    """
    targets = set()
    for dirpath, dirnames, filenames in os.walk(releases_root):
        for name in list(dirnames) + list(filenames):
            p = os.path.join(dirpath, name)
            if os.path.islink(p):
                targets.add(os.path.realpath(p))
    return frozenset(targets)


def busy_targets(known_targets, squeue_output=None):
    """Field names that appear in a queued or running SLURM job name.

    Job names follow ``<target><proposal>-o<obs>-<phase>-<role>`` (the standing
    naming rule in the sbatch headers), so a prefix test against the registry is
    enough and does not need to parse the rest.  ``squeue_output`` is injectable
    so this is testable without a scheduler.
    """
    if squeue_output is None:
        try:
            squeue_output = subprocess.run(
                ['squeue', '-h', '-u', getpass.getuser(), '-o', '%j'],
                capture_output=True, text=True, timeout=60).stdout
        except (OSError, subprocess.SubprocessError) as ex:
            raise RetentionError(
                f'could not ask squeue which fields are busy ({ex}); refusing '
                f'to plan a deletion blind.  Pass --assume-idle to override '
                f'when you know the queue is empty.') from ex
    busy = set()
    for line in (squeue_output or '').splitlines():
        job = line.strip()
        for target in known_targets:
            if job.startswith(target):
                busy.add(target)
    return frozenset(busy)


def guard_for(roots, known_targets, *, releases_root=None, min_age_days=30.0,
              protect_globs=(), squeue_output=None, assume_idle=False):
    """Build the Guard for a real cleanup run."""
    targets = frozenset()
    if releases_root and os.path.isdir(releases_root):
        targets = release_symlink_targets(releases_root)
    busy = frozenset() if assume_idle else busy_targets(known_targets,
                                                        squeue_output)
    return Guard(release_targets=targets, busy_fields=busy,
                 min_age_days=min_age_days, protect_globs=tuple(protect_globs))


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------

@dataclass
class Candidate:
    path: str
    size: int
    mtime: float
    rule: str
    reason: str
    vetoed_by: str = None

    @property
    def deletable(self):
        return self.vetoed_by is None

    def as_dict(self):
        return {'path': self.path, 'size': self.size, 'mtime': self.mtime,
                'rule': self.rule, 'reason': self.reason,
                'vetoed_by': self.vetoed_by}


_MOSAIC_LABEL_RE = re.compile(
    r'(?:m\d+|iter\d+)(?=_daophot_[a-z]+_mergedcat_residual_i2d\.fits$)')


def _directory_context(dirpath, filenames):
    """Facts a rule needs about the directory it is looking at.

    ``mosaic_labels`` is every label with a mergedcat residual i2d on disk, and
    ``final_label`` is the highest of them -- both read from the directory
    itself so a field that stopped early is judged by what it actually
    produced, never by the nominal phase list.
    """
    labels = set()
    for name in filenames:
        m = _MOSAIC_LABEL_RE.search(name)
        if m:
            labels.add(m.group(0))
    final = max(labels, key=_label_ordinal) if labels else None
    return {'siblings': frozenset(filenames),
            'mosaic_labels': frozenset(labels),
            'final_label': final}


def plan(roots, *, guard=None, rules=DEFAULT_RULES, enabled=None,
         include_vetoed=False, follow_symlinks=False):
    """Walk ``roots`` and return the Candidates the policy selects.

    Every candidate carries the rule that selected it and the reason, and a
    vetoed candidate keeps ``vetoed_by`` so a reviewer can see what the guard
    stopped.  This function only reads.
    """
    guard = guard if guard is not None else Guard()
    out = []
    seen_dirs = set()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root,
                                                    followlinks=follow_symlinks):
            real_dir = os.path.realpath(dirpath)
            if real_dir in seen_dirs:
                # brick/cloudc/wd1 are reachable by several routes (the /orange
                # symlink, the /blue path, and <field>/F<X> == <field>/
                # mastDownload/JWST/F<X>).  Without this a naive walk counts --
                # and would offer to delete -- the same bytes up to four times.
                dirnames[:] = []
                continue
            seen_dirs.add(real_dir)
            if '/releases/' in dirpath + '/':
                dirnames[:] = []
                continue
            ctx = _directory_context(dirpath, filenames)
            for name in filenames:
                path = os.path.join(dirpath, name)
                rule, why = classify(path, ctx, rules=rules, enabled=enabled)
                if rule is None:
                    continue
                try:
                    st = os.lstat(path)
                except OSError:
                    continue
                veto = guard.veto(path, st, rule)
                if veto and not include_vetoed:
                    continue
                out.append(Candidate(path=path, size=st.st_size,
                                     mtime=st.st_mtime, rule=rule.name,
                                     reason=why, vetoed_by=veto))
    return out


def plan_quarantine_directories(roots, *, guard=None, min_age_days=90.0,
                                include_vetoed=False):
    """Whole directories whose NAME says their contents are superseded.

    Kept separate from :func:`plan` because the unit is a directory: the size is
    the tree's, the age is the newest file in it (a backup someone is still
    adding to is not stale), and a single release symlink pointing inside vetoes
    the whole thing.
    """
    guard = guard if guard is not None else Guard(min_age_days=min_age_days)
    out = []
    seen = set()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            if '/releases/' in dirpath + '/':
                dirnames[:] = []
                continue
            for name in list(dirnames):
                if not QUARANTINE_DIR_RE.match(name):
                    continue
                d = os.path.join(dirpath, name)
                real = os.path.realpath(d)
                if real in seen:
                    continue
                seen.add(real)
                dirnames.remove(name)          # do not descend into it
                size, newest, veto = _tree_stats(d, guard)
                if veto is None:
                    age_days = (_now() - newest) / 86400.0
                    if age_days < max(min_age_days, guard.min_age_days):
                        veto = (f'newest file is only {age_days:.1f} d old '
                                f'(floor {max(min_age_days, guard.min_age_days):.0f} d)')
                if veto and not include_vetoed:
                    continue
                out.append(Candidate(
                    path=d, size=size, mtime=newest, rule='quarantine_dir',
                    reason=f'directory name {name!r} declares its contents '
                           f'superseded',
                    vetoed_by=veto))
    return out


def _tree_stats(directory, guard):
    """(bytes, newest mtime, veto reason) for a whole tree."""
    total = 0
    newest = 0.0
    veto = None
    for dirpath, _dirnames, filenames in os.walk(directory):
        for name in filenames:
            p = os.path.join(dirpath, name)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            total += st.st_size
            newest = max(newest, st.st_mtime)
            if veto is None and os.path.realpath(p) in guard.release_targets:
                veto = f'a published release symlinks to {p}'
    if veto is None:
        for fieldname in guard.busy_fields:
            if f'/{fieldname}/' in directory + '/':
                veto = (f'field {fieldname} has a queued or running SLURM '
                        f'chain')
                break
    return total, newest, veto


# --------------------------------------------------------------------------
# Applying
# --------------------------------------------------------------------------

def apply(candidates, *, dry_run=True, manifest_path=None, on_error='raise'):
    """Delete the candidates.  Refuses to do anything without a manifest.

    The manifest is written and flushed BEFORE the first unlink, so an
    interrupted run still says exactly what it was about to remove.  Vetoed
    candidates are recorded and skipped.
    """
    import shutil

    deletable = [c for c in candidates if c.deletable]
    summary = {'planned': len(candidates), 'deletable': len(deletable),
               'bytes': sum(c.size for c in deletable), 'deleted': 0,
               'failed': 0, 'dry_run': bool(dry_run)}
    if manifest_path:
        with open(manifest_path, 'w') as fh:
            json.dump({'written': datetime.datetime.now().isoformat(),
                       'dry_run': bool(dry_run),
                       'summary': summary,
                       'candidates': [c.as_dict() for c in candidates]},
                      fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
    if dry_run:
        return summary
    if not manifest_path:
        raise RetentionError(
            'refusing to delete without a manifest path: the manifest is the '
            'only record of what was removed and why')
    for c in deletable:
        try:
            if os.path.isdir(c.path) and not os.path.islink(c.path):
                shutil.rmtree(c.path)
            else:
                os.unlink(c.path)
            summary['deleted'] += 1
        except OSError as ex:
            summary['failed'] += 1
            if on_error == 'raise':
                raise RetentionError(f'failed to remove {c.path}: {ex}') from ex
    return summary


# --------------------------------------------------------------------------
# In-run helpers (used by the phase loop; see cataloging.py)
# --------------------------------------------------------------------------

def _stem_prefix(proposal_id, field, filtername):
    from jwst_gc_pipeline.mast_names import jw_prefix
    from jwst_gc_pipeline.photometry.naming import _inst_token
    return (f'{jw_prefix(proposal_id)}-o{field}_t001_'
            f'{_inst_token(filtername)}_')


def spent_mergedcat_frames(pipeline_dir, *, proposal_id, field, filtername,
                           label):
    """Per-frame mergedcat renders for ``label`` whose i2d mosaic exists.

    Scoped by the ``-o{field}_t001_`` prefix so a directory shared by two
    observations never offers up the other observation's frames.
    """
    pre = _stem_prefix(proposal_id, field, filtername)
    mid = f'-{filtername.lower()}-'
    i2d = glob.glob(os.path.join(
        pipeline_dir,
        f'{pre}*{mid}*_{label}_daophot_*_mergedcat_residual_i2d.fits'))
    if not i2d:
        return []
    out = []
    for what in ('residual', 'model'):
        out.extend(glob.glob(os.path.join(
            pipeline_dir,
            f'{pre}*{mid}*_{label}_daophot_*_mergedcat_{what}.fits')))
    return sorted(p for p in out if not _is_protected_name(p))


def superseded_perframe_products(pipeline_dir, *, proposal_id, field,
                                 filtername, label):
    """Per-frame residual/model for ``label``, for a label already superseded.

    The caller decides that ``label`` is superseded -- in the phase loop it is
    the phase before the one whose mosaic just landed.  This function only
    resolves that decision to paths, and never returns a mergedcat render or an
    i2d (the ``_mergedcat_`` exclusion and the protected-suffix check).
    """
    pre = _stem_prefix(proposal_id, field, filtername)
    mid = f'-{filtername.lower()}-'
    out = []
    for what in ('residual', 'model'):
        out.extend(glob.glob(os.path.join(
            pipeline_dir, f'{pre}*{mid}*_{label}_daophot_*_{what}.fits')))
    return sorted(p for p in out
                  if '_mergedcat_' not in os.path.basename(p)
                  and not _is_protected_name(p))
