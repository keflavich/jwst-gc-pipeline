"""Rerun-skip decision engine.

Given a stage's RECORDED provenance (from its ``.prov.json`` sidecar) and a
freshly-computed CURRENT provenance (same schema), decide the MINIMAL action for
that stage, then propagate the seeding cascade across the stage DAG.

Verdicts
--------
``NO_PROVENANCE`` -- no recorded sidecar; cannot prove anything -> must run.
``SKIP``          -- every input facet matches; reuse the recorded product.
``RESTAMP``       -- only non-WCS header metadata changed; re-write the header
                     stamp, recompute nothing.
``REPROJECT``     -- only the WCS moved (data identical) and we are NOT re-tying
                     at a seeding stage: refresh ``x,y -> ra,dec`` on the existing
                     catalog (``astrometry_utils.reproject_xy_to_world``); no PSF
                     fit.
``REFIT``         -- science data / code / params / an upstream *data* facet /
                     the previous stage's catalog changed: re-run this cataloging
                     stage's fit.
``RE_REDUCE``     -- an imaging input (jwst version / CRDS context / reduction
                     code / params) changed: re-run the parent ``jwst`` pipeline.
``BLOCKED``       -- a bulk WCS shift is being introduced at a FROZEN stage
                     (m3+); illegal per the astrometry checkpoint ladder.

The seeding-cascade invariant
-----------------------------
Each cataloging phase N seeds from N-1 (background map + residual detection image
+ previous catalog).  Therefore any verdict that CHANGES a stage's output
(``REFIT``/``RE_REDUCE``) forces every downstream cataloging stage to ``REFIT``
too.  A ``REPROJECT`` only refreshes RA/Dec on unchanged detector positions, so
it cascades as ``REPROJECT`` (a rigid, all-band-consistent RA/Dec refresh leaves
relative/cross-band seed geometry intact).  See ``VERSIONING_PROVENANCE.md``.
"""
import argparse
import glob
import json
import os
import warnings

from . import prov_sidecar

# The pipeline DAG, in dependency order.
STAGES = ('imaging', 'm12', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8')
CATALOG_STAGES = STAGES[1:]
# Stages where a bulk astrometric shift may be (re-)introduced pre-seed.  A WCS
# change here in 'reseed' mode reseeds everything downstream.  (Mirrors
# astrometry_checkpoint.CORRECTION_STAGES = m1/m2/m12.)
SEED_CORRECTION_STAGES = ('m12',)

NO_PROVENANCE = 'NO_PROVENANCE'
SKIP = 'SKIP'
RESTAMP = 'RESTAMP'
REPROJECT = 'REPROJECT'
REFIT = 'REFIT'
RE_REDUCE = 'RE_REDUCE'
BLOCKED = 'BLOCKED'

# Cost ordering (higher = more expensive); used to keep the strongest verdict
# when the cascade and the local decision disagree.
_RANK = {NO_PROVENANCE: 6, RE_REDUCE: 5, REFIT: 4, REPROJECT: 2, RESTAMP: 1,
         SKIP: 0, BLOCKED: 7}


class StageDecision:
    """A per-stage verdict with its reasons."""

    def __init__(self, stage, verdict, reasons=None, conditional=False):
        self.stage = stage
        self.verdict = verdict
        self.reasons = list(reasons or [])
        # conditional=True: the verdict depends on a result only knowable AFTER
        # running an upstream stage (the imaging re-reduce byte-identity check).
        self.conditional = conditional

    def as_dict(self):
        return {'stage': self.stage, 'verdict': self.verdict,
                'reasons': self.reasons, 'conditional': self.conditional}

    def __repr__(self):
        c = ' (conditional)' if self.conditional else ''
        return f'<{self.stage}: {self.verdict}{c} -- {"; ".join(self.reasons)}>'


def _facet_diff(rec_up, cur_up):
    """Which facets differ between two ``{stage: {data,wcs,meta}}`` maps.

    Returns a dict ``{'data':bool,'wcs':bool,'meta':bool,'prev':bool}`` where
    ``prev`` marks a change in a cataloging upstream's catalog content (its
    ``data`` facet), distinguished from the imaging *frame* data facet by the
    upstream stage name.
    """
    out = {'data': False, 'wcs': False, 'meta': False, 'prev': False}
    keys = set(rec_up) | set(cur_up)
    for up in keys:
        r = (rec_up.get(up) or {})
        c = (cur_up.get(up) or {})
        for facet in ('data', 'wcs', 'meta'):
            if r.get(facet) != c.get(facet):
                if facet == 'data' and up in CATALOG_STAGES:
                    out['prev'] = True
                else:
                    out[facet] = True
    return out


def decide_stage(stage, recorded, current, wcs_change_mode='posthoc'):
    """Decide the minimal action for a single stage (pure input comparison).

    Parameters
    ----------
    stage : str
    recorded : dict or None
        The recorded provenance record (``prov_sidecar`` schema), or None.
    current : dict
        A freshly-computed record with the same schema (its ``outputs`` are
        ignored -- they are unknown until the stage runs; only ``inputs`` drive
        the decision).
    wcs_change_mode : {'posthoc','reseed'}
        How to treat a WCS-only upstream change with identical data.  'posthoc'
        (default): the offsets table was corrected AFTER a finished run and we
        only refresh RA/Dec (REPROJECT).  'reseed': the correction is applied at
        a seeding stage to RE-TIE, which moves seeds (REFIT at m12, and the
        cascade re-fits downstream).

    Returns
    -------
    StageDecision
    """
    if recorded is None:
        return StageDecision(stage, NO_PROVENANCE,
                             ['no recorded provenance sidecar'])

    ri, ci = recorded.get('inputs', {}), current.get('inputs', {})
    reasons = []

    env_changed = ri.get('env', {}) != ci.get('env', {})
    code_changed = ri.get('code') != ci.get('code')
    params_changed = ri.get('params') != ci.get('params')
    d = _facet_diff(ri.get('upstream', {}), ci.get('upstream', {}))

    if stage == 'imaging':
        if env_changed:
            reasons.append(
                f"env changed: {ri.get('env')} -> {ci.get('env')}")
        if code_changed:
            reasons.append('reduction code changed')
        if params_changed:
            reasons.append('reduction params changed')
        if reasons:
            return StageDecision(stage, RE_REDUCE, reasons)
        return StageDecision(stage, SKIP, ['all imaging inputs unchanged'])

    # --- cataloging stage ---
    if code_changed:
        reasons.append('cataloging code changed')
    if params_changed:
        reasons.append('cataloging params changed')
    if d['data']:
        reasons.append('upstream frame science data changed')
    if d['prev']:
        reasons.append('previous-stage catalog changed (reseed)')
    if reasons:
        return StageDecision(stage, REFIT, reasons)

    if d['wcs']:
        if wcs_change_mode == 'reseed':
            if stage in SEED_CORRECTION_STAGES:
                return StageDecision(
                    stage, REFIT,
                    ['bulk WCS shift re-tied at seeding stage -> reseed'])
            # A reseed shift at a frozen stage is illegal (checkpoint ladder).
            return StageDecision(
                stage, BLOCKED,
                ['bulk WCS shift at FROZEN stage (m3+); forbidden by the '
                 'astrometry checkpoint ladder'])
        return StageDecision(
            stage, REPROJECT,
            ['WCS moved, science data identical -> refresh x,y->ra,dec only'])

    if d['meta']:
        return StageDecision(stage, RESTAMP,
                             ['only non-WCS header metadata changed'])

    return StageDecision(stage, SKIP, ['all inputs unchanged'])


def propagate_cascade(decisions):
    """Apply the seeding cascade to an ordered list of ``StageDecision``.

    Once a cataloging stage will change its output (REFIT/RE_REDUCE, or a
    RE_REDUCE at imaging that changes frame data), every LATER cataloging stage
    must REFIT (it seeds from the changed stage).  A REPROJECT propagates as
    REPROJECT.  A NO_PROVENANCE downstream is left as-is (it must run regardless).
    Returns a new list; input is not mutated.
    """
    by_stage = {d.stage: d for d in decisions}
    ordered = [by_stage[s] for s in STAGES if s in by_stage]
    out = []
    cascade = None  # None | REFIT | REPROJECT
    for d in ordered:
        new = StageDecision(d.stage, d.verdict, list(d.reasons), d.conditional)
        if cascade == REFIT and d.stage in CATALOG_STAGES:
            if _RANK[REFIT] > _RANK[new.verdict]:
                new.verdict = REFIT
                new.reasons.append('cascade: an upstream stage re-fits (reseed)')
        elif cascade == REPROJECT and d.stage in CATALOG_STAGES:
            if new.verdict in (SKIP, RESTAMP):
                new.verdict = REPROJECT
                new.reasons.append('cascade: upstream RA/Dec refreshed')
        # Update the running cascade from THIS stage's (possibly escalated) verdict.
        if new.verdict in (REFIT, RE_REDUCE):
            cascade = REFIT
        elif new.verdict == REPROJECT and cascade != REFIT:
            cascade = REPROJECT
        # RE_REDUCE at imaging is conditional: only cascades if the re-reduced
        # frame data actually differ (unknowable until it runs).
        if d.stage == 'imaging' and new.verdict == RE_REDUCE:
            new.conditional = True
            new.reasons.append(
                'downstream REFIT is CONDITIONAL: re-run imaging, then re-plan '
                '-- if frame science data is byte-identical, cataloging SKIPs')
        out.append(new)
    return out


def plan_from_records(recorded_by_stage, current_by_stage,
                      wcs_change_mode='posthoc'):
    """Full DAG plan from recorded/current record maps ``{stage: record}``.

    ``current_by_stage`` must provide a current record (at least its ``inputs``)
    for every stage to be evaluated.  Missing recorded records yield
    NO_PROVENANCE.  Returns the cascaded list of ``StageDecision``.
    """
    decisions = []
    for stage in STAGES:
        if stage not in current_by_stage:
            continue
        decisions.append(decide_stage(
            stage, recorded_by_stage.get(stage), current_by_stage[stage],
            wcs_change_mode=wcs_change_mode))
    return propagate_cascade(decisions)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _load_records_json(path):
    with open(path) as fh:
        data = json.load(fh)
    # Accept either {stage: record} or a list of records with a 'stage' key.
    if isinstance(data, list):
        return {r['stage']: r for r in data}
    return data


def _scan_sidecars(directory):
    """Return ``{stage: record}`` from every ``*.prov.json`` under ``directory``.

    One product per stage is expected per field.  A multi-filter/multi-module
    field tree can hold several products per stage; this collapses to one
    (last-wins) and WARNS on each collision so the ambiguity is never silent.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, '**', '*' + prov_sidecar.SIDECAR_SUFFIX),
                                 recursive=True)):
        with open(path) as fh:
            rec = json.load(fh)
        stage = rec.get('stage')
        if not stage:
            continue
        if stage in out:
            warnings.warn(
                f'--scan: multiple provenance sidecars for stage {stage!r}; '
                f'keeping the last ({os.path.basename(path)}) and ignoring the '
                f'earlier one. Scan a per-product subtree to disambiguate.',
                stacklevel=2)
        out[stage] = rec
    return out


# One-line action hint per verdict (printed in the human plan).
ACTION_HINT = {
    SKIP: 'reuse existing product',
    RESTAMP: 're-stamp header only',
    REPROJECT: 'refresh RA/Dec (reproject_xy_to_world); no PSF fit',
    REFIT: 're-run this cataloging stage (seeds downstream)',
    RE_REDUCE: 're-run imaging (parent jwst pipeline), then re-plan',
    BLOCKED: 'ILLEGAL bulk shift at a frozen stage -- investigate',
    NO_PROVENANCE: 'no sidecar -- must run to establish provenance',
}


def _print_plan(decisions):
    """Human-readable plan; a BLOCKED stage short-circuits with a banner."""
    width = max((len(d.stage) for d in decisions), default=7)
    for d in decisions:
        if d.verdict == BLOCKED:
            print('=' * 72)
            print(f'  PLAN BLOCKED at stage {d.stage}: {"; ".join(d.reasons)}')
            print('  Downstream verdicts below are NOT actionable until this is '
                  'resolved.')
            print('=' * 72)
            break
    for d in decisions:
        cond = ' [conditional]' if d.conditional else ''
        hint = ACTION_HINT.get(d.verdict, '')
        print(f'{d.stage:<{width}}  {d.verdict:<13}{cond}  {hint}')
        if d.reasons:
            print(f'{"":<{width}}    - ' + '\n'.join(
                f'{"":<{width}}      {r}' for r in d.reasons).lstrip())


def _cmd_plan(args):
    if args.field:
        from . import fieldplan
        decisions, products = fieldplan.plan_field(
            args.field, wcs_change_mode=args.wcs_change_mode,
            repo_dir=args.repo_dir, use_live_env=not args.no_live_env)
        if not products:
            raise SystemExit(f'plan --field: no *.prov.json products under '
                             f'{args.field!r} (has the field been stamped?)')
        if args.json:
            print(json.dumps([d.as_dict() for d in decisions], indent=2))
        else:
            print('NOTE: params are compared against each product\'s RECORDED '
                  'params, not your live CLI options. If you changed cataloging '
                  'options since these products were built, this plan can '
                  'under-report REFIT -- re-run the stage to be sure.')
            _print_plan(decisions)
        return

    if args.records and args.current:
        recorded = _load_records_json(args.records)
        current = _load_records_json(args.current)
    elif args.scan:
        # Compare a directory's sidecars against themselves is a no-op; the scan
        # mode reports the recorded state and flags stages lacking provenance.
        recorded = _scan_sidecars(args.scan)
        current = recorded
    else:
        raise SystemExit(
            'plan: provide --field DIR (plan a real field from disk), '
            '--records R --current C (compare two record maps), '
            'or --scan DIR (report recorded provenance state).')

    decisions = plan_from_records(recorded, current,
                                  wcs_change_mode=args.wcs_change_mode)
    if args.json:
        print(json.dumps([d.as_dict() for d in decisions], indent=2))
        return
    _print_plan(decisions)


def build_parser():
    p = argparse.ArgumentParser(
        prog='python -m jwst_gc_pipeline.versioning.rerun',
        description='Decide which pipeline stages need re-running.')
    sub = p.add_subparsers(dest='cmd', required=True)
    pl = sub.add_parser('plan', help='produce a per-stage rerun/skip plan')
    pl.add_argument('--field', help='plan a real field: directory tree to scan '
                    'for stamped products, recompute current facets from disk')
    pl.add_argument('--repo-dir', help='repo to fingerprint code against '
                    '(default: the installed package repo)')
    pl.add_argument('--no-live-env', action='store_true',
                    help='do not read the live jwst/CRDS env (imaging env '
                         'comparison falls back to the recorded env)')
    pl.add_argument('--records', help='JSON of recorded provenance {stage: record}')
    pl.add_argument('--current', help='JSON of current provenance {stage: record}')
    pl.add_argument('--scan', help='directory to scan for *.prov.json sidecars')
    pl.add_argument('--wcs-change-mode', choices=('posthoc', 'reseed'),
                    default='posthoc',
                    help="how to treat a WCS-only change: 'posthoc' (refresh "
                         "RA/Dec, default) or 'reseed' (re-tie at m12 -> refit).")
    pl.add_argument('--json', action='store_true', help='emit JSON')
    pl.set_defaults(func=_cmd_plan)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
