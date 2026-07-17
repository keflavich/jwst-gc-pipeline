"""Build a rerun-skip plan for a real field from its on-disk products.

``rerun plan --field DIR`` walks a field's product tree, and for each stage:

* reads the RECORDED provenance sidecar (what produced the product), and
* assembles a CURRENT record describing what a re-run *now* would use --
  ``code_hash`` recomputed from the repo, the live ``jwst``/CRDS environment
  (imaging), and the parents' CURRENT output facets recomputed from disk --

then feeds both to the pure engine (``rerun.plan_from_records``).  The result is
a ready-to-read plan: which stages to re-reduce / refit / reproject / skip.

Scope note: the sidecar scan keeps ONE product per stage (last-wins, with a
warning), so point ``--field`` at a per-module/-filter subtree for an
unambiguous plan.  Params comparison reuses the recorded params (a new-options
diff needs the live run options, which the plan CLI does not have).
"""
import glob
import os
import warnings

from . import fingerprint as _fp
from . import prov_sidecar
from . import rerun
from . import upstream as _up


def live_env():
    """Best-effort current imaging environment: jwst version + CRDS context.

    Returns ``{'jwst_version':..., 'crds_context':...}`` (either key omitted if
    unavailable).  Used to detect a PENDING jwst/CRDS bump: recorded env (what
    produced the product) vs the environment a re-reduction would run under now.
    """
    env = {}
    try:
        import jwst
        env['jwst_version'] = str(jwst.__version__)
    except (ImportError, AttributeError):
        pass
    try:
        import crds
        ctx = crds.get_context_name('jwst')
        if ctx:
            env['crds_context'] = str(ctx)
    except (ImportError, AttributeError, OSError, ValueError):
        pass
    return env


def scan_products(directory):
    """Map ``{stage: (product_path, recorded_record)}`` from ``*.prov.json``.

    One product per stage (last-wins, warns on collision).  ``product_path`` is
    the sidecar path with the ``.prov.json`` suffix removed.
    """
    out = {}
    suffix = prov_sidecar.SIDECAR_SUFFIX
    for sc in sorted(glob.glob(os.path.join(directory, '**', '*' + suffix),
                               recursive=True)):
        rec = prov_sidecar.read_sidecar(sc)
        if not rec:
            continue
        stage = rec.get('stage')
        if not stage:
            continue
        product = sc[:-len(suffix)]
        if stage in out:
            warnings.warn(
                f'--field: multiple products for stage {stage!r}; keeping last '
                f'({os.path.basename(product)}). Point --field at a per-product '
                f'subtree to disambiguate.', stacklevel=2)
        out[stage] = (product, rec)
    return out


def _current_facets(product_path, is_catalog=None):
    """Recompute a product's output facets from disk NOW (image or table).

    For a table catalog the facets are computed EXACTLY as ``stamp_catalog``
    does (data via ``table_hash`` with the reprojectable sky columns excluded,
    no image WCS, meta over the primary header), so a recomputed catalog facet
    matches its recorded sidecar when nothing changed.
    """
    if not is_catalog:
        try:
            return _fp.facet_hashes(product_path)
        except (OSError, ValueError, KeyError):
            pass  # fall through to the catalog path
    from astropy.io import fits
    try:
        data = _fp.table_hash(product_path,
                              exclude_cols=('ra', 'dec', 'skycoord_ra',
                                            'skycoord_dec'))
        with fits.open(product_path, memmap=True) as hdul:
            meta = _fp.meta_hash(hdul[0].header)
        return {'data': data, 'wcs': None, 'meta': meta}
    except (OSError, ValueError, KeyError):
        return None


def _current_upstream(stage, products_by_stage):
    """Pool the parents' CURRENT on-disk facets into an upstream map.

    Detects a regenerated parent even if the parent's sidecar is stale, because
    the facets are recomputed from the parent product file, not read back.
    """
    out = {}
    for parent in _up.STAGE_PARENTS.get(stage, []):
        entry = products_by_stage.get(parent)
        if entry is None:
            continue
        facets = _current_facets(entry[0], is_catalog=(parent != 'imaging'))
        if facets:
            out[parent] = facets
    return out


def build_current_record(stage, recorded, products_by_stage, repo_dir=None,
                         env=None):
    """Assemble the CURRENT record for ``stage`` (inputs a re-run would use)."""
    ri = (recorded or {}).get('inputs', {})
    try:
        code = _fp.code_hash(stage, repo_dir=repo_dir)
    except (KeyError, FileNotFoundError):
        code = ri.get('code')
    cur_env = ri.get('env', {})
    if stage == 'imaging' and env:
        cur_env = env
    return {
        'stage': stage, 'tag': 'current',
        'inputs': {
            'env': cur_env,
            'code': code,
            'params': ri.get('params'),   # reuse recorded (no live options here)
            'upstream': _current_upstream(stage, products_by_stage),
        },
        'outputs': {},
    }


def plan_field(directory, wcs_change_mode='posthoc', repo_dir=None,
               use_live_env=True):
    """Return ``(decisions, products_by_stage)`` for a field product tree."""
    products = scan_products(directory)
    env = live_env() if use_live_env else None
    recorded_by_stage = {s: rec for s, (_, rec) in products.items()}
    current_by_stage = {
        s: build_current_record(s, rec, products, repo_dir=repo_dir, env=env)
        for s, (_, rec) in products.items()
    }
    decisions = rerun.plan_from_records(
        recorded_by_stage, current_by_stage, wcs_change_mode=wcs_change_mode)
    return decisions, products
