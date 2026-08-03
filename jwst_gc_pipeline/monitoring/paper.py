"""Read the astrometry paper's own machine-readable verdicts.

The Brick astrometry paper (Overleaf ``6a521006b63a11a7e0d80fa0``, checked out at
``<brick>/astrometry_paper``) carries an analysis layer that already validates the
release products: ``scripts/post_recat_validation.py`` runs on a SLURM dependency
after re-cataloging and writes ``outputs/<date>_postrecat/summary.json``, whose
``problems`` list is the verdict.  ``config.py`` pins every threshold the analysis
uses.

So the monitor **reports that verdict; it does not re-implement it**.  Two reasons:

* the gates (cross-filter vs anchor > 30 mas, p60/p90 mode flip > 10 mas,
  degenerate-pair drift >= 0.10 mag) live in the paper's script, and a second copy
  here would drift away from the numbers the paper actually publishes; and
* every one of those gates is computed with the sanctioned window-swept
  offset-histogram over full catalogs.  Recomputing them would mean the monitor
  crossmatching catalogs itself, which is exactly the ad-hoc path
  ``CLAUDE.md`` bans.

What the monitor adds on top is the question the paper's script cannot answer
about itself: **is the verdict still about the products on disk?**  A verdict that
predates the catalogs it certifies is worse than no verdict, because it reads as a
pass.
"""
import glob
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone

#: Where the paper is checked out.  Overridable so a different working copy (or a
#: test fixture) can be pointed at.
PAPER_DIR = os.environ.get(
    'GC_MONITOR_PAPER_DIR', '/orange/adamginsburg/jwst/brick/astrometry_paper')

#: The paper covers exactly one field.  Its verdicts are attached to that field's
#: runs and to no others -- a sip/gwcs entry that happens to name another frame is
#: reported as context, never as that field's verdict.
PAPER_FIELD = 'brick'


def paper_config(paper_dir=None):
    """The paper's pinned ``config.py`` as a module, or ``None``.

    Loaded by path rather than by ``import config`` so it cannot collide with any
    other ``config`` module on the path.  It is stdlib-only, so this is cheap and
    cannot drag in astropy.
    """
    path = os.path.join(paper_dir or PAPER_DIR, 'config.py')
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location('_gc_paper_config', path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, SyntaxError, OSError, ValueError):
        return None
    return module


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def latest_postrecat(paper_dir=None):
    """``(dir, summary_dict)`` for the newest post-recat validation run.

    Newest by the ``generated`` timestamp INSIDE the summary, falling back to
    directory mtime -- the directory is named for the date it was written, but a
    rerun on the same day overwrites in place, so the recorded timestamp is the
    authority.
    """
    root = os.path.join(paper_dir or PAPER_DIR, 'outputs')
    best = (None, None, None)
    for path in glob.glob(os.path.join(root, '*_postrecat', 'summary.json')):
        rec = _read_json(path)
        if rec is None:
            continue
        stamp = rec.get('generated') or ''
        key = (stamp, _mtime(path) or 0)
        if best[0] is None or key > best[0]:
            best = (key, os.path.dirname(path), rec)
    return best[1], best[2]


def read_verdicts(paper_dir=None):
    """Every machine-readable verdict the paper has already produced.

    Nothing here is computed: each entry is a file the paper's own analysis wrote.
    """
    paper_dir = paper_dir or PAPER_DIR
    out = {'paper_dir': paper_dir, 'present': os.path.isdir(paper_dir)}
    if not out['present']:
        return out

    outputs = os.path.join(paper_dir, 'outputs')
    postrecat_dir, summary = latest_postrecat(paper_dir)
    out['postrecat'] = summary
    out['postrecat_dir'] = postrecat_dir
    out['postrecat_mtime'] = (_mtime(os.path.join(postrecat_dir, 'summary.json'))
                              if postrecat_dir else None)

    for key, name in (('sip_vs_gwcs', 'sip_vs_gwcs.json'),
                      ('anderson_vs_gwcs', 'anderson_vs_gwcs.json')):
        path = os.path.join(outputs, name)
        out[key] = _read_json(path)
        out[f'{key}_mtime'] = _mtime(path)

    # provenance.json is written per analysis-date directory; take the newest.
    provs = sorted(glob.glob(os.path.join(outputs, '*', 'provenance.json')),
                   key=lambda p: _mtime(p) or 0)
    out['provenance'] = _read_json(provs[-1]) if provs else None
    out['provenance_path'] = provs[-1] if provs else None

    cfg = paper_config(paper_dir)
    out['config'] = None if cfg is None else {
        'min_catalog_date': getattr(cfg, 'MIN_CATALOG_DATE', None),
        'min_contrast': getattr(cfg, 'MIN_CONTRAST', None),
        'same_frame_tol_mas': getattr(cfg, 'SAME_FRAME_TOL_MAS', None),
        'tie_clip_mas': getattr(cfg, 'TIE_CLIP_MAS', None),
        'qfit_max': getattr(cfg, 'JWST_QFIT_MAX', None),
        'bands': getattr(cfg, 'BANDS', None),
        'catdir': getattr(cfg, 'CATDIR', None),
    }
    return out


def catalog_freshness(verdicts):
    """``[{band, program, path, mtime, stale_vs_min_date, newer_than_verdict}]``.

    The paper records the mtime of each vetted catalog it validated.  Comparing
    that record with the file on disk NOW is the whole point: if the file has been
    rewritten since, the verdict describes a product that no longer exists.
    """
    summary = verdicts.get('postrecat') or {}
    cfg = verdicts.get('config') or {}
    min_date = cfg.get('min_catalog_date')
    verdict_mtime = verdicts.get('postrecat_mtime')

    rows = []
    for program, bands in summary.items():
        if not isinstance(bands, dict) or program in ('certifiers',):
            continue
        for band, rec in bands.items():
            if not isinstance(rec, dict) or 'path' not in rec:
                continue
            path = rec['path']
            now = _mtime(path)
            recorded = rec.get('mtime')
            # The paper's guard (provenance.check_catalog_freshness) stats the file
            # NOW and raises if it predates MIN_CATALOG_DATE, so the freshness test
            # must use the CURRENT mtime.  The recorded one is kept separately --
            # it is what tells us whether the file has been rewritten since.
            stale = None
            if min_date and now is not None:
                stale = (datetime.fromtimestamp(now).date().isoformat()
                         < str(min_date))
            rows.append({
                'program': program, 'band': band, 'path': path,
                'recorded_mtime': recorded,
                'current_mtime': now,
                'present': now is not None,
                'rewritten_since_verdict': bool(
                    now and verdict_mtime and now > verdict_mtime),
                'predates_min_catalog_date': stale,
                'min_catalog_date': min_date,
            })
    return rows


def summarize(verdicts):
    """A compact dict for the page: what the paper says, and how fresh it is."""
    summary = verdicts.get('postrecat') or {}
    bands = {}
    for program, entry in summary.items():
        if not isinstance(entry, dict) or program == 'certifiers':
            continue
        for band, rec in entry.items():
            if isinstance(rec, dict):
                bands[f'{program}/{band}'] = {
                    'vs_virac_p60': (rec.get('vs_virac_p60') or {}).get('off'),
                    'vs_virac_p90': (rec.get('vs_virac_p90') or {}).get('off'),
                    'contrast_p60': (rec.get('vs_virac_p60') or {}).get('contrast'),
                    'mode_flip_mas': rec.get('mode_flip_mas'),
                    'vs_anchor': (rec.get('vs_anchor') or {}).get('off'),
                    'mtime': rec.get('mtime'),
                }
            else:
                bands[f'{program}/{band}'] = {'status': rec}
    return {
        'generated': summary.get('generated'),
        'problems': summary.get('problems') or [],
        'certifiers': summary.get('certifiers') or {},
        'bands': bands,
        'freshness': catalog_freshness(verdicts),
        'config': verdicts.get('config'),
        'provenance': verdicts.get('provenance'),
        'sip_vs_gwcs': (verdicts.get('sip_vs_gwcs') or {}).get('as_written'),
        'anderson_vs_gwcs': verdicts.get('anderson_vs_gwcs'),
        'postrecat_dir': verdicts.get('postrecat_dir'),
        'postrecat_mtime': verdicts.get('postrecat_mtime'),
    }


def age_days(stamp):
    """Days between an ISO timestamp and now, or ``None``."""
    if not stamp:
        return None
    try:
        when = datetime.fromisoformat(str(stamp))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - when).total_seconds() / 86400.0
