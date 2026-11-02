"""Faint-star reference fields: small cutouts with fixed pass thresholds.

Four few-arcsec cutouts, one per crowding/background environment (see
``fields.yaml``), are run through the real manual chain (m12 -> m7) with a
frozen list of artificial stars injected (``--inject-stars``,
``photometry/injection.py``) and again without.  ``evaluate.py`` scores each
run on completeness, flux bias, unsubtracted-residual excess and two purity
statistics; the thresholds live with the field in ``fields.yaml`` and do not
depend on which detection/vetting parameters produced the catalog.  A
parameter change passes only if every field still passes.

Layout of one run (the cutout pipeline's own layout)::

    <basepath>/cutouts/<label>/catalogs/<filt>_merged_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits
    <basepath>/cutouts/<label>/<FILT>/pipeline/*<filt>-merged*_m7_*mergedcat_residual_i2d.fits

with ``label = run_label(field, variant, seed)``.  ``seed == 0`` is the clean
(no injection) run.
"""
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS_YAML = os.path.join(HERE, 'fields.yaml')
INJECTION_DIR = os.path.join(HERE, 'injections')


def load_config(path=FIELDS_YAML):
    """The parsed ``fields.yaml`` (defaults merged into every field)."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    defaults = {k: v for k, v in cfg.items() if k != 'fields'}
    fields = {}
    for name, spec in cfg['fields'].items():
        merged = dict(defaults)
        merged.update(spec)
        merged['name'] = name
        fields[name] = merged
    return defaults, fields


def field_spec(name, path=FIELDS_YAML):
    """One field's spec (defaults applied)."""
    return load_config(path)[1][name]


def injection_table_path(name, seed):
    """The frozen injection table for ``name`` and ``seed``."""
    return os.path.join(INJECTION_DIR, f'{name}_s{int(seed)}.ecsv')


def run_label(name, variant, seed):
    """Cutout label of one reference run; ``seed=0`` is the clean run."""
    return f'ref_{name}_{variant}_s{int(seed)}'


def basepath(spec):
    """The target's data root (``/orange/adamginsburg/jwst/<target>``)."""
    return os.path.join(spec.get('data_root', '/orange/adamginsburg/jwst'),
                        spec['target'])


def run_dir(spec, variant, seed):
    """Where the cutout pipeline writes run ``(variant, seed)``."""
    return os.path.join(basepath(spec), 'cutouts',
                        run_label(spec['name'], variant, seed))
