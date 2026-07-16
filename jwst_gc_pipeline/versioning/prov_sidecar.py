"""Read/write the ``<product>.prov.json`` provenance sidecar.

Every stage output gets a sidecar next to it recording the full provenance
record: the pipeline tag, the stage, the INPUT fingerprints (env / code / params
/ upstream facet hashes), and this product's OWN OUTPUT facet hashes
(data/wcs/meta).  The rerun engine reads the recorded record and compares it to
a freshly-computed one to decide the minimal action.

A sidecar is used (rather than only FITS keywords) because a full record carries
several 64-char hashes plus the upstream facet map -- too much for FITS cards --
and because catalog/table products and non-FITS artifacts need the same record.
The compact ``{tag, stage, data/wcs/meta}`` subset is ALSO mirrored into FITS
keywords (``GCTAG``/``GCSTAGE``/``GCDATAH``/``GCWCSH``/``GCMETAH``) by the
stamping layer, so a product remains self-describing even if the sidecar is lost.
"""
import json
import os

SCHEMA_VERSION = 1
SIDECAR_SUFFIX = '.prov.json'


def sidecar_path(product_path):
    """``/path/foo_i2d.fits -> /path/foo_i2d.fits.prov.json``."""
    return str(product_path) + SIDECAR_SUFFIX


def build_record(stage, tag, out_facets, env=None, code=None, params=None,
                 upstream=None, extra=None):
    """Assemble a provenance record dict.

    Parameters
    ----------
    stage : str            -- 'imaging', 'm12', 'm3', ...
    tag : str              -- resolved pipeline tag (release or dev).
    out_facets : dict      -- this product's output facets, e.g.
                              ``{'data': <sha>, 'wcs': <sha>, 'meta': <sha>}``.
    env : dict or None     -- environment inputs, e.g.
                              ``{'jwst_version': ..., 'crds_context': ...}``.
    code : str or None     -- ``fingerprint.code_hash(stage)``.
    params : str or None   -- ``fingerprint.params_hash(options, stage)``.
    upstream : dict or None -- upstream stage -> its output facet dict, e.g.
                              ``{'imaging': {'data':..,'wcs':..,'meta':..}}``.
    extra : dict or None   -- any additional free-form provenance.
    """
    rec = {
        'schema_version': SCHEMA_VERSION,
        'stage': stage,
        'tag': tag,
        'inputs': {
            'env': env or {},
            'code': code,
            'params': params,
            'upstream': upstream or {},
        },
        'outputs': dict(out_facets),
    }
    if extra:
        rec['extra'] = dict(extra)
    return rec


def write_sidecar(product_path, record):
    """Atomically write ``record`` to ``<product_path>.prov.json``."""
    path = sidecar_path(product_path)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write('\n')
    os.replace(tmp, path)
    return path


def read_sidecar(product_path):
    """Return the provenance record for ``product_path``, or None if absent.

    Accepts either the product path or the sidecar path directly.
    """
    path = str(product_path)
    if not path.endswith(SIDECAR_SUFFIX):
        path = sidecar_path(path)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def has_sidecar(product_path):
    return os.path.exists(sidecar_path(product_path))
