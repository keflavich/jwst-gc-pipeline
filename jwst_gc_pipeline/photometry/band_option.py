"""Per-band values of a manual-pipeline option.

An option read through this module is a plain number, which every band uses
(what such an option meant before it took per-band values), or comma-separated
``BAND=VALUE`` pairs such as ``F212N=50,F480M=0``.  A band that no pair names
uses the default of the option (``MANUAL_DEFAULTS``).  Band names are
case-insensitive and a value is a finite number >= 0.

``parse_band_values`` validates a value and normalizes it (a float for a plain
number, so a plain-number run keeps its options hash, and a dict with
lower-case band names for pairs).  ``band_value`` and ``band_mopt`` resolve it
for one band.  ``band_option_callback`` is the ``optparse`` callback that
stores the normalized value, so a malformed value stops the run when the
command line is read.
"""
import math
import optparse
from collections.abc import Mapping

from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS, mopt


def _number(raw, option, context):
    """``raw`` as a finite float >= 0; ``context`` names the text the error quotes."""
    if isinstance(raw, bool):
        raise ValueError(f"{option}: {context} is not a number")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{option}: {context} is not a number") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{option}: {context} must be a finite number >= 0")
    return value


def _is_number(text):
    try:
        float(text)
    except ValueError:
        return False
    return True


def parse_band_values(value, option):
    """Validate ``value`` of ``option`` and normalize it.

    ``'50'`` or ``50`` -> ``50.0`` (every band); ``'F212N=50,F480M=0'`` ->
    ``{'f212n': 50.0, 'f480m': 0.0}``.  A mapping that is already parsed is
    validated and returned as a new dict.  Raises ``ValueError``, naming
    ``option`` and the offending text, for a plain number mixed with pairs, a
    band named twice, an entry that is not ``BAND=VALUE``, and a value that is
    not a finite number >= 0.
    """
    if isinstance(value, Mapping):
        out = {}
        for band, val in value.items():
            key = str(band).strip().lower()
            if not key.isalnum():
                raise ValueError(f"{option}: cannot read band name {band!r} in {dict(value)!r}")
            if key in out:
                raise ValueError(f"{option}: band {key.upper()} appears twice in {dict(value)!r}")
            out[key] = _number(val, option, f"value {val!r} of band {key.upper()} in {dict(value)!r}")
        if not out:
            raise ValueError(f"{option}: empty mapping")
        return out
    if not isinstance(value, str):
        return _number(value, option, repr(value))
    text = value.strip()
    if not text:
        raise ValueError(f"{option}: {value!r} is empty")
    if '=' not in text:
        if not _is_number(text):
            raise ValueError(f"{option}: {text!r} is not a number or a list of BAND=VALUE pairs")
        return _number(text, option, repr(text))
    out = {}
    for item in text.split(','):
        band, sep, val = item.partition('=')
        key, val = band.strip().lower(), val.strip()
        if not sep and _is_number(item):
            raise ValueError(f"{option}: {text!r} mixes a plain number with BAND=VALUE pairs")
        if not sep or not key.isalnum() or not val or '=' in val:
            raise ValueError(f"{option}: cannot read {item.strip()!r} in {text!r}; "
                             f"expected BAND=VALUE")
        if key in out:
            raise ValueError(f"{option}: band {key.upper()} appears twice in {text!r}")
        out[key] = _number(val, option, f"value {val!r} of band {key.upper()} in {text!r}")
    return out


def _resolve(parsed, band, default):
    """``(value, listed)`` of a parsed option for ``band``; ``listed`` is False when a
    per-band option names other bands only and the default applies."""
    if not isinstance(parsed, dict):
        return parsed, True
    key = str(band).strip().lower()
    if key in parsed:
        return parsed[key], True
    return float(default), False


def band_value(value, band, default, option):
    """The value of ``option`` for ``band``: the plain number, the value of the pair that
    names ``band``, or ``default`` when the pairs name other bands only."""
    return _resolve(parse_band_values(value, option), band, default)[0]


def band_mopt(opts, name, band, label=None):
    """``mopt(opts, name)`` resolved for ``band``, with ``MANUAL_DEFAULTS[name]`` as the
    value of a band that the pairs do not name.

    With ``label`` and per-band pairs, print the value that ``band`` uses, so the log of
    each band's seed says which value it was built with.
    """
    flag = '--' + name.replace('_', '-')
    parsed = parse_band_values(mopt(opts, name), flag)
    value, listed = _resolve(parsed, band, MANUAL_DEFAULTS[name])
    if label and isinstance(parsed, dict):
        pairs = ','.join(f'{b.upper()}={v:g}' for b, v in parsed.items())
        print(f"[{label}] {flag} {pairs}: {str(band).upper()} uses {value:g}"
              f"{'' if listed else ' (not listed: the default)'}", flush=True)
    return value


def band_option_callback(option, opt_str, value, parser):
    """``optparse`` callback of a per-band option (``action='callback'``, ``type='string'``):
    store the normalized value (see ``parse_band_values``) under ``option.dest``."""
    try:
        setattr(parser.values, option.dest, parse_band_values(value, opt_str))
    except ValueError as err:
        raise optparse.OptionValueError(str(err)) from None
