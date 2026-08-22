"""Every filter the project observes needs a near-saturated flagging radius.

``merge_catalogs.flag_near_saturated`` looks its radius up in a hard-coded dict.
A filter missing from it raises at the END of an m7 finalize, after the
per-frame fits and the satstar consolidation are already done:

    KeyError: 'f250m'
    PERFRAME finalize: rc=1 ... (elapsed=10918s)

That is wd2 (job 39950409, 2026-08-22) -- three hours of work thrown away over a
one-line omission.  The map's own comment already said to keep it in sync with
``obs_filters``; nothing enforced it.

An audit at the time found FOUR observed filters missing, so wd2 was simply the
first field to reach the m7 that needed one:

    f090w    m92, ngc6334
    f150w2   m4, ngc6397
    f250m    wd2
    f322w2   m4, ngc6397

This test moves the discovery from three hours into a run to CI.
"""
import os
import re

import pytest
import yaml


HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGE_PY = os.path.join(HERE, 'photometry', 'merge_catalogs.py')
FIELDS_YAML = os.path.join(HERE, 'fields.yaml')


def _mapped_filters():
    """The filter keys in flag_near_saturated's radius map, read from source.

    Parsed rather than called because building the dict needs a catalog and a
    basepath; the point here is which keys EXIST.
    """
    src = open(MERGE_PY).read()
    start = src.index('radius = {# short-wave')
    block = src[start:src.index('}[filtername]', start)
                if '}[filtername]' in src[start:start + 4000]
                else src.index('        if filtername not in radius:', start)]
    return set(re.findall(r"'(f\d+[a-z0-9]*)'", block))


def _observed_filters():
    """{filter: [fields that observe it]} from the field registry."""
    doc = yaml.safe_load(open(FIELDS_YAML))
    fields = doc.get('fields', doc)
    used = {}
    for fname, fd in fields.items():
        for _prop, od in (fd.get('observations') or {}).items():
            for filt in (od.get('filters') or []):
                used.setdefault(filt.lower(), set()).add(fname)
    return {k: sorted(v) for k, v in used.items()}


def test_every_observed_filter_has_a_flagging_radius():
    """The whole point: an omission fails here, not 3 hours into an m7."""
    mapped = _mapped_filters()
    observed = _observed_filters()
    missing = {f: v for f, v in observed.items() if f not in mapped}
    assert not missing, (
        'filters observed by these fields have no near-saturated flagging '
        'radius, so their m7 finalize will KeyError after doing all its work: '
        + '; '.join(f'{f} ({", ".join(v)})' for f, v in sorted(missing.items())))


@pytest.mark.parametrize('filt,fields', sorted(_observed_filters().items()))
def test_each_observed_filter_individually(filt, fields):
    """Parametrised so a failure names the filter and who observes it."""
    assert filt in _mapped_filters(), f'{filt} observed by {fields}'


@pytest.mark.parametrize('filt', ['f090w', 'f150w2', 'f250m', 'f322w2'])
def test_the_four_that_were_missing(filt):
    """Pins the specific regression rather than trusting the sweep above, which
    would go quiet if fields.yaml ever stopped listing them."""
    assert filt in _mapped_filters()


def test_an_unknown_filter_says_how_to_fix_it():
    """A bare KeyError names the key and nothing else.  After 3 hours the log
    should say what to do."""
    from astropy.table import Table
    import astropy.units as u
    import numpy as np
    from astropy.coordinates import SkyCoord

    from jwst_gc_pipeline.photometry import merge_catalogs as MC

    cat = Table({'skycoord': SkyCoord([266.0] * u.deg, [-28.9] * u.deg)})
    sat = Table({'skycoord_fit': SkyCoord([266.0] * u.deg, [-28.9] * u.deg)})

    def _fake_load(filtername, target=None, basepath=None):
        return sat

    orig = MC.load_satstar_catalog
    MC.load_satstar_catalog = _fake_load
    try:
        with pytest.raises(KeyError) as ex:
            MC.flag_near_saturated(cat, 'f999x', target='wd2', basepath='/x')
    finally:
        MC.load_satstar_catalog = orig
    msg = str(ex.value)
    assert 'f999x' in msg
    assert 'flag_near_saturated' in msg, msg
    assert '0.55' in msg, msg


def test_nircam_entries_agree_on_one_value():
    """Every NIRCam band uses 0.55 arcsec; only MIRI scales.  A new NIRCam
    filter added with a different number is a mistake, not a choice."""
    src = open(MERGE_PY).read()
    start = src.index('radius = {# short-wave')
    end = src.index('# MIRI:', start)
    nircam = re.findall(r"'(f\d+[a-z0-9]*)':\s*([\d.]+)\*u\.arcsec",
                        src[start:end])
    assert nircam, 'could not parse the NIRCam entries'
    odd = [(f, v) for f, v in nircam if v != '0.55']
    assert not odd, f'NIRCam filters with a non-0.55 radius: {odd}'
