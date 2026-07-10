"""Drift guard: manual-pipeline defaults must live ONLY in MANUAL_DEFAULTS.

The 2026-07-09 consolidation moved ~30 duplicated optparse ``default=``
literals into ``manual_defaults.MANUAL_DEFAULTS``.  This test scans the
crowdsource_catalogs_long SOURCE (no import -- that module pulls in
jwst/stpsf and takes a minute) and fails if any option whose dest is a
MANUAL_DEFAULTS key regains a literal default, or references a key the dict
does not define.
"""
import os
import re

from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS, mopt

SRC = os.path.join(os.path.dirname(__file__), '..', 'jwst_gc_pipeline',
                   'photometry', 'crowdsource_catalogs_long.py')


def test_mopt_fallback_and_unknown_key():
    class O:
        manual_overshoot_ratio = 9.9
    assert mopt(O(), 'manual_overshoot_ratio') == 9.9
    assert mopt(object(), 'manual_overshoot_ratio') == MANUAL_DEFAULTS['manual_overshoot_ratio']
    try:
        mopt(object(), 'definitely_not_a_knob')
    except KeyError:
        pass
    else:
        raise AssertionError('unknown option name must raise KeyError')


def test_cli_defaults_read_the_dict():
    text = open(SRC).read()
    offenders = []
    for m in re.finditer(r'dest=["\'](\w+)["\']', text):
        dest = m.group(1)
        if dest not in MANUAL_DEFAULTS:
            continue
        seg = text[m.end():m.end() + 300]
        d = re.search(r'default=([^,)\s]+)', seg)
        if d and f"MANUAL_DEFAULTS['{dest}']" not in d.group(1) \
                and f'MANUAL_DEFAULTS["{dest}"]' not in d.group(1):
            offenders.append((dest, d.group(1)))
    assert not offenders, (
        f'literal CLI defaults for MANUAL_DEFAULTS-managed options (move the '
        f'value into manual_defaults.MANUAL_DEFAULTS): {offenders}')


def test_dict_keys_referenced_by_cli_exist():
    text = open(SRC).read()
    missing = sorted({k for k in re.findall(r"MANUAL_DEFAULTS\['(\w+)'\]", text)
                      if k not in MANUAL_DEFAULTS})
    assert not missing, f'CLI references undefined MANUAL_DEFAULTS keys: {missing}'
