"""The submit scripts take their environment from the caller, with the shipped
value as the default.

`export CRDS_PATH=/orange/...` overwrites whatever `config.yaml` or the caller
exported, which made the configuration decorative for that stage.  The
`${VAR:-default}` form leaves an exported value alone.  These tests pin both the
form and the default, so a rewrite cannot change which cache a stage uses.
"""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMIT_SCRIPTS = sorted((REPO_ROOT / 'scripts' / 'reduction').glob('*.sbatch'))

#: The value each variable falls back to when the caller exports nothing.  These
#: are the literals the scripts carried before the `${VAR:-...}` rewrite, so a
#: change here changes which CRDS cache a stage resolves against.
SHIPPED_DEFAULTS = {
    'CRDS_PATH': '/orange/adamginsburg/jwst/crds',
    'CRDS_SERVER_URL': 'https://jwst-crds.stsci.edu',
    'STPSF_PATH': '/orange/adamginsburg/jwst/stpsf-data',
}


def _exports(text, name):
    return re.findall(rf'^export {name}=(.*)$', text, re.M)


@pytest.mark.parametrize('script', SUBMIT_SCRIPTS, ids=lambda p: p.name)
def test_a_submit_script_defers_to_an_exported_value(script):
    text = script.read_text()
    for name, default in SHIPPED_DEFAULTS.items():
        for value in _exports(text, name):
            assert value == '${%s:-%s}' % (name, default), (
                f'{script.name}: `export {name}={value}` overwrites what the '
                f'caller exported.  Write `export {name}=${{{name}:-{default}}}` '
                f'so config.yaml and an interactive override still win.')


#: Submitters that never load a JWST reference file, so they need no cache.
NO_CRDS = {
    # column-merges the per-band m8 partials with numpy and astropy.table alone
    # (scripts/reduction/m8_merge_partials.py imports nothing from jwst).
    'submit_cataloging_m8_merge.sbatch',
}


def test_every_submitter_that_runs_the_pipeline_points_crds_somewhere():
    """CRDS reads its cache path when jwst loads, so it has to be set first."""
    stage_scripts = [s for s in SUBMIT_SCRIPTS
                     if s.name.startswith(('submit_reduction',
                                           'submit_cataloging',
                                           'submit_merge'))
                     and s.name not in NO_CRDS]
    assert stage_scripts
    missing = [s.name for s in stage_scripts
               if not _exports(s.read_text(), 'CRDS_PATH')]
    assert not missing, f'no CRDS_PATH in {missing}'


def test_the_no_crds_list_names_real_scripts():
    names = {s.name for s in SUBMIT_SCRIPTS}
    assert NO_CRDS <= names, f'stale entries: {sorted(NO_CRDS - names)}'
