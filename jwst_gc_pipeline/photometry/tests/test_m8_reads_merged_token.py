"""The m7/m8 cross-band reader must spell the token the WRITER spells.

`merge_daophot` writes the cross-band merged catalog with
`naming.merged_catalog_obs_token` (its `_obssuf`).  Two sites in cataloging.py
rebuilt that path to read it back, and both used the PER-FRAME token
(`crowdsource_catalogs_long.obs_token` -> `naming.perframe_obs_token`).

For most proposals the two agree, so nothing showed.  They differ for exactly
the `PER_OBS_MERGED_FIELDS` entries -- brick's 2221/001 and 1182/004 -- where
per-frame is `''` and merged is `_o001`/`_o004`.

The mismatch did not fail loudly.  It matched a STALE untokened sibling left
over from before #597, so brick's m8 forced fill read a ten-band file and wrote
it out under an `_o001` name: 651903 rows and all ten bands, where the correctly
tokened m7 beside it had 293925 rows and the six 2221 bands (#661).
"""
import inspect
import re

import pytest

from jwst_gc_pipeline.photometry import cataloging
from jwst_gc_pipeline.photometry.naming import (
    PER_OBS_MERGED_FIELDS, merged_catalog_obs_token, perframe_obs_token)


#: The entries where the two spellings actually diverge, derived rather than
#: hardcoded so this stays correct as the naming lists grow.  Today: brick's
#: two halves.  m4 (1979/002, 1979/003) is in PER_OBS_MERGED_FIELDS too but
#: also in PER_OBS_PERFRAME_FIELDS, so both spellings agree for it and it was
#: never exposed to this bug.
_DIVERGING = tuple((p, f) for p, f in PER_OBS_MERGED_FIELDS
                   if perframe_obs_token(p, f) != merged_catalog_obs_token(p, f))


def test_some_field_actually_diverges():
    """If nothing diverged this whole bug class would be unreachable, and the
    source test below would pass vacuously."""
    assert _DIVERGING, ('no (proposal, field) spells the merged token '
                        'differently from the per-frame one')


@pytest.mark.parametrize('proposal,field', list(_DIVERGING))
def test_the_diverging_fields_are_the_dangerous_ones(proposal, field):
    """Per-frame is empty there, so the reader built the AMBIGUOUS untokened
    name -- which is why it matched a stale sibling instead of raising."""
    assert perframe_obs_token(proposal, field) == ''
    assert merged_catalog_obs_token(proposal, field) == f'_o{field}'


@pytest.mark.parametrize('proposal,field', [
    ('2211', '028'), ('9438', '006'), ('10678', '005'),
    ('1979', '002'), ('1979', '003'),
])
def test_fields_whose_spellings_already_agree_are_unaffected(proposal, field):
    """The fix must not move any proposal where the two already matched."""
    assert perframe_obs_token(proposal, field) == merged_catalog_obs_token(proposal, field)


def test_no_site_builds_a_merged_catalog_path_from_the_perframe_token():
    """Pinned by source: the failure mode is a path that matches the WRONG file
    rather than raising, so there is no exception to assert on.

    Both cross-band sites build the name through ``crossband_merged_path``
    (#1197 added the PSF token there), so the merged-token spelling is pinned
    inside the helper and the sites are pinned to call it.
    """
    src = inspect.getsource(cataloging)
    assert '_xbsuf = _L.obs_token(' not in src, (
        'a cross-band merged path is being spelled with the per-frame token; '
        'use naming.merged_catalog_obs_token, which is what merge_daophot writes')
    helper = inspect.getsource(cataloging.crossband_merged_path)
    assert 'merged_catalog_obs_token(proposal_id, field)' in helper, (
        'crossband_merged_path must spell the obs suffix with the merged token')
    assert 'obs_token(' not in helper.replace('merged_catalog_obs_token(', ''), (
        'crossband_merged_path spells a second obs token besides the merged one')
    manual = inspect.getsource(cataloging.run_manual_pipeline)
    n = len(re.findall(r'_xb = crossband_merged_path\(', manual))
    assert n == 2, f'expected both cross-band sites to call crossband_merged_path, found {n}'


@pytest.mark.parametrize('proposal,field', list(_DIVERGING))
def test_crossband_merged_path_spells_the_merged_token(proposal, field, monkeypatch):
    """The helper itself, on the fields where the two spellings diverge."""
    monkeypatch.delenv('PSF_EPSF_CORE_DIR', raising=False)
    for label in ('m7', 'm8'):
        path = cataloging.crossband_merged_path('/x', label, 'merged', proposal, field)
        assert path.endswith(f'_{label}{merged_catalog_obs_token(proposal, field)}.fits')
        assert not path.endswith(f'_{label}.fits')


def test_the_reader_path_matches_the_writer_path_for_brick():
    """End to end on the two names: what the reader builds is what exists."""
    for proposal, field, want in (('2221', '001', '_o001'), ('1182', '004', '_o004')):
        suffix = merged_catalog_obs_token(proposal, field)
        assert suffix == want
        built = (f'basic_merged_indivexp_photometry_tables_merged_resbgsub_m7'
                 f'{suffix}.fits')
        assert built.endswith(f'_m7{want}.fits')
        # and the per-frame spelling would have built the ambiguous name
        stale = (f'basic_merged_indivexp_photometry_tables_merged_resbgsub_m7'
                 f'{perframe_obs_token(proposal, field)}.fits')
        assert stale.endswith('_m7.fits') and stale != built
