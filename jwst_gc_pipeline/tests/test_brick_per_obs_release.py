"""brick's per-observation release wiring.

Both halves of #597 shipped defects that no test exercised.  The reviewer's
mutation run made that explicit: dropping brick's `observations` key from
stage_release SURVIVED the whole suite, which is how a PR body could claim the
routing was verified while `discover_images` returned zero mosaics.
"""
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _stage_release():
    spec = importlib.util.spec_from_file_location(
        '_sr', ROOT / 'scripts' / 'release' / 'stage_release.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['_sr'] = mod
    spec.loader.exec_module(mod)
    return mod


# --- the token keys on (proposal, field), not proposal ----------------------

@pytest.mark.parametrize('proposal,field,expected', [
    ('1182', '004', '_o004'),   # brick, 4-band half
    ('2221', '001', '_o001'),   # brick, 6-band half
    # cloudc is 2221 too.  It has one proposal, one observation and no
    # collision; tokening it would rename its merged catalogs to `_o002`, which
    # COMBINED_RE does not match, so cloudc would ship NO combined catalog.
    ('2221', '002', ''),
    ('2211', '028', '_o028'),
    ('10678', '001', '_o001'),
    ('4147', '012', ''),
])
def test_merged_token_is_per_proposal_field(proposal, field, expected):
    from jwst_gc_pipeline.photometry.naming import merged_catalog_obs_token
    assert merged_catalog_obs_token(proposal, field) == expected


def test_cloudc_is_not_swept_in_with_brick():
    """Explicit: 2221 appears in the pair list only for brick's observation."""
    from jwst_gc_pipeline.photometry import naming
    props = {p for p, _f in naming.PER_OBS_MERGED_FIELDS}
    assert '2221' in props
    assert ('2221', '002') not in naming.PER_OBS_MERGED_FIELDS
    # and the whole-proposal set must not have grown to cover it
    assert '2221' not in naming.PER_OBS_MERGED_PROPOSALS
    assert '1182' not in naming.PER_OBS_MERGED_PROPOSALS


# --- a LIST proposal_prefix must survive the observations branch ------------

def test_list_prefix_keeps_full_field_untagged_images(tmp_path):
    """A LIST prefix means one field imaged by several PROPOSALS.

    brick is 1182 and 2221 over the SAME sky with disjoint filter sets, so its
    mosaics are per-FILTER full-field products.  They must NOT be tagged with an
    observation: `check_image_catalog_match` keys on (filter, observation), the
    per-filter vetted catalogs are untokened until brick is re-merged, and a tag
    makes the two sets disjoint -- zero pairs, and a provenance gate that passes
    having compared nothing.
    """
    sr = _stage_release()
    # The two proposals have DISJOINT filters, so each filter directory holds
    # exactly one mosaic: F200W is 1182's, F182M is 2221's.
    a = 'jw01182-o004_t001_nircam_clear-f200w-merged_i2d.fits'
    b = 'jw02221-o001_t001_nircam_clear-f182m-merged_i2d.fits'
    for filt, n in (('F200W', a), ('F182M', b)):
        d = tmp_path / filt / 'pipeline'
        d.mkdir(parents=True)
        (d / n).write_bytes(b'')

    cfg = {'data_dir': tmp_path,
           'proposal_prefix': ['jw01182-o004_t001_nircam_clear',
                               'jw02221-o001_t001_nircam_clear'],
           'observations': ['o004', 'o001']}
    items = sr.discover_images(cfg)
    names = {pathlib.Path(i['src']).name for i in items if 'src' in i}
    assert {a, b} <= names, 'both proposals\' mosaics must be found'
    assert all(i.get('observation') is None for i in items), (
        'a list-prefix field must not tag its images with an observation')


def test_bare_prefix_still_composes(tmp_path):
    """gc2211 carries a bare 'jw02211' and must keep composing."""
    sr = _stage_release()
    fdir = tmp_path / 'F200W' / 'pipeline'
    fdir.mkdir(parents=True)
    n = 'jw02211-o028_t001_nircam_clear-f200w-merged_i2d.fits'
    (fdir / n).write_bytes(b'')
    cfg = {'data_dir': tmp_path, 'proposal_prefix': 'jw02211',
           'observations': ['o028']}
    items = sr.discover_images(cfg)
    assert {pathlib.Path(i['src']).name for i in items if 'src' in i} == {n}


# --- brick's REAL config, not a synthetic one ------------------------------
#
# The synthetic tests above still let the reviewer's mutation survive: deleting
# brick's `observations` key from stage_release changed nothing they assert.
# These read the shipped config.

def test_brick_config_declares_its_observations():
    """Tokened merged names only reach a release through the per-pointing
    branch, which is gated on this key.  Without it brick ships no combined
    catalog at all: COMBINED_RE matches only the untokened name, which the
    naming change stops producing."""
    sr = _stage_release()
    cfg = sr.FIELDS['brick']
    assert cfg.get('observations') == ['o001', 'o004'], (
        "brick must declare both halves: o004 = 1182 (F115W/F200W/F356W/F444W), "
        "o001 = 2221 (F182M/F187N/F212N/F405N/F410M/F466N)")


def test_every_brick_observation_has_exactly_one_prefix():
    """The list branch SELECTS by `-{obs}_`, so a declared observation with no
    matching prefix silently contributes zero images -- the failure mode that
    took brick from 29 mosaics to 0, arrived at from the other side."""
    sr = _stage_release()
    cfg = sr.FIELDS['brick']
    prefixes = cfg['proposal_prefix']
    assert isinstance(prefixes, list)
    for obs in cfg['observations']:
        hits = [p for p in prefixes if f'-{obs}_' in p]
        assert len(hits) == 1, (obs, hits, prefixes)


def test_cloudc_has_no_observations_key():
    """cloudc shares proposal 2221 with brick but has no collision.  If it ever
    gains this key its untokened tables stop being found."""
    sr = _stage_release()
    assert 'observations' not in sr.FIELDS['cloudc']


# --- validation strictness, made deliberate --------------------------------

def test_a_malformed_field_raises_for_every_proposal():
    """Validation runs before the membership test, so this is not per-obs-only.

    Latent rather than live -- the only registry `obsids: '*'` is
    gc-treasury/10678, which raised on main too -- but pinned so the strictness
    is a decision rather than a side effect of where the call moved.
    """
    from jwst_gc_pipeline.photometry import naming
    for bad in ('*', 'garbage'):
        with pytest.raises(naming.ObservationFieldError):
            naming.merged_catalog_obs_token('4147', bad)   # not a per-obs proposal
        with pytest.raises(naming.ObservationFieldError):
            naming.merged_catalog_obs_token('2211', bad)   # per-obs proposal


def test_field_none_is_still_permitted():
    """`None` means 'no observation given', which is legitimate."""
    from jwst_gc_pipeline.photometry.naming import merged_catalog_obs_token
    assert merged_catalog_obs_token('4147', None) == ''
    assert merged_catalog_obs_token('2211', None) == ''
