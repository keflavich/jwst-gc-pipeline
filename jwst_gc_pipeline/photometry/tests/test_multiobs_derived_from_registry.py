"""Which proposals need an observation token in their per-frame catalog names.

`MULTIOBS_PROPOSALS` was a hand-maintained tuple of two -- 2211 and 10678, the
two someone noticed.  The registry says eight more proposals register several
obsids against a single field, and per-frame catalog names carry visit, vgroup,
exposure and detector but NOT the observation:

    f162m_nrca1_visit001_vgroup02101_exp00001_m2_daophot_basic.fits

cloudef (2092) is the proven case, and the cost is data loss.  Observations 002
and 005 both use visit 001 / vgroup 02101, so the later run overwrote the
earlier: of ~64 obs-005 m2 catalogs per short-wavelength filter, 8 survive, and
F480M has none.  Its offsets table cannot be rebuilt either -- the builder
refuses to relabel obs 002's catalogs as obs 005's, which is the guard working
and the data still gone.
"""
import pytest

from jwst_gc_pipeline.photometry import naming


#: Registered with several obsids under one field, and previously untokened.
UNTOKENED_BEFORE = {
    '2092': 'cloudef 002 004 005 006 008 -- the proven data loss',
    '5365': 'sgrb2 001 002 998',
    '3958': 'sickle 001 002 007',
    '2221': 'brick 001 / cloudc 002',
    '6151': 'w51 001 002',
    '1905': 'wd1 001 003',
    '3523': 'wd2 003 005',
    '1979': 'm4 002 003',
}


@pytest.mark.parametrize('prop', sorted(UNTOKENED_BEFORE))
def test_a_multi_observation_proposal_is_recognised(prop):
    assert naming.proposal_is_multiobs(prop), UNTOKENED_BEFORE[prop]


@pytest.mark.parametrize('prop', ['2211', '10678'])
def test_the_hand_maintained_entries_still_hold(prop):
    """The tuple stays as a FLOOR, so a proposal that is multi-obs in practice
    but not yet in the registry keeps its token."""
    assert prop in naming.MULTIOBS_PROPOSALS
    assert naming.proposal_is_multiobs(prop)


@pytest.mark.parametrize('prop', ['1182', '4147', '1939', '2045'])
def test_a_single_OBSERVATION_proposal_gets_no_token(prop):
    """The token has to stay off where it is not needed: turning it on renames
    every per-frame catalog of a field that has no collision to avoid.

    "Single-observation" is not "single-pointing".  1182 is one observation
    (004) with TWO VISITS -- separate pointings taken on different dates, and
    the pair behind the brick-1182 v001 ~20" offset.  They do not collide
    because the per-frame name already carries the visit:

        f200w_nrca1_visit001_vgroup04101_exp00001_m2_daophot_basic.fits
        f200w_nrca1_visit002_...

    and both sets coexist on disk, 96 catalogs each.  What cloudef hit is
    narrower: two OBSERVATIONS that both restart at visit 001 / vgroup 02101,
    so the visit field does not separate them and nothing else in the name
    does either.
    """
    assert not naming.proposal_is_multiobs(prop)


def test_two_visits_of_one_observation_do_not_need_the_token(tmp_path):
    """The mechanism that makes 1182 safe, stated as a test rather than assumed:
    the visit is already in the name."""
    from jwst_gc_pipeline.photometry import naming as N
    assert not N.proposal_is_multiobs('1182')
    # ...and the tokens are empty, so the two visits differ only by the visit
    # field -- which is present and distinct.
    assert N.vetted_obs_tokens('1182', '004', 'F200W', 'nrca1') == ('', '')


def test_the_tuple_alone_no_longer_decides():
    """The regression: with the membership test back, cloudef is unrecognised
    and its catalogs collide again."""
    assert set(UNTOKENED_BEFORE) & set(naming.MULTIOBS_PROPOSALS) == set(), (
        'these are exactly the proposals the tuple did NOT list')
    for prop in UNTOKENED_BEFORE:
        assert naming.proposal_is_multiobs(prop), prop


def test_an_unknown_proposal_is_not_multiobs():
    """Fail-quiet in the safe direction: an unregistered proposal keeps the
    single-obs naming it has always had rather than gaining a token no reader
    expects."""
    assert not naming.proposal_is_multiobs('99999')


def test_the_token_reaches_the_per_frame_name():
    """`vetted_obs_tokens` is what actually inserts it into a filename."""
    mod_tok, end_tok = naming.vetted_obs_tokens('2092', '005', 'F162M', 'nrca1')
    assert '_o005' in (mod_tok + end_tok), (mod_tok, end_tok)
    assert naming.vetted_obs_tokens('1182', '004', 'F115W', 'nrca1') == ('', '')


def test_cloudef_and_a_single_obs_field_no_longer_collide():
    """The collision itself: two observations, one basepath, the same
    (visit, vgroup, exposure, detector) tuple.  Before, both produced the same
    filename and the later run won."""
    a = naming.vetted_obs_tokens('2092', '002', 'F162M', 'nrca1')
    b = naming.vetted_obs_tokens('2092', '005', 'F162M', 'nrca1')
    assert a != b, 'obs 002 and obs 005 still write the same catalog name'
    assert '_o002' in ''.join(a) and '_o005' in ''.join(b)


# ---------------------------------------------------------------------------
# every site that decides "is this multi-obs" must ask the same function
# ---------------------------------------------------------------------------

def test_no_site_does_a_bare_membership_test_on_the_tuple():
    """The regression this file exists for, twice over.

    `proposal_is_multiobs` was added and `naming.observation_tokens` taught to
    use it -- while `crowdsource_catalogs_long.obs_token`, the site that names
    the PER-FRAME catalogs, kept its own `in MULTIOBS_PROPOSALS` test.  So
    cloudef's o005 recatalog on 2026-08-19 ran to completion and wrote 528
    UNTOKENED names, and the collision the change exists to stop was untouched.

    Two enforcement points deciding the same question independently is how they
    come to disagree -- the same shape as #442/#447, one module over.
    """
    import inspect
    from jwst_gc_pipeline.photometry import cataloging, crowdsource_catalogs_long
    from jwst_gc_pipeline.photometry import naming as N
    offenders = []
    for mod in (crowdsource_catalogs_long, cataloging, N):
        for line in inspect.getsource(mod).splitlines():
            code = line.split('#')[0]
            if 'in MULTIOBS_PROPOSALS' in code:
                # naming.proposal_is_multiobs IS the one allowed reader: it is
                # the floor lookup inside the policy itself.
                if mod is N and 'if prop in MULTIOBS_PROPOSALS' in code:
                    continue
                offenders.append(f'{mod.__name__}: {line.strip()}')
    assert not offenders, (
        'these decide multi-obs without asking proposal_is_multiobs:\n  '
        + '\n  '.join(offenders))


def test_the_per_frame_catalog_name_carries_the_observation():
    """`obs_token` is what actually goes into the per-frame filename, and it is
    the site that was missed."""
    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import obs_token
    assert obs_token('2092', '005') == '_o005'
    assert obs_token('2092', '002') == '_o002'
    assert obs_token('2092', '005') != obs_token('2092', '002')
    # single-observation proposals keep the name they have always had
    assert obs_token('1182', '004') == ''
