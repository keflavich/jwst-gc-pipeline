"""The JWST-internal reference: per-filter consensus and the reference filter."""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry.consensus_catalog import (
    InterVisitOffsetError, NoReferenceFilterError, consensus_path,
    pool_visit_consensi, promote_reference_filter, reference_consensus_path,
    reference_filter, reference_filter_rank, tie_to_reference_consensus,
    write_filter_consensus)


# --------------------------------------------------------------------------
# Which filter anchors the field.
# --------------------------------------------------------------------------

def test_the_intended_order_is_reproduced():
    """F212N > F210M > F187N > F182M > F200W > F150W.

    Note F210M beats F187N while F187N beats F200W: neither wavelength nor
    bandwidth alone produces that, so the two trade off.
    """
    intended = ['F212N', 'F210M', 'F187N', 'F182M', 'F200W', 'F150W']
    assert sorted(intended, key=reference_filter_rank) == intended


def test_bandwidth_alone_does_not_explain_the_order():
    """A rule that sorted on bandwidth first would put F187N above F210M."""
    assert reference_filter_rank('F210M') < reference_filter_rank('F187N')


def test_wavelength_alone_does_not_explain_the_order():
    """F200W is closer to Ks than F187N, and still ranks below it: F200W
    saturates the bright stars VIRAC2 measures."""
    assert reference_filter_rank('F187N') < reference_filter_rank('F200W')


@pytest.mark.parametrize('field,expected', [
    (['F182M', 'F187N', 'F212N', 'F405N', 'F410M', 'F466N'], 'F212N'),  # brick
    (['F115W', 'F200W', 'F356W', 'F444W'], 'F200W'),                    # w51
    (['F150W', 'F444W'], 'F150W'),
])
def test_the_field_picks_its_closest_match_to_virac2(field, expected):
    assert reference_filter(field) == expected


def test_the_second_intended_order_is_reproduced():
    """F277W > F140M > F115W: a long-wavelength filter can outrank a blue one.

    This is what forces LOG wavelength.  Linearly, F277W is 0.62 um from Ks and
    F140M is 0.75 -- so close that no positive long-wavelength penalty can put
    F277W first; in log, ln(2.77/2.15) < |ln(1.40/2.15)| with room to spare.
    """
    intended = ['F277W', 'F140M', 'F115W']
    assert sorted(intended, key=reference_filter_rank) == intended


def test_the_far_infrared_ranks_last():
    """No channel term is needed to get this -- log distance does it."""
    assert reference_filter_rank('F2550W') > reference_filter_rank('F1130W')
    assert reference_filter_rank('F1130W') > reference_filter_rank('F444W')


@pytest.mark.parametrize('name,micron', [
    ('F150W2', 1.50), ('F322W2', 3.22), ('F1500W', 15.00), ('F212N', 2.12),
])
def test_the_wide_double_filters_are_not_ten_times_their_wavelength(name, micron):
    """Stripping every digit turns F150W2 into 15.02 um, which ranks a 1.5 um
    SW filter below most of MIRI and puts it 0.0013 from F1500W."""
    from jwst_gc_pipeline.photometry.consensus_catalog import _filter_micron
    assert _filter_micron(name) == pytest.approx(micron)


def test_a_wide_double_outranks_the_mid_infrared():
    """The consequence of the parse above: F150W2 is a 1.5 um filter."""
    assert reference_filter_rank('F150W2') < reference_filter_rank('F1500W')
    assert reference_filter(['F1500W', 'F150W2']) == 'F150W2'


def test_a_field_with_no_filters_says_so():
    with pytest.raises(NoReferenceFilterError):
        reference_filter([])


def test_a_non_filter_entry_is_ignored_not_crashed_on():
    """NIRCam filter lists carry CLEAR as the pupil half of a pair."""
    assert reference_filter(['CLEAR', 'F212N']) == 'F212N'
    with pytest.raises(NoReferenceFilterError):
        reference_filter(['CLEAR', ''])


# --------------------------------------------------------------------------
# Pooling the per-visit consensi into one per-filter catalog.
# --------------------------------------------------------------------------

def _consensus(ra, dec, mag=None, nexp=None, scatter=None):
    """The shape build_visit_consensus returns."""
    n = len(ra)
    coords = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg)
    return dict(coords=coords,
                nexp=np.asarray(nexp if nexp is not None else [4] * n),
                scatter_mas=np.asarray(
                    scatter if scatter is not None else [3.0] * n),
                mag=np.asarray(mag if mag is not None else [20.0] * n))


def test_pooling_keeps_every_star_and_counts_the_visits():
    per_visit = {
        '001': _consensus([266.5, 266.6], [-28.7, -28.71]),
        '002': _consensus([266.8], [-28.9]),
    }
    pooled = pool_visit_consensi(per_visit)
    assert len(pooled) == 3
    assert set(pooled['n_visits']) == {1}
    assert pooled.meta['NVISITS'] == 2


def test_a_star_seen_in_two_visits_is_averaged_once():
    """Overlapping visits must not double a star in the reference catalog."""
    per_visit = {
        '001': _consensus([266.5], [-28.7]),
        '002': _consensus([266.5 + 1e-5], [-28.7]),   # ~0.03", same star
    }
    pooled = pool_visit_consensi(per_visit)
    assert len(pooled) == 1
    assert pooled['n_visits'][0] == 2
    assert 266.5 < pooled['RA'][0] < 266.5 + 1e-5


@pytest.mark.parametrize('nvisits', [2, 3, 4, 5])
def test_a_star_seen_in_N_visits_is_ONE_row_with_n_visits_N(nvisits):
    """A pairwise nearest-partner pass caps a group at two members, so a star
    in 3+ visits came out as one merged pair plus duplicate rows and n_visits
    could never exceed 2.  ngc6334/6778 and wd1/1905 have three visits."""
    per_visit = {f'{i:03d}': _consensus([266.5 + i * 3e-6], [-28.7])
                 for i in range(nvisits)}
    pooled = pool_visit_consensi(per_visit)
    assert len(pooled) == 1
    assert pooled['n_visits'][0] == nvisits


def test_two_stars_of_ONE_visit_are_never_merged():
    """Association is across visits.  Two genuinely close stars seen in the
    same exposure set are two stars, however near."""
    per_visit = {'001': _consensus([266.5, 266.5 + 1e-5], [-28.7, -28.7])}
    pooled = pool_visit_consensi(per_visit)
    assert len(pooled) == 2


def test_a_transitive_chain_never_merges_two_stars_of_one_visit():
    """A(v1) - B(v2) - C(v1), each link inside the radius.  Growing the group
    transitively without checking would put A and C in one row."""
    per_visit = {
        '001': _consensus([266.5, 266.5 + 8e-5], [-28.7, -28.7]),
        '002': _consensus([266.5 + 4e-5], [-28.7]),
    }
    pooled = pool_visit_consensi(per_visit)
    assert len(pooled) == 2
    assert sorted(pooled['n_visits']) == [1, 2]


def test_one_star_in_one_visit_does_not_crash():
    """A scalar SkyCoord has no len(); the checkpoint used to die on TypeError."""
    per_visit = {'001': dict(coords=SkyCoord(266.5 * u.deg, -28.7 * u.deg),
                             nexp=np.array([3]), scatter_mas=np.array([2.0]),
                             mag=np.array([19.0]))}
    pooled = pool_visit_consensi(per_visit)
    assert len(pooled) == 1
    assert pooled['n_visits'][0] == 1


def test_the_uncertainty_is_carried_not_dropped():
    """This catalog is what other filters tie to; a tie whose reference has no
    stated precision cannot be given a tolerance."""
    per_visit = {'001': _consensus([266.5], [-28.7], nexp=[9], scatter=[6.0])}
    pooled = pool_visit_consensi(per_visit)
    assert pooled['scatter_mas'][0] == pytest.approx(6.0)
    assert pooled['err_mas'][0] == pytest.approx(2.0)      # 6 / sqrt(9)
    assert pooled['n_exposures'][0] == 9


def test_a_single_exposure_star_gets_NaN_error_not_zero():
    """An identically-zero uncertainty free-passes a QC gate."""
    per_visit = {'001': _consensus([266.5], [-28.7], nexp=[1], scatter=[np.nan])}
    pooled = pool_visit_consensi(per_visit)
    assert np.isnan(pooled['err_mas'][0])


def test_magnitudes_are_averaged_in_flux():
    per_visit = {
        '001': _consensus([266.5], [-28.7], mag=[20.0]),
        '002': _consensus([266.5 + 1e-6], [-28.7], mag=[15.0]),
    }
    pooled = pool_visit_consensi(per_visit)
    flux_mean = -2.5 * np.log10(0.5 * (10 ** -8.0 + 10 ** -6.0))
    assert pooled['refmag'][0] == pytest.approx(flux_mean)
    assert pooled['refmag'][0] < 17.5      # not the magnitude-space 17.5


def test_pooling_nothing_says_so():
    with pytest.raises(ValueError, match='no visit consensus'):
        pool_visit_consensi({'001': None})


def _grid(ra0, dec0, n=400, seed=0):
    rng = np.random.default_rng(seed)
    return (ra0 + rng.uniform(-0.005, 0.005, n),
            dec0 + rng.uniform(-0.005, 0.005, n))


def test_visits_that_are_not_on_a_common_frame_refuse_to_pool():
    """Averaging two visits offset by X lands every shared star at X/2 and
    inflates the row count with unmerged duplicates -- silently, in the very
    catalog the other filters tie to."""
    ra, dec = _grid(266.5, -28.7)
    shift = 0.5 / 3600.0                      # 500 mas, well past gross
    per_visit = {'001': _consensus(ra, dec),
                 '002': _consensus(ra, dec + shift)}
    with pytest.raises(InterVisitOffsetError, match='gross|mas from'):
        pool_visit_consensi(per_visit)


def test_the_inter_visit_offset_is_recorded_even_when_it_passes():
    """The measurements are what make a FINE tolerance choosable later."""
    ra, dec = _grid(266.5, -28.7)
    per_visit = {'001': _consensus(ra, dec),
                 '002': _consensus(ra, dec)}
    pooled = pool_visit_consensi(per_visit)
    assert pooled.meta['IVMAXMAS'] < 20.0
    assert pooled.meta['ANCHORVI'] == '001'
    assert 'IV_002' in pooled.meta


# --------------------------------------------------------------------------
# On disk.
# --------------------------------------------------------------------------

def test_the_written_catalog_says_what_it_is(tmp_path):
    table = pool_visit_consensi({'001': _consensus([266.5], [-28.7])})
    path = write_filter_consensus(str(tmp_path), 'F212N', table)
    assert path == consensus_path(str(tmp_path), 'F212N')
    back = Table.read(path)
    assert back.meta['FILTER'] == 'F212N'
    assert back.meta['CONSTYPE'] == 'per-filter JWST consensus'
    assert 'skycoord' in back.colnames        # what load_reference_catalog reads


def test_two_proposals_sharing_a_target_directory_do_not_collide(tmp_path):
    """ngc6334's 6778 and 7213 share catalogs/, a filter list AND obsid 001, at
    reference epochs 1.6 yr apart -- so the token must be the PROPOSAL."""
    from jwst_gc_pipeline.photometry.consensus_catalog import consensus_obs_token
    a = consensus_path(str(tmp_path), 'F200W',
                       obs_token=consensus_obs_token('6778', '001'))
    b = consensus_path(str(tmp_path), 'F200W',
                       obs_token=consensus_obs_token('7213', '001'))
    assert a != b
    assert reference_consensus_path(str(tmp_path), obs_token='_j6778') \
        != reference_consensus_path(str(tmp_path), obs_token='_j7213')


def test_two_observations_sharing_a_directory_do_not_collide(tmp_path):
    """cloudef/2092 runs obs 002 and 005 through one directory with one filter
    list.  The legacy per-frame obs_token does not cover 2092, so the consensus
    token falls back to _o<obsid> rather than letting them overwrite."""
    from jwst_gc_pipeline.photometry.consensus_catalog import consensus_obs_token
    a = consensus_path(str(tmp_path), 'F210M',
                       obs_token=consensus_obs_token('2092', '002'))
    b = consensus_path(str(tmp_path), 'F210M',
                       obs_token=consensus_obs_token('2092', '005'))
    assert a != b
    assert a.endswith('f210m_o002_consensus.fits')


def test_no_observation_means_no_token(tmp_path):
    from jwst_gc_pipeline.photometry.consensus_catalog import consensus_obs_token
    assert consensus_path(str(tmp_path), 'F212N',
                          obs_token=consensus_obs_token(None, None)) \
        == consensus_path(str(tmp_path), 'F212N')


def test_a_nameless_filter_refuses_to_be_written(tmp_path):
    """`unknown_consensus.fits` cannot be tied to, promoted, or told apart from
    the next filter that lands there."""
    table = pool_visit_consensi({'001': _consensus([266.5], [-28.7])})
    with pytest.raises(ValueError, match='no filter name'):
        write_filter_consensus(str(tmp_path), '', table)


def test_promoting_the_reference_filter_names_the_filter_it_chose(tmp_path):
    table = pool_visit_consensi({'001': _consensus([266.5], [-28.7])})
    write_filter_consensus(str(tmp_path), 'F212N', table)
    chosen, path = promote_reference_filter(str(tmp_path),
                                            ['F405N', 'F212N', 'F182M'])
    assert chosen == 'F212N'
    assert path == reference_consensus_path(str(tmp_path))
    back = Table.read(path)
    assert back.meta['REFFILT'] == 'F212N'
    assert back.meta['CONSTYPE'] == 'JWST reference-filter consensus'


def test_promoting_without_the_reference_filters_catalog_refuses(tmp_path):
    """Tying every filter to a silently-absent reference is the failure this
    ladder exists to prevent."""
    table = pool_visit_consensi({'001': _consensus([266.5], [-28.7])})
    write_filter_consensus(str(tmp_path), 'F405N', table)     # not the chosen one
    with pytest.raises(FileNotFoundError, match='F212N'):
        promote_reference_filter(str(tmp_path), ['F405N', 'F212N'])


# --------------------------------------------------------------------------
# Tying a filter to the reference.
# --------------------------------------------------------------------------

def test_a_filter_offset_from_the_reference_is_measured(tmp_path):
    rng = np.random.default_rng(42)
    ra = 266.5 + rng.uniform(-0.01, 0.01, 400)
    dec = -28.7 + rng.uniform(-0.01, 0.01, 400)
    reference = SkyCoord(ra * u.deg, dec * u.deg)
    shift_deg = 20.0 / 3600.0 / 1000.0            # 20 mas in Dec
    moved = SkyCoord(ra * u.deg, (dec + shift_deg) * u.deg)
    result = tie_to_reference_consensus(moved, reference, context='test')
    assert np.isclose(result['off_mas'], 20.0, atol=5.0)


# --------------------------------------------------------------------------
# The registry and the formula must give the same answer.
# --------------------------------------------------------------------------

def _registry_filters(proposal, obs=None):
    """The proposal's filters, scoped to ``obs``'s instrument when it can be.

    `fields.yaml` carries one flat filter list per proposal, so `obs_filters`
    returns the same union for every instrument.  Ranking over that union can
    choose a band the observation never took -- and `promote_reference_filter`
    resolves the chosen band's consensus under the OBSERVATION's token, so such
    a choice names a file that cannot exist (sickle: F210M is NIRCam obs 007,
    F770W is MIRI obs 001-002, and neither token has the other's bands).

    Falls back to the union when the observation's instrument is ambiguous --
    sgrb2's obs 001 is registered under both nircam and miri -- so nothing that
    was checkable before stops being checked.
    """
    from jwst_gc_pipeline import fields as fields_mod
    if obs:
        scoped = fields_mod.filters_for_observation(None, proposal, obs)
        if scoped:
            return scoped
    out = set()
    for instrument in ('nircam', 'miri', 'niriss'):
        for by_proposal in fields_mod.obs_filters(instrument).values():
            out.update(str(f).upper() for f in by_proposal.get(proposal, []))
    return sorted(out)


def test_a_hand_set_reference_filter_is_one_the_field_actually_observes():
    """`alignment_config.FieldAlignment.reference_filter` is the band whose
    visit consensus defines the field's internal frame -- so naming a band the
    proposal never observed leaves that frame undefined.  w51/6151 said F200W
    and observes none."""
    from jwst_gc_pipeline.reduction import alignment_config as ac
    for entry in ac.ALIGNMENT_CONFIG:
        if not entry.reference_filter:
            continue
        available = _registry_filters(
            entry.proposal, (entry.fields or (None,))[0])
        if not available:
            continue          # proposal not in fields.yaml; nothing to check
        assert entry.reference_filter.upper() in available, (
            f'proposal {entry.proposal} declares reference_filter='
            f'{entry.reference_filter} but observes {available}')


def test_the_formula_reproduces_the_hand_set_reference_filters():
    """Two places answer "which band anchors this field".  They must agree, or
    the m2 consensus catalog and the reducer are anchored to different bands."""
    from jwst_gc_pipeline.reduction import alignment_config as ac
    disagreements = []
    for entry in ac.ALIGNMENT_CONFIG:
        if not entry.reference_filter:
            continue
        available = _registry_filters(
            entry.proposal, (entry.fields or (None,))[0])
        if not available:
            continue
        computed = reference_filter(available)
        if computed.upper() != entry.reference_filter.upper():
            disagreements.append(
                f'{entry.proposal}: config={entry.reference_filter} '
                f'formula={computed} from {available}')
    assert not disagreements, '\n'.join(disagreements)
