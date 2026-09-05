"""One exposure, one lineage: the untokened suffix is a TAIL of the tokened one.

``get_filenames`` globs ``{basepath}/{filt}/pipeline/jw{prop}{obs}{visit}*
{detector}*{each_suffix}.fits``.  The wildcard between the detector token and
the suffix is what lets ``module='nrca'`` stand for ``nrca1..nrca4``, and it is
also what makes the match a SUFFIX test rather than an equality test: asking
for ``o002_crf`` matches ``..._mirimage_o002_crf.fits`` and
``..._mirimage_align_o002_crf.fits`` alike.

That became reachable when #766 gave MIRI and NIRISS the untokened
``o<obs>_crf`` they actually write.  brick/F2550W holds 48
``_mirimage_align_o002_crf`` frames from an older Image3 crf-naming branch and
no bare ones (measured 2026-09-05), so today the glob returns 48 files for 48
exposures.  The next ``PipelineMIRI`` run there writes the bare spelling beside
them -- ``PipelineMIRI`` names the per-exposure crf off the ``_cal`` stem and
does not delete the older name -- and the same glob returns 96 files for 48
exposures, every star fitted and merged twice.

Both sides are pinned here: the two-lineage directory raises, and the
one-lineage directories that exist today (including brick/F2550W's legacy-only
one) keep returning their frames.
"""
import pytest

from jwst_gc_pipeline.photometry.requested_filters import (
    MixedLineageFramesError,
    assert_one_lineage_per_exposure,
    lineages_per_exposure,
)

BARE = 'jw02221002001_02101_{exp}_mirimage_o002_crf.fits'
ALIGN = 'jw02221002001_02101_{exp}_mirimage_align_o002_crf.fits'
DESTREAK = 'jw02221001001_02101_{exp}_nrcb3_destreak_o001_crf.fits'
NIRCAM_BARE = 'jw02221001001_02101_{exp}_nrcb3_o001_crf.fits'


def _names(template, n=2):
    return [template.format(exp=f'{i:05d}') for i in range(1, n + 1)]


def test_the_bare_and_the_tokened_spelling_of_one_exposure_are_refused():
    """The case #766 opened: brick/F2550W after its next MIRI reduction."""
    paths = _names(BARE) + _names(ALIGN)
    with pytest.raises(MixedLineageFramesError) as excinfo:
        assert_one_lineage_per_exposure(paths, each_suffix='o002_crf')
    message = str(excinfo.value)
    assert '2 exposure(s)' in message
    assert 'align_o002_crf' in message and 'o002_crf' in message, (
        'both spellings are named, so the operator knows what to remove')
    assert 'each_suffix=' in message


def test_a_nircam_directory_with_both_lineages_is_refused_too():
    """Not a MIRI special case.  brick, cloudc, cloudef, sickle, w51 and wd2 all
    carry two NIRCam lineages; they are safe today only because their
    ``each_suffix`` is the tokened spelling, which no other lineage ends with.
    A run pointed at the bare one by ``--each-suffix-overrides`` gets the same
    refusal."""
    with pytest.raises(MixedLineageFramesError):
        assert_one_lineage_per_exposure(_names(DESTREAK) + _names(NIRCAM_BARE),
                                        each_suffix='o001_crf')


@pytest.mark.parametrize('template,suffix', [
    (BARE, 'o002_crf'),
    (ALIGN, 'o002_crf'),
    (DESTREAK, 'destreak_o001_crf'),
])
def test_one_lineage_per_exposure_passes(template, suffix):
    """What must NOT become a refusal.

    ``ALIGN`` alone is brick/F2550W as it stands today: the legacy spelling is
    the only one on disk, the bare suffix matches it through the wildcard, and
    that is how #766 took brick/F2550W from 0 candidate frames to 48.
    """
    assert_one_lineage_per_exposure(_names(template, 4), each_suffix=suffix)


def test_different_exposures_under_different_lineages_are_not_a_conflict():
    """The check is per EXPOSURE, not per directory: two spellings that never
    describe the same exposure are two sets of frames, not a double ingest."""
    assert_one_lineage_per_exposure(
        ['jw02221002001_02101_00001_mirimage_o002_crf.fits',
         'jw02221002001_02101_00002_mirimage_align_o002_crf.fits'])


def test_product_level_names_are_not_keyed_as_one_exposure():
    """brick/F2550W's 48 ``jw02221-o002_t001_miri_f2550w_<n>_o002_crf.fits``
    share their first four tokens, because the exposure number sits in a
    different position in a product-level name.  Keying them the way a
    per-exposure name is keyed would call a whole directory one duplicated
    exposure -- a refusal with nothing behind it."""
    product = ['jw02221-o002_t001_miri_f2550w_{}_o002_crf.fits'.format(i)
               for i in range(4)]
    assert lineages_per_exposure(product) == {}
    assert_one_lineage_per_exposure(product, each_suffix='o002_crf')


def test_a_joint_field_keeps_its_two_observations_apart():
    """sgrb2 5365 MIRI is catalogued as the joint field ``002-998`` and
    ``get_filenames`` rewrites the observation token per subfield, so one call
    returns both observations' frames.  The exposure key carries the
    observation (it is the first filename token), so obs 002's exposure 1 and
    obs 998's exposure 1 are two exposures, not one under two lineages."""
    assert_one_lineage_per_exposure(
        ['jw05365002001_02101_00001_mirimage_o002_crf.fits',
         'jw05365998001_02101_00001_mirimage_o998_crf.fits'])


def test_the_conflicting_paths_are_reported_by_exposure():
    got = lineages_per_exposure(_names(BARE, 1) + _names(ALIGN, 1))
    assert got == {
        'jw02221002001_02101_00001_mirimage': {
            'o002_crf': ['jw02221002001_02101_00001_mirimage_o002_crf.fits'],
            'align_o002_crf': [
                'jw02221002001_02101_00001_mirimage_align_o002_crf.fits'],
        }}


def test_the_globber_itself_refuses_a_two_lineage_directory(tmp_path):
    """End to end through the real glob rather than a restatement of it: the
    ambiguity is a property of the pattern ``get_filenames`` builds, so the
    guard is checked where that pattern is built.  Without it this call returns
    four files for two exposures."""
    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
        get_filenames)

    pipeline = tmp_path / 'F2550W' / 'pipeline'
    pipeline.mkdir(parents=True)
    for name in _names(BARE) + _names(ALIGN):
        (pipeline / name).touch()

    with pytest.raises(MixedLineageFramesError) as excinfo:
        get_filenames(str(tmp_path), 'F2550W', '2221', '002',
                      each_suffix='o002_crf', module='mirimage',
                      visitid='001')
    assert 'get_filenames F2550W/mirimage' in str(excinfo.value)


def test_the_globber_still_returns_a_single_lineage_directory(tmp_path):
    """brick/F2550W as it is on disk today: 48 ``align_o002_crf`` frames, no
    bare ones, and the bare suffix finds them through the wildcard."""
    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
        get_filenames)

    pipeline = tmp_path / 'F2550W' / 'pipeline'
    pipeline.mkdir(parents=True)
    for name in _names(ALIGN, 3):
        (pipeline / name).touch()

    found = get_filenames(str(tmp_path), 'F2550W', '2221', '002',
                          each_suffix='o002_crf', module='mirimage',
                          visitid='001')
    assert len(found) == 3
