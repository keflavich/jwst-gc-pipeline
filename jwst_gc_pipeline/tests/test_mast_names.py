"""The 5-digit-safe proposal spellings -- ``jw_prefix``,
``proposal_id_from_filename``, ``proposal_id_from_program``.

The behavior pinned here is issue #414's fix: the filename prefix is the
proposal zero-padded to FIVE digits, exactly as MAST writes it.  For every
4-digit proposal the helper must reproduce the old ``f'jw0{proposal_id}'``
literal byte for byte (so the sweep changed no existing product name), and for
a 5-digit proposal -- 10678 (GC Treasury) and omegacen's 12587, which was
already in the registry -- it must yield ``jw10678`` where the old literal
fabricated ``jw010678``.  ``PROGRAM`` is the same padded form in the header,
so reading a proposal out of it is a de-pad, where the reduction path used the
4-digit-only slice ``PROGRAM[1:5]``.
"""
import pytest

from jwst_gc_pipeline.mast_names import (jw_prefix, proposal_id_from_filename,
                                     proposal_id_from_program)


# ---------------------------------------------------------------------------
# jw_prefix
# ---------------------------------------------------------------------------

def test_four_digit_proposal_matches_the_old_spelling():
    """``jw_prefix(2221)`` is byte-identical to the old ``f'jw0{2221}'``."""
    assert jw_prefix(2221) == 'jw02221'
    assert jw_prefix(2221) == f'jw0{2221}'


def test_five_digit_proposal_is_not_extra_padded():
    assert jw_prefix(10678) == 'jw10678'
    assert jw_prefix('10678') == 'jw10678'
    # the defect this replaces: f'jw0{10678}' == 'jw010678'
    assert jw_prefix(10678) != f'jw0{10678}'


def test_string_and_padded_string_inputs():
    assert jw_prefix('2221') == 'jw02221'
    assert jw_prefix('02221') == 'jw02221'


def test_sub_1000_proposal_pads_to_five_digits():
    """MAST pads to five whatever the width, so a 3-digit proposal is
    ``jw00618`` -- where the old literal wrote ``jw0618``.  None is registered
    here; pinned so the divergence is documented up front."""
    assert jw_prefix(618) == 'jw00618'
    assert jw_prefix(618) != f'jw0{618}'


def _registry_proposals():
    """EVERY proposal in ``fields.yaml``, including the ones the narrower views
    drop.

    ``obs_filters()`` only reports observations that declare a ``filters:``
    key, and omegacen's two (8322 and the FIVE-digit 12587) declare none -- so
    an earlier version of the registry pin below never saw a 5-digit proposal
    and could not have caught a prefix that was wrong for one.
    """
    from jwst_gc_pipeline import fields
    return {o.proposal for f in fields.FIELDS for o in f.observations}


def test_the_registry_enumeration_is_the_complete_one():
    """The pin below is only worth the set it iterates: assert the enumeration
    is a superset of both narrower registry views."""
    from jwst_gc_pipeline import fields
    everything = _registry_proposals()
    assert everything, 'registry unexpectedly empty'
    from_filters = {p for per in fields.obs_filters().values() for p in per}
    from_obsnum = {p for per in fields.project_obsnum().values() for p in per}
    assert from_filters <= everything
    assert from_obsnum <= everything


def test_every_registry_proposal_gets_its_mast_prefix():
    """Each registry proposal gets the 5-digit-padded MAST prefix, and each
    4-digit one additionally reproduces the old ``'jw0' + proposal`` spelling
    byte for byte -- that equality is what makes the sweep a no-op for the
    products already on disk.

    The equality is asserted for 1000-9999 only.  A 5-digit proposal (10678,
    and omegacen's 12587 which predates it in the registry) is exactly where
    the old spelling was wrong, and a sub-1000 proposal would pad to five
    digits (``jw00618``) where the old literal wrote ``jw0618`` -- MAST writes
    the padded form in both cases.
    """
    for p in _registry_proposals():
        pid = int(p)
        assert jw_prefix(p) == f'jw{pid:05d}', p
        if 1000 <= pid <= 9999:
            assert jw_prefix(p) == f'jw0{pid}', p
        else:
            assert jw_prefix(p) != f'jw0{pid}', p


@pytest.mark.parametrize('bad', ['brick', '', None, -1, '-2221', 123456])
def test_non_numeric_negative_or_overwide_input_raises(bad):
    with pytest.raises(ValueError):
        jw_prefix(bad)


@pytest.mark.parametrize('bad', [
    0,            # no proposal 0; the old code built 'jw00000'
    '0',
    '2_221',      # int() reads this as 2221 and the typo survives to a glob
    '+2221',
    ' 2221 ',     # a stray space in a shell wrapper's argument
    '2221\n',
    '٢٢٢١',   # Arabic-Indic digits: int() accepts these
    2221.0,       # str(2221.0) == '2221.0'
    b'2221',
])
def test_input_that_int_would_have_normalized_is_refused(bad):
    """``int()`` accepts more shapes than a proposal number has.  Each of these
    used to produce a plausible-looking prefix from an input that is not a
    proposal, which is how a typo reaches a glob and reads as 'no data'."""
    with pytest.raises(ValueError):
        jw_prefix(bad)


# ---------------------------------------------------------------------------
# proposal_id_from_program -- the header spelling of the same defect
# ---------------------------------------------------------------------------

def test_program_header_de_pads_a_four_digit_proposal():
    """PROGRAM is the padded form on every real frame; the value the pipeline
    keys on is unpadded.  Matches the old ``PROGRAM[1:5]`` slice exactly."""
    assert proposal_id_from_program('02221') == '2221'
    assert proposal_id_from_program('02221') == '02221'[1:5]
    assert proposal_id_from_program('01182') == '1182'


def test_program_header_keeps_all_five_digits_of_a_treasury_frame():
    assert proposal_id_from_program('10678') == '10678'
    # the defect this replaces: '10678'[1:5] == '0678'
    assert proposal_id_from_program('10678') != '10678'[1:5]
    assert proposal_id_from_program('12587') == '12587'


def test_program_header_tolerates_the_padding_fits_adds():
    assert proposal_id_from_program(' 10678 ') == '10678'
    assert proposal_id_from_program(2221) == '2221'


@pytest.mark.parametrize('bad', ['', 'nircam', None, '123456'])
def test_a_program_value_that_is_not_a_proposal_raises(bad):
    with pytest.raises(ValueError):
        proposal_id_from_program(bad)


def test_program_agrees_with_the_filename_on_a_real_frame():
    """The two parses read the same proposal off the same product."""
    fn = 'jw01182004001_02101_00001_nrca1_cal.fits'
    assert proposal_id_from_filename(fn) == proposal_id_from_program('01182')


# ---------------------------------------------------------------------------
# proposal_id_from_filename
# ---------------------------------------------------------------------------

def test_parse_matches_the_old_slice_on_4_digit_products():
    fn = 'jw02221001001_02101_00001_nrcb1_cal.fits'
    assert proposal_id_from_filename(fn) == '2221'
    assert proposal_id_from_filename(fn) == fn[3:7]   # the slice it replaces


def test_parse_reads_all_five_digits_of_a_treasury_product():
    fn = '/some/path/jw10678001001_02101_00001_nrca1_uncal.fits'
    assert proposal_id_from_filename(fn) == '10678'
    # the defect this replaces: basename[3:7] == '0678'


def test_parse_refuses_a_non_jw_basename():
    for bad in ('f182m_nrcb1_cal.fits', 'jw123_cal.fits', 'jwabcde_x.fits'):
        with pytest.raises(ValueError):
            proposal_id_from_filename(bad)


# ---------------------------------------------------------------------------
# filtername_from_header
#
# JWST splits one bandpass over two wheels and which wheel holds it varies, so
# neither keyword alone is the answer.  `reduction.destreak.add_background_map`
# carried its own copy of the rule that only knew six hardcoded narrow/medium
# bands; every CLEAR-pupil WIDE band fell through it to the literal 'CLEAR',
# which is why the f200w/f356w/f444w entries of `destreak.background_mapping`
# were unreachable.
# ---------------------------------------------------------------------------

from jwst_gc_pipeline.mast_names import filtername_from_header


@pytest.mark.parametrize('filt, pupil, expected', [
    # filter-wheel band, empty pupil -- NIRCam narrow/medium/wide alike
    ('F212N', 'CLEAR', 'F212N'),
    ('F182M', 'CLEAR', 'F182M'),
    ('F187N', 'CLEAR', 'F187N'),
    ('F410M', 'CLEAR', 'F410M'),
    ('F210M', 'CLEAR', 'F210M'),
    ('F360M', 'CLEAR', 'F360M'),
    ('F480M', 'CLEAR', 'F480M'),
    # THE REGRESSION: CLEAR-pupil WIDE bands.  The destreak copy returned
    # 'CLEAR' for every one of these.
    ('F115W', 'CLEAR', 'F115W'),
    ('F200W', 'CLEAR', 'F200W'),
    ('F356W', 'CLEAR', 'F356W'),
    ('F444W', 'CLEAR', 'F444W'),
    # NIRCam pupil-wheel bands parked behind a wide blocker: the pupil wins
    ('F444W', 'F405N', 'F405N'),
    ('F444W', 'F466N', 'F466N'),
    ('F444W', 'F470N', 'F470N'),
    ('F150W2', 'F162M', 'F162M'),
    ('F150W2', 'F164N', 'F164N'),
    ('F322W2', 'F323N', 'F323N'),
    # NIRISS spells its empty pupil CLEARP
    ('F480M', 'CLEARP', 'F480M'),
    ('F200W', 'CLEARP', 'F200W'),
])
def test_filtername_resolves_both_wheel_conventions(filt, pupil, expected):
    assert filtername_from_header({'FILTER': filt, 'PUPIL': pupil}) == expected


def test_a_header_with_no_pupil_keyword_uses_filter():
    """MIRI carries no PUPIL."""
    assert filtername_from_header({'FILTER': 'F2550W'}) == 'F2550W'


def test_both_wheels_empty_is_refused():
    for bad in ({'FILTER': 'CLEAR', 'PUPIL': 'CLEAR'},
                {'FILTER': 'CLEAR', 'PUPIL': 'CLEARP'},
                {'FILTER': '', 'PUPIL': ''},
                {}):
        with pytest.raises(ValueError):
            filtername_from_header(bad)


def test_the_wide_bands_would_have_matched_their_background_map_keys():
    """The concrete consequence: `background_mapping['2221']['001']` holds
    'f200w'/'f356w'/'f444w' keys that the old rule could never produce."""
    from jwst_gc_pipeline.reduction.destreak import background_mapping
    keys = background_mapping['2221']['001']
    for filt in ('F200W', 'F356W', 'F444W'):
        assert filt.lower() in keys
        resolved = filtername_from_header({'FILTER': filt, 'PUPIL': 'CLEAR'}).lower()
        assert resolved in keys, (
            f'{filt} resolves to {resolved!r}, which is not a mapping key')


def test_filtering_get_filtername_delegates_here():
    """One rule, one spelling.  `reduction.filtering.get_filtername` is the
    older public name and must not drift from it."""
    from jwst_gc_pipeline.reduction.filtering import get_filtername
    for filt, pupil in (('F212N', 'CLEAR'), ('F444W', 'F405N'),
                        ('F150W2', 'F162M'), ('F480M', 'CLEARP'),
                        ('F444W', 'CLEAR')):
        hdr = {'FILTER': filt, 'PUPIL': pupil}
        assert get_filtername(hdr) == filtername_from_header(hdr)
