"""``jw_prefix`` / ``proposal_id_from_filename`` -- the 5-digit-safe spellings.

The behavior pinned here is issue #414's fix: the filename prefix is the
proposal zero-padded to FIVE digits, exactly as MAST writes it.  For every
4-digit proposal the helper must reproduce the old ``f'jw0{proposal_id}'``
literal byte for byte (so the sweep changed no existing product name), and for
the first 5-digit proposal (10678, GC Treasury) it must yield ``jw10678``
where the old literal fabricated ``jw010678``.
"""
import pytest

from jwst_gc_pipeline.naming import jw_prefix, proposal_id_from_filename


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


def test_every_registry_proposal_keeps_its_historical_prefix():
    """The sweep is a no-op for every proposal the pipeline has processed:
    each fields.yaml proposal is 4-digit, so the padded form equals the old
    ``'jw0' + proposal`` spelling."""
    from jwst_gc_pipeline import fields
    proposals = {p for per in fields.obs_filters().values() for p in per}
    assert proposals, 'registry unexpectedly empty'
    for p in proposals:
        assert jw_prefix(p) == f'jw0{int(p)}', p


@pytest.mark.parametrize('bad', ['brick', '', None, -1, '-2221', 123456])
def test_non_numeric_negative_or_overwide_input_raises(bad):
    with pytest.raises(ValueError):
        jw_prefix(bad)


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
