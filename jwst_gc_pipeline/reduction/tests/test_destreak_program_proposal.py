"""destreak reads its proposal from PROGRAM, which MAST pads to five digits.

``background_mapping`` is keyed on the UNPADDED proposal (``'2221'``).  The
four-character PROGRAM slice these two sites used returns that key for a
4-digit proposal and drops the fifth digit of a 5-digit one -- ``'0678'`` for
the GC Treasury program 10678, ``'2587'`` for omegacen's 12587.  The miss is
silent: ``add_background_map`` prints a warning about a filter that is "not in
background mapping" and returns the frame with no background added, so the
reduce carries on and writes a degraded destreak (issue #414).

Both tests below are value tests on the live path.  They point the mapping at
a background file that does not exist, so reaching the lookup raises
``FileNotFoundError`` and failing the lookup returns quietly -- the two
outcomes are distinguishable without a 2048x2048 background map on disk.
"""
import numpy as np
import pytest
from astropy.io import fits

from jwst_gc_pipeline.reduction import destreak as D


def _frame(program, obs='001', visit='001', filtername='F212N'):
    """A minimal cal-shaped HDUList: primary header + a SCI extension."""
    primary = fits.PrimaryHDU()
    primary.header['PROGRAM'] = program
    primary.header['OBSERVTN'] = obs
    primary.header['VISIT'] = visit
    primary.header['PUPIL'] = 'CLEAR'
    primary.header['FILTER'] = filtername
    sci = fits.ImageHDU(np.zeros((8, 8), dtype='float32'), name='SCI')
    sci.header['CTYPE1'] = 'RA---TAN'
    sci.header['CTYPE2'] = 'DEC--TAN'
    sci.header['CRVAL1'] = 266.5
    sci.header['CRVAL2'] = -28.7
    sci.header['CRPIX1'] = 4
    sci.header['CRPIX2'] = 4
    sci.header['CDELT1'] = -1.75e-5
    sci.header['CDELT2'] = 1.75e-5
    return fits.HDUList([primary, sci])


def _mapping(proposal, obs='001', regionname='gc-treasury'):
    return {proposal: {obs: {'regionname': regionname,
                             'f212n': 'no_such_background_map.fits'}}}


@pytest.mark.parametrize('program, proposal', [
    ('02221', '2221'),    # every frame on disk today
    ('10678', '10678'),   # GC Treasury
    ('12587', '12587'),   # omegacen, registered before 10678
])
def test_add_background_map_finds_the_mapping_for_its_proposal(tmp_path,
                                                               program,
                                                               proposal):
    """Reaching the lookup means reaching the background file it names."""
    data = np.zeros((8, 8), dtype='float32')
    with pytest.raises(FileNotFoundError):
        D.add_background_map(data, _frame(program),
                             background_mapping=_mapping(proposal),
                             bgmap_path=str(tmp_path))


def test_add_background_map_reports_a_proposal_it_has_no_map_for(tmp_path,
                                                                 capsys):
    """The quiet path stays quiet-but-visible: a proposal with no entry warns
    and returns the data.  This is what a 5-digit frame used to get."""
    data = np.zeros((8, 8), dtype='float32')
    out = D.add_background_map(data, _frame('10678'),
                               background_mapping=_mapping('9999'),
                               bgmap_path=str(tmp_path))
    assert out is data
    assert 'not in background mapping' in capsys.readouterr().out


def test_destreak_reaches_the_background_map_for_a_five_digit_proposal(
        tmp_path, monkeypatch):
    """The second site, in ``destreak()`` itself, which decides whether to call
    ``add_background_map`` at all.  ``destreak_data`` is stubbed out so the
    frame can be 8x8 rather than a full 2048x2048 detector."""
    monkeypatch.setattr(D, 'destreak_data',
                        lambda data, **kwargs: data)
    frame = tmp_path / 'jw10678001001_02101_00001_nrca1_cal.fits'
    _frame('10678').writeto(frame)
    with pytest.raises(FileNotFoundError):
        D.destreak(str(frame), use_background_map=True,
                   background_mapping=_mapping('10678'))


def test_destreak_skips_the_background_map_when_the_proposal_is_unmapped(
        tmp_path, monkeypatch):
    """The converse at the same site: an unmapped proposal is skipped, and the
    destreak completes.  With the old slice this is what 10678 got."""
    monkeypatch.setattr(D, 'destreak_data',
                        lambda data, **kwargs: data)
    frame = tmp_path / 'jw10678001001_02101_00001_nrca1_cal.fits'
    _frame('10678').writeto(frame)
    D.destreak(str(frame), use_background_map=True,
               background_mapping=_mapping('9999'))
    assert (tmp_path / 'jw10678001001_02101_00001_nrca1_destreak.fits').exists()
