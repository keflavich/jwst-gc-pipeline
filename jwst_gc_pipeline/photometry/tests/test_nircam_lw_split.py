"""NIRCam's channel split is a WAVELENGTH, not a substring of the filter name.

`stpsf_detector_for_module` decided long-wave with

    if 'F4' in filtername.upper() or 'F3' in filtername.upper():

which is right for F3xx/F4xx and wrong for the two LW filters whose names begin
'F2': F277W (2.77 um) and F250M (2.50 um).  Both mapped to a SHORT-wave
detector, and the PSF grid build then died inside stpsf:

    RuntimeError: The requested wavelengths are too long for NIRCam short wave
    channel.

Every other GC field's LW filters are F3xx/F4xx, which is why it stayed hidden.
gc2211 images F277W in all five observations, so its mergedcat residual / model
i2d build failed there (o050 finalize 38901731) -- and that residual is the next
phase's detection image, so the run fails closed rather than degrading m3+
(#159).  The documented override is not a way out for the same reason.
"""
import pytest

from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
    NIRCAM_SW_LW_SPLIT_UM, nircam_is_longwave, stpsf_detector_for_module)

#: Every NIRCam imaging filter, with the channel the dichroic puts it on.
SHORTWAVE = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F150W2', 'F162M',
             'F164N', 'F182M', 'F187N', 'F200W', 'F210M', 'F212N']
LONGWAVE = ['F250M', 'F277W', 'F300M', 'F322W2', 'F323N', 'F335M', 'F356W',
            'F360M', 'F405N', 'F410M', 'F430M', 'F444W', 'F460M', 'F466N',
            'F470N', 'F480M']


@pytest.mark.parametrize('filt', SHORTWAVE)
def test_shortwave_filters(filt):
    assert not nircam_is_longwave(filt), filt


@pytest.mark.parametrize('filt', LONGWAVE)
def test_longwave_filters(filt):
    assert nircam_is_longwave(filt), filt


@pytest.mark.parametrize('filt', ['F277W', 'F250M'])
def test_the_two_that_the_substring_rule_got_WRONG(filt):
    """The regression, named.  Both are LW and neither contains 'F3' or 'F4'."""
    assert 'F3' not in filt and 'F4' not in filt
    assert nircam_is_longwave(filt)
    assert stpsf_detector_for_module('nrcb', filt, 'NIRCAM') == 'NRCB5'
    assert stpsf_detector_for_module('nrca', filt, 'NIRCAM') == 'NRCA5'


def test_the_substring_rule_would_still_pass_on_everything_else():
    """Why it survived: for every other filter the old rule agrees, so a test
    written from the filters this campaign already ran could not see it."""
    for filt in SHORTWAVE + LONGWAVE:
        old = ('F4' in filt or 'F3' in filt)
        new = nircam_is_longwave(filt)
        if filt in ('F277W', 'F250M'):
            assert old != new, filt
        else:
            assert old == new, filt


def test_the_width_code_does_not_move_the_wavelength():
    """F150W2 is 1.50 um (SW) and F322W2 is 3.22 um (LW); the trailing '2' is a
    bandwidth code, not part of the wavelength."""
    assert not nircam_is_longwave('F150W2')
    assert nircam_is_longwave('F322W2')


def test_the_split_is_at_the_dichroic():
    assert NIRCAM_SW_LW_SPLIT_UM == 2.4
    assert not nircam_is_longwave('F212N')      # 2.12 um
    assert nircam_is_longwave('F250M')          # 2.50 um


@pytest.mark.parametrize('filt', ['', 'CLEAR', None, 'WLP4'])
def test_an_unparseable_filter_is_not_called_longwave(filt):
    """A name that carries no wavelength must not silently select the LW
    detector; SW is the pre-existing default for anything unrecognised."""
    assert not nircam_is_longwave(filt)


def test_the_physical_detector_paths_are_untouched():
    """Only the (nrca|nrcb) family branch consults the channel; a per-frame run
    passes the physical detector and must keep mapping directly."""
    assert stpsf_detector_for_module('nrcblong', 'F277W', 'NIRCAM') == 'NRCB5'
    assert stpsf_detector_for_module('nrca3', 'F200W', 'NIRCAM') == 'NRCA3'
    assert stpsf_detector_for_module('merged', 'F277W', 'NIRCAM') is None
    assert stpsf_detector_for_module('nrcb', 'F770W', 'MIRI') == 'MIRIM'
