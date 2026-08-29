"""F150W2's bandpass overruns stpsf's SW channel limit at the default sampling.

stpsf validates the SAMPLED wavelengths of a bandpass against the channel
limits.  ``SHORT_WAVELENGTH_MAX`` is 2.3500 um and F150W2 samples out to
2.3626 um at the default 40 points, so ONE sample of the forty kills the whole
grid build.  It only ever surfaces on a COLD psf cache, which is why m4 and
ngc6397 -- the two fields whose only SW band is F150W2 -- stayed
reduced-but-uncataloged instead of failing loudly.

These tests use a stub rather than stpsf so they stay fast and hermetic; the
real numbers below are measured from stpsf 2.2.0 / webbpsf.
"""
import pytest

from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as _L


class _StubNIRCam:
    """Minimal stand-in reproducing stpsf's sampling behaviour.

    ``_get_weights`` narrows toward the band centre as ``nlambda`` falls, which
    is what makes a smaller sampling fit inside the channel.
    """
    SHORT_WAVELENGTH_MAX = 2.35e-6
    LONG_WAVELENGTH_MIN = 2.35e-6

    def __init__(self, filt, channel, lo_um, hi_um):
        self.filter, self.channel = filt, channel
        self._lo, self._hi = lo_um * 1e-6, hi_um * 1e-6

    def _get_weights(self, nlambda=40):
        import numpy as np
        # Shrink the sampled span as nlambda drops, mirroring stpsf.
        centre = 0.5 * (self._lo + self._hi)
        half = 0.5 * (self._hi - self._lo) * (nlambda / 40.0) ** 0.08
        return (np.linspace(centre - half, centre + half, max(nlambda, 2)),)


@pytest.fixture(autouse=True)
def _stub_isinstance(monkeypatch):
    """`nircam_channel_safe_psf_kwargs` guards on isinstance(nrc, NIRCam)."""
    monkeypatch.setattr(_L.webbpsf, 'NIRCam', _StubNIRCam, raising=False)


def test_f150w2_short_channel_gets_reduced_sampling():
    """The filter that overruns gets an nlambda, and it is the LARGEST that fits."""
    nrc = _StubNIRCam('F150W2', 'short', 1.0065, 2.3626)
    kw = _L.nircam_channel_safe_psf_kwargs(nrc)
    assert 'nlambda' in kw, "F150W2 must be given a reduced sampling"
    nl = kw['nlambda']
    assert nrc._get_weights(nlambda=nl)[0].max() <= nrc.SHORT_WAVELENGTH_MAX
    # Largest that fits: one more must NOT fit, or we reduced further than needed.
    assert nrc._get_weights(nlambda=nl + 1)[0].max() > nrc.SHORT_WAVELENGTH_MAX


@pytest.mark.parametrize('filt,channel,lo,hi', [
    ('F200W', 'short', 1.753, 2.231),
    ('F115W', 'short', 1.013, 1.283),
    ('F322W2', 'long', 2.427, 4.043),
    ('F444W', 'long', 3.881, 4.982),
])
def test_filters_already_inside_the_channel_are_untouched(filt, channel, lo, hi):
    """A filter that fits gets {} -- no PSF that builds today changes."""
    nrc = _StubNIRCam(filt, channel, lo, hi)
    assert _L.nircam_channel_safe_psf_kwargs(nrc) == {}


def test_impossible_pairing_raises_rather_than_silently_degrading():
    """An SW detector on a genuinely LW bandpass cannot be rescued by sampling.

    That combination means the filter/detector mapping is wrong upstream, so it
    must raise instead of shrinking nlambda toward nothing.
    """
    nrc = _StubNIRCam('F444W', 'short', 3.881, 4.982)
    with pytest.raises(ValueError, match='no nlambda'):
        _L.nircam_channel_safe_psf_kwargs(nrc)


def test_non_nircam_instrument_is_untouched():
    """MIRI/NIRISS have no dichroic; the helper must not touch them."""
    assert _L.nircam_channel_safe_psf_kwargs(object()) == {}
