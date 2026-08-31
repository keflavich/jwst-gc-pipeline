"""Keep a NIRCam PSF grid build inside the channel its filter belongs to.

Lives outside ``crowdsource_catalogs_long`` because BOTH the photometry monolith
and ``reduction.saturated_star_finding`` build PSF grids, and the monolith
already imports from that module -- so hosting the helper there would be a
cycle.  New code goes in a new module rather than the monolith anyway.
"""
import stpsf as webbpsf


def nircam_channel_safe_psf_kwargs(nrc, nlambda_default=40):
    """``psf_grid`` kwargs that keep every sampled wavelength inside the channel.

    stpsf validates the SAMPLED wavelengths of a bandpass against the channel
    limits and raises if any one of them falls outside:

        RuntimeError: The requested wavelengths are too long for NIRCam short
        wave channel.

    ``SHORT_WAVELENGTH_MAX`` is 2.3500 um.  **F150W2**, the widest SW filter,
    samples 1.0065-2.3626 um at stpsf's default 40 points -- so exactly ONE of
    the forty sits 12.6 nm past the limit and the whole grid build dies.  Every
    other SW filter this campaign uses stops well short of 2.35 (F200W reaches
    2.231), which is why nothing hit this until m4 and ngc6397 -- the two fields
    whose only SW band is F150W2 -- were cataloged for the first time with a
    COLD psf cache.  A warm cache never reaches the build, so the fields sat
    reduced-but-uncataloged rather than failing visibly.

    Reducing ``nlambda`` moves the sample points inward.  Measured on F150W2:

        nlambda   reddest sample   verdict
             9        2.3027 um    ok
            15        2.3336 um    ok
            20        2.3452 um    ok
            25        2.3522 um    RAISES
            40        2.3626 um    RAISES

    So this returns the LARGEST sampling that stays inside the channel, rather
    than a constant: spectral sampling is only reduced on the filters that
    require it, and by the least amount that works.  A filter that already fits
    gets ``{}`` and its grid is built exactly as before -- this changes no PSF
    that builds today.
    """
    if not isinstance(nrc, webbpsf.NIRCam):
        return {}
    try:
        lam = nrc._get_weights(nlambda=nlambda_default)[0]
    except (AttributeError, KeyError, ValueError):
        # Not a bandpass we can sample ahead of time; let psf_grid decide.
        return {}
    if nrc.channel == 'short':
        limit, inside = nrc.SHORT_WAVELENGTH_MAX, lambda a: a.max() <= limit
    elif nrc.channel == 'long':
        limit, inside = nrc.LONG_WAVELENGTH_MIN, lambda a: a.min() >= limit
    else:
        return {}
    if inside(lam):
        return {}
    for nl in range(nlambda_default - 1, 4, -1):
        if inside(nrc._get_weights(nlambda=nl)[0]):
            print(f"PSF: {nrc.filter} on the {nrc.channel} channel oversteps "
                  f"its limit ({limit * 1e6:.4f} um) at nlambda={nlambda_default}; "
                  f"building with nlambda={nl}", flush=True)
            return {'nlambda': nl}
    raise ValueError(
        f"no nlambda between 5 and {nlambda_default} keeps {nrc.filter} inside "
        f"the NIRCam {nrc.channel} channel (limit {limit * 1e6:.4f} um). The "
        f"filter/detector pairing is probably wrong -- check "
        f"nircam_is_longwave() and stpsf_detector_for_module().")
