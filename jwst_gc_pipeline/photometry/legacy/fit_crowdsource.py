"""LEGACY crowdsource result serialization -- BENCHMARKS ONLY.

Split out of ``legacy/photometry_step.py``.  ``save_crowdsource_results``
turns a crowdsource ``fit_im`` result tuple into the on-disk catalog
(``*_crowdsource_<suffix>.fits``) + sky/model image, mirroring
``save_photutils_results`` for the daophot side.  crowdsource remains a
genuine photometry backend; only the code LOCATION changed.

FROZEN benchmark path.  Reuses ``catalog_long`` names via the namespace copy
below, as ``photometry_step`` does.
"""
import jwst_gc_pipeline.photometry.catalog_long as _host

# Reproduce the exact module namespace the relocated code had while it lived
# in catalog_long (shared helpers + that module's imports), so every
# bare-name reference below resolves unchanged.
globals().update({_k: _v for _k, _v in vars(_host).items()
                  if not _k.startswith('__')})


def save_crowdsource_results(results, ww, filename, suffix,
                             im1, detector,
                             basepath, filtername, module, desat, bgsub, exposure_, visitid_, vgroupid_,
                             psf=None,
                             blur=False,
                             options=None,
                             fpsf="",
                             iteration_label=None):
    print("Saving crowdsource results.")
    blur_ = "_blur" if blur else ""

    stars, modsky, skymsky, psf_ = results
    stars = Table(stars)
    coords = ww.pixel_to_world(stars['y'], stars['x'])
    stars['skycoord'] = coords
    stars['x'], stars['y'] = stars['y'], stars['x']
    stars['dx'], stars['dy'] = stars['dy'], stars['dx']

    pixscale = (ww.proj_plane_pixel_area()**0.5).to(u.arcsec)
    stars['dra'] = stars['dx'] * pixscale
    stars['ddec'] = stars['dy'] * pixscale
    if visitid_ is not None and visitid_ != '':
        stars.meta['visit'] = int(visitid_[-3:])
    if vgroupid_ is not None and vgroupid_ != '':
        stars.meta['vgroup'] = vgroupid_.removeprefix('_vgroup')
    stars.meta['filename'] = filename
    stars.meta['filter'] = filtername
    stars.meta['module'] = module
    stars.meta['detector'] = detector
    stars.meta['pixscale'] = pixscale.to(u.deg).value
    stars.meta['pixscale_as'] = pixscale.to(u.arcsec).value
    stars.meta['proposal_id'] = options.proposal_id
    if exposure_:
        stars.meta['exposure'] = exposure_
    if iteration_label not in (None, ''):
        stars.meta['iteration'] = str(iteration_label)
    if visitid_:
        stars.meta['visit'] = int(visitid_[-3:])
    if vgroupid_:
        stars.meta['vgroup'] = vgroupid_.removeprefix('_vgroup')

    if 'RAOFFSET' in im1[0].header:
        stars.meta['RAOFFSET'] = im1[0].header['RAOFFSET']
        stars.meta['DEOFFSET'] = im1[0].header['DEOFFSET']
    elif 'RAOFFSET' in im1[1].header:
        stars.meta['RAOFFSET'] = im1[1].header['RAOFFSET']
        stars.meta['DEOFFSET'] = im1[1].header['DEOFFSET']

    iter_ = _iteration_token(iteration_label)
    tblfilename = (f"{basepath}/{filtername}/"
                   f"{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}{iter_}"
                   f"_crowdsource_{suffix}.fits")

    print(f"tblfilename={tblfilename}, filename={filename}, suffix={suffix}, filtername={filtername}, module={module}, desat={desat}, bgsub={bgsub}, fpsf={fpsf} blur={blur}")

    stars.write(tblfilename, overwrite=True)
    with fits.open(tblfilename, mode='update', output_verify='fix') as fh:
        fh[0].header.update(im1[1].header)
    skymskyhdu = fits.PrimaryHDU(data=skymsky, header=im1[1].header)
    modskyhdu = fits.ImageHDU(data=modsky, header=im1[1].header)
    hdul = fits.HDUList([skymskyhdu, modskyhdu])
    hdul.writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}{iter_}_crowdsource_skymodel_{suffix}.fits", overwrite=True)

    if psf is not None:
        if hasattr(psf, 'stamp'):
            psfhdu = fits.PrimaryHDU(data=psf.stamp)
            psf_fn = (f"{basepath}/{filtername}/"
                      f"{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}{iter_}"
                      f"_crowdsource_{suffix}_psf.fits")
            psfhdu.writeto(psf_fn, overwrite=True)
        else:
            raise ValueError(f"PSF did not have a stamp attribute.  It was: {psf}, type={type(psf)}")

    return stars
