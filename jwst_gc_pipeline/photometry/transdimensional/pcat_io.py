"""Write PCAT results in the experiment's standard catalog format.

``catalog.fits`` (one row per condensed source; header TOOL, CONFIG, FRAME,
PSF, NSRC, RUNTIME_S) with columns ``x, y`` (0-based full-frame pixels), ``ra,
dec`` (frame_wcs of FRAME; NaN without a WCS), ``flux`` (posterior median,
image units x pix, total flux), ``flux_err`` (posterior std), ``x_err,
y_err`` (posterior std), ``in_box``, ``prob`` (prevalence) and ``qual``
(posterior flux S/N = flux / flux_err), plus PCAT extras (``prob_cal``, the
prevalence mapped through a calibration ``prob -> fraction of real sources``,
NaN when none is given; ``x_P, y_P``, the position in the box frame P used for
``in_box``; ``flux_p16``, ``flux_p84``, ``flux_mean``, ``nmatch``,
``degree``, ``nmerge`` after :func:`.condense.merge_bright`, ``ring_flag`` /
``ring_parent`` after :func:`.rings.flag_psf_rings`, per-band
fluxes).
``samples.npz`` holds the chain; ``model.fits`` / ``residual.fits`` the
posterior median (and mean) model and residual on the cutout grid;
``run.json`` the configuration, timings, acceptance and versions.
"""
import json
import platform
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

__all__ = ['in_box', 'write_catalog', 'write_samples', 'write_images', 'write_run_json',
           'write_all', 'load_samples', 'apply_prob_cal', 'load_prob_cal']

# EVAL box of the experiment in P full-frame pixels (CONTEXT.md)
BOX_X = (64.5, 224.5)
BOX_Y = (765.5, 925.5)


def in_box(x, y, box_x=BOX_X, box_y=BOX_Y):
    x = np.asarray(x)
    y = np.asarray(y)
    return (x >= box_x[0]) & (x < box_x[1]) & (y >= box_y[0]) & (y < box_y[1])


def load_prob_cal(path):
    """Read a prevalence calibration JSON (``{'knots': {'prob': [...],
    'p_real': [...]}, ...}``) and return the dict."""
    d = json.loads(Path(path).read_text())
    if 'knots' not in d or not {'prob', 'p_real'} <= set(d['knots']):
        raise ValueError(f'{path}: no knots.prob / knots.p_real')
    d['path'] = str(path)
    return d


def apply_prob_cal(prob, cal):
    """Calibrated prevalence: piecewise-linear interpolation of the monotone
    calibration knots (``cal['knots']['prob'] -> cal['knots']['p_real']``),
    constant beyond the end knots.  ``cal=None`` gives NaN."""
    prob = np.asarray(prob, float)
    if cal is None:
        return np.full(prob.shape, np.nan)
    kx = np.asarray(cal['knots']['prob'], float)
    ky = np.asarray(cal['knots']['p_real'], float)
    if kx.size == 0 or np.any(np.diff(kx) < 0) or np.any(np.diff(ky) < -1e-12):
        raise ValueError('calibration knots must be non-decreasing in prob and p_real')
    return np.clip(np.interp(prob, kx, ky), 0.0, 1.0)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


def write_catalog(path, cond, *, x0=0, y0=0, config='', frame='', psf='', runtime_s=np.nan,
                  wcs=None, band_names=None, box=True, tool='pcat', box_wcs=None, prob_cal=None):
    """Write a condensed catalog (``condense`` output, cutout pixel coordinates).

    ``x0, y0`` convert cutout to full-frame pixels; ``wcs`` (frame_wcs of FRAME)
    gives ``ra, dec``.  ``box=False`` sets ``in_box`` True everywhere (mocks).
    ``box_wcs`` (frame_wcs of P): ``in_box`` maps ``ra, dec`` through it to P
    pixels (the experiment's rule for every frame); without it the full-frame
    ``x, y`` are taken as P pixels (correct for P and its synthetic copies).
    ``prob_cal``: calibration dict (:func:`load_prob_cal`) for ``prob_cal``.
    """
    x = np.asarray(cond['x'], float) + x0
    y = np.asarray(cond['y'], float) + y0
    n = x.size
    if wcs is not None and n:
        sc = wcs.pixel_to_world(x, y)
        ra, dec = sc.ra.deg, sc.dec.deg
    else:
        ra = dec = np.full(n, np.nan)
    if box_wcs is not None and wcs is not None and n:
        xp, yp = box_wcs.world_to_pixel(sc)
        xp, yp = np.asarray(xp, float), np.asarray(yp, float)
    else:
        xp, yp = x, y
    flux = np.asarray(cond['flux'])
    ferr = np.asarray(cond['flux_err'])
    with np.errstate(divide='ignore', invalid='ignore'):
        qual = flux[0] / ferr[0] if n else np.zeros(0)
    t = Table()
    t['x'] = x
    t['y'] = y
    t['ra'] = ra
    t['dec'] = dec
    t['flux'] = flux[0] if n else np.zeros(0)
    t['flux_err'] = ferr[0] if n else np.zeros(0)
    t['x_err'] = np.asarray(cond['x_err'], float)
    t['y_err'] = np.asarray(cond['y_err'], float)
    t['in_box'] = in_box(xp, yp) if box else np.ones(n, bool)
    t['prob'] = np.asarray(cond['prob'], float)
    t['qual'] = qual
    t['prob_cal'] = apply_prob_cal(t['prob'], prob_cal)
    t['x_P'] = xp
    t['y_P'] = yp
    t['x_med'] = np.asarray(cond['x_med'], float) + x0
    t['y_med'] = np.asarray(cond['y_med'], float) + y0
    t['flux_mean'] = np.asarray(cond['flux_mean'])[0] if n else np.zeros(0)
    t['flux_p16'] = np.asarray(cond['flux_p16'])[0] if n else np.zeros(0)
    t['flux_p84'] = np.asarray(cond['flux_p84'])[0] if n else np.zeros(0)
    t['nmatch'] = np.asarray(cond['nmatch'], int)
    t['degree'] = np.asarray(cond['degree'], int)
    if 'nmerge' in cond:
        t['nmerge'] = np.asarray(cond['nmerge'], int)
    if 'ring_flag' in cond:
        # rings.flag_psf_rings: on a PSF ring of a brighter source (diagnostic flag; rows are kept)
        t['ring_flag'] = np.asarray(cond['ring_flag'], bool)
        t['ring_parent'] = np.asarray(cond['ring_parent'], int)
    nb = flux.shape[0]
    if nb > 1:
        names = band_names or [f'b{b}' for b in range(nb)]
        for b in range(1, nb):
            t[f'flux_{names[b]}'] = flux[b]
            t[f'flux_err_{names[b]}'] = ferr[b]
    for c, unit in (('x', 'pix'), ('y', 'pix'), ('ra', 'deg'), ('dec', 'deg'),
                    ('x_err', 'pix'), ('y_err', 'pix'), ('flux', 'MJy/sr pix'),
                    ('flux_err', 'MJy/sr pix')):
        t[c].unit = unit
    t.meta.update(TOOL=tool, CONFIG=str(config)[:68], FRAME=str(frame)[:68], PSF=Path(str(psf)).name[:68],
                  NSRC=int(n), RUNTIME_S=float(runtime_s), NSAMPCND=int(cond['nsamp_used']),
                  NSAMPSED=int(cond.get('nsamp_seed', cond['nsamp_used'])),
                  PROBCAL=('none' if prob_cal is None else Path(str(prob_cal.get('path', 'given'))).name)[:68],
                  BOXRULE='radec->P pix' if box_wcs is not None and wcs is not None else 'x,y as P pix',
                  COMMENT='prob=prevalence; prob_cal=calibrated prevalence; flux=posterior median; '
                          '*_err=posterior std; qual=flux/flux_err')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import warnings
    from astropy.io.fits.verify import VerifyWarning
    with warnings.catch_warnings():
        # RUNTIME_S / NSAMPCND exceed 8 characters -> HIERARCH cards (intended)
        warnings.simplefilter('ignore', VerifyWarning)
        t.write(path, overwrite=True)
    return t


def write_samples(path, res):
    """Save the chain (``PCATResult.samples`` + acceptance) to ``.npz``."""
    s = res.samples
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in s.items()},
                        accept=json.dumps(_jsonable(res.accept)),
                        nburn=res.info.get('nburn', 0))


def load_samples(path):
    with np.load(path, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    out['accept'] = json.loads(str(out['accept']))
    return out


def write_images(outdir, res, band=0, header=None, primary='median'):
    """``model.fits`` and ``residual.fits`` (data - model).

    ``primary='median'`` (default): primary HDU = posterior median when stored
    (else the mean), extension MEAN = posterior mean.  ``primary='mean'``:
    primary = posterior mean, extension MEDIAN when stored.
    """
    if primary not in ('median', 'mean'):
        raise ValueError(f'primary {primary!r}')
    outdir = Path(outdir)
    hdr = fits.Header() if header is None else header.copy()
    hdr['BUNIT'] = 'MJy/sr'
    med = res.model_median[band]
    mean = res.model_mean[band]
    rmed = res.residual_median[band]
    rmean = res.residual_mean[band]
    if primary == 'mean':
        hm = [fits.PrimaryHDU(np.asarray(mean, np.float32), header=hdr)]
        hr = [fits.PrimaryHDU(np.asarray(rmean, np.float32), header=hdr)]
        if med is not None:
            hm.append(fits.ImageHDU(np.asarray(med, np.float32), name='MEDIAN'))
            hr.append(fits.ImageHDU(np.asarray(rmed, np.float32), name='MEDIAN'))
        hm, hr = fits.HDUList(hm), fits.HDUList(hr)
        hm[0].header['MODEL'] = 'posterior mean'
        hr[0].header['RESID'] = 'data - posterior mean model'
    else:
        prim = med if med is not None else mean
        hm = fits.HDUList([fits.PrimaryHDU(np.asarray(prim, np.float32), header=hdr),
                           fits.ImageHDU(np.asarray(mean, np.float32), name='MEAN')])
        hm[0].header['MODEL'] = 'posterior median' if med is not None else 'posterior mean'
        prim = rmed if rmed is not None else rmean
        hr = fits.HDUList([fits.PrimaryHDU(np.asarray(prim, np.float32), header=hdr),
                           fits.ImageHDU(np.asarray(rmean, np.float32), name='MEAN')])
        hr[0].header['RESID'] = ('data - posterior median model' if rmed is not None
                                 else 'data - posterior mean model')
    hm.writeto(outdir / 'model.fits', overwrite=True)
    hr.writeto(outdir / 'residual.fits', overwrite=True)


def _versions():
    import numpy
    import scipy
    import astropy
    v = dict(python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__,
             astropy=astropy.__version__, host=platform.node())
    try:
        import photutils
        v['photutils'] = photutils.__version__
    except ImportError:
        pass
    return v


def write_run_json(path, res, extra=None):
    d = dict(config=res.config, info=res.info, accept=res.accept, versions=_versions())
    if extra:
        d.update(extra)
    Path(path).write_text(json.dumps(_jsonable(d), indent=1))


def write_all(outdir, res, cond, *, x0=0, y0=0, config='', frame='', psf='', wcs=None,
              header=None, box=True, extra=None):
    """Write catalog.fits, samples.npz, model.fits, residual.fits and run.json."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t = write_catalog(outdir / 'catalog.fits', cond, x0=x0, y0=y0, config=config, frame=frame,
                      psf=psf, runtime_s=res.info['runtime_s'], wcs=wcs,
                      band_names=res.info.get('band_names'), box=box)
    write_samples(outdir / 'samples.npz', res)
    write_images(outdir, res, header=header)
    write_run_json(outdir / 'run.json', res, extra=extra)
    return t
