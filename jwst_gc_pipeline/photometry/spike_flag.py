"""Catalog-level flag for diffraction-spike artifacts in m8 catalogs (#1035).

Saturated stars leave diffraction spikes whose flux beyond the satstar stamp
(+-81 px) is picked up by the per-frame iteration-1 DAOStarFinder as beaded
chains of faint, mostly single-band sources.  This module marks (never removes)
such rows with two criteria:

``spike_wedge``
    a row with few real bands lying inside a narrow wedge around a spike
    position angle of a bright multi-band parent, out to a length that scales
    with the parent brightness.
``single_band_crowd``
    a single-band row with many other single-band rows within a small radius
    (the chains are dense in single-band detections).

Spike sky position angles are ``PA_V3 + 60k`` deg (k=0..5), measured in wd2 as
20.5 + 60k deg for PA_V3 = 140.82.  A weak strut pair sits at ``PA_V3 +- 90``.
"""
import glob
import os

import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

from jwst_gc_pipeline.atomic_io import write_table_atomic
from jwst_gc_pipeline.photometry.dedup_catalog import _arr, real_bands

#: Spike pattern offset measured from wd2 data: spikes at PA_V3 + 60k.
_MAX_FILES_PER_BAND = 50
_ARCSEC = 3600.0


def spike_position_angles(pa_v3_values, struts=False):
    """Sorted unique sky PAs (deg E of N, in [0, 360)) of the spikes.

    Six main spikes at ``PA_V3 + 60k`` per distinct roll, plus ``PA_V3 +- 90``
    when ``struts`` is True.
    """
    pa = np.atleast_1d(np.asarray(pa_v3_values, dtype=float))
    if pa.size == 0:
        return np.array([], dtype=float)
    offs = 60.0 * np.arange(6)
    if struts:
        offs = np.concatenate([offs, [90.0, -90.0]])
    out = (pa[:, None] + offs[None, :]).ravel() % 360.0
    return np.unique(np.round(out, 6))


def _cluster_pa(vals, tol=0.5):
    """Sorted PA values with entries within ``tol`` deg of an earlier one dropped.

    The comparison is circular, so 359.9 and 0.1 count as neighbours.
    """
    out = []
    for v in sorted(vals):
        if not any(abs((v - o + 180.0) % 360.0 - 180.0) <= tol for o in out):
            out.append(v)
    return np.array(out, dtype=float)


def _sample_files(files, per_group=3):
    """Pick header-read candidates spread over observations.

    Files are grouped by the ``jwPPPPPOOO`` prefix (program + observation);
    the first ``per_group`` and the last file of each group are read.  A
    plain "first 50 sorted" read can see only one program of a multi-program
    field (brick: jw01182 broad bands and jw02221 narrow bands, different
    rolls).
    """
    groups = {}
    for fn in files:
        groups.setdefault(os.path.basename(fn)[:11], []).append(fn)
    pick = []
    for g in groups.values():
        pick.extend(g[:per_group])
        pick.append(g[-1])
    return sorted(set(pick))


def _read_pa_v3(fn):
    """PA_V3 from the SCI header, else the primary header; None if absent."""
    v = None
    try:
        v = fits.getheader(fn, 'SCI').get('PA_V3')
    except KeyError:
        pass
    if v is None:
        v = fits.getheader(fn, 0).get('PA_V3')
    return v


def collect_pa_v3(basepath, bands, tol=0.5):
    """Distinct PA_V3 values (rounded to 0.1 deg) of the cal frames of ``bands``.

    Reads headers only from ``<basepath>/<BAND>/pipeline/*_cal.fits``, sampling
    each program/observation of each band (see :func:`_sample_files`, at most
    ``_MAX_FILES_PER_BAND`` files per band) so that every roll of a
    multi-program field is represented.  Values within ``tol`` deg are merged.
    Returns an empty array when nothing is found.
    """
    vals = []
    for band in bands:
        files = sorted(glob.glob(os.path.join(
            basepath, str(band).upper(), 'pipeline', '*_cal.fits')))
        for fn in _sample_files(files)[:_MAX_FILES_PER_BAND]:
            try:
                v = _read_pa_v3(fn)
            except (OSError, ValueError):
                continue
            if v is not None and np.isfinite(float(v)):
                vals.append(round(float(v), 1))
    if not vals:
        return np.array([], dtype=float)
    return _cluster_pa(vals, tol)


def _band_list(tbl, bands=None):
    allb = real_bands(tbl)
    if bands is None:
        return allb
    return [b for b in allb if b in set(bands)]


def n_real_bands(tbl, bands=None):
    """Per-row ``(n_real, mag_min)``.

    ``n_real`` counts bands with finite ``mag_vega_<b>`` that are neither
    ``mask_<b>`` nor ``forced_filled_<b>``.  ``mag_min`` is the minimum over
    all finite ``mag_vega_<b>`` (NaN when a row has none).
    """
    n = len(tbl)
    nreal = np.zeros(n, dtype=int)
    mmin = np.full(n, np.inf)
    for b in _band_list(tbl, bands):
        m = _arr(tbl, f'mag_vega_{b}')
        fin = np.isfinite(m)
        masked = (_arr(tbl, f'mask_{b}', fill=True, dtype=bool)
                  if f'mask_{b}' in tbl.colnames else np.zeros(n, bool))
        filled = (_arr(tbl, f'forced_filled_{b}', fill=False, dtype=bool)
                  if f'forced_filled_{b}' in tbl.colnames else np.zeros(n, bool))
        nreal += (fin & ~masked & ~filled).astype(int)
        mmin = np.minimum(mmin, np.where(fin, m, np.inf))
    mmin[~np.isfinite(mmin)] = np.nan
    return nreal, mmin


def _radec(tbl):
    """RA, Dec in degrees from ra/dec columns or ``skycoord_ref``."""
    if 'skycoord_ref' in tbl.colnames:
        sc = tbl['skycoord_ref']
        return np.asarray(sc.ra.deg, float), np.asarray(sc.dec.deg, float)
    for rc, dc in (('ra', 'dec'), ('RA', 'DEC')):
        if rc in tbl.colnames and dc in tbl.colnames:
            return _arr(tbl, rc), _arr(tbl, dc)
    raise KeyError('table has neither skycoord_ref nor ra/dec columns')


def _project(ra, dec):
    """Gnomonic offsets (arcsec) about the field centre; valid at any Dec."""
    ra0 = np.arctan2(np.mean(np.sin(np.radians(ra))), np.mean(np.cos(np.radians(ra))))
    dec0 = np.radians(np.mean(dec))
    r, d = np.radians(ra), np.radians(dec)
    cosc = np.sin(dec0) * np.sin(d) + np.cos(dec0) * np.cos(d) * np.cos(r - ra0)
    x = np.cos(d) * np.sin(r - ra0) / cosc
    y = (np.cos(dec0) * np.sin(d) - np.sin(dec0) * np.cos(d) * np.cos(r - ra0)) / cosc
    return np.degrees(x) * _ARCSEC, np.degrees(y) * _ARCSEC


def _wedge(ra, dec, x, y, cand, par, mmin, spike_pa, wmin_deg, w_phys, rmin, L0, m0,
           alpha, Lcap, adaptive=True, ratio=1.5, nsig=2.0, min_count=3, nseg=2,
           chunk=20000):
    """Boolean mask over all rows: candidate inside a parent's spike wedge.

    With ``adaptive`` the on-spike candidates of one (parent, radial segment)
    are flagged only when their count exceeds the count in control wedges of
    the same shape rotated by +30 deg:
    ``N_on >= max(min_count, ratio * N_off + nsig * sqrt(N_off + 1))``.
    """
    out = np.zeros(len(x), bool)
    if len(par) == 0 or len(cand) == 0 or len(spike_pa) == 0:
        return out
    xy = np.c_[x, y]
    tp = cKDTree(xy[par])
    Lpar = np.minimum(Lcap, L0 * 10 ** (-0.4 * alpha * (mmin[par] - m0)))
    rad, decr = np.radians(ra), np.radians(dec)
    ngrp = len(par) * nseg
    n_on = np.zeros(ngrp)
    n_off = np.zeros(ngrp)
    on_c, on_g = [], []
    for s in range(0, len(cand), chunk):
        c = cand[s:s + chunk]
        tc = cKDTree(xy[c])
        sdm = tc.sparse_distance_matrix(tp, Lcap, output_type='ndarray')
        ic, ip, r = sdm['i'], sdm['j'], sdm['v']
        k = (r > rmin) & (r < Lpar[ip])
        ic, ip, r = ic[k], ip[k], r[k]
        if len(r) == 0:
            continue
        cc, pp = c[ic], par[ip]
        keep = cc != pp
        cc, pp, ip, r = cc[keep], pp[keep], ip[keep], r[keep]
        # exact spherical PA of candidate seen from parent
        dra = rad[cc] - rad[pp]
        pa = np.degrees(np.arctan2(
            np.sin(dra) * np.cos(decr[cc]),
            np.cos(decr[pp]) * np.sin(decr[cc])
            - np.sin(decr[pp]) * np.cos(decr[cc]) * np.cos(dra))) % 360.0
        ph = np.abs((pa[:, None] - spike_pa[None, :] + 180.0) % 360.0 - 180.0).min(axis=1)
        w = np.maximum(wmin_deg, np.degrees(np.arctan2(w_phys, r)))
        on = ph < w
        if not adaptive:
            out[cc[on]] = True
            continue
        phc = np.abs((pa[:, None] - spike_pa[None, :] - 30.0 + 180.0) % 360.0
                     - 180.0).min(axis=1)
        off = (phc < w) & ~on
        seg = np.minimum((r / Lpar[ip] * nseg).astype(int), nseg - 1)
        g = ip * nseg + seg
        n_on += np.bincount(g[on], minlength=ngrp)
        n_off += np.bincount(g[off], minlength=ngrp)
        on_c.append(cc[on])
        on_g.append(g[on])
    if adaptive and on_c:
        on_c, on_g = np.concatenate(on_c), np.concatenate(on_g)
        thr = np.maximum(min_count, ratio * n_off + nsig * np.sqrt(n_off + 1.0))
        good = n_on[on_g] >= thr[on_g]
        out[on_c[good]] = True
    return out


def _crowd(x, y, nreal, dens_r1, dens_n1, dens_r2, dens_n2, contrast, ann):
    """Single-band rows with a local excess of other single-band rows.

    The expected count within a radius is the n_real==1 surface density in the
    annulus ``ann`` (arcsec) times the circle area; a row is flagged when its
    count reaches both the absolute minimum and ``contrast`` times expected.
    """
    one = nreal == 1
    crowd = np.zeros(len(x), bool)
    if not one.any():
        return crowd
    xy = np.c_[x, y]
    t1 = cKDTree(xy[one])
    a_in, a_out = ann
    n_ann = (t1.query_ball_point(xy, a_out, return_length=True)
             - t1.query_ball_point(xy, a_in, return_length=True))
    sigma = n_ann / (np.pi * (a_out ** 2 - a_in ** 2))
    c1 = t1.query_ball_point(xy, dens_r1, return_length=True) - one
    c2 = t1.query_ball_point(xy, dens_r2, return_length=True) - one
    e1 = sigma * np.pi * dens_r1 ** 2
    e2 = sigma * np.pi * dens_r2 ** 2
    hit = ((c1 >= dens_n1) & (c1 >= contrast * e1)) | ((c2 >= dens_n2) & (c2 >= contrast * e2))
    return one & hit


def flag_spike_artifacts(tbl, pa_v3, *, parent_mag=12.5, parent_min_real=2, max_real=2,
                         wmin_deg=2.0, w_phys_arcsec=0.25, rmin_arcsec=1.5,
                         L0_arcsec=22.0, m0=7.7, alpha=0.67, Lcap_arcsec=40.0,
                         dens_r1=1.5, dens_n1=15, dens_r2=3.0, dens_n2=60,
                         adaptive=True, ratio=1.5, nsig=2.0, min_count=3, nseg=2,
                         contrast=2.0, ann=(8.0, 20.0), crowd_near_parent=False,
                         pa_offset_deg=0.0, struts=False, verbose=True):
    """Flag likely diffraction-spike artifacts.

    Returns a dict of boolean arrays ``spike_wedge``, ``single_band_crowd`` and
    ``spike_artifact`` (their OR).  Rows are never dropped.

    ``adaptive`` makes the wedge test compare each parent's on-spike count with
    rotated control wedges (``ratio``, ``nsig``, ``min_count``; ``nseg`` radial
    segments per parent) and makes the crowd test a contrast against the local
    n_real==1 density in the annulus ``ann`` (``contrast``).  With
    ``crowd_near_parent`` the crowd flag additionally needs a bright parent
    within its length L.  ``pa_offset_deg`` rotates the spike pattern (used for
    selectivity controls).
    """
    n = len(tbl)
    nreal, mmin = n_real_bands(tbl)
    ra, dec = _radec(tbl)
    x, y = _project(ra, dec)
    spike_pa = spike_position_angles(pa_v3, struts=struts)
    with np.errstate(invalid='ignore'):
        par = np.where((mmin < parent_mag) & (nreal >= parent_min_real))[0]
    if len(spike_pa) == 0:
        if verbose:
            print('spike_flag: no PA_V3 values available; spike_wedge left all False',
                  flush=True)
        wedge = np.zeros(n, bool)
    else:
        cand = np.where(nreal <= max_real)[0]
        wedge = _wedge(ra, dec, x, y, cand, par, mmin, (spike_pa + pa_offset_deg) % 360.0,
                       wmin_deg, w_phys_arcsec, rmin_arcsec, L0_arcsec, m0, alpha,
                       Lcap_arcsec, adaptive=adaptive, ratio=ratio, nsig=nsig,
                       min_count=min_count, nseg=nseg)
    if adaptive:
        crowd = _crowd(x, y, nreal, dens_r1, dens_n1, dens_r2, dens_n2, contrast, ann)
    else:
        one = nreal == 1
        crowd = np.zeros(n, bool)
        if one.any():
            t1 = cKDTree(np.c_[x, y][one])
            xy = np.c_[x, y]
            c1 = t1.query_ball_point(xy, dens_r1, return_length=True) - one
            c2 = t1.query_ball_point(xy, dens_r2, return_length=True) - one
            crowd = one & ((c1 >= dens_n1) | (c2 >= dens_n2))
    if crowd_near_parent and crowd.any():
        if len(par):
            Lpar = np.minimum(Lcap_arcsec, L0_arcsec * 10 ** (-0.4 * alpha * (mmin[par] - m0)))
            tp = cKDTree(np.c_[x, y][par])
            ci = np.where(crowd)[0]
            near = np.zeros(n, bool)
            for j, lst in zip(ci, tp.query_ball_point(np.c_[x, y][ci], Lcap_arcsec)):
                if lst:
                    d = np.hypot(x[par[lst]] - x[j], y[par[lst]] - y[j])
                    near[j] = bool((d < Lpar[lst]).any())
            crowd &= near
        else:
            crowd[:] = False
    return {'spike_wedge': wedge, 'single_band_crowd': crowd,
            'spike_artifact': wedge | crowd}


def flag_m8_spike_artifacts(path, basepath, bands=None):
    """Add spike-flag columns to an m8_dedup FITS file (atomic rewrite).

    Adds ``spike_wedge``, ``single_band_crowd``, ``spike_artifact`` (bool) and
    ``n_real_bands`` (int).  Returns the dict of flag arrays.
    """
    from astropy.table import Table
    tbl = Table.read(path)
    if bands is None:
        bands = real_bands(tbl)
    pa_v3 = collect_pa_v3(basepath, bands)
    flags = flag_spike_artifacts(tbl, pa_v3)
    nreal, _ = n_real_bands(tbl)
    tbl['n_real_bands'] = nreal.astype(np.int16)
    for k, v in flags.items():
        tbl[k] = v
    write_table_atomic(tbl, path)
    print(f"spike_flag: {os.path.basename(path)} N={len(tbl)} PA_V3={list(pa_v3)} "
          f"wedge={int(flags['spike_wedge'].sum())} "
          f"crowd={int(flags['single_band_crowd'].sum())} "
          f"any={int(flags['spike_artifact'].sum())}", flush=True)
    return flags
