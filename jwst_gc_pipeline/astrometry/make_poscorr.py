"""
Producer of the per-row position-correction tables that ``poscorr.py`` reads (the per-detection distortion hook).

For a merged catalogue C (the exact FITS file the PM builder reads) and the per-frame catalogues that were merged
into it, every row's correction is the MEAN over the row's per-frame detections of the sky vector

    c_sky(det) = inv(M_frame) @ c_det(x_fit, y_fit)       [arcsec on the tangent plane; xi east*cos(dec), eta north]

with c_det the detector-axis correction (mas) of the chosen solution evaluated at the detection's detector position
and M_frame = J * PIXSCALE * 1000 the frame's sky(arcsec) -> detector(mas) matrix (J = linear fit of x_fit, y_fit
against the frame's own tangent-plane coordinates).  The merged position is the uniform mean of the kept frames'
positions (merge_catalogs.py: pos_weights = keepmask), so the corrected row position is  position + mean(c_sky)  to
first order.  This module estimates no distortion: the correction comes from the solution handed to it.

Association of a frame detection to a catalogue row follows merge_catalogs.combine_singleframe: mutual nearest
neighbours on the sky within --radius (default 0.10", the merge's max_offset).  Frame positions are used exactly as
on disk -- no per-exposure bulk shift is removed -- which is right when the merge applied no offsets table and no
m2 registration to those frames (sgrb2 5365: fields.yaml has no offsets_table for it, and the merged header has no
M2REG* cards).  Where that assumption fails, ``recon_*`` and ``set_consistent`` below show it.  The merge's
detection selection is replicated: a detection counts only if it survives sigma_clip(stdfunc='mad_std', axis=1) of
flux, ra and dec over the row's detections (NaN flux is never kept; a row whose every detection is clipped falls
back to all finite ones), as in combine_singleframe's keepmask.  --mode clip (default) averages that set; 'all'
averages every associated detection.  Both are written (dxi_mas/deta_mas = the chosen mode, *_all_mas = every
associated detection) so the detection-set sensitivity is in the table.

Rows with no associated detection, and saturated-star rows (their position is the satstar fit over every frame that
covers the star, not the per-frame detections), get the footprint estimate instead: the mean of the solution's sky
correction over every frame whose detector footprint contains the row's position, each frame's pixel position being
predicted by its linear sky -> pixel fit (8 px margin).  --extra CATALOG::FRAMEREGEX adds frames outside the
catalogue's own merge (e.g. the other module's) to that footprint test.  A row no frame covers gets 0.

Solutions (--solution):
  zero   c = 0 (plumbing test; a build with these tables reproduces the build without them)
  map    c = -M, M being the detector-axis map (mas) stored in --map-npz under the keys ``{kind}_{DET}_dx`` /
         ``_dy`` (kind = --map-kind; 16 x 16 cells per detector over the 2048 px, indexed [ix, iy], bilinear
         between cell centres).  The estimator that makes the maps is not part of this repository.
``run`` takes any callable ``solution(detector, x, y) -> (cx_mas, cy_mas)``; its ``name`` attribute is written to the
header.  That is how a correction that cannot live in this repository (e.g. the STDGDC delta field, which needs CRDS
reference files and the data-qa code) is turned into a table: wrap it and call ``run``.

Output FITS table (row-parallel to the catalogue; the header carries the guards ``poscorr.load_poscorr`` checks):
  row (int32), dxi_mas, deta_mas (float32; sky correction to ADD to the position), n_assoc, n_keep (int16),
  dxi_all_mas, deta_all_mas, recon_dxi_mas, recon_deta_mas (float32), foot_dxi_mas, foot_deta_mas, n_foot (the
  footprint-mean correction over every covering frame, written for every row), estimate (int16: 1 detection mean,
  2 footprint, 0 none), nmatch_cat, nmatch_good_cat (int16; the merge's own counts), set_consistent (int16; 1 where
  n_assoc == nmatch and n_keep == nmatch_good).  recon_* = catalogue position minus the mean of the kept detections'
  positions (east*cos(dec), north; mas): near 0 where the kept set is the one the merge used, ~ detection scatter / n
  where it is not.
  header: CATALOG, NROWS, RACHK/DECCHK (CRC32 of the float64 position bytes), SOLUTION, MAPFILE, MAPKIND, NFRAMES
          (frames given; one with fewer than 50 associated detections is skipped but still counted), NDET
          (associated detections used), RADIUS, MODE, F_ASSOC (fraction of rows with >= 1 detection), F_SETOK
          (fraction with set_consistent; absent, with the three count columns, when the catalogue has no nmatch
          columns), POSCONV

Re-run this whenever the catalogue or its per-frame inputs change (e.g. 5365 re-catalogued with new cores): the
builder refuses a table whose checksums do not match the catalogue it is given.

Example:
    python -m jwst_gc_pipeline.astrometry.make_poscorr \\
        --catalog /orange/.../sgrb2/catalogs/f212n_nrca_indivexp_merged_..._vetted.fits \\
        --frame-dir /orange/.../sgrb2/F212N --frame-regex '^f212n_nrca[1-4]_visit001_vgroup03101_exp\\d+_.*_basic\\.fits$' \\
        --extra '/orange/.../f212n_nrcb_..._vetted.fits::^f212n_nrcb[1-4]_visit001_..._basic\\.fits$' \\
        --solution map --map-npz maps_f212n.npz --map-kind loofull --out poscorr_nrca.fits
"""
import argparse
import os
import re

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table

from ..atomic_io import write_table_atomic
from .poscorr import position_checksums

FILT = 'f212n'
SW_DETECTORS = ('NRCA1', 'NRCA2', 'NRCA3', 'NRCA4', 'NRCB1', 'NRCB2', 'NRCB3', 'NRCB4')
NPIX = 2048           # NIRCam SW detector size (pixels)
FOOT_MARGIN = 8.0     # pixels; the footprint test uses a per-frame LINEAR sky->pixel fit, good to ~1-2 px at the edge
MIN_ASSOCIATED = 50   # a frame with fewer associated detections than this is skipped (no usable fit of its geometry)


def catalogue_counts(path, filt=FILT):
    """(nmatch, nmatch_good) of every row (frames the merge matched / kept after its clip), or (None, None)."""
    with fits.open(path, memmap=True) as h:
        d = h[1].data
        names = d.names
        for sfx in ('', f'_{filt}'):
            if f'nmatch{sfx}' in names and f'nmatch_good{sfx}' in names:
                return (np.asarray(d[f'nmatch{sfx}'], np.int32), np.asarray(d[f'nmatch_good{sfx}'], np.int32))
    return None, None


def catalogue_saturated(path):
    """Per-row bool: saturated-star rows (is_saturated | replaced_saturated), False where the columns are absent."""
    with fits.open(path, memmap=True) as h:
        d = h[1].data
        sat = np.zeros(h[1].header['NAXIS2'], bool)
        for col in ('is_saturated', 'replaced_saturated'):
            if col in d.names:
                sat |= np.asarray(d[col], bool)
    return sat


def catalogue_positions(path, filt=FILT):
    """(ra, dec) in degrees of every row, bare ('skycoord.ra') or suffixed ('skycoord_f212n.ra') columns."""
    with fits.open(path, memmap=True) as h:
        d = h[1].data
        names = d.names
        for base in ('skycoord', f'skycoord_{filt}'):
            if f'{base}.ra' in names:
                return np.asarray(d[f'{base}.ra'], float), np.asarray(d[f'{base}.dec'], float)
    raise KeyError(f'no skycoord.ra / skycoord_{filt}.ra in {path}')


class BinnedMapField:
    """A binned detector-axis map (nb x nb cells over [0, size), indexed [ix, iy]) evaluated by bilinear
    interpolation between cell centres.  Beyond the outermost cell centres (the outer half cell on each edge) the
    bilinear surface is EXTRAPOLATED linearly (``fill_value=None``), not held at the edge value.  NaN cells are
    filled by the mean of their finite neighbours (repeated up to nb times), then by 0."""

    def __init__(self, mdx, mdy, size=NPIX):
        from scipy.interpolate import RegularGridInterpolator
        mdx = np.array(mdx, float)
        mdy = np.array(mdy, float)
        nb = mdx.shape[0]
        for m in (mdx, mdy):
            for _ in range(nb):
                bad = ~np.isfinite(m)
                if not bad.any():
                    break
                pad = np.pad(m, 1, constant_values=np.nan)
                neigh = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
                with np.errstate(invalid='ignore'):
                    fill = np.nanmean(np.where(np.isfinite(neigh), neigh, np.nan), axis=0)
                m[bad] = fill[bad]
            m[~np.isfinite(m)] = 0.0
        c = (np.arange(nb) + 0.5) * size / nb
        kw = dict(method='linear', bounds_error=False, fill_value=None)
        self._fx = RegularGridInterpolator((c, c), mdx, **kw)
        self._fy = RegularGridInterpolator((c, c), mdy, **kw)

    def __call__(self, x, y):
        pts = np.stack(np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float)), axis=-1)
        return self._fx(pts), self._fy(pts)


class Solution:
    """Detector-axis correction c(x, y) in mas, called as ``solution(detector, x, y) -> (cx, cy)``.

    'zero': c = 0.   'map': c = -M, M the ``{map_kind}_{DET}_dx/_dy`` maps of the npz ``map_npz`` (a detector with no
    map raises ValueError when first asked for).  Non-finite values are returned as 0."""

    def __init__(self, name='zero', map_npz=None, map_kind=None):
        if name not in ('zero', 'map'):
            raise ValueError(f"solution {name!r}: use 'zero' or 'map' (or pass any callable to run())")
        self.name = name
        self.map_file = map_npz if name == 'map' else ''
        self.map_kind = map_kind if name == 'map' else ''
        self._fields = {}
        if name == 'map':
            if not map_npz or not map_kind:
                raise ValueError("solution 'map' needs map_npz and map_kind")
            with np.load(map_npz) as z:
                self._maps = {d: (np.array(z[f'{map_kind}_{d}_dx']), np.array(z[f'{map_kind}_{d}_dy']))
                              for d in SW_DETECTORS if f'{map_kind}_{d}_dx' in z.files}
                if not self._maps:
                    kinds = sorted({k.split('_')[0] for k in z.files})
                    raise ValueError(f'{map_npz}: no {map_kind!r} maps (kinds present: {kinds})')

    def _field(self, det):
        if det not in self._fields:
            if self.name == 'zero':
                self._fields[det] = lambda x, y: (np.zeros_like(x, float), np.zeros_like(y, float))
            elif det in self._maps:
                bm = BinnedMapField(*self._maps[det])
                self._fields[det] = lambda x, y, bm=bm: tuple(-v for v in bm(x, y))
            else:
                raise ValueError(f'{self.map_file}: no {self.map_kind!r} map for {det} (has {sorted(self._maps)})')
        return self._fields[det]

    def __call__(self, det, x, y):
        cx, cy = self._field(det.upper())(np.asarray(x, float), np.asarray(y, float))
        return np.nan_to_num(np.asarray(cx, float)), np.nan_to_num(np.asarray(cy, float))


def load_frame_all(path):
    """All rows of a per-frame catalogue with finite x, y, ra, dec (no S/N cut, as in the merge): x, y, ra, dec,
    flux_fit + header PIXSCALE, DETECTOR."""
    with fits.open(path, memmap=False) as h:
        d = h[1].data
        hdr = h[1].header
        x = np.asarray(d['x_fit'], float)
        y = np.asarray(d['y_fit'], float)
        ra = np.asarray(d['skycoord_centroid.ra'], float)
        dec = np.asarray(d['skycoord_centroid.dec'], float)
        flux = np.asarray(d['flux_fit'], np.float32)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(ra) & np.isfinite(dec)
    return dict(x=x[ok], y=y[ok], ra=ra[ok], dec=dec[ok], flux=flux[ok], pixscale=float(hdr['PIXSCALE']),
                detector=str(hdr['DETECTOR']).upper())


def associate(fr_ra, fr_dec, cat_ra, cat_dec, radius_arcsec):
    """Mutual nearest neighbours between frame detections and catalogue rows within the radius: the criterion of
    merge_catalogs.combine_singleframe (a detection's nearest row has that detection as its own nearest, and the
    separation is below max_offset).  Returns (index into the frame arrays, index into the catalogue rows)."""
    fsc = SkyCoord(fr_ra * u.deg, fr_dec * u.deg)
    csc = SkyCoord(cat_ra * u.deg, cat_dec * u.deg)
    ci, sep, _ = fsc.match_to_catalog_sky(csc)
    fi, _, _ = csc.match_to_catalog_sky(fsc)
    ok = (sep < radius_arcsec * u.arcsec) & (fi[ci] == np.arange(len(ci)))
    return np.nonzero(ok)[0], ci[ok]


def frame_matrix(fr, ia):
    """M = J * pixscale * 1000 (sky arcsec -> detector mas) from a linear fit of the associated detections ``ia``
    (x_fit, y_fit) against their tangent-plane coordinates about their median position.  Returns (M, J, centre)
    with centre = (ra0, dec0, x0, y0), the origin of the fit."""
    ra0 = float(np.median(fr['ra'][ia]))
    dec0 = float(np.median(fr['dec'][ia]))
    cosd = np.cos(np.radians(dec0))
    xi = (fr['ra'][ia] - ra0) * cosd * 3600.0
    eta = (fr['dec'][ia] - dec0) * 3600.0
    A = np.column_stack([np.ones_like(xi), xi, eta])
    cx, *_ = np.linalg.lstsq(A, fr['x'][ia], rcond=None)
    cy, *_ = np.linalg.lstsq(A, fr['y'][ia], rcond=None)
    J = np.array([[cx[1], cx[2]], [cy[1], cy[2]]])
    return J * (fr['pixscale'] * 1000.0), J, (ra0, dec0, float(cx[0]), float(cy[0]))


def build_stacks(catalog, frames, solution, radius, filt=FILT, verbose=True):
    """Row x frame stacks (as combine_singleframe builds them) of the detections associated to the catalogue
    rows: offx/offy = frame minus catalogue position (mas; east*cos(dec), north), flux (flux_fit),
    cxi/cet = the solution's sky correction (mas).  NaN = row not detected in that frame.  Positions are used as
    on disk (see the module docstring)."""
    cat_ra, cat_dec = catalogue_positions(catalog, filt)
    ok_cat = np.isfinite(cat_ra) & np.isfinite(cat_dec)
    ridx = np.nonzero(ok_cat)[0]
    nr, nf = len(ridx), len(frames)
    cosd_row = np.cos(np.radians(cat_dec[ridx]))
    per_frame = []
    for k, p in enumerate(frames):
        fr = load_frame_all(p)
        ia, ib = associate(fr['ra'], fr['dec'], cat_ra[ok_cat], cat_dec[ok_cat], radius)
        if len(ia) < MIN_ASSOCIATED:
            if verbose:
                print(f'  {os.path.basename(p)}: only {len(ia)} associated detections; skipped', flush=True)
            continue
        M, J, centre = frame_matrix(fr, ia)
        cdx, cdy = solution(fr['detector'], fr['x'][ia], fr['y'][ia])
        Mi = np.linalg.inv(M)
        # frame minus catalogue position (east*cos dec, north; mas)
        ox = (fr['ra'][ia] - cat_ra[ridx[ib]]) * cosd_row[ib] * 3.6e6
        oy = (fr['dec'][ia] - cat_dec[ridx[ib]]) * 3.6e6
        per_frame.append(dict(k=k, ib=ib, ox=ox, oy=oy, flux=fr['flux'][ia],
                              detector=fr['detector'], M=M, J=J, centre=centre,
                              cxi=(Mi[0, 0] * cdx + Mi[0, 1] * cdy) * 1000.0,    # arcsec -> mas on the sky
                              cet=(Mi[1, 0] * cdx + Mi[1, 1] * cdy) * 1000.0))
        if verbose:
            print(f'  {os.path.basename(p)}: {len(ia)}/{len(fr["x"])} detections associated; '
                  f'|c| rms {np.sqrt(np.mean(cdx ** 2 + cdy ** 2)):.2f} mas', flush=True)
    # row x frame stacks, as combine_singleframe builds them (offsets in mas relative to the catalogue position)
    offx = np.full((nr, nf), np.nan, np.float32)
    offy = np.full((nr, nf), np.nan, np.float32)
    flux = np.full((nr, nf), np.nan, np.float32)
    cxi = np.zeros((nr, nf), np.float32)
    cet = np.zeros((nr, nf), np.float32)
    n_det_total = 0
    for f in per_frame:
        offx[f['ib'], f['k']] = f['ox']
        offy[f['ib'], f['k']] = f['oy']
        flux[f['ib'], f['k']] = f['flux']
        cxi[f['ib'], f['k']] = f['cxi']
        cet[f['ib'], f['k']] = f['cet']
        n_det_total += len(f['ib'])
    geom = [dict(k=f['k'], detector=f['detector'], M=f['M'], J=f['J'], centre=f['centre']) for f in per_frame]
    return dict(cat_ra=cat_ra, cat_dec=cat_dec, ridx=ridx, offx=offx, offy=offy, flux=flux, cxi=cxi, cet=cet,
                n_det_total=n_det_total, geom=geom)


def frame_geoms(catalog, frames, radius, filt=FILT, verbose=True):
    """Per-frame linear sky->pixel geometry (as in build_stacks) for frames that are NOT part of ``catalog``'s own
    merge, found by associating their detections to ``catalog`` (e.g. the other module's frames against the other
    module's catalogue).  Used only by the footprint fallback."""
    cat_ra, cat_dec = catalogue_positions(catalog, filt)
    ok_cat = np.isfinite(cat_ra) & np.isfinite(cat_dec)
    geom = []
    for k, p in enumerate(frames):
        fr = load_frame_all(p)
        ia, ib = associate(fr['ra'], fr['dec'], cat_ra[ok_cat], cat_dec[ok_cat], radius)
        if len(ia) < MIN_ASSOCIATED:
            if verbose:
                print(f'  extra {os.path.basename(p)}: only {len(ia)} associated detections; skipped', flush=True)
            continue
        M, J, centre = frame_matrix(fr, ia)
        geom.append(dict(k=-1 - k, detector=fr['detector'], M=M, J=J, centre=centre))
    if verbose:
        print(f'  {len(geom)}/{len(frames)} extra frames with geometry (from {os.path.basename(catalog)})', flush=True)
    return geom


def footprint_mean(S, rows, solution, margin=FOOT_MARGIN, extra_geom=()):
    """Fallback for catalogue rows with no associated detection (saturated stars whose position comes from the
    satstar catalogue): the mean of the solution's sky correction over every frame whose detector footprint
    contains the row's position, the star's pixel position being predicted by each frame's linear sky->pixel fit.
    rows: indices into the stack rows (0..nr-1).  extra_geom: geometries of frames outside the catalogue's own
    merge (see frame_geoms).  Returns (cxi_mean, cet_mean, n_frames) arrays over `rows`."""
    cat_ra, cat_dec, ridx = S['cat_ra'], S['cat_dec'], S['ridx']
    ra, dec = cat_ra[ridx[rows]], cat_dec[ridx[rows]]
    sx = np.zeros(len(rows))
    sy = np.zeros(len(rows))
    cnt = np.zeros(len(rows), int)
    for g in list(S['geom']) + list(extra_geom):
        ra0, dec0, x0, y0 = g['centre']
        xi = (ra - ra0) * np.cos(np.radians(dec0)) * 3600.0
        eta = (dec - dec0) * 3600.0
        x = x0 + g['J'][0, 0] * xi + g['J'][0, 1] * eta
        y = y0 + g['J'][1, 0] * xi + g['J'][1, 1] * eta
        inside = (x >= margin) & (x <= NPIX - 1 - margin) & (y >= margin) & (y <= NPIX - 1 - margin)
        if not inside.any():
            continue
        cdx, cdy = solution(g['detector'], x[inside], y[inside])
        Mi = np.linalg.inv(g['M'])
        sx[inside] += (Mi[0, 0] * cdx + Mi[0, 1] * cdy) * 1000.0
        sy[inside] += (Mi[1, 0] * cdx + Mi[1, 1] * cdy) * 1000.0
        cnt[inside] += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(cnt > 0, sx / np.maximum(cnt, 1), 0.0), np.where(cnt > 0, sy / np.maximum(cnt, 1), 0.0), cnt


def keep_mask(offx, offy, flux):
    """combine_singleframe's keepmask: sigma-clip flux, ra and dec per row (stdfunc mad_std; NaN is masked);
    a row whose every detection is clipped falls back to all its finite detections."""
    finite_pos = np.isfinite(offx) & np.isfinite(offy)
    clip_f = sigma_clip(flux, stdfunc='mad_std', axis=1)
    clip_x = sigma_clip(offx, stdfunc='mad_std', axis=1)
    clip_y = sigma_clip(offy, stdfunc='mad_std', axis=1)
    keep = ~(np.ma.getmaskarray(clip_f) | np.ma.getmaskarray(clip_x) | np.ma.getmaskarray(clip_y))
    fin_all = finite_pos & np.isfinite(flux)
    all_clipped = (fin_all.sum(axis=1) > 0) & ~keep.any(axis=1)
    keep[all_clipped] = fin_all[all_clipped]
    return keep


def run(catalog, frames, solution, radius, out, mode='clip', filt=FILT, verbose=True, extra=()):
    """Write the position-correction table of ``catalog`` to ``out`` and return it.

    frames: the per-frame catalogues merged into ``catalog``.  solution: ``Solution`` or any callable
    ``(detector, x, y) -> (cx_mas, cy_mas)``; a ``name`` (and optional ``map_file`` / ``map_kind``) attribute goes
    to the header.  extra: ``(catalogue, frames)`` pairs whose frames only enter the footprint fallback."""
    S = build_stacks(catalog, frames, solution, radius, filt, verbose)
    cat_ra, cat_dec, ridx = S['cat_ra'], S['cat_dec'], S['ridx']
    offx, offy, flux, cxi, cet = S['offx'], S['offy'], S['flux'], S['cxi'], S['cet']
    n_det_total = S['n_det_total']
    n = len(cat_ra)
    finite_pos = np.isfinite(offx) & np.isfinite(offy)
    n_assoc = finite_pos.sum(axis=1)
    keep = keep_mask(offx, offy, flux)
    n_keep = keep.sum(axis=1)

    def row_mean(a, m):
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(m.sum(axis=1) > 0, np.where(m, a, 0.0).sum(axis=1) / np.maximum(m.sum(axis=1), 1), 0.0)

    use = keep if mode == 'clip' else finite_pos
    dxi_c, det_c = row_mean(cxi, use), row_mean(cet, use)
    dxi_a, det_a = row_mean(cxi, finite_pos), row_mean(cet, finite_pos)
    # footprint estimate for every row (diagnostic where detections exist); it REPLACES the detection mean for
    # rows with no associated detection and for saturated-star rows (their position is the satstar fit over every
    # frame that covers the star, not the per-frame DAO detections, which are few and unreliable there)
    extra_geom = []
    for ecat, eframes in extra:
        extra_geom += frame_geoms(ecat, eframes, radius, filt, verbose)
    fx, fy, nfoot = footprint_mean(S, np.arange(len(ridx)), solution, extra_geom=extra_geom)
    sat_row = catalogue_saturated(catalog)[ridx]
    use_foot = ((n_assoc == 0) | sat_row) & (nfoot > 0)
    dxi_c, det_c = np.where(use_foot, fx, dxi_c), np.where(use_foot, fy, det_c)
    dxi_a, det_a = np.where(use_foot, fx, dxi_a), np.where(use_foot, fy, det_a)
    rec_x, rec_y = -row_mean(offx, keep), -row_mean(offy, keep)       # catalogue minus mean of kept positions

    def full(a, dtype):
        o = np.zeros(n, dtype)
        o[ridx] = a
        return o

    t = Table()
    t['row'] = np.arange(n, dtype=np.int32)
    t['dxi_mas'] = full(dxi_c, np.float32)
    t['deta_mas'] = full(det_c, np.float32)
    t['n_assoc'] = full(np.minimum(n_assoc, 32000), np.int16)
    t['n_keep'] = full(np.minimum(n_keep, 32000), np.int16)
    t['dxi_all_mas'] = full(dxi_a, np.float32)
    t['deta_all_mas'] = full(det_a, np.float32)
    t['recon_dxi_mas'] = full(rec_x, np.float32)
    t['recon_deta_mas'] = full(rec_y, np.float32)
    t['foot_dxi_mas'] = full(fx, np.float32)           # footprint-mean correction (every row; used where n_assoc == 0)
    t['foot_deta_mas'] = full(fy, np.float32)
    t['n_foot'] = full(np.minimum(nfoot, 32000), np.int16)
    # 1 detections, 2 footprint, 0 none (int8 would be written as a FITS logical)
    t['estimate'] = full(np.where(use_foot, 2, np.where(n_assoc > 0, 1, 0)), np.int16)
    nm, nmg = catalogue_counts(catalog, filt)
    if nm is not None:
        # the merge's own counts: a row is "set-consistent" when association and clip reproduce them
        t['nmatch_cat'] = np.minimum(nm, 32000).astype(np.int16)
        t['nmatch_good_cat'] = np.minimum(nmg, 32000).astype(np.int16)
        t['set_consistent'] = ((np.asarray(t['n_assoc']) == nm) & (np.asarray(t['n_keep']) == nmg)).astype(np.int16)
    rachk, decchk = position_checksums(cat_ra, cat_dec)
    na_full = t['n_assoc']
    t.meta.update(CATALOG=os.path.abspath(catalog), NROWS=n, RACHK=rachk, DECCHK=decchk,
                  SOLUTION=getattr(solution, 'name', type(solution).__name__),
                  MAPFILE=getattr(solution, 'map_file', '') or '', MAPKIND=getattr(solution, 'map_kind', '') or '',
                  NFRAMES=len(frames), NDET=n_det_total, RADIUS=radius, MODE=mode,
                  F_ASSOC=float((na_full > 0).mean()))
    if 'set_consistent' in t.colnames:                 # a NaN card is not writable to FITS
        t.meta['F_SETOK'] = float(np.mean(t['set_consistent']))
    t.meta['POSCONV'] = 'add dxi_mas/(3.6e6*cos(dec)) to ra and deta_mas/3.6e6 to dec'
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write_table_atomic(t, out)
    if verbose:
        a = np.hypot(t['dxi_mas'], t['deta_mas'])
        has = np.asarray(na_full) > 0
        rr = np.hypot(t['recon_dxi_mas'], t['recon_deta_mas'])[np.asarray(t['n_keep']) > 0]
        print(f'wrote {out}: {n} rows, {int(has.sum())} with >=1 detection ({100 * has.mean():.1f}%), '
              f'|corr| rms {np.sqrt(np.mean(a[has] ** 2)):.3f} mas ({mode}); '
              f'reconstructed-minus-catalogue position |d| median {np.median(rr):.2f} mas, '
              f'p90 {np.percentile(rr, 90):.2f}, frac < 0.5 mas {np.mean(rr < 0.5):.3f}', flush=True)
    return t


def list_frames(frame_dir, frame_regex):
    rx = re.compile(frame_regex)
    return [os.path.join(frame_dir, n) for n in sorted(os.listdir(frame_dir)) if rx.match(n)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--catalog', required=True, help='the merged catalogue the PM builder reads')
    ap.add_argument('--frame-dir', required=True, help='directory of the per-frame catalogues merged into it')
    ap.add_argument('--frame-regex', required=True, help='regex (re.match) selecting those frames in --frame-dir')
    ap.add_argument('--solution', required=True, choices=('zero', 'map'))
    ap.add_argument('--map-npz', help="npz of maps, keys '{kind}_{DET}_dx' / '_dy' (with --solution map)")
    ap.add_argument('--map-kind', help="key prefix of the maps to use in --map-npz, e.g. loofull (with --solution map)")
    ap.add_argument('--filter', default=FILT, help='filter of the catalogue: suffix of its skycoord/nmatch columns')
    ap.add_argument('--radius', type=float, default=0.10, help='association radius in arcsec (merge max_offset)')
    ap.add_argument('--mode', default='clip', choices=('clip', 'all'),
                    help="detections averaged: those kept by the merge's sigma clip, or every associated one")
    ap.add_argument('--extra', action='append', default=[], metavar='CATALOG::FRAMEREGEX',
                    help="other frames (e.g. the other module's) whose footprint may cover this catalogue's "
                         "saturated rows: geometry fitted against CATALOG; repeatable")
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.solution == 'map' and not (a.map_npz and a.map_kind):
        ap.error('--solution map needs --map-npz and --map-kind')
    sol = Solution(a.solution, a.map_npz, a.map_kind)
    frames = list_frames(a.frame_dir, a.frame_regex)
    if not frames:
        raise SystemExit(f'no frames match {a.frame_regex!r} in {a.frame_dir}')
    print(f'{len(frames)} frames; catalogue {a.catalog}', flush=True)
    extra = []
    for spec in a.extra:
        ecat, eregex = spec.split('::', 1)
        eframes = list_frames(a.frame_dir, eregex)
        if not eframes:
            raise SystemExit(f'--extra {spec!r}: no frames in {a.frame_dir}')
        extra.append((ecat, eframes))
    run(a.catalog, frames, sol, a.radius, a.out, mode=a.mode, filt=a.filter, extra=extra)


if __name__ == '__main__':
    main()
