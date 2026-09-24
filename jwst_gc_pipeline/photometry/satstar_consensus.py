"""Saturated stars in the visit consensus (issue #957).

A saturated star has no per-frame daophot row: its core is blanked in the crf,
so daophot never detects it and ``build_visit_consensus`` -- which reads only
the per-frame daophot catalogs -- never sees it.  Its VIRAC2 counterpart is
then paired by the reference tie to a faint neighbour 150-300 mas away, and
those wrong pairs refuse the same-star region map.  On gc-treasury o040 F212N,
44% of the 0.3" consensus->VIRAC2 pairs were wrong before this channel and 26%
after it.

The satstar fit of that star exists per exposure (``*_satstar_catalog.fits``)
and repeats between exposures to ~1.6 mas rms (p50; p90 2.7 mas on o040
F212N).  This module lets such a fit into the consensus when:

* it appears in at least ``SATSTAR_CONSENSUS_MIN_EXPOSURES`` (2) exposures
  within ``SATSTAR_CONSENSUS_MATCH_ARCSEC`` (0.1") of each other, and
* its position repeats between those exposures to within
  ``SATSTAR_CONSENSUS_MAX_RMS_MAS`` (5 mas) rms.

The daophot quality cuts (qfit, flux/flux_err) do not apply: a satstar fit's
qfit measures the wings of a clipped core and is 0.3-1.5 for fits whose
positions repeat to 2 mas.  The repeatability test replaces them.

The satstar rows only EXTEND the consensus used for the reference tie.  They do
not enter the per-exposure-vs-consensus measurement, which is untouched.
"""
import os

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from jwst_gc_pipeline.photometry.visit_consensus import exposure_key

#: Two fits of one star in different exposures must lie within this of each
#: other to be linked.
SATSTAR_CONSENSUS_MATCH_ARCSEC = 0.1
#: A satstar enters the consensus only when this many exposures fit it.
SATSTAR_CONSENSUS_MIN_EXPOSURES = 2
#: ...and its per-exposure positions scatter by at most this (rms about the
#: mean, mas).
SATSTAR_CONSENSUS_MAX_RMS_MAS = 5.0
#: A satstar this close to a daophot consensus star is the same star (a
#: partially saturated core that daophot also measured); the daophot position
#: is kept and the satstar is not added a second time.
SATSTAR_CONSENSUS_EXCLUSION_ARCSEC = 0.2

#: Stages whose per-frame catalogs are written by the m12 phase, which names
#: its satstar catalogs ``_m12``.
_M12_STAGES = ("m1", "m2", "m12")


def satstar_stage_label(stage):
    """Satstar-catalog iteration label for a checkpoint stage.

    The m12 phase writes the m1 and m2 per-frame catalogs and its satstar
    catalogs as ``..._crf_m12_satstar_catalog.fits``; every later phase uses
    its own label (``m3``, ``resbgsub_m5``, ...)."""
    stage = str(stage)
    return "m12" if stage in _M12_STAGES else stage


def satstar_catalog_path(exposure_table, stage):
    """Per-exposure satstar catalog that belongs to ``exposure_table``, or None.

    The per-frame daophot catalog records its frame in ``meta['FILENAME']``;
    the satstar catalog sits beside that frame with the stage's iteration
    token.  None when the table carries no FILENAME or the file does not exist
    (a frame with no saturated star legitimately has no satstar catalog).
    """
    frame = exposure_table.meta.get("FILENAME")
    if not frame or not str(frame).endswith(".fits"):
        return None
    path = str(frame)[:-len(".fits")] + f"_{satstar_stage_label(stage)}_satstar_catalog.fits"
    return path if os.path.exists(path) else None


def _satstar_coords(tbl):
    """Finite satstar fit positions of one per-exposure satstar catalog."""
    if "skycoord_fit" not in tbl.colnames:
        return SkyCoord([] * u.deg, [] * u.deg)
    sc = SkyCoord(tbl["skycoord_fit"]).icrs
    ok = np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg)
    return sc[ok]


def load_exposure_satstars(exposure_tables, stage):
    """``{exposure_key: SkyCoord}`` of each exposure's satstar fits.

    Positions are re-projected through the frame's CURRENT GWCS (the satstar
    fit is cached across offsets-table corrections, issue #193), using the
    same reader the merge uses.  Exposures with no satstar catalog are
    absent from the result.
    """
    # Local import: merge_catalogs is large and imports this package's
    # cataloging helpers; the checkpoint path only needs it when satstars are
    # actually read.
    from jwst_gc_pipeline.photometry.merge_catalogs import (
        _read_satstar_catalog_on_current_frame)
    out = {}
    wcs_cache = {}
    for tbl in exposure_tables:
        path = satstar_catalog_path(tbl, stage)
        if path is None:
            continue
        sat = _read_satstar_catalog_on_current_frame(path, wcs_cache)
        coords = _satstar_coords(sat)
        if len(coords):
            out[exposure_key(tbl)] = coords
    return out


def _shift(coords, dra_mas, ddec_mas):
    cosd = np.cos(np.radians(coords.dec.deg))
    return SkyCoord(ra=coords.ra.deg + dra_mas / 3.6e6 / cosd,
                    dec=coords.dec.deg + ddec_mas / 3.6e6, unit="deg")


def satstar_consensus(satstars_by_exposure, exposure_offsets=None,
                      match_radius=SATSTAR_CONSENSUS_MATCH_ARCSEC * u.arcsec,
                      min_exposures=SATSTAR_CONSENSUS_MIN_EXPOSURES,
                      max_rms_mas=SATSTAR_CONSENSUS_MAX_RMS_MAS):
    """Group satstar fits across exposures and keep the repeatable ones.

    Parameters
    ----------
    satstars_by_exposure : dict
        ``{exposure_key: SkyCoord}`` from :func:`load_exposure_satstars`.
    exposure_offsets : dict, optional
        ``{exposure_key: (dra_mas, ddec_mas)}`` on-sky offset that carries the
        exposure onto the consensus frame (``vs_consensus`` of
        ``build_visit_consensus``: consensus minus exposure).  Applied before
        grouping so the satstars sit in the same frame as the daophot
        consensus stars.  Exposures absent from it are used as measured.

    Returns
    -------
    dict with ``coords`` (SkyCoord, mean position per kept star), ``nexp``
    (exposures per kept star), ``rms_mas`` (rms scatter about the mean),
    ``n_groups`` (groups seen in >= ``min_exposures`` exposures, before the
    rms cut) and ``n_rejected_rms``.
    """
    exposure_offsets = exposure_offsets or {}
    parts, owners = [], []
    for i, (key, coords) in enumerate(sorted(satstars_by_exposure.items(),
                                             key=lambda kv: str(kv[0]))):
        off = exposure_offsets.get(key)
        if off is not None and np.all(np.isfinite(off)):
            coords = _shift(coords, off[0], off[1])
        parts.append(coords)
        owners.append(np.full(len(coords), i))
    empty = dict(coords=SkyCoord([] * u.deg, [] * u.deg), nexp=np.zeros(0, int),
                 rms_mas=np.zeros(0), n_groups=0, n_rejected_rms=0)
    if not parts or sum(len(p) for p in parts) == 0:
        return empty
    ra = np.concatenate([p.ra.deg for p in parts])
    dec = np.concatenate([p.dec.deg for p in parts])
    owner = np.concatenate(owners)
    allc = SkyCoord(ra, dec, unit="deg")
    n = len(allc)

    # Link fits from DIFFERENT exposures within match_radius; the connected
    # components are the stars.
    i1, i2, _, _ = search_around_sky(allc, allc, match_radius)
    cross = owner[i1] != owner[i2]
    graph = coo_matrix((np.ones(cross.sum()), (i1[cross], i2[cross])), shape=(n, n))
    _, label = connected_components(graph, directed=False)

    order = np.argsort(label, kind="stable")
    bounds = np.flatnonzero(np.diff(label[order])) + 1
    keep_ra, keep_dec, keep_n, keep_rms = [], [], [], []
    n_groups = n_rejected = 0
    for members in np.split(order, bounds):
        if len(np.unique(owner[members])) < min_exposures:
            continue
        n_groups += 1
        # One fit per exposure: a second fit of the same exposure inside the
        # group is a deblend fragment, not an independent measurement.  Keep
        # the one nearest the group's centre.
        cdec = np.radians(np.median(dec[members]))
        x = (ra[members] - np.median(ra[members])) * np.cos(cdec) * 3.6e6
        y = (dec[members] - np.median(dec[members])) * 3.6e6
        dist = np.hypot(x, y)
        chosen = []
        for o in np.unique(owner[members]):
            sel = np.flatnonzero(owner[members] == o)
            chosen.append(sel[np.argmin(dist[sel])])
        chosen = np.asarray(chosen)
        xm, ym = x[chosen].mean(), y[chosen].mean()
        rms = float(np.sqrt(np.mean((x[chosen] - xm) ** 2 + (y[chosen] - ym) ** 2)))
        if rms > max_rms_mas:
            n_rejected += 1
            continue
        keep_ra.append(np.median(ra[members]) + xm / 3.6e6 / np.cos(cdec))
        keep_dec.append(np.median(dec[members]) + ym / 3.6e6)
        keep_n.append(len(chosen))
        keep_rms.append(rms)
    if not keep_ra:
        empty.update(n_groups=n_groups, n_rejected_rms=n_rejected)
        return empty
    return dict(coords=SkyCoord(keep_ra, keep_dec, unit="deg"),
                nexp=np.asarray(keep_n, int), rms_mas=np.asarray(keep_rms),
                n_groups=n_groups, n_rejected_rms=n_rejected)


def augment_consensus_with_satstars(cons, satcons,
                                    exclusion_radius=SATSTAR_CONSENSUS_EXCLUSION_ARCSEC * u.arcsec):
    """Consensus coords/mags for the reference tie, with satstars appended.

    A satstar within ``exclusion_radius`` of any daophot consensus star is not
    added (the daophot row already represents that star).  Appended stars get
    a NaN magnitude, which ``measure_reference_tie`` treats as unknown.

    Returns ``(coords, mag, n_added)``.
    """
    coords = cons["coords"]
    mag = cons.get("mag")
    sat = satcons["coords"]
    if len(sat) == 0:
        return coords, mag, 0
    add = np.ones(len(sat), bool)
    if len(coords):
        i_sat, _, _, _ = search_around_sky(sat, coords, exclusion_radius)
        add[i_sat] = False
    if not add.any():
        return coords, mag, 0
    new = sat[add]
    out = SkyCoord(np.concatenate([coords.icrs.ra.deg, new.ra.deg]),
                   np.concatenate([coords.icrs.dec.deg, new.dec.deg]), unit="deg")
    if mag is not None:
        mag = np.concatenate([np.asarray(mag, float), np.full(int(add.sum()), np.nan)])
    return out, mag, int(add.sum())


def consensus_with_satstars(cons, exposure_tables, satstars_by_exposure):
    """Reference-tie inputs for one visit: the daophot consensus plus its
    repeatable satstars.

    ``cons`` is ``build_visit_consensus`` output for ``exposure_tables``;
    ``satstars_by_exposure`` may hold other visits' exposures too, and only
    this visit's are used.  Each exposure's satstars are carried onto the
    consensus frame by that exposure's own ``vs_consensus`` offset.

    Returns ``(coords, mag, summary)``; ``summary`` is JSON-able for the
    checkpoint record.
    """
    keys = {exposure_key(t) for t in exposure_tables}
    mine = {k: v for k, v in satstars_by_exposure.items() if k in keys}
    offsets = {}
    for e in cons.get("exposures", []):
        vs = e.get("vs_consensus")
        if vs is not None and vs.get("ok", True):
            offsets[tuple(e["key"])] = (float(vs["dra"]), float(vs["ddec"]))
    sat = satstar_consensus(mine, exposure_offsets=offsets)
    coords, mag, n_added = augment_consensus_with_satstars(cons, sat)
    summary = dict(
        n_exposures_with_satstars=len(mine),
        n_groups_multi_exposure=int(sat["n_groups"]),
        n_rejected_rms=int(sat["n_rejected_rms"]),
        n_repeatable=int(len(sat["coords"])),
        n_added=int(n_added),
        median_rms_mas=(float(np.median(sat["rms_mas"]))
                        if len(sat["rms_mas"]) else float("nan")),
        match_arcsec=SATSTAR_CONSENSUS_MATCH_ARCSEC,
        min_exposures=SATSTAR_CONSENSUS_MIN_EXPOSURES,
        max_rms_mas=SATSTAR_CONSENSUS_MAX_RMS_MAS,
        exclusion_arcsec=SATSTAR_CONSENSUS_EXCLUSION_ARCSEC,
    )
    return coords, mag, summary
