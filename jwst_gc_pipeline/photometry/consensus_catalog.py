"""The JWST-internal astrometric reference: per-filter consensus catalogs.

VIRAC2 is the absolute frame, and it is the limit on how well anything can be
tied to it: ~40 mas per star, propagated from a 2014.0 reference epoch. The
JWST data are far better than that against *themselves* — the same stars, in the
same filter, minutes apart — so a catalog built from them is both deeper than
VIRAC2 and internally more precise. Two products come out of that:

**The per-filter consensus catalog.** ``build_visit_consensus`` measures one
``(visit, filter)`` at a time, because detecting a misaligned exposure means
comparing it against its *own* visit's other exposures. Those per-visit
consensi are then pooled here into one catalog per filter,
``catalogs/<filter>_consensus.fits``, written at the m2 checkpoint (after the
m12 merge, the first per-frame catalogs).

**The JWST reference-filter consensus.** One filter of the field is the
reference: whichever is closest to VIRAC2 in both wavelength and in which stars
it leaves unsaturated (:func:`reference_filter`). Its per-filter consensus,
``catalogs/jwst_reference_consensus.fits``, is what every *other* filter ties
to. Only that one filter ties to VIRAC2. A filter tied through it inherits no
VIRAC2 per-star error and no proper-motion propagation, so the tie should be
much tighter than a direct VIRAC2 tie — how much tighter is an open question
this module records rather than assumes.

Promotion is a separate, explicit step (``scripts/reduction/promote_reference_consensus.py``)
because it needs the field's whole filter list, which one per-filter m2
checkpoint does not have.

See ``docs/JWST_CONSENSUS_CATALOG.md``.
"""
import os
import re

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table

from .astrometry_offsets import measure_offset

#: VIRAC2's astrometry is VVV Ks-monitoring; Ks is 2.15 um.  Closeness to it is
#: the first thing that makes a JWST filter a good match.
VIRAC2_KS_MICRON = 2.15

#: How much of a filter's light gets through, which decides which stars
#: saturate.  A narrow filter keeps VIRAC2's bright stars unsaturated, and
#: those are the stars the tie is made on.
_WIDTH_RANK = {'N': 0, 'M': 1, 'W': 2}

#: What one bandwidth class is worth, in ln(lambda) from Ks.  The two criteria
#: TRADE OFF rather than one outranking the other, and this is the exchange
#: rate.  Distance is measured in LOG wavelength, which is what makes the
#: ordering consistent across the whole range: in linear wavelength no positive
#: long-wavelength penalty can put F277W above F140M, because 0.62 um from Ks
#: looks worse than 0.75 um while ln(2.77/2.15) < ln(1.40/2.15).
#:
#: Solved from the intended orderings
#: F212N > F210M > F187N > F182M > F200W > F150W and F277W > F140M > F115W:
#:
#:     F210M before F187N  ->  0.0235 + w < 0.1394  ->  w < 0.116
#:     F182M before F200W  ->  0.1665 + w < 0.0723 + 2w  ->  w > 0.094
#:     F277W before F140M  ->  0.2532 + 2w < 0.4293 + w  ->  w < 0.176
#:
#: so 0.094 < w < 0.116; 0.105 sits in the middle.  No separate channel term is
#: needed: log distance already puts MIRI last, and it lets a narrow long-
#: wavelength filter outrank a wide blue one on its merits rather than by fiat.
#:
#: Both bounds are set by a comparison that decides a REAL field's anchor --
#: F210M vs F187N (sickle, w51) above, F182M vs F200W (ngc6334/7213) below --
#: so those two fields' picks are effectively the free parameter here.  A third
#: pair, F164N vs F182M, is separated by only 0.00086 in rank and flips at
#: w >= 0.1045, so 0.105 sits just past that flip; one step either way down the
#: ranking is still a reasonable anchor.
_WIDTH_COST_LOG = 0.105

#: ``F212N`` / ``F150W2`` / ``F1130W``: digits, then the bandwidth class, then
#: an optional trailing ``2`` (the NIRCam wide-double filters).  Parsing by
#: "strip every digit" turns F150W2 into 15.02 um -- a 10x error that ranks a
#: 1.5 um SW filter below most of MIRI.
_FILTER_RE = re.compile(r'^F(\d+)([NMW])(\d?)$', re.IGNORECASE)

#: An inter-visit residual this large means the visits are not on a common
#: frame, and pooling them would bake HALF of it into every shared star's
#: position (the pooled position lands at the midpoint).  This is a GROSS gate
#: in the sense CLAUDE.md uses for the sparse-Gaia cross-check: it catches
#: "these visits were never tied", not a few-mas imperfection.  The measured
#: per-visit values are recorded in the output meta either way, which is what
#: makes a FINE threshold choosable later.
GROSS_INTER_VISIT_MAS = 100.0


class NoReferenceFilterError(ValueError):
    """The field has no filter that can anchor the others."""


class InterVisitOffsetError(ValueError):
    """The visits of one filter are not on a common frame, so pooling them
    would bias every shared star."""


def _filter_micron(filtername):
    """``'F212N'`` -> 2.12, ``'F150W2'`` -> 1.50, ``'F1130W'`` -> 11.30.

    The digits are the wavelength in units of 0.01 um.  The trailing ``2`` of
    a wide-double filter is part of the NAME, not the number.
    """
    m = _FILTER_RE.match(str(filtername).strip())
    if not m:
        raise ValueError(f'{filtername!r} is not a JWST filter name')
    return int(m.group(1)) / 100.0


def _filter_width_class(filtername):
    """``N`` -> 0, ``M`` -> 1, ``W``/``W2`` -> 2; how much light gets through,
    which is what decides whether VIRAC2's bright stars saturate."""
    m = _FILTER_RE.match(str(filtername).strip())
    if not m:
        raise ValueError(f'{filtername!r} is not a JWST filter name')
    return _WIDTH_RANK[m.group(2).upper()]


def reference_filter_rank(filtername):
    """How good an anchor this filter is for the field's others; lower is better.

    ``|ln(lambda / Ks)| + 0.105 * bandwidth_class``.  Reproduces both intended
    orderings: ``F212N > F210M > F187N > F182M > F200W > F150W`` and
    ``F277W > F140M > F115W``.
    """
    return (abs(np.log(_filter_micron(filtername) / VIRAC2_KS_MICRON))
            + _WIDTH_COST_LOG * _filter_width_class(filtername))


def reference_filter(filternames):
    """Which of a field's filters anchors the others.

    Closest to VIRAC2 in the two senses that matter: wavelength, and which
    stars it leaves unsaturated.  Entries that are not filter names (``CLEAR``,
    ``''``) are ignored rather than crashing the sort.
    """
    usable, unparsed = [], []
    for name in filternames:
        try:
            reference_filter_rank(name)
        except ValueError:
            if str(name).strip():
                unparsed.append(str(name))
            continue
        usable.append(name)
    if not usable:
        raise NoReferenceFilterError(
            'no rankable filter among ' + (repr(list(filternames)) or '[]')
            + (f' (unparsed: {unparsed})' if unparsed else '')
            + ', so no reference filter can be chosen')
    return sorted(usable, key=reference_filter_rank)[0]


def consensus_obs_token(proposal_id, obsid):
    """The filename token that keeps two observations' consensus catalogs apart.

    Prefers ``crowdsource_catalogs_long.obs_token``, which already encodes the
    cases where the disambiguator must be the PROPOSAL rather than the
    observation: ngc6334's 6778 and 7213 share a target directory, a filter
    list AND obsid ``001``, so ``_o001`` would not separate them and ``_j7213``
    does.

    Where that helper returns nothing, this falls back to ``_o<obsid>``.  It
    covers only the proposals whose per-FRAME product names were already
    colliding; cloudef/2092 has two obsids under one directory and is not among
    them, so its consensus catalogs would overwrite each other.  These files are
    new, so naming every one of them by observation costs nothing and does not
    touch a legacy filename.
    """
    from .crowdsource_catalogs_long import obs_token as _legacy_obs_token

    token = _legacy_obs_token(proposal_id, obsid)
    if token:
        return token
    obsid = str(obsid or '').strip()
    return f'_o{obsid}' if obsid else ''


def consensus_path(basepath, filtername, obs_token=''):
    """Where one filter's pooled consensus catalog lives.

    ``obs_token`` is the per-observation disambiguator from
    ``crowdsource_catalogs_long.obs_token`` (``_j7213``, ``_o023``, ...).  It is
    not decorative: ngc6334's proposals 6778 and 7213 share a target directory
    AND a filter list at reference epochs 1.6 yr apart, and cloudef/2092 has two
    obsids under one directory -- without the token the second m2 checkpoint
    silently overwrites the first field's reference catalog.
    """
    token = str(obs_token or '')
    return os.path.join(basepath, 'catalogs',
                        f'{str(filtername).lower()}{token}_consensus.fits')


def reference_consensus_path(basepath, obs_token=''):
    """Where the field's reference-filter consensus lives."""
    token = str(obs_token or '')
    return os.path.join(basepath, 'catalogs',
                        f'jwst_reference{token}_consensus.fits')


def _as_coord_array(coords):
    """A scalar ``SkyCoord`` (a one-star visit) has no ``len``; make it 1-D."""
    if coords is None:
        return None
    return coords.reshape(1) if coords.isscalar else coords


def _visit_arrays(cons, n):
    """Pull the per-star columns ``build_visit_consensus`` produces, filling
    the ones an older/partial caller may not carry."""
    out = {}
    for key in ('nexp', 'scatter_mas', 'mag'):
        val = cons.get(key)
        if val is None:
            out[key] = np.full(n, np.nan)
        else:
            arr = np.atleast_1d(np.asarray(val, dtype=float))
            out[key] = arr if len(arr) == n else np.full(n, np.nan)
    return out


def _measure_inter_visit(per_visit, visits, context=''):
    """Each visit's bulk offset from the pooling anchor (the first visit).

    Recorded, not applied.  ``pool_visit_consensi`` deliberately does not
    re-tie the visits -- each is already on the frame the checkpoint that built
    it verified, and a second correction here would fold avoidable noise into
    the positions that are meant to BE the reference.  But averaging visits
    that are NOT on a common frame lands every shared star at the midpoint, so
    the premise has to be checked rather than assumed.

    Measured with the density-immune offset histogram (``measure_offset``, with
    the window sweep), never a nearest-neighbour median.  Against a dense
    catalog the histogram peak over-reads by a few mas (see the
    ``histogram-vs-samestar-offset-bias`` note), which is why the gate this
    feeds is GROSS.
    """
    anchor = visits[0]
    out = {anchor: dict(off_mas=0.0, dra_mas=0.0, ddec_mas=0.0,
                        anchor=True, measured=True)}
    for visit in visits[1:]:
        res = measure_offset(_as_coord_array(per_visit[visit]['coords']),
                             _as_coord_array(per_visit[anchor]['coords']),
                             sweep=True,
                             context=f'{context} visit {visit} vs {anchor}')
        if res is None:
            # Too few stars to form a histogram peak.  Reported as
            # could-not-verify rather than folded into the gate as a 0: an
            # unmeasurable tie is not a passing one, and the count lands in the
            # output meta so a reader can see the check did not run.
            out[visit] = dict(off_mas=float('nan'), dra_mas=float('nan'),
                              ddec_mas=float('nan'), anchor=False,
                              measured=False)
            continue
        out[visit] = dict(off_mas=float(res['off']),
                          dra_mas=float(res['dra']),
                          ddec_mas=float(res['ddec']),
                          contrast=float(res.get('contrast', np.nan)),
                          anchor=False, measured=True)
    return out


def _group_across_visits(coords, origin, match_radius):
    """Group the SAME star seen in several visits; never merge two stars of one
    visit.

    ``search_around_sky`` (ALL pairs within the radius), never a
    nearest-partner match: a single-partner pass caps a group at two members,
    so a star seen in three visits came out as one merged pair plus a duplicate
    row and ``n_visits`` could never exceed 2.  ngc6334/6778 and wd1/1905 both
    have three visits today.  (``search_around_sky`` is also the method the
    repo's NN-median guard deliberately does not flag, because it is the basis
    of the sanctioned offset-histogram stacking.)

    Union is greedy in order of separation and REFUSES to merge two groups that
    already share a visit: association is across visits, so two genuinely close
    stars in the same visit stay two stars however near.  That also stops a
    transitive chain A(v1)-B(v2)-C(v1) from quietly merging A with C.
    """
    n = len(coords)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    i1, i2, sep, _ = search_around_sky(coords, coords, match_radius)
    keep = i1 < i2
    i1, i2, sep = i1[keep], i2[keep], sep[keep]
    order = np.argsort(sep.arcsec)

    visits_of = {i: {int(origin[i])} for i in range(n)}
    for k in order:
        a, b = find(int(i1[k])), find(int(i2[k]))
        if a == b or visits_of[a] & visits_of[b]:
            continue
        parent[b] = a
        visits_of[a] |= visits_of.pop(b)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [np.asarray(v) for _, v in sorted(groups.items())]


def _mean_direction(coords):
    """Mean position as a unit-vector average, so RA wrap is not a special
    case (no current field sits near RA=0, but the reference catalog is not
    where to leave that latent)."""
    xyz = coords.cartesian.xyz.value
    mean = xyz.mean(axis=1)
    norm = np.linalg.norm(mean)
    if not np.isfinite(norm) or norm == 0:
        return float(coords.ra.deg[0]), float(coords.dec.deg[0])
    c = SkyCoord(*(mean / norm), representation_type='cartesian').icrs
    c.representation_type = 'spherical'
    return float(c.ra.deg), float(c.dec.deg)


def _mean_mag(mags):
    """Average magnitudes in FLUX, not in magnitudes."""
    finite = mags[np.isfinite(mags)]
    if not len(finite):
        return np.nan
    return float(-2.5 * np.log10(np.mean(10.0 ** (-0.4 * finite))))


def pool_visit_consensi(per_visit, match_radius=0.2 * u.arcsec,
                        gross_inter_visit_mas=GROSS_INTER_VISIT_MAS,
                        context=''):
    """One filter's per-visit consensi -> one catalog for the filter.

    ``per_visit`` maps a visit to the dict ``build_visit_consensus`` returns
    (``coords``, ``nexp``, ``scatter_mas``).  Stars seen in more than one visit
    are averaged; each row records how many visits and how many exposures saw
    it, and what the contributing positions' scatter was, so a consumer can
    both filter on redundancy and know the precision of what it is tying to.

    Raises ``InterVisitOffsetError`` when a visit sits more than
    ``gross_inter_visit_mas`` from the anchor: pooling visits that are not on a
    common frame puts every shared star at the midpoint of the disagreement,
    and this catalog is meant to BE the thing other filters tie to.  The
    measured per-visit offsets are written to the meta regardless.
    """
    visits, coords_by_visit = [], {}
    for visit, cons in sorted(per_visit.items()):
        if cons is None:
            continue
        coords = _as_coord_array(cons.get('coords'))
        if coords is None or len(coords) == 0:
            continue
        visits.append(visit)
        coords_by_visit[visit] = coords
    if not visits:
        raise ValueError('no visit consensus to pool')

    inter_visit = (_measure_inter_visit(per_visit, visits, context=context)
                   if len(visits) > 1 else
                   {visits[0]: dict(off_mas=0.0, dra_mas=0.0, ddec_mas=0.0,
                                    anchor=True, measured=True)})
    measured = [v['off_mas'] for v in inter_visit.values() if v['measured']]
    n_unmeasured = sum(1 for v in inter_visit.values() if not v['measured'])
    worst = max(measured) if measured else float('nan')
    if measured and worst > gross_inter_visit_mas:
        bad = {k: round(v['off_mas'], 1) for k, v in inter_visit.items()
               if v['measured'] and v['off_mas'] > gross_inter_visit_mas}
        raise InterVisitOffsetError(
            f'visits {bad} sit more than {gross_inter_visit_mas:g} mas from '
            f'anchor visit {visits[0]}; pooling them would bake half of that '
            f'into every star seen in both. {context}'.strip())

    coords = SkyCoord(
        ra=np.concatenate([coords_by_visit[v].ra.deg for v in visits]) * u.deg,
        dec=np.concatenate([coords_by_visit[v].dec.deg for v in visits]) * u.deg,
        frame='icrs')
    origin = np.concatenate([np.full(len(coords_by_visit[v]), i)
                             for i, v in enumerate(visits)])
    cols = {k: np.concatenate([_visit_arrays(per_visit[v],
                                             len(coords_by_visit[v]))[k]
                               for v in visits])
            for k in ('nexp', 'scatter_mas', 'mag')}

    groups = _group_across_visits(coords, origin, match_radius)
    ra = np.empty(len(groups))
    dec = np.empty(len(groups))
    n_visits = np.empty(len(groups), dtype=int)
    n_exposures = np.empty(len(groups))
    scatter = np.empty(len(groups))
    err = np.empty(len(groups))
    mag = np.empty(len(groups))
    for k, g in enumerate(groups):
        ra[k], dec[k] = _mean_direction(coords[g])
        n_visits[k] = len({int(origin[i]) for i in g})
        n_exposures[k] = np.nansum(cols['nexp'][g])
        finite_scatter = cols['scatter_mas'][g][np.isfinite(cols['scatter_mas'][g])]
        scatter[k] = (np.sqrt(np.mean(finite_scatter ** 2))
                      if len(finite_scatter) else np.nan)
        # Error on the MEAN of the contributing per-exposure positions.  With
        # one exposure there is no scatter to divide, so this stays NaN rather
        # than claiming 0 -- an identically-zero uncertainty free-passes a QC
        # gate (see the std=0 forced-position note).
        n_eff = n_exposures[k]
        err[k] = (scatter[k] / np.sqrt(n_eff)
                  if np.isfinite(scatter[k]) and np.isfinite(n_eff) and n_eff > 1
                  else np.nan)
        mag[k] = _mean_mag(cols['mag'][g])

    out = Table()
    out['RA'] = ra
    out['DEC'] = dec
    out['n_visits'] = n_visits
    out['n_exposures'] = n_exposures
    out['scatter_mas'] = scatter
    out['err_mas'] = err
    out['refmag'] = mag
    out['skycoord'] = SkyCoord(ra * u.deg, dec * u.deg)
    out.meta['NVISITS'] = len(visits)
    out.meta['VISITS'] = ','.join(str(v) for v in visits)
    out.meta['ANCHORVI'] = str(visits[0])
    out.meta['IVMAXMAS'] = float(worst)
    out.meta['IVGROSS'] = float(gross_inter_visit_mas)
    out.meta['IVNOMEAS'] = int(n_unmeasured)
    for visit, res in inter_visit.items():
        out.meta[f'IV_{visit}'] = (
            'NOT MEASURABLE (too few stars for a histogram peak)'
            if not res['measured'] else
            f"off={res['off_mas']:.2f} dra={res['dra_mas']:+.2f} "
            f"ddec={res['ddec_mas']:+.2f} mas")
    return out


def write_filter_consensus(basepath, filtername, table, absolute_tie=None,
                           obs_token=''):
    """Persist one filter's consensus catalog, atomically.

    ``absolute_tie`` is the ``measure_reference_tie`` result for this filter
    when it has one, so the file records what it is tied to and how well.
    """
    from ..atomic_io import write_table_atomic

    name = str(filtername).strip().upper()
    if not name:
        raise ValueError(
            'refusing to write a consensus catalog with no filter name: the '
            'file is keyed by filter and an "unknown" one cannot be tied to, '
            'promoted, or told apart from the next filter that lands here')
    path = consensus_path(basepath, name, obs_token=obs_token)
    table = Table(table)
    table.meta['FILTER'] = name
    table.meta['CONSTYPE'] = 'per-filter JWST consensus'
    table.meta['OBSTOKEN'] = str(obs_token or '')
    if absolute_tie is not None:
        table.meta['TIEDTO'] = str(absolute_tie.get('reference', 'VIRAC2'))
        table.meta['TIEMAS'] = float(absolute_tie.get('off_mas', np.nan))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_table_atomic(table, path, format='fits')
    return path


def promote_reference_filter(basepath, filternames, obs_token=''):
    """Copy the chosen filter's consensus to the field's reference catalog.

    Returns ``(reference_filter, path)``.  Raises when that filter has no
    consensus on disk: the alternative is tying every other filter to a
    silently-absent reference, which is the shape of failure this whole
    ladder exists to prevent.

    Called by ``scripts/reduction/promote_reference_consensus.py`` once every
    filter's m2 checkpoint has run -- not by the checkpoint itself, which sees
    one filter and cannot know which of the field's filters ranks best.
    """
    from ..atomic_io import write_table_atomic

    chosen = reference_filter(filternames)
    source = consensus_path(basepath, chosen, obs_token=obs_token)
    if not os.path.exists(source):
        raise FileNotFoundError(
            f'{chosen} is this field\'s reference filter (closest to VIRAC2 in '
            f'wavelength and in which stars it leaves unsaturated), but its '
            f'consensus catalog {source} is not there.  Run the m2 checkpoint '
            f'for {chosen} before tying the other filters.')
    table = Table.read(source)
    table.meta['CONSTYPE'] = 'JWST reference-filter consensus'
    table.meta['REFFILT'] = str(chosen).upper()
    path = reference_consensus_path(basepath, obs_token=obs_token)
    write_table_atomic(table, path, format='fits')
    return chosen, path


def tie_to_reference_consensus(coords, reference_coords, context=''):
    """Measure one filter's offset from the field's reference-filter consensus.

    The density-immune histogram, as everywhere else.  Returns the
    ``measure_offset`` result with ``off_mas`` added.

    This tie carries neither VIRAC2's ~40 mas per-star error nor its
    proper-motion propagation, so it should come out far tighter than the same
    filter's tie to VIRAC2.  How tight is not yet established, so nothing here
    gates on a threshold -- the number is measured and recorded, and the
    tolerance is set once there are measurements to set it from.
    """
    result = measure_offset(coords, reference_coords, sweep=True,
                            context=context)
    # measure_offset already reports on-sky milliarcsec: `dra`/`ddec` are the
    # components and `off` their magnitude.  Aliased here so a caller reading
    # `off_mas` cannot mistake it for degrees or arcsec.
    result['off_mas'] = float(result['off'])
    return result
