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
this module records rather than assumes (see ``ties`` in the written meta).

See ``docs/JWST_CONSENSUS_CATALOG.md``.
"""
import os

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
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
#:     F182M before F200W  ->  0.1665 + w < 0.0728 + 2w  ->  w > 0.094
#:     F277W before F140M  ->  0.2532 + 2w < 0.4293 + w  ->  w < 0.176
#:
#: so 0.094 < w < 0.116; 0.105 sits in the middle.  No separate channel term is
#: needed: log distance already puts MIRI last, and it lets a narrow long-
#: wavelength filter outrank a wide blue one on its merits rather than by fiat.
_WIDTH_COST_LOG = 0.105

class NoReferenceFilterError(ValueError):
    """The field has no filter that can anchor the others."""


def _filter_micron(filtername):
    """``'F212N'`` -> 2.12.  The digits are the wavelength in units of 0.01 um
    below 1000, and 0.1 um above (F1130W is 11.30 um)."""
    digits = ''.join(c for c in str(filtername) if c.isdigit())
    if not digits:
        raise ValueError(f'{filtername!r} carries no wavelength')
    return int(digits) / 100.0


def reference_filter_rank(filtername):
    """How good an anchor this filter is for the field's others; lower is better.

    ``|ln(lambda / Ks)| + 0.105 * bandwidth_class``.  Reproduces both intended
    orderings: ``F212N > F210M > F187N > F182M > F200W > F150W`` and
    ``F277W > F140M > F115W``.
    """
    name = str(filtername).upper()
    micron = _filter_micron(name)
    width = _WIDTH_RANK.get(name[-1], _WIDTH_RANK['W'])
    return (abs(np.log(micron / VIRAC2_KS_MICRON))
            + _WIDTH_COST_LOG * width)


def reference_filter(filternames):
    """Which of a field's filters anchors the others.

    Closest to VIRAC2 in the two senses that matter: wavelength, and which
    stars it leaves unsaturated.
    """
    usable = [f for f in filternames if str(f).strip()]
    if not usable:
        raise NoReferenceFilterError(
            'no filters given, so no reference filter can be chosen')
    return sorted(usable, key=reference_filter_rank)[0]


def consensus_path(basepath, filtername):
    """Where one filter's pooled consensus catalog lives."""
    return os.path.join(basepath, 'catalogs',
                        f'{str(filtername).lower()}_consensus.fits')


def reference_consensus_path(basepath):
    """Where the field's reference-filter consensus lives."""
    return os.path.join(basepath, 'catalogs', 'jwst_reference_consensus.fits')


def pool_visit_consensi(per_visit, match_radius=0.2 * u.arcsec):
    """One filter's per-visit consensi -> one catalog for the filter.

    ``per_visit`` maps a visit to the dict ``build_visit_consensus`` returns.
    Stars seen in more than one visit are averaged; each row records how many
    visits it was seen in, so a consumer can require more than one.

    Visits are pooled WITHOUT re-measuring an offset between them: each visit's
    consensus is already tied to the same frame by the checkpoint that built it,
    and re-tying here would fold a second, unnecessary measurement into
    positions that are meant to be the reference.
    """
    visits = [v for v, cons in sorted(per_visit.items())
              if cons is not None and len(cons.get('coords', ())) > 0]
    if not visits:
        raise ValueError('no visit consensus to pool')

    coords = SkyCoord(
        ra=np.concatenate([per_visit[v]['coords'].ra.deg for v in visits]) * u.deg,
        dec=np.concatenate([per_visit[v]['coords'].dec.deg for v in visits]) * u.deg,
        frame='icrs')
    origin = np.concatenate([np.full(len(per_visit[v]['coords']), i)
                             for i, v in enumerate(visits)])
    mags = []
    for v in visits:
        mag = per_visit[v].get('mag')
        n = len(per_visit[v]['coords'])
        mags.append(np.full(n, np.nan) if mag is None
                    else np.asarray(mag, dtype=float))
    mag = np.concatenate(mags)

    ra, dec, nvis, mean_mag = _average_repeats(coords, origin, mag, match_radius)
    out = Table()
    out['RA'] = ra
    out['DEC'] = dec
    out['n_visits'] = nvis
    out['refmag'] = mean_mag
    out['skycoord'] = SkyCoord(ra * u.deg, dec * u.deg)
    out.meta['NVISITS'] = len(visits)
    out.meta['VISITS'] = ','.join(str(v) for v in visits)
    return out


def _average_repeats(coords, origin, mag, match_radius):
    """Average stars that appear in more than one visit.

    Nearest-pair association, which is safe here for the reason the module
    header gives: every visit consensus is already on the same frame, so the
    nearest source within 0.2" is the same star.  (This is association, not
    offset measurement -- the banned pattern is measuring a SHIFT from nearest
    pairs.)
    """
    ra = coords.ra.deg
    if len(coords) < 2:
        # match_to_catalog_sky(nthneighbor=2) needs a second neighbour to find.
        with np.errstate(invalid='ignore'):
            return (ra.copy(), coords.dec.deg.copy(),
                    np.ones(len(coords), dtype=int), np.asarray(mag, dtype=float))
    idx, sep, _ = coords.match_to_catalog_sky(coords, nthneighbor=2)
    dec = coords.dec.deg
    taken = np.zeros(len(coords), dtype=bool)
    out_ra, out_dec, out_n, out_mag = [], [], [], []
    for i in range(len(coords)):
        if taken[i]:
            continue
        partner = int(idx[i])
        group = [i]
        if (sep[i] < match_radius and not taken[partner]
                and origin[partner] != origin[i]):
            group.append(partner)
        taken[group] = True
        out_ra.append(float(np.mean(ra[group])))
        out_dec.append(float(np.mean(dec[group])))
        out_n.append(len({int(origin[g]) for g in group}))
        with np.errstate(invalid='ignore'):
            out_mag.append(float(np.nanmean(mag[group])))
    return (np.array(out_ra), np.array(out_dec),
            np.array(out_n, dtype=int), np.array(out_mag))


def write_filter_consensus(basepath, filtername, table, absolute_tie=None):
    """Persist one filter's consensus catalog, atomically.

    ``absolute_tie`` is the ``measure_reference_tie`` result for this filter
    when it has one, so the file records what it is tied to and how well.
    """
    from ..atomic_io import write_table_atomic

    path = consensus_path(basepath, filtername)
    table = Table(table)
    table.meta['FILTER'] = str(filtername).upper()
    table.meta['CONSTYPE'] = 'per-filter JWST consensus'
    if absolute_tie is not None:
        table.meta['TIEDTO'] = str(absolute_tie.get('reference', 'VIRAC2'))
        table.meta['TIEMAS'] = float(absolute_tie.get('off_mas', np.nan))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_table_atomic(table, path, format='fits')
    return path


def promote_reference_filter(basepath, filternames):
    """Copy the chosen filter's consensus to the field's reference catalog.

    Returns ``(reference_filter, path)``.  Raises when that filter has no
    consensus on disk: the alternative is tying every other filter to a
    silently-absent reference, which is the shape of failure this whole
    ladder exists to prevent.
    """
    from ..atomic_io import write_table_atomic

    chosen = reference_filter(filternames)
    source = consensus_path(basepath, chosen)
    if not os.path.exists(source):
        raise FileNotFoundError(
            f'{chosen} is this field\'s reference filter (closest to VIRAC2 in '
            f'wavelength and in which stars it leaves unsaturated), but its '
            f'consensus catalog {source} is not there.  Run the m2 checkpoint '
            f'for {chosen} before tying the other filters.')
    table = Table.read(source)
    table.meta['CONSTYPE'] = 'JWST reference-filter consensus'
    table.meta['REFFILT'] = str(chosen).upper()
    path = reference_consensus_path(basepath)
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
