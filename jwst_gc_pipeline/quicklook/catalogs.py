"""Where the F212N/F480M photometry for a Treasury pointing comes from.

Two sources, checked in that order per POINTING:

* **jicama** -- this pipeline's own per-filter catalogs under the field's
  ``catalogs/`` directory.  These are the ones we want.
* **MAST** -- the ``*_cat.ecsv`` source catalogs delivered with a Level-3
  association.

The choice is made per pointing rather than once for the survey.  Program 10678
is being reduced tile by tile, so at any moment some pointings have pipeline
catalogs and some do not; a global "jicama or MAST" switch would either throw
away the pipeline work that exists or refuse to plot the tiles it does not
cover yet.  Which source each pointing used is recorded and shown on the page.

As of 2026-09-14 the MAST side is empty for 10678: the tiles land level-2 only,
with no image3 association, so there are no ``_cat.ecsv`` files to read (see the
``treasury-tiles-land-level2-only`` note).  The reader below is written to the
documented ``SourceCatalogStep`` schema and is exercised by a synthetic file in
the tests; it has not been run against a delivered 10678 catalog, because there
is not one.  It is here so that the page keeps working if MAST products arrive
before the pipeline reaches these tiles, which is the order the last two
programs arrived in.

**Magnitudes are Vega throughout**, matching the pipeline's ``mag_vega_<band>``
columns and the convention the rest of the survey's plots use.  Nothing here
reports AB.  A catalog that carries AB is converted (see ``AB_ZEROPOINT_JY``);
a catalog that carries raw instrumental flux is converted from that.
"""
import math
import os
import re
from pathlib import Path

import numpy as np

#: The colour-magnitude diagram this page is about.  ``blue`` is also the
#: magnitude axis, so the CMD is F212N vs F212N-F480M.
BLUE_BAND = 'f212n'
RED_BAND = 'f480m'
BANDS = (BLUE_BAND, RED_BAND)

#: AB's definition: 3631 Jy.  Only used to undo an AB magnitude back to a flux
#: before re-expressing it in Vega -- we never publish AB here.
AB_ZEROPOINT_JY = 3631.0

#: SVO Filter Profile Service Vega zeropoints (Jy), read 2026-09-14, cached so
#: a build does not fail when SVO is unreachable.  ``vega_zeropoint_jy`` asks
#: SVO first and falls back to these: the live service is the authority, and a
#: hardcoded number that silently goes stale is how a photometric offset gets
#: baked into a plot nobody re-derives.
#:
#: The implied AB-Vega offsets are F212N +1.827 and F480M +3.437 mag, which is
#: the size of the error made by plotting AB and labelling it Vega.
FALLBACK_VEGA_ZEROPOINT_JY = {
    'f212n': 674.83167374035,
    'f480m': 153.22388044927,
}

#: Seconds to wait on SVO before falling back to the cached zeropoints.
SVO_TIMEOUT_S = 30

#: Match radius for pairing a F212N detection with its F480M counterpart.
#: NIRCam SW pixels are 0.031" and LW 0.063", and both catalogs are on the same
#: pointing's own astrometric solution, so a real pair lands well inside this.
#: Widening it in a field this crowded buys wrong pairs, not completeness.
DEFAULT_MATCH_ARCSEC = 0.10

#: ``<filter>_<module>_o<NNN>_indivexp_merged[_resbgsub]_m<N>_dao_basic[_...]``
#:
#: ``lineage`` captures what sits between ``indivexp_merged`` and ``_m<N>``.
#: Empty for the plain chain, ``resbgsub`` for the residual-background-
#: subtracted one.  It is captured rather than absorbed into ``.*?`` because
#: **the two are different reductions, not successive passes of one**, so
#: their stage numbers are not on a common scale and comparing them as
#: integers is meaningless.
_JICAMA = re.compile(
    r'^(?P<filt>f\d{3,4}[a-z])_(?P<module>nrca|nrcb|merged)_o(?P<obs>\d{3})_'
    r'indivexp_merged(?:_(?P<lineage>[a-z]+))?'
    r'_m(?P<stage>\d+)_dao_basic(?P<tail>.*)\.fits$')

#: Products that share the naming convention but are not photometry.  An
#: ``_i2dseed`` file is the seed source LIST that a merge stage starts from, so
#: it carries positions and a placeholder flux; it outranks nothing, but a
#: stage present only as a seed would otherwise be selected as if it were the
#: catalog.
_NOT_PHOTOMETRY = ('_i2dseed',)

#: Module preference: ``merged`` covers both NIRCam modules of the pointing, so
#: it is the whole tile; a single-module file is the whole tile only for the
#: pointings that were observed with one module.  Preferring it means a
#: two-module pointing is never plotted as half of itself.
_MODULE_RANK = {'merged': 2, 'nrca': 1, 'nrcb': 0}


def _jicama_rank(path):
    """Sort key for picking one file per (pointing, band).

    WIDER module first, then later merge stage, then ``_vetted`` over raw.

    Coverage leads, and that ordering is load-bearing rather than a preference.
    The modules advance through the merge stages independently, so a pointing
    routinely has ``nrca`` at m3 while ``merged`` is still at m2 -- o127 and
    o129 both did on 2026-09-14.  Ranking by stage first picks the single
    module, which plots HALF the tile under the whole tile's name: o127 went
    from 86k matched pairs to 49k, and the diagram would have been labelled
    GC_127 either way.  A later merge stage is a refinement of the same stars;
    a missing module is missing sky.
    """
    m = _JICAMA.match(Path(path).name)
    tail = m.group('tail')
    return (_MODULE_RANK[m.group('module')], int(m.group('stage')),
            1 if 'vetted' in tail else 0)


def lineage_of(path):
    """``'plain'`` or the reduction variant token (e.g. ``'resbgsub'``).

    Kept separate from the stage because they are NOT the same axis.  On
    2026-09-15 o132 and o135 had ``resbgsub_m5`` alongside a plain ``m4``, and
    ranking by stage number alone silently preferred the resbgsub chain for
    those two fields while the other eight stayed plain -- a diagram pooling
    two reductions, with nothing saying so.  Reported rather than resolved:
    which chain a release should use is not this module's call.
    """
    m = _JICAMA.match(Path(path).name)
    return (m.group('lineage') or 'plain') if m else 'unknown'


def find_jicama(catalog_dir, bands=BANDS):
    """``{obsid: {band: path}}`` for the pipeline catalogs under a directory.

    Only pointings with a catalog in EVERY requested band are returned: a
    colour needs both, and a pointing that has F212N but no F480M yet (o133 and
    o138 as of this writing) belongs in the "not ready" list rather than in a
    half-drawn diagram.
    """
    catalog_dir = Path(catalog_dir)
    if not catalog_dir.is_dir():
        return {}
    found = {}
    for path in sorted(catalog_dir.glob('*.fits')):
        m = _JICAMA.match(path.name)
        if not m or m.group('filt') not in bands:
            continue
        if any(tag in m.group('tail') for tag in _NOT_PHOTOMETRY):
            continue
        found.setdefault(f"o{m.group('obs')}", {}).setdefault(
            m.group('filt'), []).append(path)
    out = {}
    for obsid, per_band in found.items():
        if not all(b in per_band for b in bands):
            continue
        out[obsid] = {b: max(per_band[b], key=_jicama_rank) for b in bands}
    return out


def _mast_bands(mast_dir, bands=BANDS):
    """``{obsid: {band: path}}`` for MAST catalogs, complete or not."""
    mast_dir = Path(mast_dir)
    if not mast_dir.is_dir():
        return {}
    out = {}
    for path in sorted(mast_dir.rglob('*_cat.ecsv')):
        m = re.match(r'^jw\d+-o(?P<obs>\d{3})_.*_(?P<optics>[^_]+)_cat\.ecsv$',
                     path.name)
        if not m:
            continue
        tokens = set(m.group('optics').lower().split('-'))
        band = next((b for b in bands if b in tokens), None)
        if band is not None:
            out.setdefault(f"o{m.group('obs')}", {})[band] = path
    return out


def find_mast(mast_dir, bands=BANDS):
    """``{obsid: {band: path}}`` for MAST ``*_cat.ecsv`` source catalogs.

    A level-3 catalog is named ``jw<prop>-o<NNN>_t<NNN>_nircam_<optics>_cat.ecsv``
    and the band is whichever requested filter appears in the optics element --
    read that way rather than by position, because a pupil-wheel band arrives as
    ``clear-f212n`` on some products and ``f212n-clear`` on others.

    Only pointings complete in every band, matching ``find_jicama``.
    ``_mast_bands`` is the unfiltered view, for reporting what is missing.
    """
    return {o: p for o, p in _mast_bands(mast_dir, bands).items()
            if all(b in p for b in bands)}


def choose_sources(jicama, mast):
    """Per pointing, the pipeline catalogs if it has them, else MAST.

    ``{obsid: {'source': 'jicama'|'mast', 'paths': {band: path}}}``.  Both
    inputs contain only pointings complete in every band, so the choice here is
    only about which source, never about whether there is enough to plot;
    ``incomplete_pointings`` reports the tiles that are still short a band.
    """
    chosen = {}
    for obsid in sorted(set(jicama) | set(mast)):
        if obsid in jicama:
            chosen[obsid] = {'source': 'jicama', 'paths': jicama[obsid]}
        else:
            chosen[obsid] = {'source': 'mast', 'paths': mast[obsid]}
    return chosen


def incomplete_pointings(catalog_dir, mast_dir, bands=BANDS):
    """Pointings that have one band but not the other, and which they have.

    Read straight off the filenames, so it costs a directory listing and names
    the tiles that are one reduction away from joining the plot.
    """
    have = {}
    for path in Path(catalog_dir).glob('*.fits') if Path(catalog_dir).is_dir() else ():
        m = _JICAMA.match(path.name)
        if (m and m.group('filt') in bands
                and not any(t in m.group('tail') for t in _NOT_PHOTOMETRY)):
            have.setdefault(f"o{m.group('obs')}", set()).add(m.group('filt'))
    # NOT `find_mast`, which pre-filters to pointings complete in every band:
    # feeding its output back in here meant a MAST pointing short a band was
    # filtered out before it could be reported, so the one function whose job
    # is naming incomplete pointings could never name a MAST one.
    for obsid, paths in _mast_bands(mast_dir, bands).items():
        have.setdefault(obsid, set()).update(paths)
    return {o: sorted(set(bands) - b) for o, b in sorted(have.items())
            if set(bands) - b}


_SVO_CACHE = {}


def vega_zeropoint_jy(band, allow_network=True):
    """SVO Vega zeropoint in Jy, falling back to the cached constant.

    The same number ``merge_catalogs`` uses, reached the same way, so a
    magnitude computed here and one read out of an ``m7`` table agree.
    """
    band = band.lower()
    if band in _SVO_CACHE:
        return _SVO_CACHE[band]
    if allow_network:
        try:
            from astroquery.svo_fps import SvoFps
            # The cached constants exist for an unreachable service, and a hang
            # has no path to them: without a timeout the build stops rather
            # than falls back.
            SvoFps.TIMEOUT = min(getattr(SvoFps, 'TIMEOUT', SVO_TIMEOUT_S) or
                                 SVO_TIMEOUT_S, SVO_TIMEOUT_S)
            table = SvoFps.get_filter_list('JWST')
            table.add_index('filterID')
            zp = float(table.loc[f'JWST/NIRCam.{band.upper()}']['ZeroPoint'])
            _SVO_CACHE[band] = zp
            return zp
        except (ImportError, KeyError, OSError, ValueError) as err:
            print(f"note: SVO zeropoint lookup for {band} failed ({err}); "
                  f"using the cached value")
    if band not in FALLBACK_VEGA_ZEROPOINT_JY:
        raise KeyError(f"no cached Vega zeropoint for {band}")
    _SVO_CACHE[band] = FALLBACK_VEGA_ZEROPOINT_JY[band]
    return _SVO_CACHE[band]


def _pixel_area_sr(meta):
    """Pixel solid angle, from whichever key the writer left behind.

    ``merge_catalogs`` stores ``pixelscale_deg2`` (an area); the per-filter
    tables carry ``PIXSCALE``, a side length in arcsec.  Both appear on disk.
    """
    if 'pixelscale_deg2' in meta:
        import astropy.units as u
        return float(u.Quantity(meta['pixelscale_deg2']).to(u.sr).value)
    for key in ('PIXSCALE', 'pixelscale_arcsec'):
        if key in meta:
            arcsec = float(getattr(meta[key], 'value', meta[key]))
            return (arcsec * math.pi / (180.0 * 3600.0)) ** 2
    raise KeyError('no pixel scale in catalog metadata '
                   f'(have {sorted(meta)[:12]}...)')


def vega_from_instrumental(flux, meta, band, allow_network=True):
    """Instrumental PSF flux (summed MJy/sr pixels) -> Vega magnitude.

    ``flux`` is a sum over pixels each in MJy/sr, so multiplying by the pixel
    solid angle gives MJy and 1e6 of those gives Jy -- the FWHM does not enter,
    because this is an integral and not a peak.  Same arithmetic as
    ``merge_catalogs``, kept here because the per-filter tables these read have
    not been through that step.
    """
    flux = np.asarray(flux, dtype=float)
    flux_jy = flux * _pixel_area_sr(meta) * 1e6
    zp = vega_zeropoint_jy(band, allow_network=allow_network)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(flux_jy > 0, -2.5 * np.log10(flux_jy / zp), np.nan)


def _skycoord(table):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    for name in ('skycoord', 'sky_centroid', 'sky_centroid_icrs'):
        if name in table.colnames:
            col = table[name]
            if isinstance(col, SkyCoord):
                return col
            return SkyCoord(col)
    for ra, dec in (('ra', 'dec'), ('RA', 'DEC'), ('sky_ra', 'sky_dec')):
        if ra in table.colnames and dec in table.colnames:
            return SkyCoord(np.asarray(table[ra], dtype=float) * u.deg,
                            np.asarray(table[dec], dtype=float) * u.deg)
    raise KeyError(f'no sky position in {table.colnames[:10]}...')


def load_band(path, band, allow_network=True):
    """``(SkyCoord, vega_mag)`` for one band of one pointing.

    Handles the three shapes that turn up: a pipeline per-filter table
    (instrumental ``flux``), a pipeline merged table (``mag_vega``), and a MAST
    source catalog (``aper_total_flux`` in Jy, or an AB magnitude).
    """
    from astropy.table import Table
    table = Table.read(path)
    coords = _skycoord(table)
    if 'mag_vega' in table.colnames:
        mag = np.asarray(table['mag_vega'], dtype=float)
    elif f'mag_vega_{band}' in table.colnames:
        mag = np.asarray(table[f'mag_vega_{band}'], dtype=float)
    elif 'aper_total_flux' in table.colnames:
        # MAST delivers this in Jy.
        import astropy.units as u
        col = table['aper_total_flux']
        flux_jy = np.asarray(u.Quantity(col, u.Jy).value
                             if col.unit else col, dtype=float)
        zp = vega_zeropoint_jy(band, allow_network=allow_network)
        with np.errstate(divide='ignore', invalid='ignore'):
            mag = np.where(flux_jy > 0, -2.5 * np.log10(flux_jy / zp), np.nan)
    elif 'aper_total_abmag' in table.colnames:
        # Undo AB back to a flux and re-express in Vega rather than applying a
        # difference of zeropoints: same answer, but it cannot be read as an
        # offset that might be stale.
        ab = np.asarray(table['aper_total_abmag'], dtype=float)
        flux_jy = AB_ZEROPOINT_JY * 10.0 ** (-0.4 * ab)
        zp = vega_zeropoint_jy(band, allow_network=allow_network)
        with np.errstate(divide='ignore', invalid='ignore'):
            mag = np.where(flux_jy > 0, -2.5 * np.log10(flux_jy / zp), np.nan)
    elif 'flux' in table.colnames:
        mag = vega_from_instrumental(table['flux'], table.meta, band,
                                     allow_network=allow_network)
    else:
        raise KeyError(f'{path}: no flux or magnitude column '
                       f'in {table.colnames[:12]}...')
    return coords, mag


def pair_bands(blue_coords, blue_mag, red_coords, red_mag,
               max_sep_arcsec=DEFAULT_MATCH_ARCSEC):
    """Nearest-neighbour association of the two bands' detections.

    This is source association between two filters of the SAME pointing on the
    SAME astrometric solution -- pairing a star with itself -- and no offset is
    derived from it.  It is not the dense-nearest-neighbour-median astrometry
    that CLAUDE.md rule #1 forbids: nothing here reduces the separations to a
    number, and the separations are used only to reject pairs.

    Nearest-neighbour and NOT one-to-one: several blue detections can claim the
    same red source, since nothing rejects a red source already taken.  In a
    field this crowded that happens.  Acceptable for a density map -- it
    duplicates a red magnitude rather than inventing one -- and it would not be
    for anything that counted sources or measured a luminosity function.

    Returns ``(blue_mag, red_mag)`` for the matched pairs.
    """
    import astropy.units as u
    if len(blue_coords) == 0 or len(red_coords) == 0:
        return np.empty(0), np.empty(0)
    idx, sep, _ = blue_coords.match_to_catalog_sky(red_coords)
    keep = sep < max_sep_arcsec * u.arcsec
    b = np.asarray(blue_mag, dtype=float)[keep]
    r = np.asarray(red_mag, dtype=float)[np.asarray(idx)[keep]]
    good = np.isfinite(b) & np.isfinite(r)
    return b[good], r[good]


def pointing_cmd(paths, max_sep_arcsec=DEFAULT_MATCH_ARCSEC, allow_network=True):
    """``(colour, magnitude)`` arrays for one pointing: F212N-F480M vs F212N."""
    blue_coords, blue_mag = load_band(paths[BLUE_BAND], BLUE_BAND,
                                      allow_network=allow_network)
    red_coords, red_mag = load_band(paths[RED_BAND], RED_BAND,
                                    allow_network=allow_network)
    b, r = pair_bands(blue_coords, blue_mag, red_coords, red_mag,
                      max_sep_arcsec=max_sep_arcsec)
    return b - r, b


def catalog_mtime(paths):
    """Newest input mtime, so a rebuild can be skipped when nothing changed."""
    return max((os.path.getmtime(p) for p in paths.values()), default=0.0)
