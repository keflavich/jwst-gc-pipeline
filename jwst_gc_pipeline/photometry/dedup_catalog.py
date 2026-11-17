"""Post-merge de-duplication of the cross-band merged photometry catalog.

The cross-band merge (``merge_catalogs.merge_daophot``) builds its reference
coordinate list as a union across filters and keeps only *mutual* nearest-
neighbour matches.  In crowded fields a single physical star can be split into
two reference rows: each band's detection attaches to whichever row is its
mutual nearest, orphaning the other row into a phantom non-detection (e.g.
bright in F187N, "missing" in F405N even though F405N detected it 0.05" away).

This module collapses those split rows *after* the merge, without touching the
merge logic, and -- crucially -- without merging genuinely resolved binaries.
Two nearby rows are merged only when their band coverage is **complementary**:
no band independently detects *both* rows with discrepant magnitudes.  A band
that resolved the pair into two distinct-brightness sources blocks the merge,
so real binaries are preserved.

Decision per close pair (within ``link_radius``):
  * no same-band collision (no band detects both)        -> merge (split star)
  * collisions but all |dmag| < ``dmag_collision``        -> merge (redundant)
  * any collision with |dmag| >= ``dmag_collision``       -> keep both (binary)

Components are validated as a whole: if any band has >=2 members detected with
a magnitude spread >= ``dmag_collision``, the whole component is left intact.
Components larger than ``max_component`` are also left intact (avoids chaining
through dense clumps).

Known disagreement with the truth (#1124): where the SW bands resolve a close
double and the LW bands see one blend, the per-band positions of the fainter
member scatter by up to ~40 mas and the cross-band merge attaches each band to
one of the two reference rows inconsistently.  Each row then carries a mix of
the two stars' bands, the pair has no same-band collision, and dedup collapses
it.  The surviving row keeps the lowest-emag detection per band, so it can
combine, for example, F115W from one star with F162M from the other and drops
the other star's values.  This behaviour is unchanged.
``flag_complementary_splits`` marks such pairs (``complementary_split``,
``split_group``) on the m7 catalog, where both rows still exist, and again on
the dedup output for pairs that dedup leaves split because the component is
too large or contains a band-resolved pair.  The flag uses ``link_radius`` only, so a
complementary pair that dedup links solely through the wider satstar radius is
not flagged.

The output records where each band of a merged row came from (``dedup_src_<band>``
and related columns, see ``dedup_merged_catalog``), so a row that mixes two
pre-dedup rows can be identified.
"""
import os
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

# real photometric bands that can carry an independent detection (mask/mag).
# synthetic line-subtracted bands are derived and recomputed post-merge.
SYNTHETIC = ('410m405', '405m410', '182m187', '187m182')
# (continuum, line) pairs and the line-subtraction scale used by the merge.
LINESUB = {
    '410m405': ('f410m', 'f405n', 0.11),
    '182m187': ('f182m', 'f187n', 0.11),
}


class _UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _arr(t, name, fill=np.nan, dtype=float):
    col = t[name]
    if hasattr(col, 'filled'):
        col = col.filled(fill)
    return np.asarray(col, dtype=dtype)


def real_bands(tbl):
    """Real (non-synthetic) photometric bands present, via mag_vega_ columns."""
    out = []
    for c in tbl.colnames:
        if c.startswith('mag_vega_'):
            b = c[len('mag_vega_'):]
            if b not in SYNTHETIC:
                out.append(b)
    return out


def _band_columns(tbl, bands):
    """Map band -> list of columns suffixed _<band> (longest suffix wins)."""
    allsuf = sorted(set(bands) | set(SYNTHETIC), key=len, reverse=True)
    byband = {b: [] for b in allsuf}
    for c in tbl.colnames:
        if c.startswith('dedup_'):
            continue  # provenance columns of an earlier dedup run
        for b in allsuf:
            if c.endswith('_' + b):
                byband[b].append(c)
                break
    return byband


def _detection_mask(tbl, bands):
    """det[b] = INDEPENDENT firm detection: finite magnitude, not masked, and
    NOT an m8 forced fill.

    Forced fills are derived at the reference position after the merge -- they
    are not independent evidence that a band resolved anything.  Counting them
    as detections poisoned both dedup tests: a satstar row whose partner band
    was filled on the star-subtracted residual (~5 mag faint) "collided" with
    the partner's real detection and got binary-protected, so the split rows
    that most needed merging were exactly the ones the fill locked apart.
    """
    det = {}
    for b in bands:
        finite = np.isfinite(_arr(tbl, f'mag_vega_{b}'))
        if f'mask_{b}' in tbl.colnames:
            masked = _arr(tbl, f'mask_{b}', fill=True, dtype=bool)
        else:
            masked = np.zeros(len(tbl), bool)
        filled = (_arr(tbl, f'forced_filled_{b}', fill=False, dtype=bool)
                  if f'forced_filled_{b}' in tbl.colnames
                  else np.zeros(len(tbl), bool))
        det[b] = finite & ~masked & ~filled
    return det


def _check_mas(value, what):
    """``value`` as a finite positive float (mas), else ``ValueError``.

    A non-positive, NaN or infinite separation never blocks a link, so the
    guard would be silently off while the run looks as if it were guarded.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: {value!r} is not a number") from None
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"{what}: {value!r} must be a finite number > 0 mas "
                         f"(anything else would switch the guard off)")
    return v


def parse_band_resolve_mas(text):
    """Parse and validate a ``band_resolve_mas`` value.

    ``''``/``None`` -> ``None`` (guard off); ``'60'`` or ``60`` -> 60.0 (every
    band); ``'f212n=60,f200w=50'`` -> ``{'f212n': 60.0, 'f200w': 50.0}``.  A
    number or mapping that is already parsed is validated and returned.
    Raises ``ValueError`` for a malformed string, an empty band name or a
    separation that is not a finite number > 0, so a typo fails when the
    options are read rather than after the run.
    """
    if text is None:
        return None
    if isinstance(text, dict):
        if not text:
            raise ValueError("band_resolve_mas: empty mapping")
        out = {}
        for band, val in text.items():
            band = str(band).strip().lower()
            if not band:
                raise ValueError("band_resolve_mas: empty band name")
            out[band] = _check_mas(val, f"band_resolve_mas[{band}]")
        return out
    if not isinstance(text, str):
        return _check_mas(text, "band_resolve_mas")
    text = text.strip()
    if not text:
        return None
    if '=' not in text:
        return _check_mas(text, "band_resolve_mas")
    out = {}
    for item in text.split(','):
        band, sep, val = item.partition('=')
        band = band.strip().lower()
        if not sep or not band or '=' in val:
            raise ValueError(f"band_resolve_mas: cannot read {item!r} in "
                             f"{text!r}; expected BAND=MAS,BAND=MAS")
        out[band] = _check_mas(val, f"band_resolve_mas[{band}]")
    return out


def check_winner_emag_ratio(ratio):
    """``ratio`` as ``None`` or a finite float > 0, else ``ValueError``.

    NaN, 0 and negative values make the primary keep every independent band
    whatever the challenger's emag; a ratio above 1 is the default rule
    (the best challenger's emag never exceeds the primary's), so it warns.
    """
    if ratio is None:
        return None
    try:
        r = float(ratio)
    except (TypeError, ValueError):
        raise ValueError(f"winner_emag_ratio: {ratio!r} is not a number") from None
    if not np.isfinite(r) or r <= 0:
        raise ValueError(f"winner_emag_ratio: {ratio!r} must be a finite number "
                         f"> 0 (use e.g. 0.8; omit it for the lowest-emag rule)")
    if r > 1:
        warnings.warn(f"winner_emag_ratio={r:g} > 1 reproduces the default "
                      f"lowest-emag rule (the challenger's emag never exceeds "
                      f"the primary's); use a value <= 1 for hysteresis",
                      UserWarning, stacklevel=2)
    return r


def _band_resolve_thresholds(band_resolve_mas, tbl, bands):
    """Map band -> resolved-in-band separation (mas) for bands that can be tested.

    ``band_resolve_mas`` is ``None`` (guard off), a single number applied to
    every band, or a mapping band -> mas (bands absent from the mapping are
    not guarded).  A requested band that cannot be tested (not a band of this
    table, or no ``skycoord_<band>`` SkyCoord column) is reported with a
    warning and dropped, and the guard is reported as OFF when none is left.
    """
    band_resolve_mas = parse_band_resolve_mas(band_resolve_mas)
    if band_resolve_mas is None:
        return {}
    if isinstance(band_resolve_mas, dict):
        want = band_resolve_mas
    else:
        want = {b: band_resolve_mas for b in bands}
    out = {}
    dropped = []
    for b, v in want.items():
        col = f'skycoord_{b}'
        if b not in bands:
            dropped.append(f"{b} (not a band of this table: {', '.join(bands)})")
        elif col not in tbl.colnames or not isinstance(tbl[col], SkyCoord):
            dropped.append(f"{b} (no {col} SkyCoord column)")
        else:
            out[b] = v
    if dropped:
        msg = (f"[dedup] band_resolve_mas: not applied to "
               f"{'; '.join(dropped)}")
        if not out:
            msg += "; the resolved-in-band guard is OFF"
        warnings.warn(msg, UserWarning, stacklevel=2)
    return out


def _band_sep_mas(tbl, band, i, j):
    """Separation (mas) between ``skycoord_<band>`` of rows ``i`` and ``j``."""
    col = tbl[f'skycoord_{band}']
    return col[i].separation(col[j]).to_value(u.mas)


def _recompute_synthetic(tbl, bands, rows):
    """Recompute line-subtracted synthetic bands for ``rows`` only.

    flux_jy_<syn> = flux_jy_<cont> - scale * flux_jy_<line>  (forward), and
    flux_jy_<rev> = flux_jy_<line> - flux_jy_<syn>           (reverse).
    The AB offset and Vega zeropoint are additive constants in magnitude
    (mag = -2.5 log10(flux_jy) + const); each constant is backed out of the
    catalog's existing finite rows, so no SVO filter table is needed.  Only
    the given ``rows`` (those whose constituent fluxes were reassigned during
    de-duplication) are updated; untouched singletons keep their merge-time
    synthetic values exactly.
    """
    if not np.any(rows):
        return

    def _const(old_mag, old_flux):
        with np.errstate(invalid='ignore', divide='ignore'):
            c = old_mag + 2.5 * np.log10(np.where(old_flux > 0, old_flux, np.nan))
        return np.nanmedian(c)

    for syn, (cont, line, scale) in LINESUB.items():
        fcont, fline = f'flux_jy_{cont}', f'flux_jy_{line}'
        rev = f'{line[1:-1]}m{cont[1:-1]}'  # e.g. 405m410
        if fcont not in tbl.colnames or fline not in tbl.colnames:
            continue
        new = {syn: _arr(tbl, fcont) - _arr(tbl, fline) * scale}
        new[rev] = _arr(tbl, fline) - new[syn]
        for nm, newflux in new.items():
            fcol = f'flux_jy_{nm}'
            with np.errstate(invalid='ignore', divide='ignore'):
                logf = -2.5 * np.log10(np.where(newflux > 0, newflux, np.nan))
            for magcol in (f'mag_ab_{nm}', f'mag_vega_{nm}'):
                if magcol in tbl.colnames and fcol in tbl.colnames:
                    const = _const(_arr(tbl, magcol), _arr(tbl, fcol))
                    tbl[magcol][rows] = (logf + const)[rows]
            if fcol in tbl.colnames:
                tbl[fcol][rows] = newflux[rows]


def _position_spread_mas(tbl, bands, det, src, swapped_rows):
    """Largest pairwise ``skycoord_<band>`` separation (mas) per row.

    Only rows in ``swapped_rows`` (at least one band supplied by another row)
    get a value; the others are 0.  Bands count when the supplying row is an
    independent detection in that band.
    """
    n = len(tbl)
    spread = np.zeros(n)
    rows = np.flatnonzero(swapped_rows)
    if rows.size == 0:
        return spread
    coords = {}
    for b in bands:
        col = f'skycoord_{b}'
        if col not in tbl.colnames or not isinstance(tbl[col], SkyCoord):
            continue
        sel = src[b][rows]
        ok = det[b][sel]
        ra = np.where(ok, tbl[col].ra.deg[sel], np.nan)
        dec = np.where(ok, tbl[col].dec.deg[sel], np.nan)
        coords[b] = SkyCoord(ra * u.deg, dec * u.deg)
    names = list(coords)
    best = np.zeros(rows.size)
    for k, a in enumerate(names):
        for c in names[k + 1:]:
            sep = coords[a].separation(coords[c]).to_value(u.mas)
            best = np.fmax(best, sep)
    spread[rows] = best
    return spread


def _band_state(tbl, bands):
    """Per-band detection masks, magnitudes and saturation flags of ``tbl``.

    Returns ``(det, mags, is_sat, is_sat_any)``: ``det[b]`` the independent
    detection mask (``_detection_mask``), ``mags[b]`` the Vega magnitudes,
    ``is_sat[b]`` the subtracted/flagged saturated-star flag of band ``b`` and
    ``is_sat_any`` its union over every band (the wide link radius keys on it).
    """
    n = len(tbl)
    det = _detection_mask(tbl, bands)
    mags = {b: _arr(tbl, f'mag_vega_{b}') for b in bands}
    is_sat_any = np.zeros(n, bool)
    for c in tbl.colnames:
        if c.startswith('replaced_saturated_') or c.startswith('is_saturated_'):
            is_sat_any |= _arr(tbl, c, fill=False, dtype=bool)
    is_sat = {b: np.zeros(n, bool) for b in bands}
    for b in bands:
        for c in (f'replaced_saturated_{b}', f'is_saturated_{b}'):
            if c in tbl.colnames:
                is_sat[b] |= _arr(tbl, c, fill=False, dtype=bool)
    return det, mags, is_sat, is_sat_any


def _complementary_pairs(sc, bands, det, mags, is_sat, is_sat_any, *,
                         link_radius, sat_link_radius, dmag_collision,
                         sat_dmag_collision, verbose=False):
    """Candidate close pairs of rows and the per-pair collision test.

    Shared by ``dedup_merged_catalog`` (which collapses linked pairs) and
    ``flag_complementary_splits`` (which marks the strictly complementary
    ones), so both use one definition of "close pair" and "same-band
    collision".

    Returns ``(i1, i2, sep, link, n_shared)`` over the unique unordered pairs
    ``i1 < i2``: ``link`` is True where no shared band blocks the pair and
    ``n_shared`` counts the bands that detect both rows.
    """
    n0 = len(sc)
    wide = max(link_radius, sat_link_radius)
    i1, i2, sep, _ = sc.search_around_sky(sc, wide)
    # keep unique unordered pairs i<j
    keep = i1 < i2
    i1, i2, sep = i1[keep], i2[keep], sep[keep]
    # beyond link_radius only satstar-involved pairs qualify (see sat_link_radius)
    pairsat = is_sat_any[i1] | is_sat_any[i2]
    keep = (sep <= link_radius) | (pairsat & (sep <= sat_link_radius))
    n_satpairs = int((keep & (sep > link_radius)).sum())
    i1, i2, sep = i1[keep], i2[keep], sep[keep]
    if verbose:
        print(f"[dedup] {n0} rows; {len(i1)} candidate pairs within "
              f"{link_radius} (+{n_satpairs} satstar pairs out to "
              f"{sat_link_radius})", flush=True)

    # --- pairwise link decision: complementary coverage or near-identical ---
    # saturated-involved pairs use the looser collision threshold.  A pair
    # linked only by the wide satstar radius and detected in two or more of
    # the same bands is two stars when, in a shared band, both rows or
    # neither are saturated; on wd2 the looser threshold merged such
    # neighbours 0.2-0.46" apart that dolphot resolves, dropping one star's
    # band measurements (#1038).  Two cases remain duplicates of one star
    # and keep the looser threshold: a band where exactly one row is
    # saturated (satstar fit + daophot detection; cloudc: 0.13" apart along
    # one direction), and a pair sharing one band (a one-band satstar
    # fragment; on wd2 dolphot has a star at only one of the two rows for
    # 157 of 165 such pairs).
    pair_thresh = np.where(is_sat_any[i1] | is_sat_any[i2],
                           sat_dmag_collision, dmag_collision)
    n_shared = np.zeros(len(i1), int)
    for b in bands:
        n_shared += det[b][i1] & det[b][i2]
    multi_wide = (sep > link_radius) & (n_shared >= 2)
    link = np.ones(len(i1), bool)
    for b in bands:
        both = det[b][i1] & det[b][i2]
        if both.any():
            dmag = np.abs(mags[b][i1] - mags[b][i2])
            one_sat = is_sat[b][i1] ^ is_sat[b][i2]
            pair_thresh_b = np.where(multi_wide & ~one_sat, 0.0, pair_thresh)
            # a real collision (resolved, discrepant) blocks the link
            block = both & ~(dmag < pair_thresh_b)
            link &= ~block
    return i1, i2, sep, link, n_shared


def flag_complementary_splits(tbl, radius=0.10 * u.arcsec):
    """Mark close row pairs whose bands are strictly complementary (#1124).

    A pair is flagged when its rows lie within ``radius`` and no band detects
    both, with each row detecting at least one band.  That is the signature of
    the cross-band merge attaching one star's bands (or two stars' bands) to
    the wrong reference row.  A pair that shares a detected band is excluded
    outright, whether the shared band has near-identical magnitudes (a
    duplicate) or discrepant ones (a resolved binary), so no magnitude
    threshold enters the flag.

    Only the plain ``radius`` is used, not dedup's wider satstar radius: a
    complementary pair 0.1" to 0.5" apart that dedup links only through the
    satstar radius is not flagged.

    "Detection" is the one ``dedup_merged_catalog`` uses: a finite, unmasked
    magnitude that is not marked ``forced_filled_<band>``.  At m7 time the
    ``forced_filled_*`` and ``independently_detected_*`` columns do not exist
    yet (m8 and ``cataloging.annotate_independent_detection`` write them
    later), so a cross-band-seeded force-fit in an m7 row counts as a
    detection there; the forced-fill exclusion is operative on the
    ``m8_dedup`` output only.

    Adds two columns to ``tbl`` in place and returns it:

    ``complementary_split`` : bool
        The row belongs to at least one flagged pair.
    ``split_group`` : int
        Shared by the rows of one pair or chain of flagged pairs, equal to the
        lowest member row index at flag time; -1 where unflagged.  It is a
        label, not a row pointer: dedup drops rows, and the quality-cut
        (``_qualcuts``/oksep) tables select rows after the flag is computed,
        so a flagged row can lack its partner there.

    Flux and magnitude columns are not touched, and repeated calls give the
    same columns.  A table without ``skycoord_ref`` or without real bands gets
    all-False/-1 columns.

    Raises
    ------
    ValueError
        If any ``skycoord_ref`` coordinate is non-finite.
    """
    n = len(tbl)
    split = np.zeros(n, bool)
    group = np.full(n, -1, int)
    bands = real_bands(tbl)
    if n > 1 and 'skycoord_ref' in tbl.colnames:
        sc = SkyCoord(tbl['skycoord_ref'])
        nbad = int((~(np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg))).sum())
        if nbad:
            raise ValueError(
                f"skycoord_ref has {nbad} non-finite coordinate(s); the "
                f"complementary-split flag needs a position for every row")
    if n > 1 and bands and 'skycoord_ref' in tbl.colnames:
        det, mags, is_sat, is_sat_any = _band_state(tbl, bands)
        nbands_det = np.sum([det[b] for b in bands], axis=0)
        # the collision thresholds are inert here: pairs that share a band
        # are dropped by the n_shared == 0 test below, and a pair with no
        # shared band has no collision to threshold
        i1, i2, _, _, n_shared = _complementary_pairs(
            sc, bands, det, mags, is_sat, is_sat_any,
            link_radius=radius, sat_link_radius=0 * u.arcsec,
            dmag_collision=0.0, sat_dmag_collision=0.0)
        sel = ((n_shared == 0)
               & (nbands_det[i1] > 0) & (nbands_det[i2] > 0))
        i1, i2 = i1[sel], i2[sel]
        if len(i1):
            uf = _UnionFind(n)
            for a, b in zip(i1, i2):
                uf.union(int(a), int(b))
            involved = np.unique(np.concatenate([i1, i2]))
            lowest = {}
            for idx in involved:  # ascending, so the first seen is the lowest
                lowest.setdefault(uf.find(int(idx)), int(idx))
            for idx in involved:
                group[idx] = lowest[uf.find(int(idx))]
            split[involved] = True
    tbl['complementary_split'] = split
    tbl['split_group'] = group
    return tbl


def dedup_merged_catalog(in_path, out_path, *,
                         link_radius=0.10 * u.arcsec,
                         sat_link_radius=0.5 * u.arcsec,
                         dmag_collision=0.1,
                         sat_dmag_collision=0.5,
                         max_component=4,
                         winner_emag_ratio=None,
                         band_resolve_mas=None,
                         write_provenance=True,
                         verbose=True):
    """Collapse split-source duplicate rows in a cross-band merged catalog.

    Parameters
    ----------
    in_path, out_path : str
        Input merged catalog and output (de-duplicated) catalog paths.
    link_radius : Quantity
        Maximum separation for two rows to be considered the same source.
    sat_link_radius : Quantity
        Wider link radius used when EITHER row of a pair is a subtracted
        saturated star (``replaced_saturated_<band>``/``is_saturated_<band>``
        in any band).  Saturated-core centroids scatter by ~0.2" (Brick
        F182M: the partner band's independent detection sits a median 0.214"
        from the satstar row, 78% within 0.5"), so at the default 0.10" the
        satstar row and the partner-band detection stay SPLIT -- the m8 fill
        then "fills" the satstar row's other bands on the star-subtracted
        residual, planting ~5-mag-wrong colors on the CMD bright end.  The
        wide link between rows that share two or more bands is blocked by
        any shared band where both rows, or neither, are saturated, whatever
        the dmag: two multi-band detections 0.1-0.5" apart are two stars
        (#1038).  A shared band where exactly one row is saturated (a
        satstar fit and a daophot detection of the same star), and a pair
        sharing a single band (a one-band fragment of the star), still use
        ``sat_dmag_collision``.  Set to 0 (or <= link_radius) to disable.
    dmag_collision : float
        A band that detects both rows with |dmag| >= this is treated as a
        genuine resolved pair and blocks the merge (binary protection).
    sat_dmag_collision : float
        Collision threshold used instead of ``dmag_collision`` when either
        row is saturated: saturated-star magnitudes carry a few-tenths-mag
        scatter, so the tight threshold mistakes the SAME star's two rows
        for a resolved pair.  Beyond ``link_radius`` see ``sat_link_radius``
        for the bands it applies to.
    max_component : int
        Connected components larger than this are left intact.
    winner_emag_ratio : float or None
        Hysteresis on the per-band winner (#1278).  By default each band of
        the merged row is copied from the member with the lowest
        ``emag_ab_<band>``, so a near-tie can move a band's position by
        ~0.1" for no photometric gain.  With a ratio (e.g. 0.8) the primary
        keeps its own measurement of a band when it is an independent
        detection with a finite ``emag_ab_<band>``, unless the best
        challenger's emag is below ``ratio * primary_emag``.  A primary
        without an independent detection in the band still takes the best
        challenger.  ``None`` keeps the lowest-emag rule.  Must be a finite
        number > 0 (``ValueError`` otherwise: NaN, 0 and negative values would
        make the primary keep every band); a value above 1 reproduces the
        default rule and warns.
    band_resolve_mas : float, dict or None
        Resolved-in-band guard (#1278).  A band that detects both rows
        independently with ``skycoord_<band>`` positions further apart than
        this many mas counts as a collision and blocks the link whatever the
        magnitude difference, in the pairwise test and in the whole-component
        check.  A mapping gives a per-band value and leaves other bands
        unguarded (the long-wavelength bands, with a PSF FWHM of ~0.16", are
        too coarse for this test).  ``None`` disables the guard.  A value that
        is not a finite number > 0 raises ``ValueError``; a requested band
        that is not in the table or has no ``skycoord_<band>`` column is
        dropped with a warning (and the guard is reported as OFF when none is
        left), so a mistyped band cannot leave the guard silently off.
    write_provenance : bool
        Add ``dedup_primary`` (pre-dedup row index of the surviving row),
        ``dedup_src_<band>`` (pre-dedup row index that supplied the band),
        ``dedup_band_swapped_<band>`` (the band came from another row) and
        ``dedup_pos_spread_mas`` (see below).  The indices refer to rows of
        ``in_path``, the m8 file of the same run, and the output header key
        ``DEDUPSRC`` names that file.  ``dedup_pos_spread_mas`` is the
        largest pairwise separation between the output ``skycoord_<band>``
        of the independently detected bands, for rows with at least one
        swapped band, and 0 otherwise.  The header also records ``DEDUPN0``
        (the row count of ``in_path``, so a reader can tell whether the
        indices still match a rewritten m8), ``DEDUPRAT`` and ``DEDUPRES``
        (the options, when set).

    Notes
    -----
    ``n_filt_independent``, when present, is recomputed from the
    ``independently_detected_<band>`` columns for the rows that took a band
    from another row, since the scalar is not a ``_<band>`` column and would
    otherwise keep the primary's pre-swap count (#1329).
    """
    winner_emag_ratio = check_winner_emag_ratio(winner_emag_ratio)
    band_resolve_mas = parse_band_resolve_mas(band_resolve_mas)
    t = Table.read(in_path)
    n0 = len(t)
    bands = real_bands(t)
    det, mags, is_sat, is_sat_any = _band_state(t, bands)
    nbands_det = np.sum([det[b] for b in bands], axis=0)

    sc = SkyCoord(t['skycoord_ref'])
    i1, i2, sep, link, _ = _complementary_pairs(
        sc, bands, det, mags, is_sat, is_sat_any,
        link_radius=link_radius, sat_link_radius=sat_link_radius,
        dmag_collision=dmag_collision, sat_dmag_collision=sat_dmag_collision,
        verbose=verbose)
    # resolved-in-band guard (#1278): a band that detects both rows
    # independently with positions further apart than its threshold blocks
    # the link whatever the magnitude difference
    resolve = _band_resolve_thresholds(band_resolve_mas, t, bands)
    resolved_any = np.zeros(len(i1), bool)
    for b in bands:
        if b not in resolve:
            continue
        both = det[b][i1] & det[b][i2]
        if both.any():
            resolved = np.zeros(len(i1), bool)
            resolved[both] = _band_sep_mas(t, b, i1[both], i2[both]) > resolve[b]
            resolved_any |= resolved
    # pairs the magnitude test alone would link but a resolved band blocks
    # (independent of the band order)
    n_resolved_blocked = int((link & resolved_any).sum())
    link &= ~resolved_any
    li1, li2 = i1[link], i2[link]
    if verbose:
        print(f"[dedup] {link.sum()} pairs pass the complementary/identical "
              f"test" + (f" ({n_resolved_blocked} blocked only by the "
                         f"resolved-in-band guard)" if resolve else ""),
              flush=True)

    # --- connected components over linked pairs ---
    uf = _UnionFind(n0)
    for a, b in zip(li1, li2):
        uf.union(int(a), int(b))
    roots = np.array([uf.find(int(k)) for k in range(n0)])
    # group members by root, only for involved rows
    involved = np.unique(np.concatenate([li1, li2])) if len(li1) else np.array([], int)
    from collections import defaultdict
    comps = defaultdict(list)
    for idx in involved:
        comps[roots[idx]].append(int(idx))

    # --- validate components; collect band winners for the safe ones ---
    primaries = []          # surviving primary index per merged component
    drop = np.zeros(n0, bool)
    n_merged_col = np.ones(n0, int)
    # per-band assignment: (primary_idx, winner_idx) where winner != primary
    assign = {b: ([], []) for b in bands}
    skipped_binary = 0
    skipped_large = 0

    emag = {}
    for b in bands:
        ec = f'emag_ab_{b}'
        emag[b] = _arr(t, ec) if ec in t.colnames else np.full(n0, np.nan)

    for root, members in comps.items():
        members = sorted(members)
        if len(members) > max_component:
            skipped_large += 1
            continue
        m = np.array(members)
        # whole-component safety: any band with >=2 detected members spanning
        # >= dmag_collision is a resolved multiple -> leave intact
        unsafe = False
        comp_thresh = (sat_dmag_collision if is_sat_any[m].any()
                       else dmag_collision)
        for b in bands:
            dm = m[det[b][m]]
            if dm.size >= 2:
                mv = mags[b][dm]
                if np.nanmax(mv) - np.nanmin(mv) >= comp_thresh:
                    unsafe = True
                    break
                if b in resolve and any(
                        _band_sep_mas(t, b, int(p), int(q)) > resolve[b]
                        for k, p in enumerate(dm) for q in dm[k + 1:]):
                    unsafe = True
                    break
        if unsafe:
            skipped_binary += 1
            continue
        # primary = most detected bands, tie-break brightest summed flux proxy
        primary = m[np.argmax(nbands_det[m])]
        primaries.append(primary)
        n_merged_col[primary] = len(m)
        for b in bands:
            dm = m[det[b][m]]
            if dm.size == 0:
                continue
            # best detection = lowest emag (fallback: first)
            ev = emag[b][dm]
            winner = dm[np.nanargmin(ev)] if np.isfinite(ev).any() else dm[0]
            if (winner != primary and winner_emag_ratio is not None
                    and det[b][primary] and np.isfinite(emag[b][primary])
                    and not emag[b][winner] < winner_emag_ratio * emag[b][primary]):
                # near-tie: the primary keeps its own measurement of this band
                winner = primary
            if winner != primary:
                assign[b][0].append(primary)
                assign[b][1].append(winner)
        # everything in the component except the primary is dropped
        for idx in m:
            if idx != primary:
                drop[idx] = True

    # --- apply band assignments (vectorised per column) ---
    byband = _band_columns(t, bands)
    changed = np.zeros(n0, bool)
    # pre-dedup row index supplying each band of each row (own index unless swapped)
    src = {b: np.arange(n0) for b in bands}
    for b in bands:
        prim, win = assign[b]
        if not prim:
            continue
        prim = np.array(prim)
        win = np.array(win)
        changed[prim] = True
        src[b][prim] = win
        for col in byband[b]:
            t[col][prim] = t[col][win]

    t['n_merged'] = n_merged_col
    if 'n_filt_independent' in t.colnames and changed.any():
        # the scalar is not a ``_<band>`` column, so the swap above left the
        # primary's pre-swap count next to the winner's independently_detected_*
        indep_cols = [c for c in t.colnames
                      if c.startswith('independently_detected_')]
        if indep_cols:
            n_indep = np.sum([_arr(t, c, fill=False, dtype=bool)
                              for c in indep_cols], axis=0)
            t['n_filt_independent'][changed] = n_indep[changed]
    if write_provenance:
        t['dedup_primary'] = np.arange(n0)
        for b in bands:
            t[f'dedup_src_{b}'] = src[b]
            t[f'dedup_band_swapped_{b}'] = src[b] != np.arange(n0)
        t['dedup_pos_spread_mas'] = _position_spread_mas(t, bands, det, src, changed)
        t.meta['DEDUPSRC'] = ('pre-dedup row indices refer to '
                              + os.path.basename(str(in_path)))
        t.meta['DEDUPN0'] = n0
        if winner_emag_ratio is not None:
            t.meta['DEDUPRAT'] = float(winner_emag_ratio)
        if resolve:
            t.meta['DEDUPRES'] = ','.join(f'{b}={v:g}' for b, v in resolve.items())

    out = t[~drop]
    # recompute synthetic line-sub bands only for rows whose fluxes changed
    _recompute_synthetic(out, bands, changed[~drop])
    # pairs that dedup left split (too-large component, binary protection)
    # are flagged on the output rows; any m7 values are stale after the drop
    flag_complementary_splits(out, radius=link_radius)

    if verbose:
        ncomp = len(primaries)
        print(f"[dedup] merged {drop.sum()} rows into {ncomp} primaries; "
              f"kept {skipped_binary} components as resolved binaries, "
              f"{skipped_large} as too-large; "
              f"{n0} -> {len(out)} rows; {int(changed.sum())} kept rows take "
              f"at least one band from another row; "
              f"{int(out['complementary_split'].sum())} rows flagged "
              f"complementary_split", flush=True)
    out.write(out_path, overwrite=True)
    return out_path
