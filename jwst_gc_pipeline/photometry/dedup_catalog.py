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
"""
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


def dedup_merged_catalog(in_path, out_path, *,
                         link_radius=0.10 * u.arcsec,
                         sat_link_radius=0.5 * u.arcsec,
                         dmag_collision=0.1,
                         sat_dmag_collision=0.5,
                         max_component=4,
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
        complementary-coverage / dmag-collision binary protection below
        applies to these pairs with the looser ``sat_dmag_collision``.  Set
        to 0 (or <= link_radius) to disable.
    dmag_collision : float
        A band that detects both rows with |dmag| >= this is treated as a
        genuine resolved pair and blocks the merge (binary protection).
    sat_dmag_collision : float
        Collision threshold used instead of ``dmag_collision`` when either
        row is saturated: saturated-star magnitudes carry a few-tenths-mag
        scatter, so the tight threshold mistakes the SAME star's two rows
        for a resolved pair.
    max_component : int
        Connected components larger than this are left intact.
    """
    t = Table.read(in_path)
    n0 = len(t)
    bands = real_bands(t)
    det = _detection_mask(t, bands)
    mags = {b: _arr(t, f'mag_vega_{b}') for b in bands}
    nbands_det = np.sum([det[b] for b in bands], axis=0)

    # any-band subtracted/flagged saturated star (for the wide link radius)
    is_sat_any = np.zeros(n0, bool)
    for c in t.colnames:
        if c.startswith('replaced_saturated_') or c.startswith('is_saturated_'):
            is_sat_any |= _arr(t, c, fill=False, dtype=bool)

    sc = SkyCoord(t['skycoord_ref'])
    wide = max(link_radius, sat_link_radius)
    i1, i2, sep, _ = sc.search_around_sky(sc, wide)
    # keep unique unordered pairs i<j
    keep = i1 < i2
    i1, i2, sep = i1[keep], i2[keep], sep[keep]
    # beyond link_radius only satstar-involved pairs qualify (see sat_link_radius)
    pairsat = is_sat_any[i1] | is_sat_any[i2]
    keep = (sep <= link_radius) | (pairsat & (sep <= sat_link_radius))
    n_satpairs = int((keep & (sep > link_radius)).sum())
    i1, i2 = i1[keep], i2[keep]
    if verbose:
        print(f"[dedup] {n0} rows; {len(i1)} candidate pairs within "
              f"{link_radius} (+{n_satpairs} satstar pairs out to "
              f"{sat_link_radius})", flush=True)

    # --- pairwise link decision: complementary coverage or near-identical ---
    # saturated-involved pairs use the looser collision threshold
    pair_thresh = np.where(is_sat_any[i1] | is_sat_any[i2],
                           sat_dmag_collision, dmag_collision)
    link = np.ones(len(i1), bool)
    for b in bands:
        both = det[b][i1] & det[b][i2]
        if both.any():
            dmag = np.abs(mags[b][i1] - mags[b][i2])
            # a real collision (resolved, discrepant) blocks the link
            block = both & ~(dmag < pair_thresh)
            link &= ~block
    li1, li2 = i1[link], i2[link]
    if verbose:
        print(f"[dedup] {link.sum()} pairs pass the complementary/identical "
              f"test", flush=True)

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
    for b in bands:
        prim, win = assign[b]
        if not prim:
            continue
        prim = np.array(prim)
        win = np.array(win)
        changed[prim] = True
        for col in byband[b]:
            t[col][prim] = t[col][win]

    t['n_merged'] = n_merged_col

    out = t[~drop]
    # recompute synthetic line-sub bands only for rows whose fluxes changed
    _recompute_synthetic(out, bands, changed[~drop])

    if verbose:
        ncomp = len(primaries)
        print(f"[dedup] merged {drop.sum()} rows into {ncomp} primaries; "
              f"kept {skipped_binary} components as resolved binaries, "
              f"{skipped_large} as too-large; "
              f"{n0} -> {len(out)} rows", flush=True)
    out.write(out_path, overwrite=True)
    return out_path
