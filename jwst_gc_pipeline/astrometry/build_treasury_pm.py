"""
Driver: JWST-epoch x JWST-epoch proper motions via flystar, for a field whose
only long-baseline counterpart is another JWST program rather than a ground
catalog (VIRAC2/GNS) -- e.g. Sgr B2 (program 5365) x GC Treasury (program
10678), or Cloud e/f (program 2092) x GC Treasury, once GC Treasury source
photometry exists over that footprint.

Unlike build_multiepoch_pm.py (JWST x GNS x VIRAC2, VIRAC2 defines the
reference frame and supplies a pm PRIOR for matching), neither epoch here has
an independent proper-motion catalog. The frame tie is therefore a full
6-parameter affine fit (multiepoch_pm.affine_tie), not a pm-propagated match
-- a translation-only tie leaves any relative rotation, plate-scale, or shear
difference between the two epochs' astrometric solutions in the residual,
which reads as a spurious COHERENT proper motion across the field.

CAVEAT this shares with affine_tie (see its docstring): a real bulk/rotation/
shear velocity field is ALSO linear to first order across a few arcmin, so
the tie cannot distinguish real coherent motion from a plate-scale/shear
astrometric error and removes both by construction. The output pm_ra/pm_dec
here are RELATIVE to the mean motion + mean shear of the tie's own bright/
compact matching sample, not an absolute frame -- e.g. a rotation curve or
bulk streaming measurement from this catalog will read artificially close to
zero. The fitted per-observation tie coefficients (A, B) are stored in
pm.meta['tie_diag_json'] precisely so this can be checked/undone later.

Example:
    python -m jwst_gc_pipeline.astrometry.build_treasury_pm \\
        --src /orange/.../sgrb2/catalogs/f212n_..._vetted.fits --src-epoch 2024.684 \\
        --ref /orange/.../gc-treasury/catalogs/..._o127.fits \\
              /orange/.../gc-treasury/catalogs/..._o129.fits \\
        --ref-epoch 2026.696 --filter f212n \\
        --out /orange/.../sgrb2/astrometry_diag/pm_flystar/pm_sgrb2_treasury_f212n.fits
"""
import argparse
import os
import json
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, concatenate, search_around_sky
import astropy.units as u

from . import multiepoch_pm as M

# Run defaults, chosen 2026-11-18 from the Sgr B2 5365 x 10678 completeness
# census (each is reverted per run by --assign nearest, --rescue-inflated-err 0,
# --flux-ratio-cap 0.10, which reproduce the earlier tables):
#   mincost  keeps the claimant of a contested ref row whose position and flux
#            fit, where 'nearest' drops every claimant (4.3% of the Sgr B2 src
#            stars);
#   rescue   uses rows whose flux_err fails the S/N cut but whose flux is
#            >= 8 x the local noise (photutils flag-16 group fits that lost
#            their covariance); pairs with a rescued row stay non-trustworthy;
#   cap 0.5  a companion fainter than half the star's flux no longer
#            disqualifies it from 'trustworthy' (0.10 before).
# The cap leaves comp_flux_ratio / src_comp_flux_ratio in the table, so a
# consumer can re-threshold; pm.meta records ASSIGN, RESCUE_K, flux_ratio_cap.
DEFAULT_ASSIGN = 'mincost'
DEFAULT_RESCUE_K = 8.0
DEFAULT_FLUX_RATIO_CAP = 0.5


def _col(t, base, filt):
    """Return column name: bare `base` if present, else `base_{filt}`."""
    if base in t.colnames:
        return base
    c = f'{base}_{filt}'
    if c in t.colnames:
        return c
    raise KeyError(f'neither {base!r} nor {c!r} in {t.colnames}')


# std_ra/std_dec are DEGREES (merge_catalogs.py scatter of avg_ra/avg_dec, no
# FITS unit set); the conversion to arcsec, the degenerate-0.0 guard and the
# optional cos(dec) scaling live in multiepoch_pm.std_columns_to_arcsec so that
# this driver and load_jwst_m7 share one site.  Reading them as arcsec shrinks
# the combined pm_ra_err/pm_dec_err by ~3600x (jwst-gc-pipeline#140, #922).
DEG_TO_ARCSEC = M.STD_DEG_TO_ARCSEC


def _sat_flags(t, filt):
    """``replaced_saturated`` (bare or ``_{filt}``) as bool, or all-False
    when the catalog has no such column (older catalogs, synthetic tests)."""
    try:
        c = _col(t, 'replaced_saturated', filt)
    except KeyError:
        return np.zeros(len(t), bool)
    v = np.ma.filled(np.ma.asarray(t[c]), False)
    return np.asarray(v, bool)


def _load_one(path, filt, epoch, sn_cut, obs_index, rescue_k=None):
    """One catalog's usable rows as a dict (sc, ex, ey, flux, mag, sat, flux_noise, ...).

    A row is usable when ``flux / flux_err > sn_cut``.  With ``rescue_k`` it is
    also usable when ``flux >= rescue_k * flux_noise`` -- ``flux_noise`` being
    the local noise of ``multiepoch_pm.local_flux_noise`` -- although its own
    ``flux_err`` fails the cut: a group fit that exhausted its LM budget loses
    its parameter covariance (photutils flags bit 16) and reports a ``flux_err``
    that makes even a star at 8x the local noise read S/N < 5.  Those rows come
    back with ``err_rescued`` True (the key is present only when ``rescue_k`` is
    given).
    """
    t = Table.read(path)
    sat = _sat_flags(t, filt)
    sc_col = _col(t, 'skycoord', filt)
    fl_col = _col(t, 'flux', filt)
    fe_col = _col(t, 'flux_err', filt)
    _col(t, 'std_ra', filt)     # raise KeyError when absent, as for the columns above
    _col(t, 'std_dec', filt)
    sc = SkyCoord(t[sc_col])
    flux = np.asarray(t[fl_col], float)
    flux_err = np.asarray(t[fe_col], float)
    ex, ey = M.std_columns_to_arcsec(t, filt, dec_deg=sc.dec.deg,
                                     apply_cosdec=M.STD_RA_APPLY_COSDEC)
    flux_noise = M.local_flux_noise(sc.ra.deg, sc.dec.deg, flux, flux_err)
    finite = np.isfinite(sc.ra.deg) & np.isfinite(flux) & (flux > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        healthy = np.isfinite(flux_err) & (flux_err > 0) & (flux / flux_err > sn_cut)
        rescued = (finite & ~healthy & (flux >= rescue_k * flux_noise)
                   if rescue_k is not None else np.zeros(len(t), bool))
    ok = finite & (healthy | rescued)
    cat = dict(sc=sc[ok],
               ex=np.where(np.isfinite(ex[ok]), ex[ok], 0.005),
               ey=np.where(np.isfinite(ey[ok]), ey[ok], 0.005),
               flux=flux[ok], mag=-2.5 * np.log10(flux[ok]), sat=sat[ok],
               flux_noise=flux_noise[ok],
               epoch=epoch, n=int(ok.sum()), obs_index=obs_index)
    if rescue_k is not None:
        cat['err_rescued'] = rescued[ok]
    return cat


def _stack(cats, keep=None):
    """Concatenate the per-row arrays the PM builder reads from several ``_load_one``
    dicts (``err_rescued`` only when every dict has it), then apply ``keep``."""
    keys = ['ex', 'ey', 'flux', 'mag', 'sat', 'flux_noise', 'obs_index']
    if all('err_rescued' in c for c in cats):
        keys.append('err_rescued')
    out = {}
    for k in keys:
        arr = (np.concatenate([np.full(c['n'], c[k]) for c in cats]) if k == 'obs_index'
               else np.concatenate([c[k] for c in cats]))
        out[k] = arr if keep is None else arr[keep]
    return out


def load_filter_catalog(paths, filt, epoch, sn_cut=5.0, dedup_radius=0.05, rescue_k=None):
    """Load one or more per-obs/per-field catalogs, keep one filter's usable
    (S/N > sn_cut) sources, concatenate. Works for both the bare-column
    single-filter vetted catalogs (skycoord/flux/...) and the suffixed
    multi-filter merged catalogs (skycoord_f212n/flux_f212n/...).

    This is only ever called for ``src`` (a single-epoch field), which
    normally means a single path -- but the CLI accepts more than one, and
    if those ever cover overlapping sky (as opposed to ``ref``'s deliberate
    multi-observation concatenation, which is untangled downstream by
    build_pm_catalog_2epoch's same_star_radius), the same star would get one
    row per path here with nothing to merge them back down, inflating
    matched/trustworthy counts with duplicate PM rows for one physical star.
    Source-association dedup here, not a correction -- no reduce derived
    from the match, just which duplicate rows are kept.
    """
    cats = [_load_one(p, filt, epoch, sn_cut, i, rescue_k=rescue_k) for i, p in enumerate(paths)]
    sc = concatenate([c['sc'] for c in cats]) if len(cats) > 1 else cats[0]['sc']
    prefer = (np.concatenate([c['err_rescued'] for c in cats]).astype(np.int64)
              if all('err_rescued' in c for c in cats) else None)
    keep = _dedup_mask(sc, dedup_radius, prefer=prefer) if len(cats) > 1 else None
    if keep is not None:
        sc = sc[keep]
    return dict(sc=sc, epoch=epoch, n=int(len(sc)), **_stack(cats, keep))


def _dedup_mask(sc, radius, prefer=None):
    """Boolean mask keeping one row per CONNECTED COMPONENT of positions in
    ``sc`` within ``radius`` (arcsec) of each other -- the lowest-indexed
    row of each component.  With ``prefer`` (a per-row integer, lower wins)
    the kept row is the one with the lowest ``prefer`` and, among those, the
    lowest index; the rescue option passes its rescued flag so that a row
    added by the rescue never displaces a healthy duplicate of the same star.

    A single-nearest-neighbour version of this (each row paired only with
    its own closest other row) can leave two rows of the same 3+-row group
    both marked "not a duplicate of the other": on a near-collinear chain
    A-B-C with ~30 mas steps, A's nearest is B and B's nearest is A (a
    mutual pair, so B is dropped), but C's nearest is B, an ALREADY-DROPPED
    row that never gets checked against A -- C survives even though it is
    well within radius of A transitively through B. Using
    search_around_sky (returns every pair within radius, not just each
    row's single closest) plus connected components -- every row within one
    component's radius chain collapses to a single kept row regardless of
    chain length, not just direct pairs.
    """
    n = len(sc)
    i, j, _, _ = search_around_sky(sc, sc, radius * u.arcsec)
    edges = i != j
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    graph = coo_matrix((np.ones(edges.sum()), (i[edges], j[edges])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    order = np.arange(n, dtype=np.int64)
    if prefer is not None:
        order = np.asarray(prefer, dtype=np.int64) * n + order
    comp_best = np.full(labels.max() + 1, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(comp_best, labels, order)
    return order == comp_best[labels]


#: A ref-vs-ref duplicate fraction above this is reported loudly. Not a hard
#: gate: jwst-gc-pipeline#958's OWN measured fractions for Sgr B2 (29-44%)
#: and Cloud e/f (30-35%) sit above this too, yet those two fields' PM
#: scatter is only 1.3-1.5x (Sgr B2) and 3.4-4x (Cloud e/f) their formal
#: errors -- roughly matching the ~2.6 mas/yr intrinsic field-star velocity
#: dispersion expected in this region, the better reference point than
#: formal error alone (pr-reviewer, PR #959 round 2 review). So duplicate-
#: row fraction ALONE is not established as the thing that breaks a fit.
#: What pr-reviewer's per-cell analysis (same review) found INSTEAD tracks
#: the affected fields (Arches/Quintuplet/Brick/Cloud c, whose PM scatter
#: is several times formal error even after accounting for real field-star
#: dispersion): the excess is SPATIALLY COHERENT and present in bright AND
#: faint stars alike -- a patch-wise frame-tie residual, not centroid noise
#: or duplication by itself, and it tracks each field's own tie rms_resid_
#: mas (a print WARNING below when it exceeds 25 mas, not a gate). Reported
#: here so a human reader CAN correlate ref_dup_fraction_max against that
#: residual and the per-cell map, not treated as sufficient on its own.
REF_DUP_WARN_FRACTION = 0.05
REF_DUP_SEP_MAS = 1.0
REF_DUP_DMAG = 1.0e-4


def _ref_pair_duplicate_fraction(cat_a, cat_b, sep_mas=REF_DUP_SEP_MAS,
                                 dmag=REF_DUP_DMAG):
    """Fraction of ``cat_a``'s rows that are a literal duplicate (position
    within ``sep_mas``, flux within ``dmag``) of some row in ``cat_b``.

    This checks the jwst-gc-pipeline#958 signature DIRECTLY, ref-vs-ref,
    rather than inferring it from src-ref matching. affine_tie's own
    duplicate-row gate (multiepoch_pm.py) cannot see this class of
    contamination at all: each call only ever compares src to ONE ref file,
    never two ref files to each other, so o105 and o108 sharing bit-
    identical rows is invisible to it (both ties can independently read
    dup_row_suspect_fraction=0.0 while o105 and o108 are >50% identical to
    each other -- caught by pr-reviewer, PR #959 review, after an earlier
    claim in this codebase's own history that the src-ref gate would catch
    this was wrong).
    """
    ia, ib, _, _ = search_around_sky(cat_a['sc'], cat_b['sc'], sep_mas * u.mas)
    if len(ia) == 0:
        return 0.0
    dup = np.abs(cat_a['mag'][ia] - cat_b['mag'][ib]) < dmag
    return len(set(ia[dup].tolist())) / cat_a['n']


def load_and_tie_ref_catalogs(ref_paths, filt, ref_epoch, src, sn_cut=5.0,
                              tie_magcut=None, tie_bright_percentile=20.0,
                              verbose=True, rescue_k=None, skip_unverified=False):
    """Load each ref path SEPARATELY and affine-tie it onto src's frame
    BEFORE combining.

    Each JWST "observation" is its own independent visit with its own guide-
    star lock, hence its own small absolute pointing (and possibly
    rotation/scale) offset.  Whether several observations cover the SAME sky
    footprint (revisits) or adjacent, partly overlapping tiles depends on the
    field: for Sgr B2 the GC Treasury o127/o128/o129/o132/o135 are tiles, and
    only o128, o129 and o135 overlap the 5365 footprint (by 50%, 49% and 32%
    of it).  Either way, concatenating them first and fitting one global tie
    only removes the AVERAGE of the different offsets, leaving each
    observation's individual residual in the data -- which then reads as a
    spatially-incoherent (by whichever observation happened to be the
    nearest-neighbor match at a given position) spurious proper motion.
    Tying each observation independently removes this before it can leak
    into the PM fit.

    A tile with no overlap with ``src`` cannot be tied
    (``multiepoch_pm.TieNotVerifiedError``).  That error propagates unless
    ``skip_unverified`` is True, which drops such an observation with a
    printed warning and records its path in ``ref['unverified_paths']``; the
    remaining observations keep their positions in ``ref_paths`` as
    ``obs_index`` and each tie diagnostic carries its ``ref_index``.
    ``rescue_k`` is passed to ``_load_one``.

    ``tie_magcut=None`` (the default) derives an ABSOLUTE cutoff from
    ``src``'s own magnitude distribution (its ``tie_bright_percentile``,
    i.e. the brightest 20% by default) rather than using a fixed literal
    value. affine_tie's own default (magcut=15) assumes a calibrated
    Vega/AB-like scale, as VIRAC/GNS use -- but mag_src/mag here is
    -2.5*log10(flux) of this pipeline's raw instrumental flux, which runs
    roughly -21 to -3 for GC Treasury (found by checking why a coherent
    ~6 mas/yr DEC bias in Cloud c's trustworthy PMs grew monotonically with
    faintness: ``mag < 15`` was true for EVERY star, so the tie's supposedly
    "bright, well-measured" calibration sample was silently the ENTIRE
    matched population, uncurated, for every field measured this way so
    far). An explicit ``tie_magcut`` still overrides this when given.
    """
    if tie_magcut is None:
        tie_magcut = float(np.percentile(src['mag'], tie_bright_percentile))
        if verbose:
            n_cut = int((src['mag'] < tie_magcut).sum())
            print(f'  tie_magcut (bright {tie_bright_percentile:g}% of src): '
                  f'{tie_magcut:.2f} -- {n_cut:,}/{src["n"]:,} src stars pass '
                  f'({100 * n_cut / src["n"]:.1f}%, vs {tie_bright_percentile:g}% '
                  f'intended -- ties are matched to REF separately per observation, '
                  f'so this is src-side selectivity only, not the eventual tie sample size)')
    cats = [_load_one(p, filt, ref_epoch, sn_cut, i, rescue_k=rescue_k) for i, p in enumerate(ref_paths)]
    ref_dup_fractions = {}
    if len(cats) > 1:
        for a in range(len(cats)):
            for b in range(len(cats)):
                if a == b:
                    continue
                frac = _ref_pair_duplicate_fraction(cats[a], cats[b])
                ref_dup_fractions[(a, b)] = frac
                if frac > REF_DUP_WARN_FRACTION:
                    print(f'  WARNING: ref[{a}] ({ref_paths[a].split("/")[-1]}) shares '
                          f'{100 * frac:.1f}% bit-identical rows with ref[{b}] '
                          f'({ref_paths[b].split("/")[-1]}) -- the jwst-gc-pipeline#958 '
                          f'signature (nominally-independent observations sharing '
                          f'detections, not independent epochs to the degree assumed). '
                          f'Weigh this against ref[{a}]\'s own tie residual below -- '
                          f'high duplication AND high residual together is the '
                          f'combination seen on fields whose trustworthy PM scatter '
                          f'came out several times their formal error.')

    tied, diags, unverified = [], [], []
    for i, p in enumerate(ref_paths):
        cat = cats[i]
        n_ref_cut = int((cat['mag'] < tie_magcut).sum())
        # match_radius here is the TIE-FITTING search radius for the bright/
        # compact magcut sample, not the final PM match radius (that is the
        # caller's own --match-radius, applied later in build_pm_catalog_
        # 2epoch). It must satisfy affine_tie's own verified-tie requirement
        # (global offset << match_radius, enforced by
        # astrometry_offsets.local_residual_map) with margin for the largest
        # real per-observation guide-star-lock offset seen on real data --
        # Sgr B2's o127/o129/o132/o135 ties run up to ~140 mas (previously
        # this raised TieNotVerifiedError at the old 0.3" radius, whose 100
        # mas safety floor a 134 mas real offset exceeded). 1.0" leaves ample
        # margin above that; the dedup/uniqueness rejection in
        # _unique_nearest_pairs / local_residual_map (not radius alone) is
        # what keeps a wider search from admitting an ambiguous pair.
        try:
            sc_tied, diag = M.affine_tie(cat['sc'], cat['mag'], src['sc'], src['mag'],
                                         magcut=tie_magcut, match_radius=1.0)
        except M.TieNotVerifiedError as exc:
            if not skip_unverified:
                raise
            print(f'  ref[{i}] {p.split("/")[-1]}: tie not verified ({exc}); dropped '
                  f'(skip_unverified)')
            unverified.append(p)
            continue
        diag['ref_index'] = i
        if verbose:
            print(f'  ref[{i}] {p.split("/")[-1]}: {n_ref_cut:,}/{cat["n"]:,} '
                  f'pass the magcut ({100 * n_ref_cut / cat["n"]:.1f}%) -- '
                  f'diag["n_match"]={diag["n_match"]:,} is the count AFTER '
                  f'this cut and the tie match radius both apply, so it is '
                  f'not directly comparable to an uncut n_match without '
                  f'rerunning affine_tie with magcut=inf.')
        cat['sc_raw'] = cat['sc']
        cat['sc'] = sc_tied
        tied.append(cat)
        # worst ref-vs-ref duplicate fraction involving this observation, in
        # either direction -- numerically usable diagnostic, not just the
        # printed warning above, so a later reader can correlate it against
        # rms_resid_mas without re-running the check.
        diag['ref_dup_fraction_max'] = max(
            [v for (a, b), v in ref_dup_fractions.items() if a == i or b == i],
            default=0.0)
        diags.append(diag)
        if verbose:
            print(f'  ref[{i}] {p.split("/")[-1]}: affine tie '
                  f'{diag["n_kept"]}/{diag["n_match"]} kept, '
                  f'median ({diag["med_dx_mas"]:+.1f},{diag["med_dy_mas"]:+.1f}) mas, '
                  f'resid rms {diag["rms_resid_mas"]:.1f} mas')
    if not tied:
        raise M.TieNotVerifiedError(
            f'none of the {len(ref_paths)} ref catalogs could be tied to src: '
            + ', '.join(os.path.basename(q) for q in unverified))
    sc = concatenate([c['sc'] for c in tied]) if len(tied) > 1 else tied[0]['sc']
    sc_raw = concatenate([c['sc_raw'] for c in tied]) if len(tied) > 1 else tied[0]['sc_raw']
    ref = dict(sc=sc, sc_raw=sc_raw, epoch=ref_epoch, n=int(sum(c['n'] for c in tied)),
               **_stack(tied))
    if unverified:
        ref['unverified_paths'] = unverified
    return ref, diags


#: |median PM(saturated) - median PM(unsaturated)| (mas/yr, either axis)
#: above which saturation_pm_offset reports the saturated population as
#: offset, if it is also SAT_PM_OFFSET_NSIGMA times its standard error.
#: Measured on the v6 Treasury tables (JWST-GC/data-qa#371):
#: Cloud c src satstars (-4.4, -23.1), Quintuplet (+8.5,
#: -38.8), against ~2 mas/yr in Brick and Sgr B2. A real cluster-vs-field
#: motion of bright stars (Arches/Quintuplet members, ~5 mas/yr relative
#: to the field) sits below this, so the threshold does not flag it.
SAT_PM_OFFSET_MAS_YR = 10.0
SAT_PM_OFFSET_NSIGMA = 5.0
SAT_PM_OFFSET_MIN_N = 20


def _median_and_err(x):
    med = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - med)))
    return med, 1.2533 * mad / np.sqrt(len(x))


def _offset_between(pm, a, b, dt, max_offset, nsigma):
    res = dict(n_sat=int(a.sum()), n_unsat=int(b.sum()), offset=False)
    for ax in ('pm_ra', 'pm_dec'):
        x = np.asarray(pm[ax], float)
        ms, es = _median_and_err(x[a])
        mu, eu = _median_and_err(x[b])
        d, e = ms - mu, float(np.hypot(es, eu))
        res[f'd_{ax}'] = round(d, 3)
        res[f'd_{ax}_err'] = round(e, 3)
        # the epoch-to-epoch position offset this PM offset implies
        res[f'd_{ax.replace("pm_", "pos_")}_mas'] = round(d * dt, 1)
        if abs(d) > max_offset and abs(d) > nsigma * e:
            res['offset'] = True
    return res


def saturation_pm_offset(pm, max_offset=SAT_PM_OFFSET_MAS_YR,
                         nsigma=SAT_PM_OFFSET_NSIGMA, min_n=SAT_PM_OFFSET_MIN_N):
    """Saturated-minus-unsaturated median trustworthy PM, per epoch.

    The frame tie (brightest 20% of src) is dominated by unsaturated stars,
    so a position bias in the saturated-star fit of ONE epoch passes
    straight into those stars' PMs as a common offset: a step in PM at the
    saturation magnitude.

    A PM difference alone cannot say WHICH epoch's positions are biased, and
    a star saturated in both epochs carries either epoch's bias. Each
    epoch's diagnostic therefore uses the rows saturated ONLY in that epoch
    (src: ``src_saturated & ~ref_saturated``; ref: the converse), compared
    against rows saturated in neither. Rows saturated in both form a third,
    'both' entry, which can show that an offset exists but cannot attribute
    it.

    Returns a list of dicts, ``which`` in ('src', 'ref', 'both'), each with
    n_sat, n_unsat, d_pm_ra/d_pm_dec (+ _err, mas/yr), the implied position
    offset d_pos_ra_mas/d_pos_dec_mas (= d_pm * baseline) and ``offset``
    (True when either axis exceeds ``max_offset`` mas/yr AND ``nsigma``
    standard errors). An entry is omitted when its saturated set or the
    neither-saturated set has < ``min_n`` trustworthy rows. When only one of
    the src/ref columns exists, rows are attributed to that epoch alone.
    """
    g = np.asarray(pm['trustworthy'], bool)
    have = {w: f'{w}_saturated' in pm.colnames for w in ('src', 'ref')}
    if not any(have.values()):
        return []
    sat = {w: (np.asarray(pm[f'{w}_saturated'], bool) if have[w]
               else np.zeros(len(pm), bool)) for w in ('src', 'ref')}
    neither = g & ~sat['src'] & ~sat['ref']
    dt = float(pm.meta.get('baseline_yr', np.nan))
    sets = {'src': g & sat['src'] & ~sat['ref'],
            'ref': g & sat['ref'] & ~sat['src'],
            'both': g & sat['src'] & sat['ref']}
    out = []
    if neither.sum() < min_n:
        return out
    for which, a in sets.items():
        if which != 'both' and not have[which]:
            continue
        if a.sum() < min_n:
            continue
        res = dict(which=which)
        res.update(_offset_between(pm, a, neither, dt, max_offset, nsigma))
        out.append(res)
    return out


def saturation_offset_epochs(diags):
    """From saturation_pm_offset's output: (epochs whose saturated positions
    are offset, unattributed).

    An epoch is blamed when its own exclusive set is offset. An offset seen
    only in the saturated-in-both set is attributed by elimination when the
    OTHER epoch's exclusive set exists and is clean (that epoch's saturated
    positions agree with its unsaturated frame), e.g. Quintuplet v6: every
    src-saturated star is also ref-saturated, but 3,011 ref-only saturated
    stars show no offset, so the 'both' offset comes from src. With neither
    exclusive set available, ``unattributed`` is True and no epoch is
    blamed.
    """
    by = {d['which']: d for d in diags}
    epochs = [w for w in ('src', 'ref') if w in by and by[w]['offset']]
    both_off = 'both' in by and by['both']['offset']
    if both_off and not epochs:
        # 'clean' = consistent with zero (|d| < nsigma*err), small next to
        # the 'both' offset (|d| < 1/4 of it) AND precise enough to have seen
        # 1/4 of it (nsigma*err < 1/4 of it), on each axis where that offset
        # is significant; on every axis also precise enough to have seen the
        # flag threshold (nsigma*err < SAT_PM_OFFSET_MAS_YR). A set merely
        # below the flag threshold can carry a real offset of its own, and a
        # noisy set is consistent with zero without showing anything.
        def clean(w):
            d, b = by.get(w), by['both']
            if d is None or d['offset']:
                return False
            for ax in ('d_pm_ra', 'd_pm_dec'):
                lim = SAT_PM_OFFSET_NSIGMA * d[f'{ax}_err']
                if abs(d[ax]) >= lim or lim >= SAT_PM_OFFSET_MAS_YR:
                    return False
                if abs(b[ax]) > SAT_PM_OFFSET_NSIGMA * b[f'{ax}_err']:
                    if abs(d[ax]) >= 0.25 * abs(b[ax]) or lim >= 0.25 * abs(b[ax]):
                        return False
            return True
        cl = [w for w in ('src', 'ref') if clean(w)]
        if len(cl) == 1:
            epochs = [{'src': 'ref', 'ref': 'src'}[cl[0]]]
            # recorded in the diag (and so in meta['sat_pm_offset_json'])
            by['both']['attributed_by_elimination'] = epochs[0]
    return epochs, (both_off and not epochs)


def build(src_paths, ref_paths, filt, src_epoch, ref_epoch, out_path,
         match_radius=0.15, tie_magcut=None, isolation_radius=None, verbose=True,
         demote_offset_saturated=True, src_isolation='flux', assign=DEFAULT_ASSIGN,
         rescue_k=DEFAULT_RESCUE_K, rescued_trustworthy=False, skip_unverified_refs=False,
         flux_ratio_cap=DEFAULT_FLUX_RATIO_CAP):
    src = load_filter_catalog(src_paths, filt, src_epoch, rescue_k=rescue_k)
    if verbose:
        print(f'  src ({filt}, epoch {src_epoch:.3f}) usable: {src["n"]:,}')
        if rescue_k is not None:
            print(f'  src rows rescued (flux_err fails the S/N cut, flux >= {rescue_k:g} x '
                  f'local noise): {int(src["err_rescued"].sum()):,}')
    ref, diags = load_and_tie_ref_catalogs(ref_paths, filt, ref_epoch, src,
                                           tie_magcut=tie_magcut, verbose=verbose,
                                           rescue_k=rescue_k,
                                           skip_unverified=skip_unverified_refs)
    if verbose:
        print(f'  ref ({filt}, epoch {ref_epoch:.3f}) usable: {ref["n"]:,} '
              f'(from {len(diags)} independently-tied observations)')
        if rescue_k is not None:
            print(f'  ref rows rescued: {int(ref["err_rescued"].sum()):,}')
        # same_star_radius (build_pm_catalog_2epoch's default 0.05" = 50 mas)
        # assumes a tied observation's own residual scatter is well below
        # that, so a re-detection of the SAME star lands within it. A tie
        # whose resid_mas approaches 50 mas can't reliably tell "same star,
        # noisy" from "different, close star" at that radius either way.
        for d in diags:
            i = d['ref_index']
            if d['rms_resid_mas'] > 25.0:
                print(f'  WARNING: ref[{i}] tie resid {d["rms_resid_mas"]:.1f} mas is '
                      f'within 2x the same_star_radius (50 mas) build_pm_catalog_2epoch '
                      f'uses to tell a re-detection of the same star from a real close '
                      f'neighbor -- isolation-cut results here are less trustworthy than '
                      f'the trustworthy flag alone suggests.')
    pm = M.build_pm_catalog_2epoch(src, ref, match_radius=match_radius,
                                   isolation_radius=isolation_radius,
                                   flux_ratio_cap=flux_ratio_cap,
                                   src_isolation=src_isolation, assign=assign,
                                   rescued_trustworthy=rescued_trustworthy)
    pm.meta['filter'] = filt
    pm.meta['flux_ratio_cap'] = float(flux_ratio_cap)
    pm.meta['frame'] = ('relative: mean motion + mean shear of the affine tie\'s own '
                        'bright/compact matching sample subtracted per observation; '
                        'NOT an absolute frame -- see build_treasury_pm module docstring')
    # Per-observation fitted tie coefficients (A, B: dx/dy = A0/B0 + A1/B1*x +
    # A2/B2*y, arcsec) plus match/residual diagnostics -- numerically usable
    # (json.loads this), not just a log string, so a later reader can see
    # exactly how much linear motion/shear each tie removed and add it back
    # if needed.
    pm.meta['tie_diag_json'] = json.dumps(diags)
    pm.meta['ref_paths_json'] = json.dumps([os.path.basename(q) for q in ref_paths])
    pm.meta['src_paths_json'] = json.dumps([os.path.basename(q) for q in src_paths])
    if rescue_k is not None:
        pm.meta['rescue_k'] = float(rescue_k)
    if ref.get('unverified_paths'):
        pm.meta['ref_unverified_json'] = json.dumps(
            [os.path.basename(q) for q in ref['unverified_paths']])
    # Saturated-star PM offset check (see saturation_pm_offset). An offset
    # saturated population is a catalog-side position bias, not a motion:
    # report it, and (by default) remove the affected rows from
    # 'trustworthy' so downstream products do not carry a PM step at the
    # saturation magnitude. 'trustworthy_presat' always holds the flag
    # before this step.
    sat_diag = saturation_pm_offset(pm)
    pm['trustworthy_presat'] = pm['trustworthy'].copy()
    for d in sat_diag:
        if verbose or d['offset']:
            label = ('saturated in both epochs' if d['which'] == 'both'
                     else f'saturated in {d["which"]} only')
            print(f'  saturated-star PM offset ({label} vs neither): '
                  f'n_sat={d["n_sat"]:,} n_unsat={d["n_unsat"]:,} '
                  f'd_pm_ra={d["d_pm_ra"]:+.2f}+-{d["d_pm_ra_err"]:.2f} '
                  f'd_pm_dec={d["d_pm_dec"]:+.2f}+-{d["d_pm_dec_err"]:.2f} mas/yr '
                  f'(position {d["d_pos_ra_mas"]:+.1f}, {d["d_pos_dec_mas"]:+.1f} mas)')
    epochs, unattributed = saturation_offset_epochs(sat_diag)
    pm.meta['sat_pm_offset_json'] = json.dumps(sat_diag)
    demote = np.zeros(len(pm), bool)
    for w in epochs:
        print(f'  WARNING: {w}-epoch saturated-star-replaced positions are offset '
              f'from the unsaturated frame (PM step at the saturation magnitude); '
              f're-catalog the {w} epoch with the current saturated_star_finding '
              f'before using bright-star PMs.'
              + (' (attributed by elimination; assumes depth-independent bias)'
                 if any(d.get('attributed_by_elimination') == w for d in sat_diag)
                 else ''))
        demote |= np.asarray(pm[f'{w}_saturated'], bool)
    if unattributed:
        print('  WARNING: stars saturated in BOTH epochs show a PM offset, but '
              'cannot attribute it to an epoch (too few stars saturated in only '
              'one epoch); check both epochs\' saturated-star positions.')
        demote |= (np.asarray(pm['src_saturated'], bool) &
                   np.asarray(pm['ref_saturated'], bool))
    if demote_offset_saturated and demote.any():
        n_dem = int((pm['trustworthy'] & demote).sum())
        pm['trustworthy'] &= ~demote
        print(f'  {n_dem:,} saturated rows removed from trustworthy '
              f'(trustworthy_presat keeps them).')
    pm.write(out_path, overwrite=True)
    ntrust = int(pm['trustworthy'].sum())
    if verbose:
        g = pm['trustworthy']
        print(f'  matched {len(pm):,}  trustworthy {ntrust:,}  '
              f'dt {pm.meta["baseline_yr"]:.3f} yr')
        if ntrust > 0:
            print(f'  trustworthy pm_tot median {np.median(pm["pm_tot"][g]):.2f} mas/yr '
                  f'(systematic-tie sanity: should be a few mas/yr for CMZ stars, '
                  f'not tens -- if it is, suspect a frame-tie residual, not real motion)')
    print(f'  wrote {out_path}')
    return pm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, nargs='+', help='epoch-0 catalog path(s)')
    ap.add_argument('--src-epoch', type=float, required=True)
    ap.add_argument('--ref', required=True, nargs='+', help='epoch-1 catalog path(s)')
    ap.add_argument('--ref-epoch', type=float, required=True)
    ap.add_argument('--filter', required=True, help='filter, e.g. f212n')
    ap.add_argument('--out', required=True)
    ap.add_argument('--match-radius', type=float, default=0.15)
    ap.add_argument('--isolation-radius', type=float, default=None,
                    help='beam (x3) for the trustworthy isolation check; '
                         'defaults to --match-radius. Widen this for a src '
                         'field with real sub-arcsec structure (e.g. a dense '
                         'cluster core) where match_radius alone is too small '
                         'to catch a blended-but-undetected companion.')
    ap.add_argument('--src-isolation', choices=('flux', 'any'), default='flux',
                    help="source-epoch half of the isolation check: 'flux' "
                         "(default) ignores src neighbours fainter than "
                         "--flux-ratio-cap x the star's flux (as on the ref "
                         "side); 'any' is the pre-2026-10-20 rule, any src "
                         "neighbour in the beam rejects the star")
    ap.add_argument('--flux-ratio-cap', type=float, default=DEFAULT_FLUX_RATIO_CAP,
                    help='companion-to-star flux ratio below which a neighbour inside '
                         'the isolation beam does not disqualify the star from '
                         f"'trustworthy' (default {DEFAULT_FLUX_RATIO_CAP:g}; 0.10, "
                         'i.e. contrast >= 10, reproduces the tables made before '
                         '2026-11-18)')
    ap.add_argument('--keep-offset-saturated', action='store_true',
                    help='report a saturated-star PM offset but leave those '
                         'rows trustworthy (default: remove them)')
    ap.add_argument('--assign', choices=('nearest', 'mincost'), default=DEFAULT_ASSIGN,
                    help="src-ref pairing: 'mincost' keeps, of the src rows that "
                         "claim one ref row, the one whose position and flux fit "
                         "(multiepoch_pm._gated_mincost_pairs); 'nearest' drops "
                         "every claimant of a ref row that two src rows both call "
                         f"nearest (the pre-2026-11-18 rule). Default: {DEFAULT_ASSIGN}")
    ap.add_argument('--rescue-inflated-err', type=float, default=DEFAULT_RESCUE_K, metavar='K',
                    help='also use rows whose flux_err fails the S/N cut but whose '
                         'flux is >= K x the local noise (group fits that lost '
                         f'their covariance). Default: {DEFAULT_RESCUE_K:g}; 0 turns it off')
    ap.add_argument('--rescued-trustworthy', action='store_true',
                    help='let pairs with a rescued row be trustworthy (default: '
                         'they are matched but not trustworthy)')
    ap.add_argument('--skip-unverified-refs', action='store_true',
                    help='drop a --ref catalog whose tie to --src cannot be verified '
                         '(e.g. a tile with no overlap) instead of aborting')
    args = ap.parse_args()
    if args.rescue_inflated_err < 0:
        ap.error('--rescue-inflated-err must be >= 0 (0 turns the rescue off)')
    build(args.src, args.ref, args.filter, args.src_epoch, args.ref_epoch, args.out,
         match_radius=args.match_radius, isolation_radius=args.isolation_radius,
         demote_offset_saturated=not args.keep_offset_saturated,
         src_isolation=args.src_isolation, assign=args.assign,
         rescue_k=args.rescue_inflated_err or None, rescued_trustworthy=args.rescued_trustworthy,
         skip_unverified_refs=args.skip_unverified_refs, flux_ratio_cap=args.flux_ratio_cap)


if __name__ == '__main__':
    main()
