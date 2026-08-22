"""Exposures excluded from EVERYTHING — imaging and analysis alike.

An exposure lands here when the data are bad at the source, not when a stage
happens to dislike them: guiding failures, tracking errors, anything a
re-reduction reproduces because the defect is already in the ramp.  Excluding
one is therefore a decision about the observation, taken once, and both halves
of the pipeline read it from here so they cannot disagree about which exposures
exist — the same reason ``destreak_policy`` is a module rather than a flag.

This is deliberately NOT a quality cut.  Nothing here is computed, nothing is
thresholded, and no stage may add to it: a stage that dislikes an exposure has
its own recorded ways to say so (``unverified``, the correction floor, the
vetting cuts), all of which keep the exposure and describe it.  Exclusion is the
one action that makes an exposure cease to exist for the survey, so it is spelled
out by hand with a reason attached.

The key is the exposure's own identity — ``jw<proposal><obs><visit>_<vgroup>_<exp>``
— which every product of that exposure carries as its filename stem, from
``_uncal`` through ``_crf`` and on into the per-exposure catalogs.  Matching on
the stem therefore excludes the exposure everywhere without needing a separate
rule per product type.
"""

#: ``exposure stem -> why``.  Add an entry only with a reason that names the
#: defect in the DATA.
EXCLUDED_EXPOSURES = {
    'jw02045001001_02101_00004': (
        'arches 2045/001 exposure 4: tracking errors. Excluded from all imaging '
        'and analysis (operator decision, 2026-08-21). Its symptoms were visible '
        'downstream long before the cause was named -- an astrometric tie that '
        'oscillated at ~50 mas through eight re-tie iterations without trending '
        '(jwst-gc-pipeline#409), formal errors of +-0.5-2.2 mas against +-0.01 '
        'for every other exposure of the field, and a whole-exposure displacement '
        'across six detectors at once. It also carries a snowball storm (337 '
        'connected components >100 px against 20-23 for its siblings, largest '
        '28,127 px). Both are consequences of the tracking failure, and neither '
        'is removable by re-reduction because both are in the ramp.'),
}


def exposure_stem(path):
    """The ``jw<proposal><obs><visit>_<vgroup>_<exp>`` stem of a product path.

    Returns ``None`` for anything not named that way — mosaics
    (``jw02045-o001_t001_...``), association files, reference catalogues.  Those
    are per-observation products rather than per-exposure ones, so an exposure
    exclusion cannot be expressed against them and must not be guessed at.
    """
    import os
    import re

    base = os.path.basename(str(path))
    m = re.match(r'(jw\d{11}_[0-9a-z]{5}_\d{5})_', base)
    return m.group(1) if m else None


def is_excluded(path):
    """Is this product's exposure excluded from the survey?"""
    stem = exposure_stem(path)
    return stem is not None and stem in EXCLUDED_EXPOSURES


def exclusion_reason(path):
    """Why, or ``None`` if the exposure is not excluded."""
    stem = exposure_stem(path)
    return EXCLUDED_EXPOSURES.get(stem) if stem else None


def drop_excluded(paths, label=''):
    """``(kept, dropped)``, announcing any drop.

    Announcing is the point: an exposure vanishing silently is
    indistinguishable from one that was never reduced, which is the failure this
    project already has a hard rule against ("cataloging must hard-crash on any
    dropped exposure").  An exclusion is the one sanctioned drop, so it says so
    every time rather than relying on someone remembering this file exists.
    """
    kept, dropped = [], []
    for p in paths:
        (dropped if is_excluded(p) else kept).append(p)
    if dropped:
        stems = sorted({exposure_stem(p) for p in dropped})
        where = f'{label}: ' if label else ''
        print(f'{where}excluded {len(dropped)} product(s) of {len(stems)} '
              f'exposure(s) {", ".join(stems)} -- '
              f'{EXCLUDED_EXPOSURES[stems[0]].split(".")[0]}', flush=True)
    return kept, dropped


# --------------------------------------------------------------------------
# NO catalog-name matcher lives here, deliberately
# --------------------------------------------------------------------------
#
# Per-exposure CATALOGS are named
# ``f212n_nrca1_visit001_vgroup02101_exp00004_m2_daophot_basic.fits`` -- visit,
# vgroup, exposure and detector, but NO proposal, observation or field.  A
# matcher keyed on that triple looks like it identifies an exposure and does
# not: ``visit001_vgroup02101_exp00004`` is an ordinary first-visit-group
# exposure that eight other fields also have.  Counted 2026-08-21:
#
#     wd1 150   brick 126   cloudef 120   sgrc 82
#     quintuplet 74   sgra 56   cloudc 6   sickle 6
#
# So such a matcher would have reported 150 good wd1 catalogs as arches
# exposure 4 and dropped them, with a message naming the wrong field.
#
# The need it was written for is real but already met on disk: arches exposure
# 4 had catalogs through m7 written before the exclusion existed, and all 396 of
# its derived products were renamed `.EXCLUDED_tracking_errors_*` with a
# receipt, so no merge glob reaches them.  A future caller that genuinely needs
# to recognise the catalog spelling must key it by FIELD as well and take the
# field from its caller -- until then, a ready-to-misfire matcher whose whole
# purpose is to drop data does not belong here.
