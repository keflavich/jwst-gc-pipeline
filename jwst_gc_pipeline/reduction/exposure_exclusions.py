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

    # gc2211 observation 023 -- the WHOLE observation.  All four exposures carry
    # serious tracking errors (operator decision, 2026-08-22, on the raw _cal
    # pixels).  Every exposure is listed individually because that is the unit
    # this module excludes; the effect is that observation 023 contributes
    # nothing to the survey.
    #
    # exp4 is the extreme case: bright unsaturated stars are drawn into long
    # curved streaks several times the PSF width, on ALL SIX detectors checked
    # (nrca1-4, nrcb1, nrcb2) -- common to the focal plane, so guiding rather
    # than a detector effect.  exp1 is smeared on all six as well, and exp2/exp3,
    # which read as the "good" pair against their own siblings, are themselves
    # degraded against a clean observation: o046's stars are round and crisp at
    # ~750-980 usable isolated bright stars per detector, where o023 yields 2-35.
    #
    # It was assessed by eye because three summary statistics could not tell the
    # streaked frames from a clean control: second moments at 19x19 px read exp4
    # 1.689 against a clean exp2 1.655 (the trails are longer than the box, so
    # the moments truncate), the same moments at 41x41 put every exposure of
    # every observation in 1.47-1.67 including the o046 control, and R80 read
    # exp4 18.1 px against a control 17.5-17.7 (in a bright crowded field an
    # 80%-enclosed radius measures the stamp, not the star).
    #
    # Downstream symptoms, all now explained: a ~150 mas per-exposure split
    # (exp1 154.6, exp2 13.5, exp3 11.3, exp4 138.8 mas vs the visit consensus),
    # consensus scatter 37.41 mas against ~1 mas for healthy fields, and an
    # m2 checkpoint that could not write its correction because the four nrca
    # detectors disagreed by 73.7 mas (jwst-gc-pipeline#484).
    #
    # NOT excluded, deliberately: gc2211 o049 exposure 4 and o028 exposure 2.
    # Both were flagged by the same qfit/source-count proxies, and the pixels
    # clear them -- o049 exp4 has a small visible defect judged recoverable at
    # <10%, and its stars are round at n~750-980 per detector.  The proxies
    # flagged four exposures across the program; the images support these four
    # (one observation) and not those two.
    'jw02211023001_02201_00001': (
        'gc2211 023 exposure 1: tracking errors across the whole observation. '
        'Excluded from all imaging and analysis (operator decision, 2026-08-22, '
        'assessed on the raw _cal pixels -- see the block above and issue #484). '
        'All four exposures of 023 are excluded; the observation contributes nothing.'),
    'jw02211023001_02201_00002': (
        'gc2211 023 exposure 2: tracking errors across the whole observation. '
        'Excluded from all imaging and analysis (operator decision, 2026-08-22, '
        'assessed on the raw _cal pixels -- see the block above and issue #484). '
        'All four exposures of 023 are excluded; the observation contributes nothing.'),
    'jw02211023001_02201_00003': (
        'gc2211 023 exposure 3: tracking errors across the whole observation. '
        'Excluded from all imaging and analysis (operator decision, 2026-08-22, '
        'assessed on the raw _cal pixels -- see the block above and issue #484). '
        'All four exposures of 023 are excluded; the observation contributes nothing.'),
    'jw02211023001_02201_00004': (
        'gc2211 023 exposure 4: tracking errors across the whole observation. '
        'Excluded from all imaging and analysis (operator decision, 2026-08-22, '
        'assessed on the raw _cal pixels -- see the block above and issue #484). '
        'All four exposures of 023 are excluded; the observation contributes nothing.'),
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
