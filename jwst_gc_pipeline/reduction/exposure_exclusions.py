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


def _excluded_visit_keys():
    """``{(visit, vgroup, exposure)}`` of the excluded exposures.

    The PER-EXPOSURE CATALOG names are built differently from the frames --
    ``f212n_nrca1_visit001_vgroup02101_exp00004_m2_daophot_basic.fits`` carries
    visit / vgroup / exposure / detector but NOT proposal or observation -- so
    the frame stem cannot match them.  Within one field's directory the triple
    is unique, which is enough.
    """
    import re
    out = set()
    for stem in EXCLUDED_EXPOSURES:
        m = re.match(r'jw\d{5}\d{3}(\d{3})_([0-9a-z]{5})_(\d{5})$', stem)
        if m:
            out.add(m.groups())
    return out


def is_excluded_catalog_name(path):
    """Is this a PER-EXPOSURE CATALOG of an excluded exposure?

    Needed because the exclusion has to survive on disk: arches exposure 4 has
    catalogs through m7 written before it was excluded, and a merge globbing
    ``*_exp*_m*_daophot_basic.fits`` would ingest them.
    """
    import os
    import re

    m = re.search(r'_visit(\d{3})_vgroup([0-9a-z]{5})_exp(\d{5})_',
                  os.path.basename(str(path)))
    return bool(m) and m.groups() in _excluded_visit_keys()


def is_excluded_any(path):
    """Either spelling: a frame/product stem, or a per-exposure catalog name."""
    return is_excluded(path) or is_excluded_catalog_name(path)
