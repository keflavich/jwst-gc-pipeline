#!/usr/bin/env python
"""Rename superseded mosaics so they stop matching ``*.fits`` globs.

A mosaic left behind by a superseded reduction carries that reduction's
astrometry, which is usually wrong by arcseconds.  Left under a live ``*.fits``
name it is indistinguishable from a current product to every glob in the tree,
so a reader picks it up by accident.  Renaming it out of ``*.fits`` is what
makes that impossible; the file itself is kept, and the rename is reversible
from the log this script writes.

Two independent rules select a file, because a superseded mosaic can be
recognised in two different ways:

RULE 1 -- BY NAME.  ``*reproject*i2d*.fits``, ``*realigned-to-*.fits``,
``*-merged-reproject-*.fits``: names that say outright which retired alignment
path produced them.  (Generalisation of the brick-only 2026-07-03 scratch pass,
repo-root ``_stale_rename.py``: 192 renamed, log
``<field>/_stale_rename_2026-07-03.log``.  That pass missed
``merged-reproject-vvv_i2d`` -- the canonical known-bad 2023 brick files --
because its globs were narrower than these.)

RULE 2 -- BY RETIRED PRODUCT FAMILY.  A superseded mosaic need not say so in
its name.  cloudc carries

    F405N/pipeline/jw02221-o002_t001_nircam_f405n-f444w_i2d.fits   2023-07-11

a 2023 product that ties to VIRAC2 (the VVV-based near-infrared reference
catalogue used for these fields) **4.1 arcsec** out of place, where every
current cloudc mosaic ties at 32-33 mas.

Its header reads ``FILTER=F444W, PUPIL=F405N`` -- and so does its live 2026
replacement, ``clear-f405n-merged_i2d.fits``.  F444W is the blocking filter for
the F405N narrowband, not a foreign filter, so an earlier version of this note
claiming "F444W is not a cloudc filter and no F444W exposure exists on disk" was
simply wrong.  What separates the two files is the NAMING VINTAGE -- the 2023
reduction wrote the pupil pair into the product token (``f405n-f444w``) where
the current one writes ``clear-f405n-merged``.  So for this file the
(band, pointing, product) key is distinguishing filename generations rather than
different kinds of product, and it is the 365-day age guard plus the 4.1 arcsec
measurement, not the key, that make the case for quarantining it.  Nothing
about its NAME distinguishes
it from a live product: it carries the canonical level-3 form
``jw<prop>-o<obs>_t<NNN>_<instrument>_<band>-<module>_i2d.fits``, so rule 1 does
not see it and a ``*_i2d.fits`` glob does (issue #339).

The question rule 2 asks is NOT "is this older than its neighbours" -- that is
the version that nearly deleted live data, see below.  It is **is this KIND of
product still being made**:

  * no member of the file's own (band, pointing, product) family is newer than
    the reduction-campaign floor = (the field's newest primary mosaic) - 21 days
    -- i.e. the current generation does not produce this product at all; AND
  * the file is more than MIN_ORPHAN_AGE_DAYS (365) older than the field's
    newest primary mosaic.

"pointing" is the ``jw<proposal>-o<observation>`` prefix; "product" is the band
and module token, ``clear-f200w-merged`` / ``clear-f200w-nrca`` / the pupil-pair
form ``f405n-f444w``.  Both belong in the key, and the second one is the one
that matters most:

  * BY PRODUCT.  A merged mosaic and its two per-module mosaics are three
    distinct live deliverables, and the merged drizzle routinely runs later in a
    campaign than the per-module ones -- so the merged mosaic ends up the OLDEST
    primary product of its band while being the band's headline deliverable.
    wd1's F200W merged mosaic (2026-06-13, 5.1 GB) sits 18 days behind its own
    nrca/nrcb siblings (2026-07-01).  An earlier version of this rule, which
    ranked a product against every primary mosaic of its band and pointing, put
    that file THREE DAYS from being quarantined, and would have taken
    cloudef's and quintuplet's per-module mosaics with it.
  * BY POINTING.  Several pointings share a band directory and are reduced weeks
    apart without any being superseded: ngc6334 keeps proposals 6778 and 7213 in
    one F200W directory, sickle keeps observations o001/o002/o003 in one F1130W
    directory.  Ignoring the pointing flagged nine live products under the
    SUPERSEDED design, which ranked a product against its band's newest.

    Under the family rule the pointing no longer protects live data, and saying
    otherwise would overstate what this key does: merging pointings raises a
    family's newest, so a family looks live MORE often and the rule selects
    FEWER files.  Dropping it can only cause false NEGATIVES -- one pointing's
    currency masking another's staleness -- which is what its test pins.

The 365-day age guard is what separates a retired family from a live one that
merely finished early, and it is LOAD BEARING: 47 live primary mosaics across
the archive are held back from quarantine by that constant alone.  The numbers
are in MIN_ORPHAN_AGE_DAYS below -- 3.26x clear of the closest live product
(sickle MIRI at 111.9 days), 1.17x clear of the nearest orphan (w51 at 426.7).
Do not change it without reading them.

Scope is narrow by construction -- only the primary drizzle products
(``PRIMARY_MOSAIC_RE``), never the per-exposure ``_outlier_i2d`` intermediates
(20-184 of them per cloudc band) and never the photometry byproducts
(``_data_i2d``, ``*mergedcat*``).

Each rule keeps its OWN reference and campaign floor, so adding rule 2 leaves
rule 1's selection bit-identical: measured over all 17 fields with a
``*/pipeline`` directory, 55 files before and 55 after, plus rule 2's 8.  The
two rules are evaluated INDEPENDENTLY and their results unioned; letting rule 1
claim a file it then skips for want of its own reference hid files that rule 2
selects.

Known limitation, stated so it is not mistaken for coverage: rule 2 cannot see a
whole POINTING that is stale, because every one of its mosaics is then its own
family's newest.  That is the conservative direction, and it is what
``check_generation_span`` and the release freshness gate look at instead.

(An earlier version of this note claimed the opposite of what the rule now does
-- that a band directory whose ONLY primary mosaic is an orphan would be kept
because it is its own reference.  Under the (band, pointing, product) key that
is exactly how such an orphan IS caught: cloudc's F405N file is the sole member
of its family and is selected.  A stale safety notice in a script that renames
science data reads as a guarantee it does not give, which is why it is called
out here rather than quietly deleted.)

⚠ THREE OF THE EIGHT current selections are the ONLY primary mosaic of their
(band, pointing): brick F405N, F410M and F466N each hold exactly one
``jw02221-o002`` primary mosaic and it is the 2023 file.  Quarantining those
leaves that band and pointing with no primary mosaic at all, and none of the
three has had its astrometry measured -- they are selected on generation alone.

They CAN be rebuilt: each of those directories holds 64 ``jw02221*_cal.fits``
calibrated exposures (cloudc F405N holds 32), so a re-drizzle would regenerate
them.  An earlier version of this notice said no constituent exposures survived,
which told the operator a decision was permanent when it is not.  The rename is
reversible in any case -- the file is kept and the log records it.

Decide it before running with --execute all the same: it is still the removal of
a band and pointing's only product.

Band token is parsed from either NIRCam ('clear-f182m-', 'f405n-f444w-'
pupil forms) or MIRI ('_f2550w') filename conventions.

Usage:
  python rename_stale_mosaics.py --field brick [--field cloudc ...] [--execute]

Dry-run by default; --execute renames and appends to
``<fieldpath>/_stale_rename_<date>.log``.
"""
import argparse
import calendar
import glob
import os
import re
import time

DAY = 86400
BASE = '/orange/adamginsburg/jwst'
# band token: 'clear-f182m-', '-f405n-' (pupil form), or MIRI '_f2550w'
BAND_RES = (re.compile(r'clear-([a-z0-9]+)[-_]', re.I),
            re.compile(r'[-_](f\d{3,4}[mnw])[-_.]', re.I))
PATTERNS = ('*reproject*i2d*.fits', '*realigned-to-*.fits',
            '*-merged-reproject-*.fits')

#: A PRIMARY drizzle product: the level-3 association output itself, ending at
#: ``_i2d.fits`` immediately after the band/module token.  Excludes by
#: construction the per-exposure ``jw02221002001_02201_00001_nrcalong_*_i2d``
#: intermediates (no ``-o<obs>_t<NNN>_`` infix) and every photometry byproduct
#: (``_data_i2d``, ``*_m<N>_daophot_*_mergedcat_*_i2d``), which carry extra
#: tokens after the band.  Together they are 20-184 live files per cloudc band
#: (F770W 20, F405N 184), all of which a looser pattern would quarantine.
PRIMARY_MOSAIC_RE = re.compile(
    r'^(?P<pointing>jw\d+-o\d+)_t\d+_[a-z]+_(?P<product>[a-z0-9-]+)_i2d\.fits$')

#: How much older than the field's current generation a mosaic must be before it
#: counts as an orphan of a retired reduction rather than a product that was
#: simply written earlier in the same campaign.
#:
#: This is the guard that keeps the rule off live data, and it is LOAD BEARING
#: rather than a comfortable separator.  Measured over all 17 fields at the
#: default --campaign-days 21:
#:
#:   47 LIVE primary mosaics belong to a family the current generation no longer
#:   writes, and are held back from quarantine by THIS CONSTANT ALONE.
#:
#:   The margin on the live side is set by the file NEAREST to being taken,
#:   i.e. the LARGEST age gap among those 47 -- not the smallest:
#:
#:   closest live case   sickle F770W and F1500W (jw03958-o003), 111.91 days
#:                       behind their field's newest primary mosaic
#:                       -> margin 365/111.91 = 3.26x
#:   next                sickle F1130W (same pointing), 111.82 days;
#:                       then cloudc F2550W at 59.4 and six more sickle MIRI
#:                       products at ~50
#:   tightest ORPHAN     w51's 2025-06-06 merged-reproject, 426.7 days
#:                       -> margin 426.7/365 = 1.17x
#:   the other seven     2023 products in 2026 fields, 1115.6-1152.7 days
#:
#: So the two populations are separated by 111.9 days against 426.7 -- a factor
#: of 3.8, not the "two orders of magnitude" one earlier version of this comment
#: claimed nor the 18 another did, and the guard sits nearer the orphan edge
#: than the live edge either way.  Lowering it does NOT start taking live data
#: 47 files at a time: the first casualties are the three sickle MIRI products
#: at ~112 days, alone, and 44 files sit at 59 days or less.  Raising it drops
#: w51's orphan first.
#:
#: Two earlier versions of this comment got the live-side number wrong, in the
#: direction that makes the guard look safer than it is, so both are recorded
#: here rather than quietly replaced:
#:
#:   * wd1's merged F200W mosaic, "18 days behind its own per-module siblings",
#:     was cited as the worst live case.  Doubly wrong: wd1 has NO
#:     family-retired primary mosaics at the default settings, so nothing there
#:     depends on this constant at all.  The (band, pointing, product) key is
#:     what protects wd1; this constant is what protects the other 47.
#:   * brick F2550W at 23.3 days was then cited instead, for a claimed 15.7x.
#:     That is the SMALLEST gap in the held-back set -- its safest member --
#:     because the measurement sorted ascending and took the first row.  The
#:     margin is set by the nearest miss, which is 4.8x closer than that.
#:
#: Every number above is printed by
#:
#:     rename_stale_mosaics.py --audit-age-guard --field <each field>
#:
#: which measures through the rule's OWN references (`field_generations`) rather
#: than a hand-written scan.  Both retracted numbers came from hand-written
#: scans.  Re-run it before moving this constant.
MIN_ORPHAN_AGE_DAYS = 365

#: Renaming to this suffix takes the file out of every ``*.fits`` glob, which is
#: the protection that matters: no reader selects it by accident any more.  It
#: does NOT make the bytes unopenable -- ``fits.open`` does not care about the
#: extension -- so anyone who reaches for one of these has to name it
#: explicitly, which is the point.
SUFFIX = '.bad'

#: Quarantine conventions this tree has used, kept as documentation and as the
#: invariant asserted below.  NOTE: `is_quarantined` does not read this list --
#: it tests `not basename.endswith('.fits')`, which every one of these satisfies
#: by construction.  The list is what makes that test's correctness
#: checkable.  Counted under ``*/*/pipeline`` on 2026-08-10:
#: ``*_stale`` 465, of which ``*_badastrometry_stale`` is 327 -- so 138 carry the
#: bare form, which is what the 2026-07-03 brick pass wrote ("EXECUTE -- 192
#: stale, 29 kept").  ``*.bad`` 0, this convention being new.  (A whole-tree
#: count is larger and did not complete in reasonable time on this filesystem;
#: the number quoted here is the one that was actually measured.)
#: ``release_freshness`` recognises ``_badastrometry_stale`` and ``.bad`` but NOT
#: the bare ``_stale``; that gap is real and is reported on #339 rather than
#: fixed here, since closing it reclassifies files in every release listing.
QUARANTINE_SUFFIXES = (SUFFIX, '_badastrometry_stale', '_stale')
if any(s.endswith('.fits') for s in QUARANTINE_SUFFIXES):      # pragma: no cover
    raise RuntimeError(
        "a quarantine suffix that leaves the name ending in '.fits' would let "
        "the file back into every glob this script exists to remove it from")


def mt(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return None


def fmt(t):
    return time.strftime('%Y-%m-%d', time.localtime(t)) if t else 'NONE'


def band_of(name):
    for rex in BAND_RES:
        m = rex.search(name)
        if m:
            return m.group(1).lower()
    return None


def is_quarantined(basename):
    """True for a file some pass has already renamed out of ``*.fits``.

    Every quarantine convention this script has used appends AFTER the
    extension (``.fits.bad``, ``.fits_badastrometry_stale``), so the test is
    simply that the name no longer ends in ``.fits``.  Written out rather than
    left to the globs, so that a future suffix inserted before the extension
    instead of after it fails here loudly instead of silently re-entering the
    candidate set.
    """
    return not basename.endswith('.fits')


def date_header(path):
    """The pipeline's own ``DATE`` for this product, as a unix time, or None.

    Read lazily so the script still imports and dry-runs where astropy is not
    installed, and header-only so a 150 MB mosaic costs one small read.
    """
    try:
        from astropy.io import fits
    except ImportError:
        return None
    try:
        value = fits.getheader(path).get('DATE')
    except OSError:
        return None
    if not value:
        return None
    # FITS DATE is UTC.  `time.mktime` would read it as LOCAL time, putting a
    # 4-5 hour zone offset (and a DST discontinuity) between a DATE-derived
    # generation and an mtime-derived one.  `calendar.timegm` is the UTC inverse.
    try:
        return calendar.timegm(time.strptime(str(value)[:19],
                                             '%Y-%m-%dT%H:%M:%S'))
    except ValueError:
        return None


def generation(path):
    """When this product was made: ``(unix_time, clock)``, clock in {DATE, mtime}.

    ``DATE`` is what the pipeline stamped when it wrote the file and is what
    survives a copy or a restore; mtime is the fallback for a product that has
    none (or that astropy cannot open).  Reported so a comparison between two
    files measured on different clocks is visible rather than assumed.
    """
    stamped = date_header(path)
    if stamped is not None:
        return stamped, 'DATE'
    return mt(path), 'mtime'


def product_key(basename):
    """``(pointing, product)`` for a primary mosaic, else None.

    ``pointing`` is ``jw<proposal>-o<observation>``; ``product`` is the band and
    module token, e.g. ``clear-f200w-merged``, ``clear-f200w-nrca``, or the
    pupil-pair form ``f405n-f444w``.  BOTH are needed to decide what a product's
    own generation is, and getting this wrong is the way this script destroys
    live data.

    One band directory routinely holds SEVERAL live primary mosaics, and they are
    written at different times without any of them being superseded.

    By product: the merged mosaic and the two per-module ones are three distinct
    deliverables, and the merged drizzle often runs later in a campaign than the
    per-module ones.  wd1's F200W merged mosaic is 18 days older than its own
    nrca/nrcb siblings, and it is the 5.1 GB primary deliverable of that band.
    Ranking it against them -- which an earlier version of this did -- put it
    three days away from being quarantined.  (Under the family rule the age
    guard now covers wd1's 18 days on its own; the product token's job is to
    stop a current product of the same pointing MASKING a retired one.)

    By pointing: several pointings share a band directory and are reduced at
    different times without any of them being superseded:
    sickle's ``F1130W/pipeline`` carries observations o001, o002 and o003 of
    proposal 3958, and ngc6334's ``F200W/pipeline`` carries proposals 6778 and
    7213 side by side (they share a target directory and a filter list).
    Comparing a mosaic's generation against the directory's newest, rather than
    against its OWN pointing's newest, would report six live ngc6334/7213
    products and three live sickle MIRI ones as superseded -- measured, before
    this scoping was added.
    """
    m = PRIMARY_MOSAIC_RE.match(basename)
    return (m.group('pointing'), m.group('product')) if m else None


def primary_mosaics(banddir):
    """The band's primary drizzle products -- see ``PRIMARY_MOSAIC_RE``."""
    return sorted(p for p in glob.glob(f'{banddir}/*_i2d.fits')
                  if PRIMARY_MOSAIC_RE.match(os.path.basename(p))
                  and not is_quarantined(os.path.basename(p)))


def field_generations(field, campaign_days=21):
    """Everything the two rules need to know about one field's generations.

    ``(banddirs, band_of_dir, dref1, fam_newest, field_newest, campaign1,
    campaign2)``, or None when the field has no ``*/pipeline`` directories.

    Extracted so ``--audit-age-guard`` measures the age guard's margin through
    the SAME references the rule uses.  An audit that recomputes them
    independently can report a margin the rule does not have, which is how the
    two retracted numbers in ``MIN_ORPHAN_AGE_DAYS`` were arrived at.
    """
    banddirs = glob.glob(f'{BASE}/{field}/*/pipeline')
    if not banddirs:
        return None
    dref1, band_of_dir = {}, {}
    fam_newest = {}          # (band, pointing, product) -> newest generation
    field_newest = None      # the field's current generation, any primary mosaic
    for d in banddirs:
        band = os.path.basename(d.rsplit('/pipeline', 1)[0]).lower()
        band_of_dir[d] = band
        g = [x for x in glob.glob(f'{d}/*-merged_data_i2d.fits')
             if 'nrca' not in os.path.basename(x)
             and 'nrcb' not in os.path.basename(x)]
        dref1[band] = max((mt(x) for x in g), default=None)
        for p in primary_mosaics(d):
            gen = generation(p)[0]
            if gen is None:
                continue
            key = (band,) + product_key(os.path.basename(p))
            fam_newest[key] = max(gen, fam_newest.get(key, gen))
            field_newest = gen if field_newest is None else max(field_newest, gen)
    campaign1 = max((v for v in dref1.values() if v), default=None)
    campaign1 = (campaign1 - campaign_days * DAY) if campaign1 else None
    campaign2 = (field_newest - campaign_days * DAY) if field_newest else None
    return (banddirs, band_of_dir, dref1, fam_newest, field_newest,
            campaign1, campaign2)


def age_guard_rows(field, campaign_days=21):
    """Every primary mosaic of ``field`` whose product family is RETIRED.

    ``[(age_days, path), ...]`` -- age behind the field's newest primary mosaic.
    These are exactly the files that clause (a) of rule 2 selects, so
    ``MIN_ORPHAN_AGE_DAYS`` alone decides each one: below it the file is LIVE and
    kept, at or above it the file is quarantined.  The guard's margin is
    therefore the gap between the LARGEST age below the constant (the live
    product nearest to being taken) and the SMALLEST age at or above it.
    """
    gens = field_generations(field, campaign_days)
    if gens is None:
        return []
    banddirs, band_of_dir, _dref1, fam_newest, field_newest, _c1, campaign2 = gens
    if field_newest is None:
        return []
    rows = []
    for d in banddirs:
        band = band_of_dir[d]
        for p in primary_mosaics(d):
            gen = generation(p)[0]
            if gen is None:
                continue
            key = (band,) + product_key(os.path.basename(p))
            if campaign2 is not None and fam_newest.get(key, 0) >= campaign2:
                continue                      # family still live -- not rule 2's
            rows.append(((field_newest - gen) / DAY, p))
    return rows


def audit_age_guard(fields, campaign_days=21):
    """Print what MIN_ORPHAN_AGE_DAYS is actually holding, and its two margins.

    The constant's comment cites numbers that must be re-measured before it is
    moved; this is how.  Two earlier revisions of that comment quoted the live
    margin from the WRONG end of the distribution -- the safest member of the
    held-back set rather than the one nearest to being taken -- so the sort
    order here is load bearing and is asserted in the tests.
    """
    held, caught = [], []
    for field in fields:
        for age, path in age_guard_rows(field, campaign_days):
            (held if age < MIN_ORPHAN_AGE_DAYS else caught).append(
                (age, field, os.path.basename(path)))
    held.sort(reverse=True)                   # nearest miss first
    caught.sort()                             # tightest orphan first
    print(f"MIN_ORPHAN_AGE_DAYS = {MIN_ORPHAN_AGE_DAYS}, "
          f"--campaign-days {campaign_days}, {len(fields)} field(s)")
    print(f"\nLIVE primary mosaics of a retired family, held back by this "
          f"constant alone: {len(held)}")
    for age, field, base in held[:5]:
        print(f"  {age:9.2f} d  {field:10s} {base}")
    print(f"\nQuarantined by it: {len(caught)}")
    for age, field, base in caught[:5]:
        print(f"  {age:9.2f} d  {field:10s} {base}")
    if held:
        print(f"\nlive margin   {MIN_ORPHAN_AGE_DAYS}/{held[0][0]:.2f} = "
              f"{MIN_ORPHAN_AGE_DAYS / held[0][0]:.2f}x  "
              f"({held[0][1]} {held[0][2]})")
    if caught:
        print(f"orphan margin {caught[0][0]:.2f}/{MIN_ORPHAN_AGE_DAYS} = "
              f"{caught[0][0] / MIN_ORPHAN_AGE_DAYS:.2f}x  "
              f"({caught[0][1]} {caught[0][2]})")
    if held and caught:
        print(f"populations separated by {held[0][0]:.1f} d against "
              f"{caught[0][0]:.1f} d -- a factor of "
              f"{caught[0][0] / held[0][0]:.1f}")
    return held, caught


def rename_stale_for_field(field, execute=False, campaign_days=21):
    pipe = f'{BASE}/{field}'
    gens = field_generations(field, campaign_days)
    if gens is None:
        print(f"[{field}] no */pipeline dirs under {pipe}; skipping")
        return []
    (banddirs, band_of_dir, dref1, fam_newest, field_newest,
     campaign1, campaign2) = gens

    # Each rule carries its OWN reference generation, and they are deliberately
    # not shared.  Rule 1's is the one this script has always used -- the band's
    # `*-merged_data_i2d.fits` mtime -- so rule 1 selects exactly the files it
    # selected before this rule 2 was added.  Broadening rule 1's reference to
    # the primary mosaics as well was tried and measured: archive-wide it moves
    # rule 1's selection from 55 files to 229, because a band with no
    # `*-merged_data_i2d.fits` currently SKIPS its candidates rather than
    # judging them.  That may well be worth doing, but it is a separate decision
    # about 178 files and does not belong in a fix for one orphan (#339).
    #
    # (The references themselves are computed in `field_generations`, which
    # `--audit-age-guard` shares so the audited margin is the rule's own.)

    # RULE 1: named retired-alignment products, anywhere in the band directory.
    named, generational = {}, {}
    for d in banddirs:
        for pat in PATTERNS:
            for p in glob.glob(f'{d}/{pat}'):
                named[p] = 'retired-path name'
    # RULE 2: a primary mosaic belonging to a RETIRED PRODUCT FAMILY.
    #
    # Not "older than its siblings" -- that is what put wd1's 5.1 GB merged
    # F200W mosaic three days from deletion, because a merged drizzle routinely
    # runs later in a campaign than the per-module ones it is compared against.
    # The question is instead whether this KIND of product is still being made:
    #
    #   (a) no member of its own (band, pointing, product) family is newer than
    #       the campaign floor -- the current generation does not produce it; AND
    #   (b) it is more than MIN_ORPHAN_AGE_DAYS older than the field's newest
    #       primary mosaic.
    #
    # (b) is what separates a retired family from a live one that happens to sit
    # just under the floor.  See MIN_ORPHAN_AGE_DAYS for the margin.
    for d in banddirs:
        band = band_of_dir[d]
        for p in primary_mosaics(d):
            gen = generation(p)[0]
            key = (band,) + product_key(os.path.basename(p))
            if gen is None or field_newest is None:
                continue
            family_live = campaign2 is None or fam_newest.get(key, 0) >= campaign2
            old_enough = gen < field_newest - MIN_ORPHAN_AGE_DAYS * DAY
            if not family_live and old_enough:
                generational[p] = 'retired product family'

    # The two rules are evaluated INDEPENDENTLY and unioned, not first-wins.
    # Letting rule 1 claim a file and then skip it for want of its own reference
    # hides anything rule 2 would have taken.  Measured on the current tree
    # exactly ONE file is in both sets -- w51's
    # `clear-f150w-merged-reproject_i2d.fits`, which is both named and orphaned
    # -- so the union recovers one file today, not the dozen an earlier count
    # claimed (that count predated the 365-day age guard, which excludes
    # ngc6334's `merged-reproject` products).
    plan, kept, seen = [], 0, set()
    for cands, which in ((named, 1), (generational, 2)):
        for f, why in sorted(cands.items()):
            base = os.path.basename(f)
            if is_quarantined(base) or f in seen:
                continue
            band = band_of_dir.get(os.path.dirname(f)) or band_of(base)
            if which == 1:
                ref, campaign, clock, refname = (dref1.get(band), campaign1,
                                                 'mtime', "the band's data_i2d")
                fm = mt(f)
                missing = 'no-data_i2d'
            else:
                ref, campaign, clock = field_newest, campaign2, generation(f)[1]
                refname = "the field's newest primary mosaic"
                fm = generation(f)[0]
                missing = 'no-reference-mosaic'
            if band is None or ref is None or fm is None:
                print(f"  SKIP [{field}] {base} "
                      f"[rule{which}: {'no-band' if band is None else missing}]")
                continue
            if which == 2 or (fm < ref - DAY
                              and (campaign is None or fm < campaign)):
                seen.add(f)
                plan.append((f, fm, ref, why, clock, refname))
            else:
                kept += 1

    print(f"[{field}] {'EXECUTE' if execute else 'DRY RUN'}: "
          f"{len(plan)} superseded, {kept} current kept "
          f"(campaign floors {fmt(campaign1)} / {fmt(campaign2)})")
    skipped = 0
    log = None
    if execute and plan:
        log = open(f'{pipe}/_stale_rename_{time.strftime("%Y-%m-%d")}.log', 'a')
        log.write(f"# rename_stale_mosaics.py {time.strftime('%Y-%m-%d %H:%M')}\n")
    for f, fm, ref, why, clock, refname in plan:
        line = (f"  {fmt(fm)} [{clock}] ({refname} {fmt(ref)}) "
                f"{why}: {os.path.basename(f)}")
        print(('RENAME' if execute else 'would rename') + line)
        if not execute:
            continue
        dst = f + SUFFIX
        # NEVER overwrite an earlier quarantine.  `os.rename` replaces its
        # destination silently, so a product regenerated under a name that was
        # already quarantined would destroy the quarantined bytes on the next
        # run -- of a tool whose whole premise is that it is reversible.  Same
        # guard, same reasoning, as quarantine_pre_obstoken_catalogs.py:206.
        if os.path.exists(dst):
            print(f"    SKIP: {os.path.basename(dst)} already exists -- "
                  f"not overwriting an earlier quarantine")
            log.write(f"SKIP {f} -> {dst} (destination exists)\n")
            skipped += 1
            continue
        try:
            os.rename(f, dst)
        except OSError as ex:
            print(f"    FAILED to rename {f}: {ex}")
            log.write(f"FAILED {f} -> {dst}: {ex}\n")
            continue
        log.write(f"RENAME {f} -> {dst}  ({why}; {clock} {fmt(fm)}, "
                  f"{refname} {fmt(ref)})\n")
        _write_reason_sidecar(dst, why, clock, fm, ref, refname)
    if log:
        log.close()
    if skipped:
        print(f"[{field}] {skipped} of {len(plan)} left in place "
              f"(a quarantine of that name already exists)")
    return plan


def _write_reason_sidecar(path, why, clock, fm, ref, refname):
    """Record WHY this file was quarantined, next to the file itself.

    The run log lives under the field root and is easy to lose track of; whoever
    finds a ``.bad`` file years later needs the reason to be attached to it.
    Never overwrites an existing sidecar, so re-running cannot erase the first
    (and more informative) reason.
    """
    note = path + '.why.txt'
    if os.path.exists(note):
        return
    with open(note, 'w') as fh:
        fh.write(
            f"Quarantined by scripts/reduction/rename_stale_mosaics.py on "
            f"{time.strftime('%Y-%m-%d %H:%M')}.\n"
            f"Reason: {why}.\n"
            f"This product's generation ({clock}) is {fmt(fm)}; the newest "
            f"primary mosaic in its band is {fmt(ref)}.\n"
            f"It carries the astrometry of a superseded reduction and must not "
            f"be read as if it were current.  To undo, drop the '{SUFFIX}' "
            f"suffix; see <field>/_stale_rename_*.log.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--field', action='append', required=True,
                    help='field dir name under /orange/adamginsburg/jwst '
                         '(repeatable)')
    ap.add_argument('--execute', action='store_true',
                    help='actually rename (default: dry run)')
    ap.add_argument('--campaign-days', type=int, default=21)
    ap.add_argument('--audit-age-guard', action='store_true',
                    help='report what MIN_ORPHAN_AGE_DAYS holds back and its '
                         'two margins, and rename nothing')
    args = ap.parse_args()
    if args.audit_age_guard:
        audit_age_guard(args.field, campaign_days=args.campaign_days)
        return
    for field in args.field:
        rename_stale_for_field(field, execute=args.execute,
                               campaign_days=args.campaign_days)


if __name__ == '__main__':
    main()
