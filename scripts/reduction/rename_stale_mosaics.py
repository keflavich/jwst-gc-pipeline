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

  THE REPROJECT FAMILY CARRIES A PRECONDITION, and it applies to rule 2 as
  well.  A name containing ``reproject`` says which PATH wrote the file, not
  whether that file is the stale one: ``align_to_catalogs`` computes it from a
  reference catalog independently of the frames, so where the frames were never
  corrected the reproject holds the only tie the band has and the plain
  ``*-merged_i2d.fits`` beside it is at the raw ``assign_wcs`` frame.  Neither
  file's mtime distinguishes the two directions.  Such a file is therefore
  selected only when ALL FOUR of these hold: its (proposal, observation) is
  registered in ``ALIGNMENT_CONFIG``; its own pointing's frames carry a
  non-zero baked ``RAOFFSET``/``DEOFFSET``; the plain sibling EXISTS; and that
  sibling POSTDATES those frames, so it can have inherited what they carry.
  The last two are the ones that look at the file taking over -- the first two
  are both statements about the frames, and a band can hold the reproject alone
  (ngc6334 F115W does) or hold a plain mosaic drizzled before the correction
  was baked.  See ``reproject_supersession`` and issue #724.

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
    the reduction-campaign floor = (the field's newest primary mosaic FROM THE
    SAME INSTRUMENT) - 21 days -- i.e. the current generation does not produce
    this product at all; AND
  * the file is more than MIN_ORPHAN_AGE_DAYS (365) older than that same
    per-instrument newest.

Both references are per instrument because NIRCam and MIRI are reduced on
independent campaigns; clocking a MIRI product against a NIRCam re-drizzle ages
it out for activity it has no part in.  See ``field_generations``.

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
merely finished early, and it is LOAD BEARING: 28 live primary mosaics across
the archive are held back from quarantine by that constant alone.  The numbers
are in MIN_ORPHAN_AGE_DAYS below -- 5.85x clear of the closest live product
(sickle MIRI at 62.4 days), 1.17x clear of the nearest orphan (w51 at 426.7).
Do not change it without reading them.  The reference is per instrument, which
is what keeps a field's NIRCam campaign from ageing out its MIRI products.

Scope is narrow by construction -- only the primary drizzle products
(``PRIMARY_MOSAIC_RE``), never the per-exposure ``_outlier_i2d`` intermediates
and never the photometry byproducts (``_data_i2d``, ``*mergedcat*``).  Together
those come to 20-184 non-primary ``*_i2d.fits`` per cloudc band; the
``*outlier*`` subset alone is 0-144, and quoting that subset's range for the
whole set is a mistake this note has made twice.

Each rule keeps its OWN reference and campaign floor, so adding rule 2 leaves
rule 1's selection bit-identical: measured over all 17 fields with a
``*/pipeline`` directory, 55 files before and 55 after, plus rule 2's 8.  The
two rules are evaluated INDEPENDENTLY and their results unioned; letting rule 1
claim a file it then skips for want of its own reference hid files that rule 2
selects.

Known limitation, stated so it is not mistaken for coverage.  It is in the
conservative direction -- the rule misses orphans, it does not take live data --
and it is a gap ``check_generation_span`` and the release freshness gate are the
things that look at instead.

RULE 2 IS WEAK ON A WHOLE INSTRUMENT THAT IS STALE.  Its reference is that
instrument's own newest primary mosaic, computed from the very set being judged,
so if a field's entire MIRI set is superseded then the newest member of that set
becomes the yardstick for the rest of it.

Stated exactly, because an earlier version of this paragraph claimed more than
the rule delivers ("nothing is selected at any --campaign-days ... invisible"):
what is protected is the instrument's NEWEST product, which can never be more
than 0 days behind itself.  Older members of the same stale set ARE still
selected once they fall MIN_ORPHAN_AGE_DAYS behind it -- so a MIRI set whose own
products span more than a year is partially visible, and a set drizzled within
one year of itself is entirely invisible.  The claim of total invisibility was
tested only on a fixture whose two members shared one age, which is the
degenerate case.

This is the PRICE of the per-instrument reference, and it is deliberate: the
alternative clocked MIRI against NIRCam and let ordinary NIRCam reduction walk
sole-copy MIRI mosaics toward quarantine (see MIN_ORPHAN_AGE_DAYS).  Trading a
missed orphan for a protected live file is the right way round, but it IS a
trade.  No field is in this state today -- every field's newest MIRI primary
mosaic is from 2026-06 or later -- so it is a forward-looking hole rather than a
present one.

A WHOLE POINTING that is stale is NOT a second case of this, and one earlier
version of this note said it was.  The pointing appears only in the family key;
it is not part of either reference, so a stale pointing is judged against the
instrument's newest like everything else.  brick is the live example: every
NIRCam primary mosaic of ``jw02221-o002`` is a 2023 product, and all three are in
the current selection.

But being family-retired is not sufficient, and a second earlier version of this
note over-corrected into saying it was ("IS caught whenever any other pointing of
that instrument is current").  The 365-day guard still applies.  sickle
``jw03958-o003`` is the counterexample and it sits in this comment's own margin
table: all three of its MIRI primaries are family-retired while o001 and o002 are
current, and none is selected, because they are 62 days old rather than 365.
That file IS the nearest miss the 5.85x below is measured from.

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

--only: EXECUTE THE SUBSET YOU HAVE MEASURED
--------------------------------------------
"Measure before quarantining.  The date of reduction is not strong enough signal
alone." (#339, 2026-08-10).  Rule 2 selects on generation, which is not a
measurement, and the seven files it selects today do NOT all clear the 4 arcsec
bar that instruction was given under:

    cloudc  jw02221-o002_..._f405n-f444w_i2d      4122 mas   } meet the bar
    brick   jw02221-o002_..._f405n-f444w_i2d      7862 mas   }
    brick   jw02221-o002_..._clear-f410m_i2d      7841 mas   }
    brick   jw02221-o002_..._f444w-f466n_i2d      7858 mas   }
    brick   jw02221-o001_..._f405n-f444w_i2d       163 mas   } measured, real,
    brick   jw02221-o001_..._clear-f410m_i2d       157 mas   } 25x the current
    brick   jw02221-o001_..._f444w-f466n_i2d       163 mas   } generation, but
                                                             } not arcsecond-scale

Without a way to name files, ``--execute`` was all seven or none, so following
that instruction was not expressible in this tool.  ``--only <basename>``
(repeatable) restricts the run to the named files.

It REFUSES rather than proceeds when a name matches nothing selected: a typo,
or a file the rules stopped selecting, would otherwise rename nothing and exit
0, which is indistinguishable from a successful run.  Names withheld by the
filter are printed, so the log records that the run was deliberately partial.
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

#: The ``align_to_catalogs`` reproject family, which needs its OWN precondition.
#:
#: A ``*-merged-reproject_i2d.fits`` sits beside a plain ``*-merged_i2d.fits``
#: on a frame up to ~2" away, and NEITHER the name nor either file's mtime says
#: which of the two is current.  The reproject is whatever ``align_to_catalogs``
#: last wrote, computed from a reference catalog and INDEPENDENT of the frames:
#:
#:   * where the field's frames carry a baked correction, the plain mosaic was
#:     drizzled from corrected frames and is the current product -- the reproject
#:     predates it and is what this script exists to take out of ``*.fits``;
#:   * where the frames were never corrected, the plain mosaic is at the raw
#:     ``assign_wcs`` frame and the reproject holds THE ONLY TIE THAT EXISTS.
#:     Quarantining it there removes the field's astrometry.
#:
#: So neither rule's generational argument decides this family: a reproject can
#: be the older file and still be the only tied one (ngc6334, wd1), and it can
#: be the newer file and still be superseded.  What decides it is state on disk
#: -- see `reproject_supersession`, which is applied to BOTH rules so the two
#: cannot disagree about one file (#724).
#:
#: "The frames carry a shift" is NOT on its own "the plain mosaic has it": the
#: mosaic only inherits the tie at the level-3 re-drizzle that runs after
#: ``fix_alignment``.  So the gate also requires the plain sibling to EXIST and
#: to postdate those frames; without that it concludes the plain mosaic is the
#: current tie having never looked at the plain mosaic.
REPROJECT_RE = re.compile(r'reproject', re.I)

#: Fallback pointing parse for a name ``PRIMARY_MOSAIC_RE`` does not match.
#: A reproject product that some other pass has already tagged
#: (``..._i2d_im0_badastrom.fits``) keeps its ``jw<prop>-o<obs>`` prefix but no
#: longer ends at ``_i2d.fits``, so the strict pattern misses it.
POINTING_RE = re.compile(r'^(?P<pointing>jw\d+-o\d+)_')

#: Working-copy frame suffixes searched for a baked ``RAOFFSET``.  ``_align``
#: and ``_destreak`` are the two lineages a field can be reduced on (see the
#: sickle two-lineage note); ``_crf`` is the outlier-rejected copy of whichever
#: one ran, written by the same level-3 run that drizzles the mosaic, and
#: carries the same header cards.
#:
#: ALL of them are globbed and the union is ordered by mtime -- see
#: `_pointing_frames`.  Searching them in order and stopping at the first that
#: matches reads the STALE lineage on the four fields that carry two.
FRAME_SUFFIXES = ('_align.fits', '_destreak.fits', '_crf.fits')

#: How many of a pointing's frames to read before concluding it carries no
#: correction.  The issue proposing this precondition said "one frame header per
#: field"; a handful rather than one, because a field with a per-exposure jitter
#: channel can leave an individual exposure at exactly 0 while the rest of the
#: pointing carries the bulk, and one unlucky draw would then read the whole
#: field as uncorrected.  Any non-zero frame answers "corrected"; it takes all
#: of them reading zero to answer "not".
FRAME_SAMPLE = 6

#: Suffixes some OTHER pass has already used to take a product out of service
#: while leaving the name ending in ``.fits``.  ``_im0_badastrom`` is what the
#: m2 astrometry checkpoint writes when it stale-tags a mosaic it has just
#: invalidated, and it writes a ``.why.json`` sidecar NEXT TO the tagged name.
#:
#: THIS IS A POLICY DECISION ABOUT 32 FILES ON SIX FIELDS, not the four m4
#: reprojects that motivated it.  Measured by diffing the dry run of
#: ``origin/main`` against this revision over the 25 fields with a
#: ``*/pipeline`` directory (2026-09-06) -- 275 selected before, 235 after, and
#: 32 of that difference of 40 is this list:
#:
#:   brick    32 -> 26   (jw01182-o004 x4, jw02221-o001 x2)
#:   sgrb2    44 -> 33   (jw05365-o001, F150W..F480M)
#:   sgrc     40 -> 32   (jw04147-o012)
#:   ngc6397   8 ->  6   (jw01979-o001)
#:   m4       16 -> 12   (jw01979-o002/o003)
#:   m92      13 -> 12   (jw01334-o001 F090W)
#:
#: Three reasons the withheld files are still quarantined, so that skipping
#: them does not leave a live-looking product in ``*.fits``:
#:
#:   1. ``release_freshness.QUARANTINE_GLOBS`` already carries
#:      ``{stem}_im0_badastrom*.fits``, so the release path treats them as
#:      quarantined under the name they have.  Renaming one to
#:      ``..._im0_badastrom.fits.bad`` takes it OUT of that glob -- the rename
#:      would REMOVE the recognition it was meant to add.
#:   2. The ``.why.json`` sits beside the tagged name.  Renaming the product
#:      and not the sidecar orphans the first tag's explanation; renaming both
#:      rewrites another pass's audit record.
#:   3. Double-tagging loses which pass made the call.
#:
#: They are named in the output rather than skipped silently -- see the
#: ``already tagged`` line in `rename_stale_for_field`.  Note that this list is
#: the COMPLEMENT of ``QUARANTINE_SUFFIXES``: every member here ends in
#: ``.fits`` by construction (that is what makes it a foreign marker rather
#: than one of ours), which is exactly what the ``RuntimeError`` below forbids
#: for the suffixes this script WRITES.  The assertion below states both halves
#: so the two lists cannot be confused for each other.
FOREIGN_QUARANTINE_MARKERS = ('_im0_badastrom.fits',)
if not all(m.endswith('.fits') for m in FOREIGN_QUARANTINE_MARKERS):  # pragma: no cover
    raise RuntimeError(
        "FOREIGN_QUARANTINE_MARKERS names markers OTHER passes write, which "
        "leave the name ending in '.fits'; one that does not is this script's "
        "own convention and belongs in QUARANTINE_SUFFIXES")

#: A PRIMARY drizzle product: the level-3 association output itself, ending at
#: ``_i2d.fits`` immediately after the band/module token.  Excludes by
#: construction the per-exposure ``jw02221002001_02201_00001_nrcalong_*_i2d``
#: intermediates (no ``-o<obs>_t<NNN>_`` infix) and every photometry byproduct
#: (``_data_i2d``, ``*_m<N>_daophot_*_mergedcat_*_i2d``), which carry extra
#: tokens after the band.  Together they are 20-184 live files per cloudc band
#: (F770W 20, F405N 184), all of which a looser pattern would quarantine.
PRIMARY_MOSAIC_RE = re.compile(
    r'^(?P<pointing>jw\d+-o\d+)_t\d+_(?P<instrument>[a-z]+)_'
    r'(?P<product>[a-z0-9-]+)_i2d\.fits$')

#: How much older than its own instrument's current generation a mosaic must be
#: before it counts as an orphan of a retired reduction rather than a product
#: that was simply written earlier in the same campaign.
#:
#: This is the guard that keeps the rule off live data, and it is LOAD BEARING
#: rather than a comfortable separator.  Measured over all 17 fields at the
#: default --campaign-days 21:
#:
#:   28 LIVE primary mosaics belong to a family their instrument's current
#:   generation no longer writes, and are held back by THIS CONSTANT ALONE.
#:
#:   The margin on the live side is set by the file NEAREST to being taken,
#:   i.e. the LARGEST age gap among those 28 -- not the smallest:
#:
#:   closest live case   sickle F770W and F1500W (jw03958-o003), 62.35 days
#:                       behind their field's newest MIRI primary mosaic
#:                       -> margin 365/62.35 = 5.85x
#:   next                sickle F1130W (same pointing), 62.26 days; then two
#:                       sgrb2 NIRCam merged-reproject products at 45.72
#:   tightest ORPHAN     w51's 2025-06-06 merged-reproject, 426.7 days
#:                       -> margin 426.7/365 = 1.17x
#:   the other seven     2023 products in 2026 fields, 1115.6-1152.7 days
#:
#: So the two populations are separated by 62.4 days against 426.7 -- a factor
#: of 6.8, and the guard sits nearer the orphan edge than the live edge.
#: Lowering it does NOT start taking live data 28 files at a time: the first
#: casualties are the three sickle MIRI products at ~62 days, alone.  Raising it
#: drops w51's orphan first.
#:
#: The reference is PER INSTRUMENT, and the margin depends on that.  Against a
#: field-wide newest the same measurement read 47 held back at 3.26x -- because
#: 22 of those 47 were MIRI products clocked against a NIRCam newest, nine of
#: them the only primary mosaic in their band directory.  MIRI and NIRCam are
#: reduced on independent campaigns, so that arrangement let ordinary NIRCam
#: activity walk sole copies toward quarantine with no code or configuration
#: change -- the margin eroded by the calendar.  See `field_generations`.
#:
#: Two earlier versions of this comment got the live-side number wrong, in the
#: direction that makes the guard look safer than it is, so both are recorded
#: here rather than quietly replaced:
#:
#:   * wd1's merged F200W mosaic, "18 days behind its own per-module siblings",
#:     was cited as the worst live case.  Doubly wrong: wd1 has NO
#:     family-retired primary mosaics at the default settings, so nothing there
#:     depends on this constant at all.  The (band, pointing, product) key is
#:     what protects wd1; this constant is what protects the other 28.
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
    """True for a file some pass has already taken out of service.

    Every quarantine convention THIS script has used appends AFTER the
    extension (``.fits.bad``, ``.fits_badastrometry_stale``), so the test is
    simply that the name no longer ends in ``.fits``.  Written out rather than
    left to the globs, so that a future suffix inserted before the extension
    instead of after it fails here loudly instead of silently re-entering the
    candidate set.

    Other passes tag a product without taking it out of ``*.fits`` at all: the
    m2 astrometry checkpoint renames a mosaic it has invalidated to
    ``..._im0_badastrom.fits`` and writes the reason to a ``.why.json`` beside
    that name.  Those are already quarantined -- by a marker this script does
    not own -- and renaming one again would double-tag it and orphan the first
    tag's explanation, so they are recognised here rather than re-selected.
    ``FOREIGN_QUARANTINE_MARKERS`` is the list.
    """
    if any(basename.endswith(m) for m in FOREIGN_QUARANTINE_MARKERS):
        return True
    return not basename.endswith('.fits')


def pointing_of(basename):
    """``jw<proposal>-o<observation>`` for a level-3 product name, else None."""
    m = PRIMARY_MOSAIC_RE.match(basename) or POINTING_RE.match(basename)
    return m.group('pointing') if m else None


def proposal_and_obs(pointing):
    """``('6778', '001')`` from ``'jw06778-o001'``, else ``(None, None)``.

    The proposal is un-padded because ``ALIGNMENT_CONFIG`` keys on the bare
    program number (``'6778'``, ``'1979'``, ``'10678'``) while the product name
    carries it zero-padded to five digits.  The observation keeps its padding,
    which is the form the config's ``fields`` tuples use (``('001',)``).
    """
    if not pointing:
        return None, None
    prop, _, obs = pointing[2:].partition('-o')
    if not prop.isdigit() or not obs.isdigit():
        return None, None
    return str(int(prop)), obs


def _nonzero(card):
    """True for a header card that is a number other than zero.

    A missing card and a card of 0.0 both mean "this frame carries no shift".
    A card that is not a number at all (a string written by hand) is not a
    correction either, and must not raise here: this function decides whether a
    science product is renamed, so an odd header has to fall through to "no",
    which keeps the file.
    """
    if card is None:
        return False
    try:
        return float(card) != 0.0
    except (TypeError, ValueError):
        return False


def _pointing_frames(banddir, pointing):
    """Every working-copy frame of ``pointing`` in ``banddir``, NEWEST FIRST.

    All of ``FRAME_SUFFIXES`` are globbed and the union ordered by mtime, rather
    than searching them in order and stopping at the first that matches.  Four
    fields carry TWO reduction lineages in one directory (cloudc, cloudef,
    sickle, brick -- SW on destreak, LW on align), and on those the retired
    lineage's frames are still on disk carrying no ``RAOFFSET`` at all:

        sgrb2/F182M   jw05365001001_11101_00001_nrca1
            _align    2024-09-22   RAOFFSET absent
            _destreak 2026-08-24   RAOFFSET=-0.01154  DEOFFSET=-0.12106
        cloudc/F405N  jw02221002001_02201_00001_nrcalong
            _align    2024-01-11   RAOFFSET absent
            _destreak 2026-08-25   RAOFFSET=0.47905   DEOFFSET=7.87626

    First-suffix-wins reads the 2024 copy and reports the pointing as
    uncorrected.  That errs toward KEEPING the file, so nothing was ever
    mis-renamed by it, but the reason an operator reads is then false about the
    field -- and the newest frame's mtime is the clock the plain-sibling check
    below is measured against, so reading the wrong lineage would misdate that
    comparison as well.
    """
    prop, obs = proposal_and_obs(pointing)
    if prop is None:
        return []
    from jwst_gc_pipeline.mast_names import jw_prefix
    # jw_prefix, not string surgery on the pointing: it pads to five digits the
    # way MAST does, so a 5-digit program (10678) and one below 1000 both come
    # out right.  proposal_and_obs has already parsed `prop` two lines above,
    # and the repo's guard test refuses a bare f'jw{...}' for this reason.
    stem = f'{jw_prefix(prop)}{obs}'
    paths = []
    for suffix in FRAME_SUFFIXES:
        paths.extend(glob.glob(f'{banddir}/{stem}*{suffix}'))
    return sorted(set(paths), key=lambda p: (-(mt(p) or 0), p))


def frames_carry_correction(banddir, pointing, sample=FRAME_SAMPLE):
    """Do this pointing's working frames carry a baked, NON-ZERO shift?

    ``(True/False, detail, newest_mtime)``, or ``(None, detail, ...)`` when no
    frame could be read -- which is a different answer from "no correction" and
    is treated as such by the caller.  ``newest_mtime`` is the mtime of the
    newest frame of this pointing, whatever it answered: it is the clock
    ``reproject_supersession`` measures the plain sibling against, since a
    mosaic drizzled before the frames were touched cannot have inherited what
    they now carry.

    Reads the SCI header of up to ``sample`` frames of ``pointing`` in
    ``banddir``.  ``fix_alignment`` writes ``RAOFFSET``/``DEOFFSET`` there as it
    applies the shift, so their presence and value is the frames' own record of
    whether this pointing was corrected:

      * absent            -- ``fix_alignment`` never ran on this frame;
      * present and 0.0   -- it ran and applied nothing (no row in the offsets
                            table, or a row of zero), so the frame is still on
                            the raw ``assign_wcs`` frame;
      * present, non-zero -- the frame was moved, and any mosaic drizzled from
                            it since carries that tie.

    The first two are both "not corrected" for this purpose: what matters is
    whether the PLAIN mosaic inherited a tie, not whether a step ran.

    Scoped to ``pointing``, not to the directory: one band directory routinely
    holds several pointings that are aligned independently (ngc6334 keeps
    proposals 6778 and 7213 in one ``F200W/pipeline``; m4 keeps observations
    o002 and o003 in one ``F150W2/pipeline``), and reading a neighbour's frames
    would answer for the wrong data.

    ``DEOFFSET`` counts as well as ``RAOFFSET``.  The precondition is stated in
    #724 as "a non-zero ``RAOFFSET``", and taken literally that misses a
    correction that is pure declination -- a real shape for a field whose tie is
    dominated by one axis -- which would read a corrected pointing as
    uncorrected and keep a superseded mosaic.  Erring that way is the safe
    direction, but it is still wrong, so both cards are read.
    """
    try:
        from astropy.io import fits
        from astropy.io.fits.verify import VerifyError
    except ImportError:
        return None, 'astropy unavailable', None
    prop, obs = proposal_and_obs(pointing)
    if prop is None:
        return None, f'unparseable pointing {pointing!r}', None
    paths = _pointing_frames(banddir, pointing)
    if not paths:
        return None, f'no {"/".join(FRAME_SUFFIXES)} frame for {pointing}', None
    newest = mt(paths[0])
    read = 0
    for path in paths[:sample]:
        # A truncated or malformed FITS raises `VerifyError` or `ValueError`
        # rather than `OSError`; letting either escape would abort the whole
        # field's pass instead of keeping this one file.
        try:
            header = fits.getheader(path, ext=1)
        except (OSError, IndexError, KeyError, ValueError, VerifyError):
            continue
        read += 1
        ra, dec = header.get('RAOFFSET'), header.get('DEOFFSET')
        if _nonzero(ra) or _nonzero(dec):
            return True, (f'{os.path.basename(path)} RAOFFSET={ra} '
                          f'DEOFFSET={dec}'), newest
    if not read:
        return None, f'could not read any of {len(paths)} frame header(s)', newest
    return False, (f'{read} of {len(paths)} {pointing} frame(s) read, '
                   f'all RAOFFSET/DEOFFSET absent or 0.0'), newest


def plain_sibling(path):
    """The plain ``*_i2d.fits`` a reproject is claimed to be superseded BY.

    ``align_to_catalogs`` writes ``<product>-reproject<tail>`` beside the
    level-3 output ``<product>_i2d.fits``, so the sibling's name is the part of
    the reproject's name before ``-reproject`` plus ``_i2d.fits``.  That covers
    every form on the archive:

        clear-f150w-merged-reproject_i2d.fits            -> clear-f150w-merged_i2d.fits
        clear-f405n-merged-reproject_vvvcat.fits         -> clear-f405n-merged_i2d.fits
        clear-f405n-merged-reproject_i2d_im0_badastrom.fits
                                                         -> clear-f405n-merged_i2d.fits
        clear-f182m-merged-reproject-vvv_i2d.fits        -> clear-f182m-merged_i2d.fits

    ``None`` when the name carries ``reproject`` without the ``-reproject``
    token this derivation needs, which the caller treats as "cannot tell" and
    therefore as keep.
    """
    base = os.path.basename(path)
    token = base.find('-reproject')
    if token < 0:
        return None
    return os.path.join(os.path.dirname(path), base[:token] + '_i2d.fits')


def reproject_supersession(path, banddir):
    """Is this reproject superseded? ``(bool, why, frames_mtime)``.

    The precondition #724 asks for, and the reason it is not a date comparison:
    see ``REPROJECT_RE``.  ALL FOUR halves must hold, and the last two are
    about the file that is supposed to take over.

    1. The field is REGISTERED in ``ALIGNMENT_CONFIG``.  An unregistered
       (proposal, observation) is left at the raw ``assign_wcs`` frame by
       construction -- that is the failure mode ``alignment_config`` exists to
       make visible -- so nothing drizzled from its frames is tied to anything,
       and the reproject is the only tied copy of that band.
    2. Its FRAMES carry a non-zero baked shift.  Registration alone is not
       enough: ngc6334 was registered on 2026-09-01 and its frames still read
       ``RAOFFSET=0.0`` because no reduction has run since, so its plain
       mosaics are still untied and its reprojects are still the only tie.
    3. THE PLAIN SIBLING EXISTS.  1 and 2 are both statements about the FRAMES;
       neither of them looks at the mosaic that is supposed to inherit the tie,
       and a band can hold the reproject alone.  That state is on disk today --
       ``ngc6334/F115W/pipeline`` holds
       ``jw07213-o001_..._merged-reproject_i2d_im0_badastrom.fits`` and no plain
       sibling -- and renaming there leaves the band with no mosaic at all.
    4. THE PLAIN SIBLING POSTDATES THE FRAMES.  "The frames carry a shift" and
       "the plain mosaic inherited it" are different claims: ``fix_alignment``
       bakes the shift into the frames, and only the level-3 re-drizzle AFTER
       that puts it into the mosaic.  Between those two steps -- or if the
       re-drizzle fails, or has not been submitted -- the plain mosaic is a
       pre-correction product and the reproject is still the band's only tie.
       ngc6334 (13 reprojects) and wd1 (7) are one re-reduction away from
       exactly that shape, which is when this pass is next expected to run.

       The comparison is the plain mosaic's `generation` (its ``DATE``, else
       mtime) against the mtime of the pointing's NEWEST frame -- newest, not
       the frame that answered, because any frame written after the mosaic
       means the mosaic predates the current state of the pointing.  A tie is
       allowed (``>=``): a drizzle writes its ``_crf`` frames and its mosaic in
       one run, seconds apart in either order.

    Answering "keep" is the conservative outcome -- a kept file is renameable
    later, a renamed tie is a field's astrometry removed -- so every uncertain
    case (unreadable frames, no astropy, an unparseable name, an underivable or
    undatable sibling) answers keep.

    The third value is the newest frame mtime, so the caller can put the date
    the file was actually judged against into the log and the ``.why.txt``.
    """
    base = os.path.basename(path)
    pointing = pointing_of(base)
    prop, obs = proposal_and_obs(pointing)
    if prop is None:
        return False, f'cannot read a pointing out of {base!r}', None
    try:
        from jwst_gc_pipeline.reduction.alignment_config import resolve
    except ImportError as ex:
        return False, f'alignment_config unavailable ({ex})', None
    cfg = resolve(prop, obs)
    if cfg is None:
        return False, (f'{pointing} is NOT in ALIGNMENT_CONFIG -- untied '
                       f'frames, so this reproject is the only tie'), None
    corrected, detail, frames_mtime = frames_carry_correction(banddir, pointing)
    if corrected is None:
        return False, f'{pointing} frame state unknown ({detail})', frames_mtime
    if not corrected:
        return (False, (f'{pointing} is registered ({cfg.reference_frame}/'
                        f'{cfg.source}) but its frames carry NO baked shift '
                        f'({detail}) -- this reproject is the only tie'),
                frames_mtime)
    tied = (f'{pointing} is registered ({cfg.reference_frame}/{cfg.source}) '
            f'and its frames carry a baked shift ({detail})')
    plain = plain_sibling(path)
    if plain is None:
        return (False, (f'{tied}, but no plain sibling name can be derived '
                        f'from {base!r} -- cannot tell what would take over'),
                frames_mtime)
    sib = os.path.basename(plain)
    if not os.path.exists(plain):
        return False, (f'{tied}, but {sib} DOES NOT EXIST -- this reproject is '
                       f'the band\'s only mosaic'), frames_mtime
    plain_gen, plain_clock = generation(plain)
    if plain_gen is None or frames_mtime is None:
        return (False, f'{tied}, but {sib} cannot be dated against them',
                frames_mtime)
    if plain_gen < frames_mtime:
        return False, (f'{tied}, but {sib} ({plain_clock} {fmt(plain_gen)}) '
                       f'PREDATES them ({fmt(frames_mtime)}) -- it was drizzled '
                       f'before the tie was baked, so it has not inherited it '
                       f'and this reproject is still the only tie'), frames_mtime
    return True, (f'{tied}, and {sib} ({plain_clock} {fmt(plain_gen)}) '
                  f'postdates them ({fmt(frames_mtime)}) -- the plain mosaic '
                  f'is the current tie'), frames_mtime


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


def instrument_of(basename):
    """``'nircam'`` / ``'miri'`` from a level-3 product name, else ``'?'``.

    The instrument sits between the target token and the band in the canonical
    form ``jw<prop>-o<obs>_t<NNN>_<instrument>_<band>_i2d.fits``, which is the
    only form ``PRIMARY_MOSAIC_RE`` matches, so this reads the token out of
    ``PRIMARY_MOSAIC_RE`` ITSELF rather than re-describing it in a second regex.
    That coupling matters: if the two drifted, every file whose name the second
    pattern did not recognise would land in one shared ``'?'`` bucket and be
    clocked against every other unrecognised file -- silently restoring the
    cross-instrument reference the per-instrument one exists to remove.
    ``test_the_instrument_token_comes_from_the_primary_mosaic_pattern`` pins it.
    """
    m = PRIMARY_MOSAIC_RE.match(basename)
    return m.group('instrument') if m else '?'


def field_generations(field, campaign_days=21):
    """Everything the two rules need to know about one field's generations.

    ``(banddirs, band_of_dir, dref1, fam_newest, newest_by_instrument,
    campaign1, campaign2_by_instrument)``, or None when the field has no
    ``*/pipeline`` directories.

    Extracted so ``--audit-age-guard`` measures the age guard's margin through
    the SAME references the rule uses.  An audit that recomputes them
    independently can report a margin the rule does not have, which is how the
    two retracted numbers in ``MIN_ORPHAN_AGE_DAYS`` were arrived at.

    Rule 2's reference is PER INSTRUMENT, and that is load bearing.  Clocking
    everything against the field's newest primary mosaic of ANY instrument makes
    a field's NIRCam activity age out its MIRI products: NIRCam and MIRI are
    reduced on independent campaigns, so a NIRCam-only re-drizzle moves a clock
    that no MIRI product has any part in.  Measured on the field-wide reference,
    22 of the 47 live mosaics the age guard was holding back were MIRI judged
    against a NIRCam newest, nine of them the ONLY primary mosaic in their band
    directory -- so continued NIRCam reduction alone would eventually have
    quarantined sole copies, with no code or configuration change.  Per
    instrument, a MIRI product is judged against MIRI's campaign.
    """
    banddirs = glob.glob(f'{BASE}/{field}/*/pipeline')
    if not banddirs:
        return None
    dref1, band_of_dir = {}, {}
    fam_newest = {}          # (band, pointing, product) -> newest generation
    newest = {}              # instrument -> that instrument's current generation
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
            base = os.path.basename(p)
            key = (band,) + product_key(base)
            fam_newest[key] = max(gen, fam_newest.get(key, gen))
            inst = instrument_of(base)
            newest[inst] = max(gen, newest.get(inst, gen))
    campaign1 = max((v for v in dref1.values() if v), default=None)
    campaign1 = (campaign1 - campaign_days * DAY) if campaign1 else None
    campaign2 = {i: g - campaign_days * DAY for i, g in newest.items()}
    return (banddirs, band_of_dir, dref1, fam_newest, newest,
            campaign1, campaign2)


def age_guard_rows(field, campaign_days=21):
    """Every primary mosaic of ``field`` whose product family is RETIRED.

    ``[(age_days, path), ...]`` -- age behind the field's newest primary mosaic
    OF THE SAME INSTRUMENT (see ``field_generations``).
    These are exactly the files that clause (a) of rule 2 selects, so
    ``MIN_ORPHAN_AGE_DAYS`` alone decides each one: below it the file is LIVE and
    kept, at or above it the file is quarantined.  The guard's margin is
    therefore the gap between the LARGEST age below the constant (the live
    product nearest to being taken) and the SMALLEST age at or above it.
    """
    gens = field_generations(field, campaign_days)
    if gens is None:
        return []
    banddirs, band_of_dir, _dref1, fam_newest, newest, _c1, campaign2 = gens
    if not newest:
        return []
    rows = []
    for d in banddirs:
        band = band_of_dir[d]
        for p in primary_mosaics(d):
            gen = generation(p)[0]
            if gen is None:
                continue
            base = os.path.basename(p)
            inst = instrument_of(base)
            ref = newest.get(inst)
            floor = campaign2.get(inst)
            if ref is None:
                continue
            key = (band,) + product_key(base)
            if floor is not None and fam_newest.get(key, 0) >= floor:
                continue                      # family still live -- not rule 2's
            rows.append(((ref - gen) / DAY, p))
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
            # `<=`, not `<`: rule 2's test is `gen < ref - MIN * DAY`, i.e. an
            # age of exactly MIN days is NOT caught.  The audit exists to agree
            # with the rule, so the boundary has to match it.
            (held if age <= MIN_ORPHAN_AGE_DAYS else caught).append(
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


class UnmatchedOnlyName(Exception):
    """``--only`` named a file this run did not select.

    Raised rather than warned.  A name that matches nothing silently renames
    nothing and exits 0, which reads exactly like "the rename ran" -- and the
    operator who typed the name is executing on data, so the one outcome that
    must not be silent is "your instruction reached no file".
    """

    def __init__(self, field, unmatched, available):
        self.field = field
        self.unmatched = sorted(unmatched)
        self.available = sorted(available)
        super().__init__(
            f"[{field}] --only named {len(self.unmatched)} file(s) that this "
            f"run did not select: {', '.join(self.unmatched)}.\n"
            f"  selected here: "
            f"{', '.join(self.available) if self.available else '(nothing)'}\n"
            f"  Run without --only to see the full selection; a name that "
            f"matches nothing would rename nothing and exit 0.")


def _apply_only(plan, only, field):
    """Restrict ``plan`` to the basenames in ``only``.

    Does NOT refuse on a name it does not hold: with several ``--field`` a name
    belongs to exactly one of them, so per-field refusal would fire on every
    other field.  The refusal lives in ``main``, which checks the names against
    the union of every field's selection BEFORE anything is renamed.
    """
    if not only:
        return plan
    want = {os.path.basename(n) for n in only}
    available = {os.path.basename(p[0]) for p in plan}
    kept = [p for p in plan if os.path.basename(p[0]) in want]
    withheld = sorted(available - want)
    if withheld:
        print(f"[{field}] --only: {len(withheld)} selected file(s) LEFT IN "
              f"PLACE by the filter:")
        for name in withheld:
            print(f"    withheld: {name}")
    return kept


def plan_stale_for_field(field, campaign_days=21):
    """What the rules select for ``field``, with nothing printed and nothing
    renamed.  Used to validate ``--only`` before the executing pass runs."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return rename_stale_for_field(field, execute=False,
                                      campaign_days=campaign_days)


def rename_stale_for_field(field, execute=False, campaign_days=21, only=None):
    pipe = f'{BASE}/{field}'
    gens = field_generations(field, campaign_days)
    if gens is None:
        print(f"[{field}] no */pipeline dirs under {pipe}; skipping")
        return []
    (banddirs, band_of_dir, dref1, fam_newest, newest_by_instrument,
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
            base = os.path.basename(p)
            key = (band,) + product_key(base)
            inst = instrument_of(base)
            ref = newest_by_instrument.get(inst)
            floor = campaign2.get(inst)
            if gen is None or ref is None:
                continue
            family_live = floor is None or fam_newest.get(key, 0) >= floor
            old_enough = gen < ref - MIN_ORPHAN_AGE_DAYS * DAY
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
    #
    # That shared file is also why the reproject precondition below is applied
    # to BOTH rules rather than to rule 1 alone: with the gate on rule 1 only,
    # rule 1 would keep w51's F150W reproject (its frames are untied, so it is
    # that band's only tie) while rule 2 quarantined the same file on age.  A
    # union of two rules needs the family's policy to sit OUTSIDE both of them.
    plan, kept, seen = [], 0, set()
    for cands, which in ((named, 1), (generational, 2)):
        for f, why in sorted(cands.items()):
            base = os.path.basename(f)
            if f in seen:
                continue
            if is_quarantined(base):
                # A marker ANOTHER pass wrote is worth naming: it is the reason
                # 32 files across six fields are withheld from this pass, and
                # skipping them silently is what made that invisible.  A name
                # this script's own conventions produced (`.bad`, `_stale`) is
                # not printed -- it is the record of a previous run of this
                # tool and there is nothing for an operator to decide about it.
                if any(base.endswith(mk) for mk in FOREIGN_QUARANTINE_MARKERS):
                    print(f"  SKIP [{field}] {base} [rule{which}: already "
                          f"tagged by another pass; see its .why.json]")
                    seen.add(f)
                continue
            band = band_of_dir.get(os.path.dirname(f)) or band_of(base)
            # The reproject family is decided by STATE, not by a clock, and by
            # the SAME state under both rules -- see `REPROJECT_RE`.  Placed
            # ahead of either rule's reference so that a reproject whose band
            # has no `data_i2d` (rule 1 would SKIP it) and a reproject old
            # enough to be an orphan (rule 2 would take it regardless) get one
            # answer rather than two.  w51's `clear-f150w-merged-reproject` is
            # the file that made this necessary: it is the one member of both
            # rules' sets, and its own pointing's frames read RAOFFSET=0.0.
            if REPROJECT_RE.search(base):
                superseded, gate_why, frames_mtime = reproject_supersession(
                    f, os.path.dirname(f))
                # `seen` on BOTH branches: a file in both rules' sets gets one
                # answer and one line, not two identical ones (w51 printed its
                # KEEP twice and counted it twice in "current kept").
                seen.add(f)
                if not superseded:
                    print(f"  KEEP [{field}] {base} [rule{which}: {gate_why}]")
                    kept += 1
                    continue
                # The reference slot is the FRAMES' date, which is what the gate
                # judged this file against.  It used to hold the file's own
                # mtime, so the log line and the `.why.txt` beside every rename
                # read "its generation is X; the frames it is judged against is
                # X" -- the same date twice, and never the frames'.
                plan.append((f, mt(f), frames_mtime, f'{why}; {gate_why}',
                             'mtime', 'the frames it is judged against'))
                continue
            if which == 1:
                ref, campaign, clock, refname = (dref1.get(band), campaign1,
                                                 'mtime', "the band's data_i2d")
                fm = mt(f)
                missing = 'no-data_i2d'
            else:
                _inst = instrument_of(base)
                ref, campaign = (newest_by_instrument.get(_inst),
                                 campaign2.get(_inst))
                clock = generation(f)[1]
                refname = f"the field's newest {_inst} primary mosaic"
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

    # Rule 2's floor is per instrument (see `field_generations`), so name each.
    floors2 = ", ".join(f"{i} {fmt(v)}" for i, v in sorted(campaign2.items())) \
        or "NONE"
    print(f"[{field}] {'EXECUTE' if execute else 'DRY RUN'}: "
          f"{len(plan)} superseded, {kept} current kept "
          f"(campaign floors: rule 1 {fmt(campaign1)}; rule 2 {floors2})")
    # Applied AFTER the count above, so the log always records how many files
    # the rules selected as well as how many the operator acted on.
    plan = _apply_only(plan, only, field)
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
            f"This product's generation ({clock}) is {fmt(fm)}; {refname} is "
            f"{fmt(ref)}.\n"
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
    ap.add_argument('--only', action='append', default=[], metavar='BASENAME',
                    help='act on these selected files only (repeatable). '
                         'Refuses if a name matches nothing this run selected. '
                         'Use it to quarantine the subset whose astrometry has '
                         'actually been measured (#339)')
    args = ap.parse_args()
    if args.audit_age_guard:
        audit_age_guard(args.field, campaign_days=args.campaign_days)
        return
    if args.only:
        # Check every name against the union of what the rules select across
        # ALL the named fields, and refuse BEFORE anything is renamed.  Doing
        # it after would leave the matched names already quarantined while
        # reporting a failure, which is the worst of both.
        available = set()
        for field in args.field:
            available |= {os.path.basename(p[0])
                          for p in plan_stale_for_field(
                              field, campaign_days=args.campaign_days)}
        unmatched = {os.path.basename(n) for n in args.only} - available
        if unmatched:
            raise UnmatchedOnlyName(', '.join(args.field), unmatched, available)
    for field in args.field:
        rename_stale_for_field(field, execute=args.execute,
                               campaign_days=args.campaign_days,
                               only=args.only or None)


if __name__ == '__main__':
    main()
