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

RULE 2 -- BY GENERATION.  A superseded mosaic need not say so in its name.
cloudc carries

    F405N/pipeline/jw02221-o002_t001_nircam_f405n-f444w_i2d.fits   2023-07-11

which is a 2023 artifact of a reduction that paired F405N against F444W as its
continuum -- F444W is not a cloudc filter and no F444W frame exists on disk to
have built it from -- and it ties to VIRAC2 **4.1 arcsec** out of place, where
every current-generation cloudc mosaic ties at 32-33 mas.  Nothing about its
NAME distinguishes it from a live product: it carries the canonical level-3
form ``jw<prop>-o<obs>_t<NNN>_<instrument>_<band>_i2d.fits``, so rule 1 does not
see it and a ``*_i2d.fits`` glob does (issue #339).

Rule 2 therefore compares a mosaic's GENERATION against its own band AND
pointing's current
one.  Scope is deliberately narrow -- only the primary drizzle products
(``PRIMARY_MOSAIC_RE``), never the per-exposure ``_outlier_i2d`` intermediates
(171-188 of them per cloudc band) and never the photometry byproducts
(``_data_i2d``, ``*mergedcat*``).

Each rule keeps its OWN reference and campaign floor, so adding rule 2 leaves
rule 1's selection bit-identical: measured archive-wide over 14 fields, 55 files
before and 55 after, plus rule 2's 4.

SUPERSEDED (rule 2) = BOTH of:
  * older by > 1 day than the newest primary mosaic of its own BAND AND
    POINTING, where the pointing is the ``jw<proposal>-o<observation>`` prefix
    (so same-run products made hours apart are never flagged), AND
  * older than the field's reduction-campaign floor = (newest primary mosaic
    across all bands and pointings) - 21 days (so a whole in-progress campaign
    is never flagged just because one band finished later).

Both conditions are required and neither is redundant.  cloudc's F2550W band
holds one MIRI mosaic dated 2026-06-11 against a field campaign of 2026-08-09:
59 days below the campaign floor, and correctly kept, because it is its own
band's newest.  Conversely the F405N orphan is 3 years below both.

Scoping the comparison to the POINTING is what keeps the rule from firing on a
band directory that legitimately holds several: without it, an archive-wide dry
run flagged six live ngc6334 products (F200W and F470N, where proposals 6778 and
7213 share a directory) and three live sickle MIRI ones (F770W/F1130W/F1500W,
where observations o001/o002/o003 sit together).  All nine are current products
of a pointing that was simply reduced earlier than its neighbour.

A whole pointing that is genuinely stale is therefore NOT caught -- each of its
mosaics is its own pointing's newest.  That is the same conservative gap as the
single-mosaic case below, and it is what ``check_generation_span`` and the
release freshness gate look at instead.

Generation is read from the FITS ``DATE`` header -- the time the pipeline wrote
the product -- falling back to mtime when the header cannot be read.  mtime
alone is not enough: a copy or a restore resets it, and DATE is the quantity
``check_generation_span`` already uses to decide a staged image set's
generation.  Which clock was used is printed per file.

Known limitation, stated so it is not mistaken for coverage: rule 2 needs a
sibling to compare against.  A band directory whose ONLY primary mosaic is an
orphan has nothing newer in it, so the orphan is its own reference and is kept.
That is the conservative direction (no false positives), but it is a gap.

Band token is parsed from either NIRCam ('clear-f182m-', 'f405n-f444w-'
pupil forms) or MIRI ('_f2550w') filename conventions.

Usage:
  python rename_stale_mosaics.py --field brick [--field cloudc ...] [--execute]

Dry-run by default; --execute renames and appends to
``<fieldpath>/_stale_rename_<date>.log``.
"""
import argparse
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
#: tokens after the band.  Matching those would quarantine hundreds of live
#: intermediate products per band.
PRIMARY_MOSAIC_RE = re.compile(
    r'^(?P<pointing>jw\d+-o\d+)_t\d+_[a-z]+_[a-z0-9-]+_i2d\.fits$')

#: Renaming to this suffix takes the file out of every ``*.fits`` glob, which is
#: the protection that matters: no reader selects it by accident any more.  It
#: does NOT make the bytes unopenable -- ``fits.open`` does not care about the
#: extension -- so anyone who reaches for one of these has to name it
#: explicitly, which is the point.
SUFFIX = '.bad'

#: Already-quarantined files, under this or any earlier convention, are skipped
#: rather than renamed again.  ``_badastrometry_stale`` is what this script
#: wrote before 2026-08 and what the 192 brick files from the 2026-07-03 pass
#: still carry; ``release_freshness`` looks for both.
QUARANTINE_SUFFIXES = (SUFFIX, '_badastrometry_stale', '_stale')


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
    assert all(not s.endswith('.fits') for s in QUARANTINE_SUFFIXES)
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
    try:
        return time.mktime(time.strptime(str(value)[:19], '%Y-%m-%dT%H:%M:%S'))
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


def pointing_of(basename):
    """``jw<proposal>-o<observation>`` for a primary mosaic, else None.

    One band directory routinely holds mosaics of SEVERAL pointings, and they
    are reduced at different times without any of them being superseded:
    sickle's ``F1130W/pipeline`` carries observations o001, o002 and o003 of
    proposal 3958, and ngc6334's ``F200W/pipeline`` carries proposals 6778 and
    7213 side by side (they share a target directory and a filter list).
    Comparing a mosaic's generation against the directory's newest, rather than
    against its OWN pointing's newest, would report six live ngc6334/7213
    products and three live sickle MIRI ones as superseded -- measured, before
    this scoping was added.
    """
    m = PRIMARY_MOSAIC_RE.match(basename)
    return m.group('pointing') if m else None


def primary_mosaics(banddir):
    """The band's primary drizzle products -- see ``PRIMARY_MOSAIC_RE``."""
    return sorted(p for p in glob.glob(f'{banddir}/*_i2d.fits')
                  if PRIMARY_MOSAIC_RE.match(os.path.basename(p))
                  and not is_quarantined(os.path.basename(p)))


def rename_stale_for_field(field, execute=False, campaign_days=21):
    pipe = f'{BASE}/{field}'
    banddirs = glob.glob(f'{pipe}/*/pipeline')
    if not banddirs:
        print(f"[{field}] no */pipeline dirs under {pipe}; skipping")
        return []

    # Each rule carries its OWN reference generation, and they are deliberately
    # not shared.  Rule 1's is the one this script has always used -- the band's
    # `*-merged_data_i2d.fits` mtime -- so rule 1 selects exactly the files it
    # selected before this rule 2 was added.  Broadening rule 1's reference to
    # the primary mosaics as well was tried and measured: archive-wide it moves
    # rule 1's selection from 55 files to 229, because a band with no
    # `*-merged_data_i2d.fits` currently SKIPS its candidates rather than
    # judging them.  That may well be worth doing, but it is a separate decision
    # about 178 files and does not belong in a fix for one orphan (#339).
    dref1, dref2, band_of_dir = {}, {}, {}
    for d in banddirs:
        band = os.path.basename(d.rsplit('/pipeline', 1)[0]).lower()
        band_of_dir[d] = band
        g = [x for x in glob.glob(f'{d}/*-merged_data_i2d.fits')
             if 'nrca' not in os.path.basename(x)
             and 'nrcb' not in os.path.basename(x)]
        dref1[band] = max((mt(x) for x in g), default=None)
        # Rule 2's reference: the newest primary mosaic of the SAME BAND AND
        # POINTING.  It cannot key on `*-merged_data_i2d.fits` -- that product
        # is absent in twelve field/band combinations (issue #256), and cloudc
        # F405N, which holds this issue's 4.1" orphan, is one of them.  Keying
        # on a product missing exactly where the orphan lives is why the orphan
        # was reported `[no-data_i2d]` and skipped instead of flagged.
        for p in primary_mosaics(d):
            gen = generation(p)[0]
            if gen is None:
                continue
            key = (band, pointing_of(os.path.basename(p)))
            dref2[key] = max(gen, dref2.get(key, gen))
    # The campaign floor stays on rule 1's clock so rule 1 is unchanged; rule 2
    # gets its own from the primary mosaics, for the same reason as above.
    campaign1 = max((v for v in dref1.values() if v), default=None)
    campaign1 = (campaign1 - campaign_days * DAY) if campaign1 else None
    campaign2 = max((v for v in dref2.values() if v), default=None)
    campaign2 = (campaign2 - campaign_days * DAY) if campaign2 else None

    # RULE 1: named retired-alignment products, anywhere in the band directory.
    cands = {}
    for d in banddirs:
        for pat in PATTERNS:
            for p in glob.glob(f'{d}/{pat}'):
                cands.setdefault(p, 'retired-path name')
    # RULE 2: primary mosaics older than their own pointing's current generation.
    for d in banddirs:
        for p in primary_mosaics(d):
            cands.setdefault(p, 'superseded generation')

    plan, kept = [], 0
    for f, why in sorted(cands.items()):
        base = os.path.basename(f)
        if is_quarantined(base):
            continue
        # the directory is authoritative for which band's reference applies;
        # the name is the fallback for a product filed somewhere unexpected
        band = band_of_dir.get(os.path.dirname(f)) or band_of(base)
        if why == 'retired-path name':
            ref, campaign, clock = dref1.get(band), campaign1, 'mtime'
            fm = mt(f)
            missing = 'no-data_i2d'
        else:
            pointing = pointing_of(base)
            ref, campaign = dref2.get((band, pointing)), campaign2
            fm, clock = generation(f)
            missing = 'no-reference-mosaic'
        if band is None or ref is None or fm is None:
            print(f"  SKIP [{field}] {base} "
                  f"[{'no-band' if band is None else missing}]")
            continue
        if fm < ref - DAY and (campaign is None or fm < campaign):
            plan.append((f, fm, ref, why, clock))
        else:
            kept += 1

    print(f"[{field}] {'EXECUTE' if execute else 'DRY RUN'}: "
          f"{len(plan)} superseded, {kept} current kept "
          f"(campaign floors {fmt(campaign1)} / {fmt(campaign2)})")
    log = None
    if execute and plan:
        log = open(f'{pipe}/_stale_rename_{time.strftime("%Y-%m-%d")}.log', 'a')
        log.write(f"# rename_stale_mosaics.py {time.strftime('%Y-%m-%d %H:%M')}\n")
    for f, fm, ref, why, clock in plan:
        line = (f"  {fmt(fm)} [{clock}] (band current {fmt(ref)}) "
                f"{why}: {os.path.basename(f)}")
        print(('RENAME' if execute else 'would rename') + line)
        if execute:
            os.rename(f, f + SUFFIX)
            log.write(f"RENAME {f} -> {f}{SUFFIX}  ({why}; {clock} {fmt(fm)}, "
                      f"band current {fmt(ref)})\n")
            _write_reason_sidecar(f + SUFFIX, why, clock, fm, ref)
    if log:
        log.close()
    return plan


def _write_reason_sidecar(path, why, clock, fm, ref):
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
    args = ap.parse_args()
    for field in args.field:
        rename_stale_for_field(field, execute=args.execute,
                               campaign_days=args.campaign_days)


if __name__ == '__main__':
    main()
