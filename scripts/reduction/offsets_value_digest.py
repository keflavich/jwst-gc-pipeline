#!/usr/bin/env python
"""Digest the VALUES of an offsets table, ignoring provenance and formatting.

`run_field_retie_loop.sh` decides whether an iteration re-tied anything by
comparing the offsets table before and after the m2 checkpoint.  It used an
md5sum of the whole file, which answers a different question: **did any byte
change**.  A checkpoint that re-stamps `prov_date` on a row it did not move
changes bytes, so the loop reads "a re-tie was made", re-reduces, and
re-measures an identical residual -- the observable recorded on issue #272,
where three consecutive rounds each reported 15 corrections and left the
`dra`/`ddec` cells untouched.

This digests only the shift columns, keyed by row identity so a re-ordered or
re-serialised table digests the same:

    (Visit, Exposure, Filter, Module, Vgroup) -> dra, ddec, dra (arcsec), ddec (arcsec)

Values are rounded to `NDIGITS` decimal places of a degree/arcsecond before
hashing.  Without that, a re-write that changes the last float digit of an
unchanged quantity reads as movement, which is the same false positive the
md5sum has -- sgrc spent four passes of ~7 h each on it.

One offsets table can hold more than one observation, and then "did the table
change" is the wrong question a second way.  The 10678 treasury registers
`fields=None` (alignment_config.py), so all 139 observations write into ONE
`Offsets_JWST_Brick10678_consensus.csv`; two tiles re-tieing at the same time
would each read the other's correction as its own re-tie and re-reduce for
nothing.  `--observation NNN` keeps only the rows belonging to that observation
-- the `Visit` cell carries it as characters 8-10 of `jwPPPPPOOOVVV`.  The
fixed-point check beside this one is already scoped that way (`--obs-token
o$FIELD`).

Scoping only ever HIDES a change that is attributably some other observation's.
A row whose `Visit` cannot be parsed is KEPT, so an unattributable change still
reads as movement; a request to scope a table with no `Visit` column at all
fails (exit 2) rather than quietly digesting every row as if it were this
observation's.

Usage:
    offsets_value_digest.py PATH        -> prints a hex digest, or `none` when
                                           the file is absent
    offsets_value_digest.py PATH --observation 088
                                        -> the same, over that observation's
                                           rows only
Exit status is 0 for both, so the caller's `set -e` sees a missing table the
same way `md5sum ... || echo none` did.  A table that exists and cannot be
parsed exits 2 with a message on stderr: the caller must not treat "I could not
read it" as "it did not change".
"""
import argparse
import hashlib
import os
import re
import sys

#: Decimal places kept per column.  The tables carry degrees in `dra`/`ddec`
#: and arcseconds in the `(arcsec)` pair; 9 places is ~4 microarcsec on the
#: former and a nanoarcsecond on the latter, far below the sub-mas quantities
#: any correction expresses, while absorbing float re-serialisation.
NDIGITS = 9

#: Columns whose values ARE the shift.  Everything else in the table -- the
#: identity columns, `prov_*`, comments -- is deliberately excluded.
VALUE_COLUMNS = ("dra", "ddec", "dra (arcsec)", "ddec (arcsec)")

#: Columns that identify the row the shift belongs to.  Vgroup is included
#: because cloudc's table has two vgroups per exposure and the four-column key
#: is ambiguous for half its rows (issue #272).
KEY_COLUMNS = ("Visit", "Exposure", "Filter", "Module", "Vgroup")

#: A visit id is `jw` + 5-digit proposal + 3-digit observation + 3-digit visit,
#: e.g. `jw07213001001`.  The observation is the only part `--observation`
#: needs, so the proposal never has to be passed in.
VISIT_RE = re.compile(r"^jw\d{5}(\d{3})")


def _observation_of(visit):
    """The 3-digit observation a `Visit` cell names, or None when unparseable."""
    match = VISIT_RE.match(str(visit).strip())
    return match.group(1) if match else None


def normalise_observation(observation):
    """`12` and `012` name the same observation; the table writes `012`."""
    if observation is None:
        return None
    text = str(observation).strip()
    if not text.isdigit() or len(text) > 3:
        raise ValueError(
            f"observation {observation!r} is not a 1-3 digit observation number; "
            f"a visit id carries it as characters 8-10 of jwPPPPPOOOVVV")
    return text.zfill(3)


def _cell(value):
    """One cell as a stable string: a number rounded, anything else verbatim."""
    text = str(value).strip()
    if text in ("", "--", "nan", "NaN", "None", "masked"):
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number != number:            # NaN: distinct from absent, stable to hash
        return "nan"
    return f"{round(number, NDIGITS):.{NDIGITS}f}"


def digest(path, observation=None):
    """Hex digest over the table's shift values, or ``"none"`` when absent.

    With ``observation`` set, only the rows whose ``Visit`` names that
    observation are digested, so a sibling observation sharing the table
    cannot present its correction as this one's re-tie.  Left unset, every
    row is digested, which is what every single-observation field does today.
    """
    # Validated before the existence check: a malformed observation is an
    # operator error, and reporting it as "none" (an absent table) would let the
    # loop run on with the scope silently not applied.
    observation = normalise_observation(observation)
    if not path or not os.path.exists(path):
        return "none"
    from astropy.table import Table
    table = Table.read(path)
    keys = [c for c in KEY_COLUMNS if c in table.colnames]
    values = [c for c in VALUE_COLUMNS if c in table.colnames]
    if not values:
        raise ValueError(
            f"{path} carries none of {VALUE_COLUMNS}; digesting it would return "
            f"the same hash for every table and report every iteration as "
            f"'no re-tie'")
    if observation is not None and "Visit" not in table.colnames:
        raise ValueError(
            f"{path} has no 'Visit' column, so its rows cannot be attributed to "
            f"observation {observation}; digesting all of them would report "
            f"another observation's correction as this one's re-tie")
    rows = []
    for row in table:
        if observation is not None:
            row_obs = _observation_of(row["Visit"])
            # An unparseable Visit is KEPT: scoping may hide a change that
            # BELONGS to another observation, never one that cannot be placed.
            if row_obs is not None and row_obs != observation:
                continue
        identity = "|".join(_cell(row[c]) for c in keys)
        rows.append(identity + "=" + ",".join(_cell(row[c]) for c in values))
    # Sorted: a table re-serialised in a different row order holds the same
    # corrections, and the loop must not read that as a re-tie.
    payload = "\n".join(sorted(rows)) + "\n" + ",".join(values)
    if observation is not None:
        # Scope in the payload, so a digest taken for one observation is not
        # comparable with one taken for another (or with the whole file).
        payload += "\nobservation=" + observation
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--observation", default=None,
                    help="digest only the rows whose Visit names this 3-digit "
                         "observation (the shared-table case: 10678's 139 "
                         "tiles).  Omitted, every row is digested.")
    args = ap.parse_args(argv)
    try:
        print(digest(args.path, observation=args.observation))
    except (OSError, ValueError, KeyError) as exc:
        print(f"offsets_value_digest: cannot digest {args.path}: {exc}",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
