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

Usage:
    offsets_value_digest.py PATH        -> prints a hex digest, or `none` when
                                           the file is absent
Exit status is 0 for both, so the caller's `set -e` sees a missing table the
same way `md5sum ... || echo none` did.  A table that exists and cannot be
parsed exits 2 with a message on stderr: the caller must not treat "I could not
read it" as "it did not change".
"""
import argparse
import hashlib
import os
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


def digest(path):
    """Hex digest over the table's shift values, or ``"none"`` when absent."""
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
    rows = []
    for row in table:
        identity = "|".join(_cell(row[c]) for c in keys)
        rows.append(identity + "=" + ",".join(_cell(row[c]) for c in values))
    # Sorted: a table re-serialised in a different row order holds the same
    # corrections, and the loop must not read that as a re-tie.
    payload = "\n".join(sorted(rows)) + "\n" + ",".join(values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    args = ap.parse_args(argv)
    try:
        print(digest(args.path))
    except (OSError, ValueError, KeyError) as exc:
        print(f"offsets_value_digest: cannot digest {args.path}: {exc}",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
