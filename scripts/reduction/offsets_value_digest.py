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
-- the `Visit` cell carries it as characters 8-10 of `jwPPPPPOOOVVV`.  It also
takes the hyphen-joined spelling `alignment_config` registers for a JOINTLY
registered field (sickle MIRI `001-002`, sgrb2 MIRI `002-998`), which names a
SET of observations; `run_field_retie_loop.sh` passes its `FIELD` through
unchanged, so this has to accept every spelling the registry holds.  The
fixed-point check beside this one is already scoped that way (`--obs-token
o$FIELD`).

Scoping only ever HIDES a change that is attributably some other observation's.
A row whose `Visit` cannot be parsed is KEPT, so an unattributable change still
reads as movement; a request to scope a table with no `Visit` column at all
fails (exit 2) rather than quietly digesting every row as if it were this
observation's.  A scope with no rows -- an absent table, or one holding only
other observations' rows -- digests to one stable per-observation value, so a
NEIGHBOUR creating the shared table is not read as this observation's re-tie
while this observation seeding its own first rows still is.

Usage:
    offsets_value_digest.py PATH        -> prints a hex digest, or `none` when
                                           the file is absent (unscoped; with
                                           --observation an absent table reads
                                           as "this observation has no rows",
                                           which is what an existing table
                                           without them reads as too)
    offsets_value_digest.py PATH --observation 088
                                        -> the same, over that observation's
                                           rows only
    offsets_value_digest.py PATH --observation 001-002
                                        -> a JOINT registration (sickle MIRI,
                                           sgrb2 MIRI): the rows of either
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
#:
#: END-ANCHORED, and deliberately the same expression as
#: `photometry.astrometry_checkpoint._VISIT_ID_RE` -- the m2 checkpoint keys the
#: rows it WRITES on that one, and two parsers of one column that disagree put a
#: row in one observation for the writer and another for this reader.
#:
#: Unanchored (`^jw\d{5}(\d{3})`) a MALFORMED id parsed as a confident wrong
#: observation: `jw2211023001` (one digit short) read `230` and `jw002211023001`
#: (one digit long) read `102`, neither of which is any real observation, so the
#: row fell out of EVERY observation's scope and a re-tie on it was invisible to
#: every loop.  Anchored, those ids do not parse at all and the row is KEPT
#: below -- the direction that cannot hide movement.  `visit_obs_key` treats the
#: same ids as bare visits for the same reason (#284).
VISIT_RE = re.compile(r"jw\d{5}(\d{3})\d{3}\s*$", re.IGNORECASE)

#: What `--observation` may spell: an observation number, or several joined by
#: `-` for a JOINT registration.  Mirrors `photometry.naming`'s
#: `_OBSERVATION_FIELD_RE`.
OBSERVATION_FIELD_RE = re.compile(r"\d+(?:-\d+)*")


def _observation_of(visit):
    """The 3-digit observation a `Visit` cell names, or None when unparseable."""
    match = VISIT_RE.match(str(visit).strip())
    return match.group(1) if match else None


def normalise_observation(observation):
    """The observations a `--observation` token names, as a tuple of `NNN`.

    `12` and `012` name the same observation; the table writes `012`.

    A JOINT registration names SEVERAL, hyphen-joined: `alignment_config`
    registers sickle's MIRI as field `001-002` and sgrb2's as `002-998`, and
    `run_field_retie_loop.sh` passes its `FIELD` straight through.  Rejecting
    the hyphen made every digest call exit 2, which that loop reads fail-open
    as "the table changed" -- for the whole run, so its "no SHIFT VALUE changed
    -> this is NOT a checkpoint re-tie -> STOPPING" branch could never be
    reached and the loop re-reduced to MAXITER.  That is issue #272's
    behaviour, produced by the file written to remove it.

    The decomposition is the campaign's usual one (`str(field).split('-')`, as
    in `naming.observation_field_token` and
    `mast_obs_scope.observation_scope_mask`), spelled out here rather than
    imported so this script keeps standing on astropy alone: an ImportError
    inside it would exit 2 on every call, which is the failure just described.
    `test_the_joint_spelling_agrees_with_naming_observation_field_token` pins
    the two spellings together.

    A part wider than three digits still raises.  `observation_field_token`
    accepts one (it only pads), but here an unmatchable observation would scope
    the digest to zero attributable rows and report every iteration as "no
    re-tie" -- silent, and on the stopping side.
    """
    if observation is None:
        return None
    text = str(observation).strip()
    if not OBSERVATION_FIELD_RE.fullmatch(text):
        raise ValueError(
            f"observation {observation!r} is not an observation number, or "
            f"several joined by '-' (a joint registration such as 001-002); a "
            f"visit id carries one as characters 8-10 of jwPPPPPOOOVVV")
    parts = tuple(dict.fromkeys(f"{int(part):03d}" for part in text.split("-")))
    wide = [part for part in parts if len(part) > 3]
    if wide:
        raise ValueError(
            f"observation {observation!r} names {wide}, which no 3-digit "
            f"observation field can equal; scoping to it would digest no rows "
            f"and report every iteration as 'no re-tie'")
    return parts


def _empty_scope_digest(observations):
    """The digest of a scope that holds no rows.

    Column-independent on purpose.  It has to be reachable from an ABSENT
    table, where the value columns a future table will carry are unknowable,
    and it has to equal what an EXISTING table with none of this observation's
    rows gives -- otherwise the two differ and the difference reads as a
    re-tie.
    """
    payload = "no rows\nobservation=" + "-".join(observations)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    observations = normalise_observation(observation)
    if not path or not os.path.exists(path):
        if observations is None:
            # Exactly what `md5sum ... || echo none` reported, for every
            # single-observation field: an absent table is its own state.
            return "none"
        # SCOPED, an absent table reads as "this observation has no rows",
        # which is the same state an existing table with none of its rows is
        # in.  Reporting them differently makes the shared table's first night
        # a false re-tie for every tile: tile 088 writes nothing, a NEIGHBOUR's
        # m2 CREATES the file, and 088's before/after goes `none` -> hex, so it
        # re-reduces for a change that is not its own.  139 treasury tiles
        # share one table that does not exist yet, so that is the first
        # iteration of most of them.
        #
        # The seeding signal survives: this observation's OWN first rows still
        # take the digest from the empty value to a populated one, which is
        # what `test_an_observation_with_no_rows_yet_...` pins.
        return _empty_scope_digest(observations)
    from astropy.table import Table
    table = Table.read(path)
    keys = [c for c in KEY_COLUMNS if c in table.colnames]
    values = [c for c in VALUE_COLUMNS if c in table.colnames]
    if not values:
        raise ValueError(
            f"{path} carries none of {VALUE_COLUMNS}; digesting it would return "
            f"the same hash for every table and report every iteration as "
            f"'no re-tie'")
    if observations is not None and "Visit" not in table.colnames:
        raise ValueError(
            f"{path} has no 'Visit' column, so its rows cannot be attributed to "
            f"observation {'-'.join(observations)}; digesting all of them would "
            f"report another observation's correction as this one's re-tie")
    rows = []
    for row in table:
        if observations is not None:
            row_obs = _observation_of(row["Visit"])
            # An unparseable Visit is KEPT: scoping may hide a change that
            # BELONGS to another observation, never one that cannot be placed.
            if row_obs is not None and row_obs not in observations:
                continue
        identity = "|".join(_cell(row[c]) for c in keys)
        rows.append(identity + "=" + ",".join(_cell(row[c]) for c in values))
    if observations is not None and not rows:
        # Same value an absent table gives -- see above.
        return _empty_scope_digest(observations)
    # Sorted: a table re-serialised in a different row order holds the same
    # corrections, and the loop must not read that as a re-tie.
    payload = "\n".join(sorted(rows)) + "\n" + ",".join(values)
    if observations is not None:
        # Scope in the payload, so a digest taken for one observation is not
        # comparable with one taken for another (or with the whole file).
        payload += "\nobservation=" + "-".join(observations)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--observation", default=None,
                    help="digest only the rows whose Visit names this "
                         "observation (the shared-table case: 10678's 139 "
                         "tiles).  A joint registration names several, "
                         "hyphen-joined: 001-002.  Omitted, every row is "
                         "digested.")
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
