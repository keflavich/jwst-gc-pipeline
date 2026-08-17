"""What astrometric solution a release's frames are on, and where it came from.

THE THING THAT MUST BE PRESERVED IS THE TABLE, NOT THE FITS
===========================================================

A detector frame is released as a symlink into the live pipeline tree, so its
bytes are whatever the current reduction last wrote.  That is deliberate (see
``exposure_bundle``): a re-reduction rewrites those headers rather than
producing a different observation, and copying tens of GB per release to freeze
a WCS would be an enormous cost for a few changed bytes.

But it means a release cannot, by itself, say what solution its frames carry.
The recoverable part is small: the per-exposure pointing corrections live in one
CSV of a few kB (``Offsets_JWST_Brick<prop>_{consensus,VIRAC2locked}.csv``), and
that table -- not the FITS -- is what has to be frozen per version.  With it, any
version's astrometry is reconstructible; without it, "the frames as they were in
v1.2" is unrecoverable the moment the pipeline re-headers them.

So the table is staged as a COPY and checksummed, exactly like a mosaic, while
the frames stay symlinks.  Cost: kilobytes per release.

WHICH TABLE, AND THE THREE HONEST ANSWERS
=========================================

``alignment_config.offsets_table_path`` is the single authority, and it has three
outcomes, all of which a reader needs told apart:

    a table path      the field is tied through that table
    ``''``            the field is registered but uses no offsets table
                      (its channel is ``none``) -- the frames are on whatever
                      frame the reduction left them
    no config entry   the field is NOT in ``ALIGNMENT_CONFIG`` at all, so
                      nothing ever applied a correction and the frames sit at
                      the raw ``assign_wcs`` frame

The third is not hypothetical and is the reason this module states it rather
than defaulting to silence: proposal 1939 was missing from ``ALIGNMENT_CONFIG``,
so every sgra mosaic sat ~14.8" off VIRAC2 and Gaia while its own m2 offsets
table went unread.  A release that ships frames from such a field and says
nothing is asserting, by omission, that they are tied.

WHAT THE ACCURACY NUMBER IS AND IS NOT
======================================

The m2 checkpoint records a reference tie per filter.  It is reported here
against BOTH references, because in the Galactic Center they disagree for a known
reason and quoting one alone misleads:

    vs_sparse   the Gaia-only subset.  Sparse, so noisier per-star, but the
                offset-histogram peak against it is unbiased.
    vs_full     the full VIRAC2 catalog.  Dense, and two catalogs tracing the
                same clustered field make a correlated wrong-pair background
                that pulls the histogram peak by several mas -- arches reads
                1.2 mas vs sparse and 9.3 mas vs dense on the same frames.

The dense number is quoted so nobody rediscovers it as a defect, labelled as the
artifact it is.  Neither is a per-star error bar: they are bulk ties.
"""
import datetime
import glob
import hashlib
import json
import os
from pathlib import Path

#: Manifest category for the frozen pointing-correction table.
ASTROMETRY_CATEGORY = "astrometry"


def _sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _proposal_and_observations(field_cfg):
    """``(proposal, [observation, ...])`` for the alignment lookup."""
    prefixes = field_cfg.get("proposal_prefix") or []
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    import re
    out = []
    observations = field_cfg.get("observations")
    for prefix in prefixes:
        match = re.match(r"^jw0?(?P<prop>\d{4,5})(?:-o?(?P<obs>\d{3}))?", str(prefix))
        if match is None:
            continue
        prop = match.group("prop").lstrip("0")
        if observations:
            out += [(prop, str(o).lstrip("o").zfill(3)) for o in observations]
        elif match.group("obs"):
            out.append((prop, match.group("obs")))
    return out


def offsets_table(field_cfg):
    """``(path, state)`` for the field's pointing-correction table.

    ``state`` is ``"table"``, ``"no-table"`` (registered, channel ``none``) or
    ``"unregistered"`` (absent from ``ALIGNMENT_CONFIG``).  ``path`` is ``None``
    for the latter two, and may name a file that does not exist -- an absent
    table for a registered field is itself worth reporting.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from jwst_gc_pipeline.reduction import alignment_config

    basepath = str(field_cfg["data_dir"])
    for prop, obs in _proposal_and_observations(field_cfg):
        if alignment_config.resolve(prop, obs) is None:
            continue
        path = alignment_config.offsets_table_path(basepath, prop, obs)
        if path:
            return Path(path), "table"
        return None, "no-table"
    return None, "unregistered"


def reference_ties(field_cfg):
    """``{FILTER: {...}}`` -- the m2 checkpoint's reference tie per filter.

    Best-effort: a field with no checkpoint records simply reports none, which
    the page then says out loud rather than implying a tie was measured.
    """
    checkpoints = Path(field_cfg["data_dir"]) / "astrometry_checkpoints"
    out = {}
    for path in sorted(glob.glob(str(checkpoints / "checkpoint_m2_*_latest.json"))):
        try:
            record = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            continue
        visits = record.get("visits") or []
        if not visits:
            continue
        tie = (visits[0] or {}).get("reference_tie") or {}
        filt = record.get("filtername")
        if not filt:
            continue
        entry = {"date": record.get("date"),
                 "correcting": bool(record.get("correcting")),
                 "n_corrections": len(record.get("corrections") or [])}
        for key, label in (("vs_sparse", "sparse_mas"), ("vs_full", "dense_mas")):
            leg = tie.get(key) or {}
            if leg.get("off") is not None:
                entry[label] = round(float(leg["off"]), 2)
                entry[label.replace("_mas", "_contrast")] = leg.get("contrast")
        # A filter appears both with and without an observation suffix
        # (`checkpoint_m2_F212N_latest` and `checkpoint_m2_F212N_o001_latest`),
        # and they are written at different times: arches F212N reads 3.88 mas in
        # the unsuffixed record from 2026-08-01 and 1.22 mas in the o001 record
        # from 2026-08-16, after the offsets table was corrected. The NEWEST wins
        # -- an older record describes a solution the frames are no longer on.
        prior = out.get(filt)
        if prior is None or str(entry.get("date") or "") >= str(prior.get("date") or ""):
            out[filt] = entry
    return out


def collect(field, field_cfg):
    """The full provenance record for one field."""
    path, state = offsets_table(field_cfg)
    record = {"field": field, "state": state, "table": None,
               "ties": reference_ties(field_cfg)}
    if path is not None:
        record["table"] = {
            "name": path.name, "path": str(path), "exists": path.is_file()}
        if path.is_file():
            stat = path.stat()
            record["table"].update(
                size_bytes=stat.st_size,
                sha256=_sha256(path),
                modified=datetime.datetime.fromtimestamp(
                    stat.st_mtime).astimezone().isoformat())
    return record


def stage_item(field, field_cfg, version, assign_dest_name="astrometry"):
    """A manifest item for the field's offsets table, or ``None``.

    Unlike an exposure this is COPIED and checksummed: it is kilobytes, and it
    is the one artifact that makes a version's astrometry reconstructible after
    the frames have been re-headered.
    """
    path, state = offsets_table(field_cfg)
    if path is None or not path.is_file():
        return None
    return {
        "category": ASTROMETRY_CATEGORY, "kind": "offsets_table",
        "filter": None, "iteration": None, "observation": None,
        "instrument": None, "src": str(path), "version": version,
        "dest": str(Path(assign_dest_name) / path.name),
    }


def summary_lines(record):
    """Plain-text provenance, for the README.  One list of short lines."""
    state = record.get("state")
    lines = []
    if state == "unregistered":
        lines += [
            "**This field is not registered in `ALIGNMENT_CONFIG`.** No pointing",
            "correction was ever applied to it, so these frames carry the raw",
            "`assign_wcs` astrometry. Proposal 1939 sat in exactly this state and",
            "its mosaics were ~14.8\" off Gaia/VIRAC2. Do not assume these frames",
            "are tied to any reference frame.",
        ]
    elif state == "no-table":
        lines += [
            "This field is registered but uses no offsets table: its frames carry",
            "whatever astrometry the reduction left them with, and there is no",
            "per-exposure correction table to preserve.",
        ]
    else:
        table = record.get("table") or {}
        if not table.get("exists"):
            lines += [
                f"The field's pointing-correction table (`{table.get('name')}`) is",
                "NOT on disk, so the solution these frames carry cannot be stated.",
            ]
        else:
            lines += [
                f"Pointing corrections: `{table['name']}`"
                f" ({table['size_bytes']} bytes, modified {table['modified'][:19]}).",
                f"  sha256 `{table['sha256']}`",
                "",
                "The table is shipped as a FROZEN COPY under `astrometry/` and is",
                "checksummed. The detector frames are NOT frozen -- they are symlinks",
                "whose headers a re-reduction rewrites -- so this table, not the FITS,",
                "is what makes this version's astrometry reconstructible later.",
            ]
    ties = record.get("ties") or {}
    if ties:
        lines += ["", "Measured reference tie (m2 checkpoint, bulk offset):"]
        for filt in sorted(ties):
            tie = ties[filt]
            bits = []
            if "sparse_mas" in tie:
                bits.append(f"{tie['sparse_mas']} mas vs Gaia(sparse)")
            if "dense_mas" in tie:
                bits.append(f"{tie['dense_mas']} mas vs VIRAC2(dense)")
            if bits:
                lines.append(f"- {filt}: " + ", ".join(bits)
                             + (f"  [{tie['date'][:10]}]" if tie.get("date") else ""))
        lines += [
            "",
            "Quote the SPARSE number. Against a DENSE reference the offset-histogram",
            "peak is pulled by several mas: two catalogs tracing the same clustered",
            "field make a correlated wrong-pair background. The dense value is listed",
            "so the discrepancy is not rediscovered as a defect. Neither is a",
            "per-star error bar; both are bulk ties.",
        ]
    else:
        lines += ["", "No m2 reference-tie record was found for this field, so no "
                  "measured accuracy is claimed here."]
    return lines
