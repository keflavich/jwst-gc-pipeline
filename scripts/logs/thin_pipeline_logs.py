#!/usr/bin/env python
"""Thin + compress historical pipeline/cataloging logs WITHOUT losing diagnostics.

Motivation: cataloging logs are dominated (~80-90% of lines) by per-source
satstar fitter spam that PR #148 silences going FORWARD (the same templates
gated behind ``SATSTAR_LOG_VERBOSE``), plus printed astropy fit tables and, in
old (2022-2023) reduction logs, webbpsf OPD/MAST-token retry spam.  This tool
applies the same judgement RETROACTIVELY:

  * every removed run of lines is replaced by a single in-place marker line
    carrying the per-template counts, so "how many sources were
    accepted/skipped here" survives thinning;
  * any line NOT matching a known zero-information template is preserved
    byte-identically;
  * nothing is ever deleted: originals are atomically replaced by the thinned
    file (mtime preserved), and ``--compress`` additionally zstd-compresses
    the survivor (original removed only after ``zstd -t`` verifies).

Safety rails:
  * DRY-RUN by default — prints a per-file and total savings report.
    Pass ``--execute`` to apply.
  * files younger than ``--min-age-days`` (default 2) are skipped;
  * files whose name embeds a SLURM job id currently in ``squeue`` are
    skipped (a live job may still be appending);
  * already-compressed files (``.zst``/``.gz``/``.xz``) are never re-thinned.

Per-source "Accepting source N with flux=..." / "Skipping source N: snr=..."
lines are only removed with ``--aggressive`` (their values are occasionally
useful when chasing a single star); the marker then records the
accepted/skipped counts, mirroring PR #148's per-frame summary line.

Typical campaign::

    # report only
    python scripts/logs/thin_pipeline_logs.py /blue/adamginsburg/adamginsburg/logs/jwst \
        /blue/adamginsburg/adamginsburg/brick_logs /blue/adamginsburg/adamginsburg/jwst/brick/logs
    # thin + compress everything older than a week, per-source lines too
    python scripts/logs/thin_pipeline_logs.py --execute --aggressive --compress \
        --min-age-days 7 <same dirs>
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

MARKER_PREFIX = "### thin_pipeline_logs:"

# --- zero-information templates (always thinnable) -------------------------
# Per-source satstar fitter spam silenced going-forward by PR #148:
DEFAULT_PATTERNS = {
    "src_center": re.compile(r"^Source \d+: center at \(x, y\) = "),
    "forced_nobkg": re.compile(r"^  forced source: no local-bkg "),
    "mask_buffer": re.compile(r"^  mask_buffer="),
    "set_bounds": re.compile(r"^Set [A-Za-z_][A-Za-z_0-9]*\.bounds = "),
    "npix_thresh": re.compile(r"^Number of pixels above threshold "),
    # bare print(np.nanmax(model_image)) lines:
    "bare_number": re.compile(r"^-?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"),
    "blank": re.compile(r"^\s*$"),
    # webbpsf OPD / MAST-token retry spam (dominates 2022-2023 reduction logs):
    "opd_spam": re.compile(
        r"^(\tURI:\t mast:JWST/product/R"
        r"|\tDate \(MJD\):\t"
        r"|\tDelta time:\t"
        r"| *MJD: \d"
        r"|User requested choosing OPD "
        r"|OPD immediately (preceding|following) the given datetime:"
        r"|Attempting to load PSF for "
        r"|psfgen load_wss_opd_by_date failed"
        r"|MAST API token invalid!"
        r"| To create a new API token)"
    ),
}
AGGRESSIVE_PATTERNS = {
    "accept_src": re.compile(r"^Accepting (forced outside-FOV )?source \d+ with flux="),
    "skip_src": re.compile(r"^Skipping source \d+: "),
}
# Printed astropy fit tables (id/group_id/... header, then divider + rows):
TABLE_HEADER = re.compile(r"^ *id group_id group_size ")
TABLE_DIVIDER = re.compile(r"^-{2,} [-\s]+$")
TABLE_ROW = re.compile(r"^ *\d+ +\d+ +\d+ [-+0-9.eEnaif ]*(False|True)?[-+0-9.eEnaif ]*$")

LOG_SUFFIXES = (".log", ".out")
COMPRESSED_SUFFIXES = (".zst", ".gz", ".xz", ".bz2")
JOBID_RE = re.compile(r"_(\d{6,})(?:_\d+)?\.(?:log|out)$")


def running_slurm_jobids():
    """Set of this user's SLURM job ids (best effort; empty off-cluster)."""
    try:
        out = subprocess.run(
            ["squeue", "--me", "--noheader", "--format=%A"],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return set()
    return {tok.strip() for tok in out.split() if tok.strip().isdigit()}


# Runs consisting ONLY of these templates carry zero information even in
# aggregate, so short (<3-line) runs of them get no marker at all — otherwise
# alternating blank/number lines would trade one spam line for one marker line.
NO_MARKER_TEMPLATES = frozenset({"blank", "bare_number"})


class RunAccumulator:
    """Collects a consecutive run of thinnable lines; emits one marker."""

    def __init__(self):
        self.counts = {}
        self.nlines = 0

    def add(self, template):
        self.counts[template] = self.counts.get(template, 0) + 1
        self.nlines += 1

    def flush(self, out):
        """Write a marker for the pending run.  Returns (removed, wrote_marker)."""
        if not self.nlines:
            return 0, False
        removed = self.nlines
        skip_marker = (removed < 3
                       and set(self.counts) <= NO_MARKER_TEMPLATES)
        if not skip_marker:
            detail = ", ".join(f"{k}x{v}" for k, v in sorted(self.counts.items()))
            out.write(f"{MARKER_PREFIX} removed {removed} lines [{detail}]\n")
        self.counts = {}
        self.nlines = 0
        return removed, not skip_marker


def classify(line, patterns, in_table):
    """Return (template_name or None, new_in_table_state)."""
    if TABLE_HEADER.match(line):
        return "fit_table", True
    if in_table:
        if TABLE_DIVIDER.match(line) or TABLE_ROW.match(line):
            return "fit_table", True
        # first non-table line ends the table block; fall through
        in_table = False
    for name, rx in patterns.items():
        if rx.match(line):
            return name, in_table
    return None, in_table


class _NullOut:
    def write(self, _s):
        pass


def thin_file(path, patterns, execute):
    """Thin one log file.

    Returns (kept_lines, removed_lines, removed_bytes, new_size).
    In dry-run mode nothing is written and new_size is the projected size.
    """
    kept = removed = removed_bytes = 0
    run = RunAccumulator()
    in_table = False
    dirname = os.path.dirname(path) or "."
    tmp = None
    out = _NullOut()
    marker_bytes = 0
    try:
        if execute:
            fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".thin.",
                                       dir=dirname)
            out = os.fdopen(fd, "w", encoding="utf-8", errors="surrogateescape")
        with open(path, "r", encoding="utf-8", errors="surrogateescape") as fin:
            for line in fin:
                if line.startswith(MARKER_PREFIX):
                    # idempotency: keep existing markers as-is
                    template = None
                else:
                    template, in_table = classify(line, patterns, in_table)
                if template is None:
                    removed_here, wrote_marker = run.flush(out)
                    removed += removed_here
                    if wrote_marker:
                        kept += 1
                        marker_bytes += 80  # ~marker line size, for the estimate
                    out.write(line)
                    kept += 1
                else:
                    run.add(template)
                    removed_bytes += len(line)
        removed_here, wrote_marker = run.flush(out)
        removed += removed_here
        if wrote_marker:
            kept += 1
            marker_bytes += 80
        if execute:
            out.close()
            out = _NullOut()
            st = os.stat(path)
            os.replace(tmp, path)
            tmp = None
            os.utime(path, (st.st_atime, st.st_mtime))
            new_size = os.path.getsize(path)
        else:
            new_size = os.path.getsize(path) - removed_bytes + marker_bytes
        return kept, removed, removed_bytes, new_size
    finally:
        if not isinstance(out, _NullOut):
            out.close()
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)


def compress_file(path, execute):
    """zstd-compress `path` in place (-> path.zst); verify before unlinking."""
    target = path + ".zst"
    if os.path.exists(target):
        return None
    if not execute:
        return -1
    if shutil.which("zstd") is None:
        raise RuntimeError("zstd not found on PATH; cannot --compress")
    subprocess.run(["zstd", "-q", "-9", "-T0", path, "-o", target], check=True)
    subprocess.run(["zstd", "-q", "-t", target], check=True)
    st = os.stat(path)
    os.utime(target, (st.st_atime, st.st_mtime))
    os.unlink(path)
    return os.path.getsize(target)


def iter_log_files(roots):
    for root in roots:
        if os.path.isfile(root):
            yield root
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in sorted(filenames):
                if fn.endswith(LOG_SUFFIXES):
                    yield os.path.join(dirpath, fn)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help="log files or directories")
    ap.add_argument("--execute", action="store_true",
                    help="apply changes (default: dry-run report only)")
    ap.add_argument("--aggressive", action="store_true",
                    help="also remove per-source Accepting/Skipping lines "
                         "(counts preserved in markers)")
    ap.add_argument("--compress", action="store_true",
                    help="zstd -9 the thinned file, verify, remove original")
    ap.add_argument("--min-age-days", type=float, default=2.0,
                    help="skip files modified more recently than this "
                         "(default 2; protects live jobs)")
    ap.add_argument("--min-size-mb", type=float, default=1.0,
                    help="skip files smaller than this (default 1 MB)")
    args = ap.parse_args(argv)

    patterns = dict(DEFAULT_PATTERNS)
    if args.aggressive:
        patterns.update(AGGRESSIVE_PATTERNS)

    live_jobs = running_slurm_jobids()
    now = time.time()
    tot_before = tot_after = tot_removed = nfiles = nskipped = 0

    for path in iter_log_files(args.paths):
        if path.endswith(COMPRESSED_SUFFIXES):
            continue
        try:
            st = os.stat(path)
        except OSError as ex:
            print(f"SKIP (stat failed: {ex}): {path}")
            nskipped += 1
            continue
        if st.st_size < args.min_size_mb * 1e6:
            continue
        if (now - st.st_mtime) < args.min_age_days * 86400:
            print(f"SKIP (younger than {args.min_age_days}d): {path}")
            nskipped += 1
            continue
        m = JOBID_RE.search(os.path.basename(path))
        if m and m.group(1) in live_jobs:
            print(f"SKIP (job {m.group(1)} still in squeue): {path}")
            nskipped += 1
            continue

        size0 = st.st_size
        kept, removed, removed_bytes, size1 = thin_file(path, patterns,
                                                        args.execute)
        frac = removed / max(kept + removed, 1)
        zsize = None
        if args.compress:
            zsize = compress_file(path, args.execute)
        final = zsize if (zsize not in (None, -1)) else size1
        tot_before += size0
        tot_after += final
        tot_removed += removed
        nfiles += 1
        ztxt = ""
        if zsize == -1:
            ztxt = " (+zstd pending)"
        elif zsize is not None:
            ztxt = f" -> {zsize/1e6:.1f}M zstd"
        print(f"{'THINNED' if args.execute else 'DRYRUN '} {path}: "
              f"{size0/1e6:.1f}M, {removed} of {kept+removed} lines "
              f"({100*frac:.0f}%) thinnable -> {size1/1e6:.1f}M{ztxt}")

    print(f"\nTOTAL: {nfiles} files, {tot_before/1e9:.2f} GB -> "
          f"{tot_after/1e9:.2f} GB "
          f"({tot_removed} lines removed; {nskipped} skipped)"
          + ("" if args.execute else "   [DRY RUN — nothing modified]"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
