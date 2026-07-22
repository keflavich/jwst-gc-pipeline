"""Tailing registration failsafe — run the per-tile local-registration scan at the
END of a field's reduction/cataloging, so a brick-1182-style localized misregistration
(a half-mosaic untied while the bulk offset reads ~0) is caught the moment the products
are built, not months later at release staging.

This wraps ``scripts/release/registration_failsafes.py --scan`` (the same per-cell
cross-band + own-catalog check the release gate uses) so any chain can call it:

    python -m jwst_gc_pipeline.photometry.registration_gate --field brick [--strict]

or, from Python at the tail of a cataloging chain::

    from jwst_gc_pipeline.photometry.registration_gate import assert_field_registered
    assert_field_registered("brick")            # raises on FAIL

Design notes
------------
* Runs the WHOLE-FIELD, all-band scan -- it is only meaningful once every band's
  merged mosaic + catalog exists, i.e. at the tail of the full cataloging pass.
* Default is **warn** (return a status); pass ``strict=True`` / ``--strict`` to
  RAISE on FAIL so it can hard-gate a chain. A scan that cannot run (e.g. <2 bands
  present) is a WARN, never a block -- it must not wedge a partial run.
* It does NOT itself use nearest-neighbour-median; the underlying failsafe is
  per-cell agreement-fraction + offset-histogram. See CLAUDE.md.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAILSAFE = _REPO_ROOT / "scripts" / "release" / "registration_failsafes.py"


class RegistrationGateError(RuntimeError):
    """Raised when the tailing registration failsafe FAILs for a field (a band is
    locally misregistered vs the other bands / its own catalog)."""


def run_registration_gate(field, *, strict=False, python=None, verbose=True):
    """Run the per-tile local-registration scan for ``field``.

    Returns ``dict(field, passed, could_not_verify, returncode)``.  ``passed`` is
    True only when the scan ran AND every band passed.  ``could_not_verify`` is True
    when the failsafe returned "warn" (it could not run -- e.g. too few bands).

    With ``strict=True``, raises :class:`RegistrationGateError` on a genuine FAIL
    (never on could-not-verify).
    """
    if not _FAILSAFE.is_file():
        msg = f"registration failsafe not found at {_FAILSAFE}"
        if strict:
            raise RegistrationGateError(msg)
        if verbose:
            print(f"WARNING: {msg}; skipping registration gate.", file=sys.stderr)
        return dict(field=field, passed=False, could_not_verify=True, returncode=None)

    # failsafe contract: rc 0 = PASS *or* could-not-verify(warn), 1 = FAIL.  rc
    # alone cannot distinguish PASS from could-not-verify, so ask the failsafe
    # for its --json result and read the "error"/"PASS" fields from it.
    fd, json_path = tempfile.mkstemp(prefix="registration_gate_",
                                     suffix=".json")
    os.close(fd)
    try:
        cmd = [python or sys.executable, str(_FAILSAFE), "--field", str(field),
               "--scan", "--json", json_path]
        if verbose:
            print(f"[registration_gate] scanning field '{field}' ...", flush=True)
        rc = subprocess.run(cmd).returncode
        scan_result = None
        try:
            with open(json_path) as fh:
                scan_result = json.load(fh)
        except (OSError, json.JSONDecodeError, ValueError):
            scan_result = None
    finally:
        try:
            os.unlink(json_path)
        except OSError:
            pass

    if rc == 1 and strict:
        raise RegistrationGateError(
            f"Registration gate FAILED for field '{field}': a band's mosaic is "
            f"locally misregistered vs the other bands / its own catalog. Do NOT "
            f"release or trust these products; re-tie + re-drizzle the offending "
            f"band. (scripts/release/registration_failsafes.py --field {field} --scan)")

    # passed is True ONLY when the scan actually ran and every band passed;
    # could_not_verify is True when the failsafe could not verify (its scan
    # reported an "error", e.g. <2 bands, or it produced no readable result).
    passed = False
    could_not_verify = False
    if rc == 0:
        if scan_result is None or scan_result.get("error"):
            could_not_verify = True
        elif scan_result.get("PASS"):
            passed = True
        else:
            # rc==0 but the scan says not-PASS: inconsistent -> unverified
            could_not_verify = True
    elif rc != 1:
        could_not_verify = True
        msg = f"registration failsafe could not run for '{field}' (rc={rc})"
        if strict:
            raise RegistrationGateError(msg)
        if verbose:
            print(f"WARNING: {msg}; not blocking.", file=sys.stderr)
    if could_not_verify and rc == 0 and verbose:
        print(f"WARNING: registration failsafe could not verify field "
              f"'{field}' (warn-tier; not blocking).", file=sys.stderr)
    return dict(field=field, passed=passed, could_not_verify=could_not_verify,
                returncode=rc)


def assert_field_registered(field, **kwargs):
    """Convenience: run the gate in strict mode (raise on FAIL)."""
    kwargs["strict"] = True
    return run_registration_gate(field, **kwargs)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Tailing registration failsafe for a field.")
    ap.add_argument("--field", required=True, help="field/target name (e.g. brick)")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero (and raise) on FAIL, to hard-gate a chain")
    args = ap.parse_args(argv)
    res = run_registration_gate(args.field, strict=args.strict)
    if res["returncode"] == 1:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
