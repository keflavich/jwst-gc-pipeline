"""Release staging must not quietly ship a short release.

Two hazards, both observed on sickle:

1. An explicitly-listed ``nircam``/``miri`` src that is no longer on disk used to be
   skipped with a bare ``continue``, so a field configured for five bands staged four
   and printed nothing.  On 2026-08-05 sickle's listed F210M ``-merged_i2d.fits`` was
   exactly that: the m2 astrometry checkpoint had renamed it to
   ``..._im0_badastrom.fits``.
2. A staged set drawn from more than one reduction generation.  v1.1 shipped sickle's
   F210M from 2026-04-19 / jwst_1535.pmap beside four bands from 2026-06-27 /
   jwst_1537.pmap.

These run on synthetic trees -- no archive files are touched.
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest
from astropy.io import fits

_SPEC = importlib.util.spec_from_file_location(
    "stage_release",
    Path(__file__).resolve().parents[3] / "scripts" / "release" / "stage_release.py")
stage_release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stage_release)

FIELD = "_synthetic_test_field"


def _write_mosaic(path, date=None, crds_ctx=None):
    """A one-pixel i2d stand-in carrying the provenance keywords we gate on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU()
    if date is not None:
        hdu.header["DATE"] = date
    if crds_ctx is not None:
        hdu.header["CRDS_CTX"] = crds_ctx
    hdu.writeto(path, overwrite=True)
    return path


def _install_field(monkeypatch, tmp_path, nircam_entries):
    cfg = {
        "data_dir": tmp_path,
        "proposal_prefix": "jw09999-o001_t001_nircam_clear",
        "no_auto_images": True,
        "skip_catalogs": True,
        "nircam": nircam_entries,
    }
    monkeypatch.setitem(stage_release.FIELDS, FIELD, cfg)
    return cfg


def test_listed_src_absent_refuses_and_names_the_filter(tmp_path, monkeypatch, capsys):
    """A listed src that is not on disk stops the stage and is named, along with the
    quarantined ``_im0_badastrom`` sibling that explains where it went."""
    pipeline = tmp_path / "F210M" / "pipeline"
    present = _write_mosaic(pipeline / "jw09999-o001_t001_nircam_clear-f187n-nrcb_i2d.fits",
                            date="2026-08-05T04:06:16.987", crds_ctx="jwst_1584.pmap")
    gone = pipeline / "jw09999-o001_t001_nircam_clear-f210m-merged_i2d.fits"
    # the checkpoint's quarantine pair, exactly as it appears in the archive
    quarantined = _write_mosaic(
        pipeline / "jw09999-o001_t001_nircam_clear-f210m-merged_i2d_im0_badastrom.fits",
        date="2026-04-19T13:21:19.806", crds_ctx="jwst_1535.pmap")
    Path(str(quarantined) + ".why.json").write_text(json.dumps({
        "original": str(gone), "renamed_to": str(quarantined),
        "reason": "m2 checkpoint corrected Offsets_JWST_Brick3958_VIRAC2locked.csv",
        "date": "2026-08-05T03:29:26Z"}))
    assert not gone.exists()

    _install_field(monkeypatch, tmp_path, [
        {"filter": "F187N", "src": str(present)},
        {"filter": "F210M", "src": str(gone)},
    ])

    rc = stage_release.main(["--field", FIELD, "--version", "v9.9-2099.01"])
    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert rc != 0, ("a listed-but-absent src must stop the stage; a zero exit means "
                     "the band was dropped silently:\n" + combined)
    assert "F210M" in combined, "the refusal must name the missing filter"
    assert str(gone) in combined, "the refusal must name the missing path"
    assert "_im0_badastrom" in combined, (
        "the refusal must name the quarantined sibling -- it is the clue that tells "
        "an operator the astrometry checkpoint took the file")
    # the band that IS present must not be blamed
    assert "F187N: listed src does not exist" not in combined


def test_all_listed_srcs_present_does_not_refuse(tmp_path, monkeypatch, capsys):
    pipeline = tmp_path / "F187N" / "pipeline"
    a = _write_mosaic(pipeline / "jw09999-o001_t001_nircam_clear-f187n-nrcb_i2d.fits",
                      date="2026-08-05T04:06:16.987", crds_ctx="jwst_1584.pmap")
    b = _write_mosaic(pipeline / "jw09999-o001_t001_nircam_clear-f210m-nrcb_i2d.fits",
                      date="2026-08-05T04:03:24.681", crds_ctx="jwst_1584.pmap")
    _install_field(monkeypatch, tmp_path, [
        {"filter": "F187N", "src": str(a)}, {"filter": "F210M", "src": str(b)}])

    rc = stage_release.main(["--field", FIELD, "--version", "v9.9-2099.01"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    assert "MIXED GENERATIONS" not in captured.out


def test_generation_span_fires_on_a_month_apart_pair(tmp_path, monkeypatch, capsys):
    """Two staged images a month apart in DATE are two reduction generations."""
    pipeline = tmp_path / "F187N" / "pipeline"
    old = _write_mosaic(pipeline / "jw09999-o001_t001_nircam_clear-f210m-nrcb_i2d.fits",
                        date="2026-04-19T13:21:19.806", crds_ctx="jwst_1535.pmap")
    new = _write_mosaic(pipeline / "jw09999-o001_t001_nircam_clear-f187n-nrcb_i2d.fits",
                        date="2026-05-20T10:00:00.000", crds_ctx="jwst_1535.pmap")
    _install_field(monkeypatch, tmp_path, [
        {"filter": "F210M", "src": str(old)}, {"filter": "F187N", "src": str(new)}])

    rc = stage_release.main(["--field", FIELD, "--version", "v9.9-2099.01"])
    captured = capsys.readouterr()
    assert rc == 0, "without the flag the span is reported, not refused"
    assert "MIXED GENERATIONS" in captured.out, (
        "a 31-day DATE span must be reported:\n" + captured.out + captured.err)
    assert "DATE span" in captured.out

    rc = stage_release.main(["--field", FIELD, "--version", "v9.9-2099.01",
                             "--refuse-mixed-generations"])
    captured = capsys.readouterr()
    assert rc != 0, ("--refuse-mixed-generations must stop the stage:\n"
                     + captured.out + captured.err)
    assert "single reduction generation" in captured.err


def test_generation_span_fires_on_mixed_crds_context(tmp_path, monkeypatch, capsys):
    """Same day, two CRDS contexts: the date leg cannot see this one."""
    pipeline = tmp_path / "F187N" / "pipeline"
    a = _write_mosaic(pipeline / "jw09999-o001_t001_nircam_clear-f187n-nrcb_i2d.fits",
                      date="2026-08-05T04:06:16.987", crds_ctx="jwst_1584.pmap")
    b = _write_mosaic(pipeline / "jw09999-o001_t001_nircam_clear-f210m-nrcb_i2d.fits",
                      date="2026-08-05T04:03:24.681", crds_ctx="jwst_1537.pmap")
    _install_field(monkeypatch, tmp_path, [
        {"filter": "F187N", "src": str(a)}, {"filter": "F210M", "src": str(b)}])

    stage_release.main(["--field", FIELD, "--version", "v9.9-2099.01"])
    captured = capsys.readouterr()
    assert "MIXED GENERATIONS" in captured.out
    assert "CRDS contexts" in captured.out


def test_nircam_and_miri_generations_are_not_compared(tmp_path, monkeypatch):
    """MIRI is reduced in its own batch, so a NIRCam-vs-MIRI date gap is expected."""
    nir = _write_mosaic(tmp_path / "F187N" / "pipeline" / "nircam_i2d.fits",
                        date="2026-08-05T04:06:16.987", crds_ctx="jwst_1584.pmap")
    mir = _write_mosaic(tmp_path / "F770W" / "pipeline" / "miri_i2d.fits",
                        date="2026-04-19T13:21:19.806", crds_ctx="jwst_1535.pmap")
    items = [
        {"category": "image", "kind": "science", "filter": "F187N",
         "instrument": "NIRCam", "src": str(nir)},
        {"category": "image", "kind": "science", "filter": "F770W",
         "instrument": "MIRI", "src": str(mir)},
    ]
    assert stage_release.check_generation_span(items, verbose=False) == []


@pytest.mark.parametrize("field", sorted(stage_release.FIELDS))
def test_every_listed_src_in_the_shipped_config_exists(field):
    """Pre-stage preflight: no field's config carries a stale path.

    OPT-IN (``STAGE_RELEASE_ARCHIVE_CHECK=1``), because it asserts against a live,
    continuously-rewritten reduction tree rather than a fixture.  A band is absent
    for as long as it takes the m2 checkpoint to quarantine and the pipeline to
    re-drizzle it -- sickle lost F210M and then F335M within two hours on
    2026-08-05 -- so in the default suite this would be red for reasons that have
    nothing to do with the code under test.  Run it deliberately before staging.
    """
    if os.environ.get("STAGE_RELEASE_ARCHIVE_CHECK") != "1":
        pytest.skip("set STAGE_RELEASE_ARCHIVE_CHECK=1 to check the live archive")
    entries = [(k, e) for k in ("nircam", "miri")
               for e in stage_release.FIELDS[field].get(k, [])]
    if not entries:
        pytest.skip(f"{field} lists no explicit srcs")
    if not stage_release.GLOBUS_COLLECTION_ROOT.is_dir():
        pytest.skip("archive not mounted")
    absent = [f"{k} {e['filter']} {e['src']}" for k, e in entries
              if not Path(e["src"]).is_file()]
    assert not absent, (
        f"{field}: explicitly-listed release source(s) are gone from disk -- staging "
        f"this field would ship a set short those bands: " + "; ".join(absent))
