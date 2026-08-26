"""A staged merged mosaic must not predate the per-module mosaics it combines.

The merged mosaic is the only product in which NIRCam module A and module B are
combined, so the inter-module seam exists nowhere else.  A merged mosaic left behind
by a re-drizzle of its own modules therefore ships a seam built from a generation of
the modules the release does not contain -- and it is invisible to
``check_generation_span``, which compares the staged BANDS against each other: one
band, one instrument, one CRDS context, DATE span zero.

Measured on disk 2026-08-25, from the DATE headers rather than mtimes:

    wd1   F200W  nrca 2026-07-01T04:08  nrcb 2026-07-01T04:26  merged 2026-06-13T13:27
    brick F356W  nrca 2026-08-22T21:44                         merged 2026-08-19T13:46

Both stage today with no generation complaint.

Every test here runs on a synthetic tree of one-pixel FITS stand-ins; nothing reads or
writes the archive.
"""
import importlib.util
from pathlib import Path

from astropy.io import fits

_SPEC = importlib.util.spec_from_file_location(
    "stage_release",
    Path(__file__).resolve().parents[3] / "scripts" / "release" / "stage_release.py")
stage_release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stage_release)

PREFIX = "jw09999-o001_t001_nircam_clear"


def _mosaic(path, date=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU()
    if date is not None:
        hdu.header["DATE"] = date
    hdu.writeto(path, overwrite=True)
    return path


def _band(tmp_path, filt, merged_date, module_dates, modules=("nrca", "nrcb")):
    """Write one band's merged + per-module mosaics; return the staged science item."""
    pipeline = tmp_path / filt.upper() / "pipeline"
    merged = _mosaic(pipeline / f"{PREFIX}-{filt}-merged_i2d.fits", merged_date)
    for module, when in zip(modules, module_dates):
        _mosaic(pipeline / f"{PREFIX}-{filt}-{module}_i2d.fits", when)
    return {"category": "image", "kind": "science", "filter": filt.upper(),
            "iteration": None, "observation": None, "instrument": "NIRCam",
            "src": str(merged)}


def test_merged_older_than_its_modules_is_reported(tmp_path, capsys):
    """The wd1 F200W state: modules re-drizzled 2026-07-01, merged still 2026-06-13."""
    item = _band(tmp_path, "f200w", "2026-06-13T13:27:19.695",
                 ["2026-07-01T04:08:13.387", "2026-07-01T04:26:17.739"])
    complaints = stage_release.check_merged_mosaic_generation([item])
    assert len(complaints) == 1, complaints
    text = complaints[0]
    assert "F200W" in text
    assert "17.6 d OLDER" in text
    # names WHICH module it is behind, so the re-drizzle target is unambiguous
    assert "nrcb" in text


def test_generation_span_alone_does_not_see_it(tmp_path):
    """The reason this check exists: the span leg reads the same tree as clean.

    Mutation guard -- delete ``check_merged_mosaic_generation`` and this state ships
    with no complaint from anything.
    """
    item = _band(tmp_path, "f200w", "2026-06-13T13:27:19.695",
                 ["2026-07-01T04:08:13.387", "2026-07-01T04:26:17.739"])
    assert stage_release.check_generation_span([item], verbose=False) == []
    assert stage_release.check_merged_mosaic_generation([item], verbose=False)


def test_merged_written_after_its_modules_passes(tmp_path):
    """One Image3 batch: the merged product is written after the modules it combines."""
    item = _band(tmp_path, "f356w", "2026-08-22T22:10:00.000",
                 ["2026-08-22T21:44:11.300", "2026-08-22T21:52:00.000"])
    assert stage_release.check_merged_mosaic_generation([item], verbose=False) == []


def test_slack_absorbs_a_slow_batch_but_not_a_generation(tmp_path):
    """0.5 d of lag is a long drizzle; 3.3 d is brick F356W's real stale merged."""
    slow = _band(tmp_path / "slow", "f405n", "2026-08-22T09:00:00.000",
                 ["2026-08-22T18:00:00.000"], modules=("nrca",))
    assert stage_release.check_merged_mosaic_generation([slow], verbose=False) == []
    stale = _band(tmp_path / "stale", "f356w", "2026-08-19T13:46:56.115",
                  ["2026-08-22T21:44:11.300"], modules=("nrca",))
    complaints = stage_release.check_merged_mosaic_generation([stale], verbose=False)
    assert len(complaints) == 1 and "3.3 d OLDER" in complaints[0]


def test_long_module_spellings_are_compared(tmp_path):
    """LW bands are `nrcalong`/`nrcblong`; a token list missing them fails open."""
    item = _band(tmp_path, "f480m", "2026-06-13T13:27:19.695",
                 ["2026-07-01T04:08:13.387", "2026-07-01T04:26:17.739"],
                 modules=("nrcalong", "nrcblong"))
    complaints = stage_release.check_merged_mosaic_generation([item], verbose=False)
    assert len(complaints) == 1 and "nrcblong" in complaints[0]


def test_single_module_band_has_nothing_to_compare(tmp_path):
    """sickle has no module A, so no merged product and no seam to be stale about.

    A band whose merged mosaic has no per-module sibling on disk is silent rather than
    a complaint -- the missing-product case is a different gate.
    """
    pipeline = tmp_path / "F210M" / "pipeline"
    merged = _mosaic(pipeline / f"{PREFIX}-f210m-merged_i2d.fits",
                     "2026-06-13T13:27:19.695")
    item = {"category": "image", "kind": "science", "filter": "F210M",
            "iteration": None, "observation": None, "instrument": "NIRCam",
            "src": str(merged)}
    assert stage_release.check_merged_mosaic_generation([item], verbose=False) == []


def test_undated_products_are_reported_as_not_compared_not_as_a_pass(tmp_path, capsys):
    """A missing DATE must not read as agreement."""
    item = _band(tmp_path, "f212n", None, ["2026-07-01T04:08:13.387"],
                 modules=("nrca",))
    complaints = stage_release.check_merged_mosaic_generation([item])
    assert complaints == []
    assert "NOT COMPARED" in capsys.readouterr().out


def test_non_science_items_are_ignored(tmp_path):
    """Residual/model i2d products are per-module byproducts, not the seam."""
    item = _band(tmp_path, "f182m", "2026-06-13T13:27:19.695",
                 ["2026-07-01T04:08:13.387"], modules=("nrca",))
    item["kind"] = "residual"
    assert stage_release.check_merged_mosaic_generation([item], verbose=False) == []
