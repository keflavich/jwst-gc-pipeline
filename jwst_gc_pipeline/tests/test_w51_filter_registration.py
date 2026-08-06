"""w51's filter list must cover the bands that actually have jw06151 data.

Issue #160: F444W and F150W hold genuine jw06151001001 frames but were absent
from the registry, so the driver never produced their catalogs -- w51/catalogs
carried 5 f444w and 2 f150w products (hand-run) against 71 for f480m.
"""
import pytest

from jwst_gc_pipeline.fields import obs_filters


def test_w51_registers_f444w_and_f150w():
    filters = obs_filters()["w51"]["6151"]
    assert "f444w" in filters, "issue #160: w51/F444W holds jw06151001001 frames"
    assert "f150w" in filters


def test_w51_filters_are_unique():
    filters = obs_filters()["w51"]["6151"]
    assert len(filters) == len(set(filters))

@pytest.mark.parametrize("band", ["f444w", "f150w"])
def test_registered_bands_have_data_if_the_tree_is_present(band):
    """Guard against registering a band with no data.  Skips off-cluster."""
    import glob
    import os
    root = "/orange/adamginsburg/jwst/w51"
    if not os.path.isdir(root):
        pytest.skip("w51 tree not present")
    frames = glob.glob(f"{root}/{band.upper()}/pipeline/jw06151*_align.fits")
    assert frames, f"{band} registered but no jw06151 frames under {root}"
