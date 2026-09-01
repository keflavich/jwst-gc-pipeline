"""cloudef / cloudef_controlfield are registered for release, and stay separated.

JWST 2092 imaged the Cloud E/F ridge (observation 002) and an offset control
field (observation 005) with the same four NIRCam bands.  Two things have to
hold for either to ship:

1. both are in ``stage_release.FIELDS`` -- ``--field`` is ``choices=sorted(FIELDS)``,
   so an unregistered field cannot be staged at all no matter how finished it is
   (issue #604);
2. the two never mix.  They are separate sky, and the ``cloudef`` tree also
   holds a few stray ``o005`` mosaics, so a release scoped by proposal alone
   would ship the control field's images as if they were the ridge's.

MIRI is withheld from both deliberately (issue #596): 2092's F770W/F2100W were
never reduced, so neither entry carries a ``miri:`` key.
"""
import sys
from pathlib import Path

import pytest

_RELEASE = Path(__file__).resolve().parents[2] / "scripts" / "release"
if str(_RELEASE) not in sys.path:
    sys.path.insert(0, str(_RELEASE))

stage_release = pytest.importorskip("stage_release")

CLOUDEF_FIELDS = ("cloudef", "cloudef_controlfield")


@pytest.mark.parametrize("field", CLOUDEF_FIELDS)
def test_field_is_registered(field):
    """Unregistered means unstageable: ``--field`` is ``choices=sorted(FIELDS)``."""
    assert field in stage_release.FIELDS, (
        f"{field} is finished on disk but absent from stage_release.FIELDS, so "
        "`stage_release --field` will reject it as an invalid choice (#604)"
    )


@pytest.mark.parametrize("field,expected", [
    ("cloudef", "jw02092-o002_t001_nircam_clear"),
    ("cloudef_controlfield", "jw02092-o005_t001_nircam_clear"),
])
def test_prefix_is_scoped_to_its_own_observation(field, expected):
    """The prefix carries the observation, which is what separates the two fields."""
    cfg = stage_release.FIELDS[field]
    assert cfg["proposal_prefix"] == expected


@pytest.mark.parametrize("field,expected", [
    ("cloudef", "cloudef"),
    ("cloudef_controlfield", "cloudef_controlfield"),
])
def test_data_dir_is_its_own_tree(field, expected):
    assert stage_release.FIELDS[field]["data_dir"].name == expected


def test_the_two_fields_do_not_share_a_prefix_or_a_tree():
    a, b = (stage_release.FIELDS[f] for f in CLOUDEF_FIELDS)
    assert a["proposal_prefix"] != b["proposal_prefix"]
    assert a["data_dir"] != b["data_dir"]


@pytest.mark.parametrize("field", CLOUDEF_FIELDS)
def test_miri_is_withheld(field):
    """2092's MIRI was never reduced -- shipping the 2024 MAST i2d is not the fix."""
    assert not stage_release.FIELDS[field].get("miri"), (
        f"{field} declares MIRI products, but 2092 F770W/F2100W have never been "
        "reduced (#596); withhold the band rather than shipping archive files"
    )


@pytest.mark.parametrize("field", CLOUDEF_FIELDS)
def test_no_per_observation_catalog_token(field):
    """Each cloudef field is ONE observation, so its catalogs carry no `_oNNN` token.

    Contrast brick, where two proposals image the same sky and the token is what
    keeps their catalogs apart.
    """
    assert not stage_release.FIELDS[field].get("observations"), (
        f"{field} is a single-observation field; an `observations` list would "
        "make discover_images tag its products per observation"
    )


@pytest.mark.parametrize("field", CLOUDEF_FIELDS)
def test_discovery_only_returns_this_fields_observation(field):
    """The real guard: stray o005 mosaics sit in the cloudef tree.

    Skips when the archive is not mounted, so this stays runnable off-cluster.
    """
    cfg = stage_release.FIELDS[field]
    if not cfg["data_dir"].exists():
        pytest.skip(f"{cfg['data_dir']} not mounted")

    obs = cfg["proposal_prefix"].split("_")[0]           # e.g. jw02092-o002
    siblings = {stage_release.FIELDS[f]["proposal_prefix"].split("_")[0]
                for f in CLOUDEF_FIELDS}
    assert obs in siblings and len(siblings) == 2, (
        f"{field}'s prefix {cfg['proposal_prefix']!r} does not name one of the two "
        f"cloudef observations; a prefix that spans both would ship the control "
        f"field's mosaics as the ridge's"
    )
    other = (siblings - {obs}).pop()

    names = [Path(str(getattr(item, "src", item))).name
             for item in stage_release.discover_images(cfg)]
    if not names:
        pytest.skip("no mosaics on disk yet")

    assert all(obs in n for n in names), \
        f"{field} discovered a product outside {obs}: {[n for n in names if obs not in n]}"
    assert not any(other in n for n in names), \
        f"{field} discovered the OTHER cloudef field's products: {[n for n in names if other in n]}"
