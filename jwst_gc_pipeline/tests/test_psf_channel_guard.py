"""Every PSF build must pass through the channel-safe kwargs.

#586 fixed F150W2's cold-cache crash at the two `get_psf_model` call sites and
missed three others.  The one that mattered --
`reduction.saturated_star_finding` -- is reached by a different path
(`load_or_make_satstar_catalog` -> `remove_saturated_stars`) and killed every
shard of the first m4 and ngc6397 runs (jobs 40691533, 40691535) with the
identical error the earlier fix was supposed to have removed.

A grep-guard rather than a behavioural test: the failure only appears on a COLD
psf cache, so a warm-cache CI run cannot reproduce it and a new unguarded call
site would ship silently.
"""
import pathlib
import re
import subprocess

CALL = re.compile(r'\.(psf_grid|calc_psf)\(')
GUARD = 'nircam_channel_safe_psf_kwargs'
ROOT = pathlib.Path(__file__).resolve().parents[2]

#: A build that constructs a NON-NIRCam instrument needs no channel guard --
#: only NIRCam has the SW/LW dichroic the helper exists for, and it returns {}
#: for anything else.  Exempt by explicit path so a NEW NIRCam site cannot hide
#: behind a broad pattern.
NON_NIRCAM_BUILDS = {
    'scripts/miri_reduction/build_large_psf_grids_miri.py',   # stpsf.MIRI()
}


def _tracked_py():
    out = subprocess.run(['git', 'ls-files', '*.py'], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [ROOT / p for p in out.split() if '/tests/' not in p]


def test_every_psf_build_is_channel_guarded():
    """A psf_grid/calc_psf call needs the guard within a few lines above it."""
    offenders = []
    for path in _tracked_py():
        if str(path.relative_to(ROOT)) in NON_NIRCAM_BUILDS:
            continue
        try:
            lines = path.read_text().splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(lines):
            if not CALL.search(line):
                continue
            window = '\n'.join(lines[max(0, i - 4):i + 4])
            if GUARD not in window:
                offenders.append(f'{path.relative_to(ROOT)}:{i + 1}: {line.strip()}')
    assert not offenders, (
        'PSF build(s) not routed through nircam_channel_safe_psf_kwargs -- a '
        'cold-cache F150W2 build will raise "requested wavelengths are too long '
        'for NIRCam short wave channel" here:\n  ' + '\n  '.join(offenders))


def test_exempt_builds_really_are_non_nircam():
    """An exemption is only valid while that file builds a non-NIRCam PSF."""
    for rel in NON_NIRCAM_BUILDS:
        text = (ROOT / rel).read_text()
        assert 'stpsf.NIRCam(' not in text, (
            f'{rel} is exempt from the channel guard but constructs a NIRCam '
            'instrument; remove the exemption and guard its PSF build')


def test_the_guard_is_importable_without_a_cycle():
    """It lives outside the monolith because reduction needs it too.

    `crowdsource_catalogs_long` imports from `saturated_star_finding`, so
    hosting the helper in the monolith and importing it back would be a cycle.
    """
    from jwst_gc_pipeline.photometry.psf_channel import (
        nircam_channel_safe_psf_kwargs as direct)
    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
        nircam_channel_safe_psf_kwargs as reexport)
    assert direct is reexport
