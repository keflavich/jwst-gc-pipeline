"""Centralized gridded-PSF storage paths.

Gridded PSFs are determined by the physics (instrument, detector/module,
filter, oversampling, blur) and do NOT change from field to field or proposal
to proposal, yet the legacy layout stored one copy per (proposal, field):

    {jwst_root}/{target}/psfs/{FILTER}_{prop}_{field}_merged_PSFgrid_oversample{N}[_blur].fits

That duplicates multi-hundred-MB grids across every field.  This module defines
a single shared store keyed only by the determining factors:

    {jwst_root}/psfs_shared/{inst}_{module}_{FILTER}_PSFgrid_oversample{N}[_blur].fits

Resolution is central-first with a fallback to the legacy per-field path so
existing grids keep loading with no migration; new grids should be written to
the central path (see ``central_*`` helpers).  Pure path logic, no astronomy
deps, so it is cheap to import and unit-test.
"""
import os

PSFS_SHARED_DIRNAME = 'psfs_shared'


def _blur_token(blur):
    return '_blur' if blur else ''


def central_psf_dir(jwst_root):
    """The shared PSF store directory (sibling of the per-target trees)."""
    return os.path.join(jwst_root, PSFS_SHARED_DIRNAME)


def central_merged_psf_grid_path(jwst_root, instrument, module, filtername,
                                 oversample=1, blur=False):
    """Central path for a merged gridded-PSF, keyed only by the physics."""
    inst = str(instrument).lower()
    name = (f'{inst}_{module}_{filtername.upper()}_PSFgrid'
            f'_oversample{oversample}{_blur_token(blur)}.fits')
    return os.path.join(central_psf_dir(jwst_root), name)


def legacy_merged_psf_grid_path(jwst_root, target, filtername, proposal_id,
                                field, oversample=1, blur=False):
    """The historical per-(proposal, field) merged gridded-PSF path."""
    name = (f'{filtername.upper()}_{proposal_id}_{field}_merged_PSFgrid'
            f'_oversample{oversample}{_blur_token(blur)}.fits')
    return os.path.join(jwst_root, target, 'psfs', name)


def resolve_merged_psf_grid_path(jwst_root, target, instrument, module,
                                 filtername, proposal_id, field,
                                 oversample=1, blur=False, exists=os.path.exists):
    """Return the merged gridded-PSF path to load.

    Read order: the central shared path if it exists, else the legacy per-field
    path.  When neither exists, returns the central path (the canonical location
    a missing grid should be created at), so error messages point at the shared
    store rather than a stale per-field name.

    ``exists`` is injectable for testing.
    """
    central = central_merged_psf_grid_path(jwst_root, instrument, module,
                                           filtername, oversample, blur)
    if exists(central):
        return central
    legacy = legacy_merged_psf_grid_path(jwst_root, target, filtername,
                                         proposal_id, field, oversample, blur)
    if exists(legacy):
        return legacy
    return central
