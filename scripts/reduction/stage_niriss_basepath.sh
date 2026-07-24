#!/bin/bash
# ---------------------------------------------------------------------------
# Stage the NIRISS cataloging basepath: <target>/niriss/ .
#
# NIRISS reduced products live under <target>/niriss/<FILTER>/pipeline/ (so they
# do not collide with the NIRCam <target>/<FILTER> trees).  The cataloging code
# (crowdsource_catalogs_long.py, with --instrument niriss) sets
# basepath=<target>/niriss/ , so it expects the usual support trees THERE too.
# This stages them:
#   reduction/ regions_/ psfs/  -> symlinks to the shared <target>/ trees
#                                  (read-only inputs; reduction/ has the NIRISS
#                                  fwhm table, psfs/ is the shared PSF cache)
#   offsets/                    -> REAL dir (NIRISS-specific; the m2 astrometry
#                                  checkpoint seeds its own consensus table here,
#                                  which must NOT collide with the NIRCam
#                                  <target>/offsets Offsets_JWST_Brick<prop>_*.csv)
#   catalogs/                   -> REAL dir (NIRISS per-band/merged catalogs go
#                                  here, separate from NIRCam) with the shared
#                                  absolute refcat(s) symlinked in so the m2
#                                  checkpoint / tweak find them.
#
# Usage: stage_niriss_basepath.sh <target>   (default sgrc)
# ---------------------------------------------------------------------------
set -euo pipefail
TARGET="${1:-sgrc}"
ROOT="/orange/adamginsburg/jwst/${TARGET}"
NIS="${ROOT}/niriss"

if [ ! -d "$ROOT" ]; then echo "no such target root: $ROOT" >&2; exit 1; fi
mkdir -p "$NIS"

# read-only shared input trees -> symlinks
for d in reduction regions_ psfs; do
    if [ -e "${ROOT}/${d}" ] && [ ! -e "${NIS}/${d}" ]; then
        ln -s "../${d}" "${NIS}/${d}"
        echo "symlink ${NIS}/${d} -> ../${d}"
    fi
done

# NIRISS-specific output trees -> real dirs
mkdir -p "${NIS}/offsets" "${NIS}/catalogs"
echo "real dir ${NIS}/offsets"
echo "real dir ${NIS}/catalogs"

# absolute reference catalog(s): symlink the shared refcat file(s) into the
# NIRISS catalogs dir so the m2 astrometry checkpoint resolves them, without
# copying the (large) files.
shopt -s nullglob
for f in "${ROOT}/catalogs/"gaia_virac2_refcat_epoch*.fits "${ROOT}/catalogs/"*reference_astrometric_catalog.fits "${ROOT}/catalogs/"twomass.fits; do
    b="$(basename "$f")"
    if [ ! -e "${NIS}/catalogs/${b}" ]; then
        ln -s "../../catalogs/${b}" "${NIS}/catalogs/${b}"
        echo "symlink ${NIS}/catalogs/${b} -> ../../catalogs/${b}"
    fi
done

echo "staged NIRISS basepath at ${NIS}"
ls -la "${NIS}"
