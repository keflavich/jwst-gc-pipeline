#!/bin/bash
# Stage a scratch basepath for a NON-DESTRUCTIVE full-frame cataloging run.
#
# Builds <SCRATCH>/ mirroring <REAL>/ but symlinking ONLY the read-only INPUT
# products (crf/cal/i2d/data_i2d frames, regions_, offsets, reduction, refcats)
# and leaving every generated OUTPUT (mergedcat*, *_satstar_catalog.fits, per-frame
# daophot, catalogs/*) to be written fresh under SCRATCH.  Satstar catalogs are
# deliberately NOT linked so they regenerate with current code (fresh sat_area).
#
# Run cataloging with GC_BASEPATH_OVERRIDE=<SCRATCH> so basepath resolves to it.
#
# Usage: stage_scratch_basepath.sh <REAL_BASEPATH> <SCRATCH_BASEPATH> <SUFFIX> <FILTER...>
#   e.g. stage_scratch_basepath.sh /orange/adamginsburg/jwst/w51 \
#          /orange/adamginsburg/jwst/w51_craterexp align_o001_crf F480M F444W
set -u
REAL=${1:?real basepath}; SCRATCH=${2:?scratch basepath}; SUFFIX=${3:?each-suffix}
shift 3
FILTERS="$@"
REAL=${REAL%/}; SCRATCH=${SCRATCH%/}
echo "REAL=$REAL  SCRATCH=$SCRATCH  SUFFIX=$SUFFIX  FILTERS=$FILTERS"

# CRITICAL: downstream per-frame key parsing does `filename.split('_')[3]` on the
# FULL path, so ANY underscore in the scratch path shifts the detector index and
# trips the collision guard.  Also the final path component must equal the target
# regionname (real basepath's last dir) so target/visit parsing matches in-place.
_real_tail=$(basename "$REAL"); _scr_tail=$(basename "$SCRATCH")
if [[ "$SCRATCH" == *_* ]]; then
  echo "FATAL: SCRATCH path contains '_' -> breaks filename.split('_') detector parse."
  echo "       Use e.g. /orange/adamginsburg/jwst-craterexp/${_real_tail} (no underscores; ends in '${_real_tail}')." >&2
  exit 2
fi
if [ "$_scr_tail" != "$_real_tail" ]; then
  echo "FATAL: SCRATCH must end in the target dir '${_real_tail}' (got '${_scr_tail}')." >&2
  exit 2
fi

link_glob () {  # srcdir dstdir pattern
  local sd="$1" dd="$2" pat="$3" n=0
  [ -d "$sd" ] || return 0
  mkdir -p "$dd"
  shopt -s nullglob
  for f in "$sd"/$pat; do
    ln -sf "$f" "$dd/$(basename "$f")" && n=$((n+1))
  done
  shopt -u nullglob
  echo "    linked $n  ($pat)"
}

# Per-filter pipeline INPUTS.
for FILT in $FILTERS; do
  rp="$REAL/$FILT/pipeline"; sp="$SCRATCH/$FILT/pipeline"
  echo "  [$FILT] staging inputs -> $sp"
  link_glob "$rp" "$sp" "*_${SUFFIX}.fits"      # per-exposure crf frames
  link_glob "$rp" "$sp" "*_cal.fits"            # cal frames (asn input)
  link_glob "$rp" "$sp" "*_data_i2d.fits"       # deep coadd / seed-gate grid
  link_glob "$rp" "$sp" "*-${FILT,,}-*_i2d.fits" # mosaic grid (lower-case filt)
  link_glob "$rp" "$sp" "*_i2d.fits"            # any other i2d grids
  # The broad *_i2d.fits glob catches generated mergedcat/model/residual i2d
  # OUTPUTS too -- purge them so a rerun writes them fresh (never through a
  # symlink to a released product).  Also drop any satstar-catalog links.
  find "$sp" -maxdepth 1 -type l \( -name "*mergedcat*" -o -name "*_model_i2d.fits" \
      -o -name "*_residual_i2d.fits" -o -name "*_residual_smoothed_bg_i2d.fits" \
      -o -name "*_satstar_catalog.fits" -o -name "*_extended_satstar_catalog.fits" \) -delete
done

# Shared read-only INPUT dirs: whole-dir symlinks are fine (never written).
for d in regions_ offsets reduction; do
  if [ -d "$REAL/$d" ] && [ ! -e "$SCRATCH/$d" ]; then
    ln -s "$REAL/$d" "$SCRATCH/$d"; echo "  linked dir $d"
  fi
done

# psfs: link EXISTING grids into a REAL scratch dir so a missing-grid regen lands
# in scratch (never clobbers the shared cache).
mkdir -p "$SCRATCH/psfs"
link_glob "$REAL/psfs" "$SCRATCH/psfs" "*.fits"

# catalogs: link only refcats + partner consolidated satstar seeds (inputs);
# all merged/vetted/consolidated OUTPUTS write fresh into this real dir.
mkdir -p "$SCRATCH/catalogs"
link_glob "$REAL/catalogs" "$SCRATCH/catalogs" "gaia_virac2_refcat*.fits"
link_glob "$REAL/catalogs" "$SCRATCH/catalogs" "gaia_refcat*.fits"
link_glob "$REAL/catalogs" "$SCRATCH/catalogs" "nircam_bootstrapped_to_vvv_refcat*.fits"
link_glob "$REAL/catalogs" "$SCRATCH/catalogs" "*_consolidated_satstar_catalog.fits"

echo "STAGE DONE: $SCRATCH"
