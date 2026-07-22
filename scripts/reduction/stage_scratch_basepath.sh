#!/bin/bash
# Stage a scratch basepath for a NON-DESTRUCTIVE reduce/catalog run.
#
# Builds <SCRATCH>/ mirroring <REAL>/ but staging ONLY read-only INPUT products,
# so every generated OUTPUT is written fresh under SCRATCH and the released tree
# is never touched.  Run the pipeline with GC_BASEPATH_OVERRIDE=<SCRATCH>.
#
# SAFETY DESIGN (see PR #143 review):
#   * ALLOWLIST, never denylist.  We link only the precise INPUT suffixes the
#     pipeline reads; we never link `*_i2d.fits` broadly (the pipeline writes
#     many `*_i2d.fits` OUTPUTS -- _residual_infilled_i2d, _satstar_flags_i2d,
#     mergedcat_*_i2d, make_starless/vetted i2dseed -- and `overwrite=True` on a
#     symlink writes THROUGH into the released tree).
#   * COPY, never symlink, anything the pipeline may rewrite in place: the
#     `offsets/` tables (the m2 checkpoint does os.replace + write(overwrite=True))
#     and the `data_i2d` grids.
#   * A final self-check FAILS the stage if any symlink under SCRATCH resolves
#     into REAL while matching a known output pattern.
#
# Usage: stage_scratch_basepath.sh <REAL_BASEPATH> <SCRATCH_BASEPATH> <MODE> <SUFFIX> <FILTER...>
#   MODE = reduce  -> only *_cal.fits linked (crf/i2d are OUTPUTS of the reduction)
#   MODE = catalog -> *_<SUFFIX>.fits (crf) + *_cal.fits linked, data_i2d copied
#   e.g. stage_scratch_basepath.sh /orange/adamginsburg/jwst/w51 \
#          /orange/adamginsburg/jwst-craterexp/w51 reduce align_o001_crf F480M
set -u
REAL=${1:?real basepath}; SCRATCH=${2:?scratch basepath}; MODE=${3:?mode reduce|catalog}
SUFFIX=${4:?each-suffix}; shift 4
FILTERS="$@"
REAL=${REAL%/}; SCRATCH=${SCRATCH%/}
case "$MODE" in reduce|catalog) ;; *) echo "FATAL: MODE must be reduce|catalog (got '$MODE')" >&2; exit 2;; esac
echo "REAL=$REAL  SCRATCH=$SCRATCH  MODE=$MODE  SUFFIX=$SUFFIX  FILTERS=$FILTERS"

# Path-parse guard: downstream per-frame keys do `filename.split('_')[3]`, so an
# underscore ANYWHERE in the path shifts the detector index.  Both REAL and
# SCRATCH must be underscore-free and end in the same target dir.
_real_tail=$(basename "$REAL"); _scr_tail=$(basename "$SCRATCH")
for p in "$REAL" "$SCRATCH"; do
  case "$p" in *_*) echo "FATAL: path '$p' contains '_' -> breaks filename.split('_') detector parse." >&2; exit 2;; esac
done
if [ "$_scr_tail" != "$_real_tail" ]; then
  echo "FATAL: SCRATCH must end in the target dir '${_real_tail}' (got '${_scr_tail}')." >&2; exit 2
fi

link_glob () {  # srcdir dstdir pattern   (symlink read-only INPUTS)
  local sd="$1" dd="$2" pat="$3" n=0
  [ -d "$sd" ] || return 0
  mkdir -p "$dd"; shopt -s nullglob
  for f in "$sd"/$pat; do ln -sf "$f" "$dd/$(basename "$f")" && n=$((n+1)); done
  shopt -u nullglob; echo "    linked $n  ($pat)"
}
copy_glob () {  # srcdir dstdir pattern   (COPY things the pipeline may overwrite)
  local sd="$1" dd="$2" pat="$3" n=0
  [ -d "$sd" ] || return 0
  mkdir -p "$dd"; shopt -s nullglob
  for f in "$sd"/$pat; do cp -f "$f" "$dd/$(basename "$f")" && n=$((n+1)); done
  shopt -u nullglob; echo "    copied $n  ($pat)"
}

# --- per-filter pipeline INPUTS (ALLOWLIST) ---
for FILT in $FILTERS; do
  rp="$REAL/$FILT/pipeline"; sp="$SCRATCH/$FILT/pipeline"
  echo "  [$FILT] staging ($MODE) -> $sp"
  link_glob "$rp" "$sp" "*_cal.fits"                       # reduction input (both modes)
  if [ "$MODE" = "catalog" ]; then
    link_glob "$rp" "$sp" "*_${SUFFIX}.fits"               # crf: cataloging input
    # data_i2d = deep-coadd astrometric/seed reference the m2 checkpoint reads.
    # COPY (never link): keep it writable-in-scratch and decoupled from REAL.
    copy_glob "$rp" "$sp" "*_data_i2d.fits"
  fi
  # NB: crf/*_i2d.fits are OUTPUTS in reduce mode and are NOT staged; the
  # reduction regenerates them fresh under SCRATCH.
done

# --- shared read-only INPUT dirs (never written by the pipeline) ---
for d in regions_ reduction; do
  if [ -d "$REAL/$d" ] && [ ! -e "$SCRATCH/$d" ]; then
    ln -s "$REAL/$d" "$SCRATCH/$d"; echo "  linked dir $d (read-only)"
  fi
done

# --- offsets/: the m2 checkpoint REWRITES these in place -> COPY, never link ---
if [ -d "$REAL/offsets" ]; then
  copy_glob "$REAL/offsets" "$SCRATCH/offsets" "*.csv"
fi

# --- psfs: load-or-make only WRITES a grid when ABSENT (never overwrite=True on
# an existing one), so symlinking existing grids is read-only in practice; a
# missing grid is generated into the real SCRATCH/psfs dir, never the cache. ---
mkdir -p "$SCRATCH/psfs"
link_glob "$REAL/psfs" "$SCRATCH/psfs" "*.fits"

# --- catalogs: refcats are read-only INPUTS; partner-band consolidated satstar
# seeds are inputs for OTHER filters only (a filter writes its OWN consolidated
# file).  COPY everything here -- a copy is safe, and it keeps the satstar-catalog
# self-check below strict (no *_satstar_catalog.fits symlink into REAL). ---
mkdir -p "$SCRATCH/catalogs"
copy_glob "$REAL/catalogs" "$SCRATCH/catalogs" "gaia_virac2_refcat*.fits"
copy_glob "$REAL/catalogs" "$SCRATCH/catalogs" "gaia_refcat*.fits"
copy_glob "$REAL/catalogs" "$SCRATCH/catalogs" "nircam_bootstrapped_to_vvv_refcat*.fits"
shopt -s nullglob
for f in "$REAL"/catalogs/*_consolidated_satstar_catalog.fits; do
  b=$(basename "$f"); own=0
  for FILT in $FILTERS; do
    lf=$(echo "$FILT" | tr 'A-Z' 'a-z')
    [ "$b" = "${lf}_consolidated_satstar_catalog.fits" ] && own=1
  done
  [ "$own" = 0 ] && cp -f "$f" "$SCRATCH/catalogs/$b"   # partner seed (input) -> copy
done
shopt -u nullglob

# --- SAFETY SELF-CHECK: no symlink under SCRATCH may resolve into REAL while
# matching an output pattern the pipeline writes (i2d outputs, mergedcat, satstar
# catalogs, offsets csv).  This is the invariant the whole staging exists to hold. ---
leak=$(find "$SCRATCH" -type l 2>/dev/null | while read -r l; do
  tgt=$(readlink -f "$l" 2>/dev/null); b=$(basename "$l")
  case "$tgt" in "$REAL"/*) ;; *) continue;; esac
  case "$b" in
    *_i2d.fits|*mergedcat*|*_satstar_catalog.fits|*_extended_satstar_catalog.fits|*.csv) echo "$l -> $tgt";;
  esac
done)
if [ -n "$leak" ]; then
  echo "FATAL: staging left symlink(s) into REAL matching output patterns (would clobber released products):" >&2
  echo "$leak" >&2
  exit 3
fi
echo "STAGE DONE (safe): $SCRATCH"
