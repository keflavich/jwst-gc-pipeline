#!/bin/bash
#SBATCH --job-name=cmz-build-hips-cat
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/cmz_build_%x_%A.out
#
# Build the CMZ-wide release products (catalog + coverage MOC + two-color HiPS,
# and optionally HATS) from a spec. The first backfill reprojects the real
# multi-GB CMZ mosaics into HiPS tiles -> a compute job, not a login-node run.
#
# Submit:
#   sbatch --job-name=cmz-build-o001 \
#     scripts/release/submit_cmz_build.sbatch /path/to/cmz_spec.json [--only hips]
#
set -euo pipefail

SPEC="${1:?usage: sbatch submit_cmz_build.sbatch <spec.json> [extra args...]}"
shift || true

export CRDS_PATH="${CRDS_PATH:-/orange/adamginsburg/jwst/crds}"
export CRDS_SERVER_URL="${CRDS_SERVER_URL:-https://jwst-crds.stsci.edu}"
export MPLBACKEND=Agg
export TMPDIR="${SLURM_TMPDIR:-/blue/adamginsburg/adamginsburg/tmp}"

# Repo root = two levels up from this script.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "CMZ build: spec=$SPEC extra='$*'  repo=$REPO"
python "$REPO/scripts/release/build_cmz_products.py" --spec "$SPEC" "$@"
echo "CMZ build done."
