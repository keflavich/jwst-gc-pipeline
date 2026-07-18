#!/bin/bash
#SBATCH --job-name=m92_deep_wingcal
#SBATCH --output=/blue/adamginsburg/adamginsburg/brick_logs/m92_deep_wingcal_%j.log
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48gb
#SBATCH --time=04:00:00

# M92 sparse-field deep-mask wing-calibration campaign (GO-1334 F090W/F150W,
# 8 SW detectors x 4 frames each): stage 1 builds empirical/model PSF stacks,
# stage 2 measures deep C(r_mask) curves to 26 px.  See m92_deep_stacks.py
# and m92_deep_calcurve.py.

set -u
PY=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STACKS=/orange/adamginsburg/jwst/m92/wingcal_deep_stacks
export PY DIR STACKS OMP_NUM_THREADS=2

mkdir -p "$STACKS"
cd "$DIR"

echo "=== stage 1: stacks ($(date)) ==="
for filt in f090w f150w; do
  for det in nrca1 nrca2 nrca3 nrca4 nrcb1 nrcb2 nrcb3 nrcb4; do
    echo "${filt}:${det}"
  done
done | xargs -P 8 -I CFG bash -c '
  cfg="CFG"; filt="${cfg%%:*}"; det="${cfg##*:}"
  echo "start ${filt} ${det} $(date)"
  "$PY" "$DIR/m92_deep_stacks.py" "$filt" "$det" 4 "$STACKS" \
      > "$STACKS/log_${filt}_${det}.txt" 2>&1
  echo "done ${filt} ${det} $(date) rc=$?"
'

echo "=== stage 2: deep calcurves ($(date)) ==="
"$PY" "$DIR/m92_deep_calcurve.py" "$STACKS" "$DIR/calcurves_deep_m92"
echo "=== finished ($(date)) ==="
