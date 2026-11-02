#!/bin/bash
#SBATCH --job-name=faint1018-peaksb-prom-guard-replay
#SBATCH --qos=astronomy-dept-b
#SBATCH --account=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=replay_prom2_%A_%a.log
# Full-field m6 vetting replay for #1018 (submit from this scripts/ directory):
#   WT_BASE=<worktree at the #1015 tip, 05b6d0a4> WT=<worktree at d29fedb2> sbatch replay_prom2.sbatch
# Writes out/<field>_<band>_<variant>.fits: the input m6 merged catalog plus a
# `kept` column and the vetting diagnostics.  `seed` is the base branch; prom2
# is this branch at its defaults (guard 4); prom2p3 / prom2p5 set guard 3 / 5.
# The W51 prom2p5 task (11) was not part of the published tables.
: "${WT_BASE:?set WT_BASE}" "${WT:?set WT}"
NAMES=(seed prom2 prom2p3 prom2p5)
OPTS=('{}' '{}'
      '{"manual_ext_star_prom_peak_min": 3.0}'
      '{"manual_ext_star_prom_peak_min": 5.0}')
FIELDS=(brick:F182M sgrb2:F187N w51:F187N)
i=$((SLURM_ARRAY_TASK_ID % 4))
fb=${FIELDS[$((SLURM_ARRAY_TASK_ID / 4))]}
f=${fb%%:*}; b=${fb#*:}
wt=$WT; [ "${NAMES[$i]}" = seed ] && wt=$WT_BASE
echo "variant ${NAMES[$i]} opts ${OPTS[$i]} field $f band $b worktree $wt host $(hostname)"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p out
REPLAY_OPTS="${OPTS[$i]}" python vet_variant.py "$wt" $f $b out/${f}_${b,,}_${NAMES[$i]}.fits | head -c 2000
