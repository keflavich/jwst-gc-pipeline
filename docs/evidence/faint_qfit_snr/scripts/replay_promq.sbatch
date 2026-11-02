#!/bin/bash
#SBATCH --job-name=faint1017-qfit-noise-bound-replay
#SBATCH --qos=astronomy-dept-b
#SBATCH --account=astronomy-dept
#SBATCH --partition=hpg-default,hpg-milan,hpg-turin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --array=0-11
#SBATCH --output=replay_promq_%A_%a.log
# Full-field m6 vetting replay for #1017 on its base (submit from this scripts/ directory):
#   WT_BASE=<worktree at pr/faint-reference-fields ef01f404> WT=<worktree at this branch> sbatch replay_promq.sbatch
# Writes out/<field>_<band>_<variant>.fits: the input m6 merged catalog plus a
# `kept` column and the vetting diagnostics.  base1016 = the base at its
# defaults; promq5b = this branch at its defaults (k = 5); promq0b = k = 0;
# promqeb = k = 1e-6, where the bound is the flat qfit_max, so promq0b & ~promqeb
# is every source whose keep depends on the bound.
# The published run was jobs 44497766 (base1016, promq5b, promq0b) and 44501077 (promqeb).
: "${WT_BASE:?set WT_BASE}" "${WT:?set WT}"
NAMES=(base1016 promq5b promq0b promqeb)
OPTS=('{}' '{}' '{"manual_ext_qfit_snr_k": 0.0}' '{"manual_ext_qfit_snr_k": 1e-6}')
FIELDS=(brick:F182M sgrb2:F187N w51:F187N)
i=$((SLURM_ARRAY_TASK_ID % 4))
fb=${FIELDS[$((SLURM_ARRAY_TASK_ID / 4))]}
f=${fb%%:*}; b=${fb#*:}
wt=$WT; [ "${NAMES[$i]}" = base1016 ] && wt=$WT_BASE
echo "variant ${NAMES[$i]} opts ${OPTS[$i]} field $f band $b worktree $wt host $(hostname)"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p out
REPLAY_OPTS="${OPTS[$i]}" python vet_variant.py "$wt" $f $b out/${f}_${b,,}_${NAMES[$i]}.fits | head -c 2000
