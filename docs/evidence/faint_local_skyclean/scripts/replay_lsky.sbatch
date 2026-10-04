#!/bin/bash
#SBATCH --job-name=faint1019-localsky-replay
#SBATCH --qos=astronomy-dept-b
#SBATCH --account=astronomy-dept
#SBATCH --partition=hpg-default,hpg-milan,hpg-turin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=replay_lsky_%A_%a.log
# Full-field m6 vetting replay on the production merged catalogs (submit from
# this directory; writes out/<field>_<band>_<variant>.fits):
#   WT_SEED=<worktree at the #1015 tip> WT_LSKY=<worktree at this branch> sbatch replay_lsky.sbatch
# seed = the #1015 base, lsky = this branch; both at pipeline defaults.
: "${WT_SEED:?set WT_SEED}" "${WT_LSKY:?set WT_LSKY}"
NAMES=(seed lsky)
WTS=("$WT_SEED" "$WT_LSKY")
FIELDS=(brick:F182M sgrb2:F187N w51:F187N)
i=$((SLURM_ARRAY_TASK_ID % 2))
fb=${FIELDS[$((SLURM_ARRAY_TASK_ID / 2))]}
f=${fb%%:*}; b=${fb#*:}
cd "${SLURM_SUBMIT_DIR}"
mkdir -p out
python vet_variant.py "${WTS[$i]}" $f $b out/${f}_${b,,}_${NAMES[$i]}.fits | head -c 2000
