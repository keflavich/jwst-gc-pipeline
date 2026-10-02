#!/bin/bash
#SBATCH --job-name=faint1021-exempt-rebased-replay
#SBATCH --qos=astronomy-dept-b
#SBATCH --account=astronomy-dept
#SBATCH --partition=hpg-default,hpg-milan,hpg-turin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=replay_fix8r_%A_%a.log
# Full-field m6 vetting replay of this branch (pipeline defaults) on the
# production merged catalogs.  Submit from
# docs/evidence/faint_snr_floor_propagated/scripts/, which holds
# vet_variant.py and the #1016 replay outputs (out/*_snrp2.fits):
#   WT=<worktree at this branch's tip> sbatch replay_exempt.sbatch
# then compare row for row with compare_replay.py.
: "${WT:?set WT}"
FIELDS=(w51:F187N brick:F182M)
fb=${FIELDS[$SLURM_ARRAY_TASK_ID]}
f=${fb%%:*}; b=${fb#*:}
cd "${SLURM_SUBMIT_DIR}"
mkdir -p out
REPLAY_OPTS='{}' python vet_variant.py "$WT" $f $b out/${f}_${b,,}_fix8r.fits | head -c 2000
