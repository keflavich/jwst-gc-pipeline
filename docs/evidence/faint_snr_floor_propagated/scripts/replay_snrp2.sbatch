#!/bin/bash
#SBATCH --job-name=faint1016-snrfloor-on-promguard-replay
#SBATCH --qos=astronomy-dept-b
#SBATCH --account=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=replay_snrp2_%A_%a.log
# Full-field m6 vetting replay of this branch on top of #1018 v2 (submit from
# the directory that holds vet_variant.py and out/):
#   WT=<worktree at this branch's tip> sbatch replay_snrp2.sbatch
# snrp2: pipeline defaults (S/N floor on flux_err_prop, peak_SB prominence
# guard 4).  snrp2pf: the same code with --manual-ext-snr-floor-per-frame;
# it must reproduce the #1018 replay (prom2) row for row.
: "${WT:?set WT}"
NAMES=(snrp2 snrp2pf)
OPTS=('{}' '{"manual_ext_snr_floor_propagated": false}')
FIELDS=(brick:F182M sgrb2:F187N w51:F187N)
i=$((SLURM_ARRAY_TASK_ID % 2))
fb=${FIELDS[$((SLURM_ARRAY_TASK_ID / 2))]}
f=${fb%%:*}; b=${fb#*:}
echo "variant ${NAMES[$i]} opts ${OPTS[$i]} field $f band $b worktree $WT host $(hostname)"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p out
REPLAY_OPTS="${OPTS[$i]}" python vet_variant.py "$WT" $f $b out/${f}_${b,,}_${NAMES[$i]}.fits | head -c 2000
