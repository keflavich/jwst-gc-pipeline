#!/bin/bash
# Realness slices, comparison table, low-quality subsets and the added-source
# gallery for this branch over the #1018 v2 replay.  Run from this directory
# after replay_snrp2.sbatch (and the prom2 / seed replays) have written out/.
#SBATCH --job-name=faint1016-snrfloor-on-promguard-analysis
#SBATCH --qos=astronomy-dept-b
#SBATCH --account=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24gb
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --output=logs/analyze_snrp2_%A_%a.log
F=(brick sgrb2 w51)
f=${F[$SLURM_ARRAY_TASK_ID]}
cd "${SLURM_SUBMIT_DIR}"
python slices.py $f snrp2@prom2 snrp2 snr
python compare.py $f
python subsets.py $f
if [ "$f" != w51 ]; then
    python added_gallery.py $f snrp2@prom2 added_${f}_snrp2.png 4
fi
