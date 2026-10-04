#!/bin/bash
#SBATCH --job-name=faint1019-localsky-analysis
#SBATCH --qos=astronomy-dept-b
#SBATCH --account=astronomy-dept
#SBATCH --partition=hpg-default,hpg-milan,hpg-turin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24gb
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --output=analyze_lsky_%A_%a.log
# After replay_lsky.sbatch (submit from this directory): realness of the
# additions (compare.py), sliced by S/N, prominence and qfit (slices.py), and
# the added-source galleries (propresid.py builds the effective PSF the
# galleries subtract, then added_gallery.py; Brick and Sgr B2).
F=(brick sgrb2 w51)
f=${F[$SLURM_ARRAY_TASK_ID]}
cd "${SLURM_SUBMIT_DIR}"
python compare.py $f
python slices.py $f lsky
if [ "$f" != w51 ]; then
    python propresid.py $f
    python added_gallery.py $f lsky added_${f}_lsky.png 4 added
fi
