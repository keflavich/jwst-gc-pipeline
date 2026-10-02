#!/bin/bash
#SBATCH --job-name=faint1018-replay-realness-galleries
#SBATCH --qos=astronomy-dept-b
#SBATCH --account=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=compare_prom2_%A_%a.log
# After replay_prom2.sbatch: realness tables (compare_<field>.json,
# slices_<field>_<variants>.json; copies in ../data/) and, for Brick and
# Sgr B2, the added / lost galleries.  Submit from this scripts/ directory.
F=(brick sgrb2 w51)
f=${F[$SLURM_ARRAY_TASK_ID]}
cd "${SLURM_SUBMIT_DIR}"
python compare.py $f
if [ $f = w51 ]; then
    python slices.py $f prom2 prom2p3
else
    python slices.py $f prom2 prom2p3 prom2p5
    python propresid.py $f   # i2d ePSF stamp for the galleries' proposed residuals
    mkdir -p gal
    python added_gallery.py $f prom2 gal/added_${f}_prom2.png 4 added
    python added_gallery.py $f prom2 gal/lost_${f}_prom2.png 4 lost
fi
