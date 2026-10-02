#!/bin/bash
#SBATCH --job-name=faint1020-loose-seed-spike-emission-figs
#SBATCH --qos=astronomy-dept-b
#SBATCH --account=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48gb
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=spikefig_%A_%a.log
F=(brick sgrb2); FL=(f182m f187n); M=(nrcb merged)
i=$SLURM_ARRAY_TASK_ID
cd "${SLURM_SUBMIT_DIR}"  # submit from this scripts/ directory
python spikefig.py ${F[$i]} ${FL[$i]} ${M[$i]} loose_seeds_${F[$i]}.png
