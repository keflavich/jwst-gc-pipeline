#!/bin/bash
#SBATCH --job-name=faint1020-spikeguard-fullframe
#SBATCH --qos=astronomy-dept-b
#SBATCH --account=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=seedspike_%A_%a.log
F=(brick sgrb2); FL=(f182m f187n); M=(nrcb merged)
i=$SLURM_ARRAY_TASK_ID
cd "${SLURM_SUBMIT_DIR}"  # submit from this scripts/ directory
python seedspike.py "${WT:?set WT to a worktree at 6761eea4}" ${F[$i]} ${FL[$i]} ${M[$i]}
