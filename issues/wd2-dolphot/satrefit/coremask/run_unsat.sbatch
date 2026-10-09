#!/bin/bash
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --cpus-per-task=2
#SBATCH --mem=24gb
#SBATCH --time=02:00:00
#SBATCH --array=0-23
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/coremask/log_unsat_%a.txt
cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/coremask
i=${SLURM_ARRAY_TASK_ID}
if [ $i -lt 8 ]; then
  B=(250M 250M 250M 250M 300M 300M 300M 300M); BAND=${B[$i]}; DET=nrcblong; E=$(( i % 4 + 1 ))
else
  if [ $i -lt 16 ]; then BAND=150W; else BAND=200W; fi
  j=$(( (i - 8) % 8 ))
  if [ $(( j / 4 )) -eq 0 ]; then DET=nrcb1; else DET=nrcb3; fi; E=$(( j % 4 + 1 ))
fi
echo $BAND $DET $E
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python -u run_unsat.py $BAND $DET $E
