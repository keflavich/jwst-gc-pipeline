#!/bin/bash
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --time=01:30:00
#SBATCH --array=0-23
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capbind/log1_%a.txt
cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capbind
# 0-7 LW (250M x4, 300M x4); 8-15 F150W (b1,b3 x 4); 16-23 F200W
i=$SLURM_ARRAY_TASK_ID
if [ $i -lt 8 ]; then
  B=(250M 250M 250M 250M 300M 300M 300M 300M); BAND=${B[$i]}; DET=nrcblong; E=$(( i % 4 + 1 ))
else
  j=$(( i - 8 )); if [ $j -lt 8 ]; then BAND=150W; else BAND=200W; j=$(( j - 8 )); fi
  if [ $(( j / 4 )) -eq 0 ]; then DET=nrcb1; else DET=nrcb3; fi; E=$(( j % 4 + 1 ))
fi
echo $BAND $DET $E
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python -u stage1.py $BAND $DET $E
