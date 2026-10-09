#!/bin/bash
#SBATCH --job-name=sclip-real
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --time=08:00:00
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=4
#SBATCH --output=/blue/adamginsburg/adamginsburg/tmp/claude-3663/-orange-adamginsburg-jwst-wd2/0e557564-8b2a-47a4-a008-5add86ab6df7/scratchpad/sclip_real/sclip_real_all_%j.log
D=/blue/adamginsburg/adamginsburg/tmp/claude-3663/-orange-adamginsburg-jwst-wd2/0e557564-8b2a-47a4-a008-5add86ab6df7/scratchpad/sclip_real
R=/blue/adamginsburg/adamginsburg/repos
cd $D
export OMP_NUM_THREADS=4 MPLBACKEND=Agg
for b in 115W 150W 162M 182M 200W 335M 410M 164N 187N 212N 323N 405N 466N; do
  echo "== F$b main $(date)"; PYTHONPATH=$R/jwst-gc-pipeline-wt-rshare python run_combine.py $b main 2>&1 | grep -v Warning | tail -8
  echo "== F$b fix $(date)";  PYTHONPATH=$R/jwst-gc-pipeline-wt-sclip  python run_combine.py $b fix  2>&1 | grep -v Warning | tail -8
done
echo "done $(date)"
