#!/bin/bash
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
#SBATCH --job-name=mkgrid
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/f150x/samp4/mk_%j.log
export PYTHONPATH=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-gridT
export STPSF_PATH=/orange/adamginsburg/repos/webbpsf/data/
export OMP_NUM_THREADS=4
cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/f150x/samp4
for f in F150W F200W; do for o in 2 4; do python make_grid.py $f nrcb1 $o; done; done
