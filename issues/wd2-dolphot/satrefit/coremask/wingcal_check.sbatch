#!/bin/bash
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --mem=8gb
#SBATCH --time=00:15:00
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/coremask/log_wingcal.txt
cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/coremask
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python -u wingcal_check.py
