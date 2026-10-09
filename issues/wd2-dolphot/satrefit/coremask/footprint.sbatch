#!/bin/bash
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/coremask/log_footprint.txt
cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/coremask
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python -u footprint.py
