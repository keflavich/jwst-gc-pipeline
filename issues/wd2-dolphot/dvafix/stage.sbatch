#!/bin/bash
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --job-name=wd2-dvx-stage
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=04:00:00
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/dvafix/logs/%x_%j.out
set -e
source ~/.bash_profile 2>/dev/null
cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/dvafix
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python stage_dvx.py dvx0 dvx1
