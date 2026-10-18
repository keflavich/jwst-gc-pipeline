#!/bin/bash
#SBATCH --account=astronomy-dept --qos=astronomy-dept-b
#SBATCH --partition=hpg-default,hpg-milan,hpg-turin
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=32gb --time=3:00:00
#SBATCH --output=<scratch>/prod_after/cloudc_diag/%x_%j.log
# cloudc F212N audits on the ng2 (outlier_detection on) m7, 2026-10-18
set -x
cd <scratch>/prod_after && python compare.py cloudc F212N out/cloudc_f212n_before_after
cd <scratch>/prod_after/cloudc_diag && python ctrl_map.py cloudc F212N cloudc_f212n_ctrl_map
python cutouts.py cloudc F212N 500 5000
python cutouts.py cloudc F212N 50 150
cd <scratch>/prod_after/oversub && python nn_oversub.py cloudc F212N after out/cloudc_f212n_after
