#!/bin/bash
#SBATCH --job-name=satstar-fit-footprint-sweep-gc2211-f200w
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=8:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/satstar-fit-footprint-sweep_%j.log

# Sweep the satstar FITTING footprint (PSFPhotometry fit_shape) while holding the
# cutout / PSF-model grid / saturated-core mask / background annulus FIXED at
# pad=81, on one gc2211 F200W frame. Measures top-20 satstar flux/position/qfit
# stability and wall-clock vs footprint (keflavich/jwst-gc-pipeline satstar
# fit-footprint decoupling study). Read-only wrt /orange products.

set -euo pipefail
PY=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
export STPSF_PATH=/orange/adamginsburg/jwst/stpsf-data/
export CRDS_PATH=/orange/adamginsburg/jwst/crds
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
cd /blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline
"$PY" scripts/satstar_deblend/fit_footprint_sweep.py "11,17,21,31,51,81"
echo "[footprint-sweep] DONE"
