#!/bin/bash
#SBATCH --job-name=wd2-1148-dcorr
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/pedsurvey/run_dcorr_%j.out
# #1148: cal - crf level under the header rate (dry pass, curve vs header vs header - D).
cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/pedsurvey || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
N=8
seq 0 $((N - 1)) | xargs -P $N -I{} sh -c "/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python -u dcorr.py {} $N >| dcorr_{}.log 2>&1"
echo ALLWORKERS_DONE
