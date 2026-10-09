#!/bin/bash
#SBATCH --job-name=wd2-1148-dryall
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/pedsurvey/run_dryall_%j.out
# #1148 review: dry pass of the ZEROFRAME rim rewrite (curve vs header rate) over w51 and the surveyed frames.
cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/pedsurvey || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
N=8
seq 0 $((N - 1)) | xargs -P $N -I{} sh -c "/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python -u dryall.py {} $N >| dryall_{}.log 2>&1"
echo ALLWORKERS_DONE
