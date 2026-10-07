#!/bin/bash
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --job-name=wd2-sat1to1-merge
#SBATCH --cpus-per-task=2
#SBATCH --mem=48gb
#SBATCH --time=03:00:00
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/sat1to1/%x_%j.out
set -e
D=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/sat1to1
B=/blue/adamginsburg/adamginsburg/repos
export PYTHONDONTWRITEBYTECODE=1
touch $D/MARKER
cd $D
python prep.py
echo "== mfs0 (5434f5e7) $(date)"
PYTHONPATH=$B/jwst-gc-pipeline-wd2main2 python run_merge.py mfs0 $B/jwst-gc-pipeline-wd2main2
echo "== mfs1 (sat1to1) $(date)"
PYTHONPATH=$B/jwst-gc-pipeline-sat1to1 python run_merge.py mfs1 $B/jwst-gc-pipeline-sat1to1
echo "== guard $(date)"
find /orange/adamginsburg/jwst/wd2 -maxdepth 3 -path '*/dolphot_benchmark' -prune -o -newer $D/MARKER -print 2>/dev/null | head -20
find /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_mainfcbg -newer $D/MARKER -print 2>/dev/null | head -20
echo "== done $(date)"
