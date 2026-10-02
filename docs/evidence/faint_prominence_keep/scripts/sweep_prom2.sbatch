#!/bin/bash
#SBATCH --job-name=faint1018-peaksb-prom-guard-replay
#SBATCH --qos=astronomy-dept-b
#SBATCH --account=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --partition=hpg-dev
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/blue/adamginsburg/adamginsburg/tmp/claude-3663/-orange-adamginsburg-jwst/3628e1c3-037c-44d2-983f-710a80e638be/scratchpad/faint2/replay/logs/sweep_prom2_%A_%a.log
S=/blue/adamginsburg/adamginsburg/tmp/claude-3663/-orange-adamginsburg-jwst/3628e1c3-037c-44d2-983f-710a80e638be/scratchpad
NAMES=(prom2 prom2p3 prom2p5)
OPTS=('{}'
      '{"manual_ext_star_prom_peak_min": 3.0}'
      '{"manual_ext_star_prom_peak_min": 5.0}')
FIELDS=(brick:F182M sgrb2:F187N)
i=$((SLURM_ARRAY_TASK_ID % 3))
fb=${FIELDS[$((SLURM_ARRAY_TASK_ID / 3))]}
f=${fb%%:*}; b=${fb#*:}
echo "variant ${NAMES[$i]} opts ${OPTS[$i]} field $f band $b host $(hostname)"
cd $S/faint2/replay
REPLAY_OPTS="${OPTS[$i]}" python vet_variant.py $S/wt-rp-prom2 $f $b out/${f}_${b,,}_${NAMES[$i]}.fits | head -c 2000
