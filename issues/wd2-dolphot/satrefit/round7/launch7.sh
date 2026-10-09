#!/bin/sh
cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit
cat ${JOBS:-out6/jobs6.txt} | xargs -P 7 -L 1 sh -c 'nice -19 timeout 3000 /blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python -u run_frames7.py refit $0 $1 $2 > out7/log7_$0_$1_$2.txt 2>&1'
