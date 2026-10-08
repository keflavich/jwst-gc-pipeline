#!/bin/bash
PY=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/apclose
for a in "main2kf F250M" "main2 F250M" "main2kf F150W" "main2 F150W"; do set -- $a; nice -19 timeout 3000 $PY -u apclose_measure.py $1 $2 > meas_$1_$2.log 2>&1 & done
wait
