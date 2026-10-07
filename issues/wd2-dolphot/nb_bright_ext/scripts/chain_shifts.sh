#!/bin/bash
# usage: bash chain_shifts.sh  -> runs ../bright_band_shift.py along the arm chain, writes chain_shifts.txt
cd "$(dirname "$0")/.." || exit 1
for p in "prod ctrl" "ctrl integ" "integ integbg" "integ integfixc" "integfixc integfc" "integfc integfcbg" "integbg integfcbg" "integfcbg mainfcbg" "mainfcbg mainfcbgkf" "prod integ" "prod mainfcbg"; do
  nice -19 python bright_band_shift.py $p 2>&1 | grep -E "^[a-z]+ -> |###|^  (10|11|12|13)-"
done >| nb_bright_ext/chain_shifts.txt
