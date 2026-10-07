#!/bin/bash
# score the two m7 outputs of job 45065002 against dolphot, then the A/B tables
set -e
cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark
M7=catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m7.fits
S=Q_integ/sat1to1
for a in mfs0 mfs1; do
    nice -19 python compare_dolphot.py --ours Q_integ/tree_$a/$M7 --tag Q_s1t1_$a --merge-twins --band-pos >| $S/cmp_$a.log 2>&1
done
nice -19 python $S/compare_ab.py >| $S/compare_ab.md 2>&1
cat $S/compare_ab.md
