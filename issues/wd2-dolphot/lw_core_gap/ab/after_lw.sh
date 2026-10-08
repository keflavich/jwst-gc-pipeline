#!/bin/bash
# Score and cut out the LW A/B arms once all merges are done.
cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/f277w_gap/dbl
Q=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ
for B in 250M 277W 300M; do
  lb=$(echo $B | tr A-Z a-z)
  args="mainfcbg=$Q/tree_mainfcbg/catalogs/f${lb}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits"
  for a in dbl0 dbl1 dbl2 dbl3; do
    f=tree_$a/catalogs/f${lb}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits
    [ -f $f ] && args="$args $a=$f"
  done
  nice -19 python score_lw.py $B $args 2>&1 | grep -v -i -e warn -e OBSGEO -e "Set DATE"
  targs=""
  for a in dbl0 dbl1 dbl2 dbl3; do
    [ -f tree_$a/catalogs/f${lb}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits ] && targs="$targs $a=$PWD/tree_$a"
  done
  nice -19 python cutouts_ab.py $B 6 $targs 2>&1 | grep -v -i -e warn -e OBSGEO -e "Set DATE" | tail -2
done
