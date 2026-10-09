cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/apclosure
for b in 200W 150W 164N 187N 212N 182M 250M 300M 410M 466N 115W 162M 277W 335M 323N 405N; do
  /usr/bin/time -f "$b %e s" nice -19 python measure.py $b 3000 > log_$b.txt 2>&1
done
