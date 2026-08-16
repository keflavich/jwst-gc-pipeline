#!/bin/bash
# Regenerate PR #390's campaign sweep.  Hand-edited counts have been wrong three
# rounds running; this prints the numbers and the failing rows together so the
# body can be pasted from one run.
cd /blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-preflight || exit 1
P=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
export PYTHONPATH=$PWD

run() {
  $P scripts/reduction/preflight_reduce_inputs.py \
     --target "$1" --proposal "$2" --obsid "$3" --filters "$4" \
     ${5:+--instrument $5} ${6:+--modules $6} 2>&1 \
   | grep -E '^OK|^MISSING|^MISMATCH'
}

specs=0 pairs=0 bad=0
FAILROWS=""
while IFS='|' read -r t p o f inst mods; do
  [ -z "$t" ] && continue
  specs=$((specs + 1))
  out=$(run "$t" "$p" "$o" "$f" "$inst" "$mods")
  n=$(printf '%s\n' "$out" | grep -c '^OK\|^MISSING')
  m=$(printf '%s\n' "$out" | grep -c '^MISSING\|^MISMATCH')
  pairs=$((pairs + n)); bad=$((bad + m))
  if [ "$m" -gt 0 ]; then
    FAILROWS="$FAILROWS$(printf '%s\n' "$out" | grep '^MISSING\|^MISMATCH')
"
  fi
done <<'SPECS'
brick|2221|001|F182M F187N F212N F405N F410M F466N||
brick|1182|004|F115W F200W F356W||
cloudc|2221|002|F182M F187N F212N F405N F410M F466N||
sgrb2|5365|001|F150W F182M F187N F212N F300M F360M F480M||
cloudef|2092|002|F162M F360M F480M||
cloudef|2092|005|F162M F360M F480M||
sgrc|4147|012|F115W F162M F182M F212N F405N||
sickle|3958|007|F187N F210M F335M F470N F480M||nrcb
sgra|1939|001|F115W F212N F405N||
arches|2045|001|F212N F323N||
quintuplet|2045|003|F212N F323N||
gc2211|2211|023|F150W F200W F277W||
gc2211|2211|028|F150W F200W F277W||
gc2211|2211|046|F150W F200W F277W||
gc2211|2211|049|F150W F200W F277W||
gc2211|2211|050|F150W F200W F277W||
cloudc|2526|021|F770W|miri|
sgrc/niriss|4147|012|F158M F200W F356W F480M|niriss|
SPECS

echo "SWEEP: $specs specs, $pairs (field, filter) pairs, $bad problems"
echo
printf '%s' "$FAILROWS"
