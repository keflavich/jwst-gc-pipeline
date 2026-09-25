#!/usr/bin/env bash
# Delete per-exposure stage-1/2 intermediates that nothing downstream reads:
#   _jump, _0_ramp_fit, _1_ramp_fit   (jump/ramp_fit step outputs)
#   _rateints, _rate                  (Detector1 outputs; only Image2 reads _rate)
#   _i2d                              (Image2's per-exposure resample, e.g.
#                                      jw10678040001_02101_00001_nrca1_i2d.fits;
#                                      the jw10678-oNNN_t001_* mosaics are untouched)
# A SKIP=0 reduce rebuilds all of them from _uncal.
#
# Kept: _uncal, _ramp (satstar ZEROFRAME/group-0), _cal (regen source),
# _destreak/_align, _crf, mosaics, catalogs.
#
# A file is deleted only when its sibling _cal.fits exists and is NEWER than it,
# and the file is more than a day old, so a reduce in flight is left alone.
# Only the top level of each DIR is scanned (not mastDownload/ etc.).
#
# Dry run by default; pass --delete to remove files.
#   delete_unneeded_files.sh /orange/adamginsburg/jwst/gc-treasury/F*/pipeline
#   delete_unneeded_files.sh --delete /orange/adamginsburg/jwst/gc-treasury/F*/pipeline
set -euo pipefail

act=(echo "would delete:"); done_msg="would be freed (dry run; pass --delete)"
[[ ${1:-} == --delete ]] && { act=(rm --); done_msg="freed"; shift; }
(( $# )) || { echo "usage: $0 [--delete] DIR..." >&2; exit 2; }

regex='.*/jw[0-9]{11}_[0-9]{5}_[0-9]{5}_[a-z0-9]+_(jump|0_ramp_fit|1_ramp_fit|rateints|rate|i2d)\.fits'
bytes=0
while IFS= read -r -d '' f; do
    IFS=_ read -r prog grp exp det _ <<< "$(basename "$f")"
    cal="$(dirname "$f")/${prog}_${grp}_${exp}_${det}_cal.fits"
    [[ $cal -nt $f ]] || { echo "keep (no newer _cal): $f"; continue; }
    bytes=$(( bytes + $(stat -c %s "$f") ))
    "${act[@]}" "$f"
done < <(find "$@" -maxdepth 1 -type f -mmin +1440 -regextype posix-extended -regex "$regex" -print0)

echo "$(( bytes / 1000000000 )) GB $done_msg"
