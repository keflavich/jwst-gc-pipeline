#!/bin/bash
# Regenerate full-frame CARTA snippets for every Galactic-Center field from the
# photometry-pipeline (cataloging.py) outputs.  Idempotent: re-run any time as
# the pipeline produces more products -- each snippet is a snapshot of the files
# that exist NOW, so re-running picks up newly-written residual/model/bg mosaics
# and catalogs.  Filters are auto-detected from the <FILTER>/pipeline dirs that
# exist under each field, so no per-field filter list to maintain.
#
# Snippets are written to ~/.carta/config/snippets/<field>-fullframe.json under
# the category jw-gc-manual.  Usage:  bash make_all_field_snippets.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
GEN="$HERE/make_carta_snippet.py"
ROOT=/orange/adamginsburg/jwst
CATEGORY=jw-gc-manual

FIELDS="sickle cloudef sgrc sgrb2 arches quintuplet sgra gc2211 wd1 wd2 w51"

for t in $FIELDS; do
    base="$ROOT/$t"
    if [ ! -d "$base" ]; then
        echo "$t: no basepath ($base); skip"
        continue
    fi
    # auto-detect filters = <FILTER> dirs that contain a pipeline/ subdir
    filters=$(ls -d "$base"/F*/pipeline 2>/dev/null \
              | sed "s|.*/jwst/$t/||; s|/pipeline||" | paste -sd, -)
    if [ -z "$filters" ]; then
        echo "$t: no filter/pipeline dirs; skip"
        continue
    fi
    python "$GEN" --base "$base" --filters "$filters" \
        --category "$CATEGORY" --name "${t}-fullframe" \
        | grep -E "wrote"
done
