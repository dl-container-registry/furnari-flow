#!/usr/bin/env bash
shopt -s nullglob
set -e

COMPUTE_FLOW=/bin/compute_flow 

if [[ $# -lt 1 || $1 = '-h' || $1 = '--help' ]]; then
    echo "USAGE: <frame_pattern> [<compute_flow_arg>]+"
    echo "Example: $0 frame%06d.jpg"
    echo 
    echo 
    "$COMPUTE_FLOW" --help
    exit 1
fi

IN=/input
OUT=/output

FRAME_PATTERN="${1:-'frame_%06d.jpg'}"; shift
OF_PATTERN="flow_%s_$FRAME_PATTERN"

mkdir -p "$OUT/"{u,v}

touch "$IN/.flow" || exit 1
rm "$IN/.flow" || exit 1

"$COMPUTE_FLOW" "$IN" "$FRAME_PATTERN" "$OF_PATTERN" $@

declare -A dim_map
dim_map[x]=u
dim_map[y]=v
for dim in x y; do
    for f in "$IN/flow_${dim}_"*.jpg; do
        filename="${f##*/}"
        mv "$f" "$OUT/${dim_map[$dim]}/${filename##flow_?_}"
    done
done

