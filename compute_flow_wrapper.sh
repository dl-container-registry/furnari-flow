#!/usr/bin/env bash

: ${COMPUTE_FLOW:=/bin/compute_flow}

if [[ $# -lt 1 || $1 = '-h' || $1 = '--help' ]]; then
    echo "USAGE: <frame_pattern> [<compute_flow_arg>]+"
    echo "Example: $0 frame%06d.jpg"
    echo
    "$COMPUTE_FLOW" --help
    exit 1
fi

IN=/input
OUT=/output

FRAME_PATTERN="$1"; shift
OF_PATTERN="flow_%s_$FRAME_PATTERN"

mkdir -p "$OUT/"{u,v}

"$COMPUTE_FLOW" "$IN" "$FRAME_PATTERN" "$OUT/$OF_PATTERN" $@

mv "$OUT/flow_x*.jpg" "$OUT/u"
mv "$OUT/flow_y*.jpg" "$OUT/v"
