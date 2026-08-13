#!/bin/sh
set -eu

umask 077
marker_path="$PGDATA/.elvis-v2-fresh-rehearsal-v1"
temporary_marker="$marker_path.tmp"
printf '%s\n' "elvis-v2-fresh-rehearsal:v1" >"$temporary_marker"
mv "$temporary_marker" "$marker_path"
