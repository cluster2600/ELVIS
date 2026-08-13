#!/bin/sh
set -eu

marker_path="$PGDATA/.elvis-v2-fresh-rehearsal-v1"
marker_value="elvis-v2-fresh-rehearsal:v1"

if [ -s "$PGDATA/PG_VERSION" ]; then
    if [ "$(cat "$PGDATA/PG_VERSION")" != "15" ]; then
        echo "rehearsal volume has an unsupported PostgreSQL version" >&2
        exit 1
    fi
    if [ ! -f "$marker_path" ] || [ "$(cat "$marker_path")" != "$marker_value" ]; then
        echo "rehearsal volume marker is absent or invalid" >&2
        exit 1
    fi
elif [ -n "$(find "$PGDATA" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    echo "rehearsal volume is not empty" >&2
    exit 1
fi

exec /usr/local/bin/docker-entrypoint.sh "$@"
