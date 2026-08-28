#!/usr/bin/env bash
# Launch one registered project. systemd calls this with the project name.
#
# The start command lives in config rather than in the unit, so adding a project
# never means writing a unit file -- and one template covers every project.
set -euo pipefail

OPS_LIB="${OPS_LIB:-/usr/local/lib/claude-projects}"
. "$OPS_LIB/common.sh"

name="${1:?project name required}"
conf="$(project_conf "$name")"
[ -f "$conf" ] || die "no config for project '$name' at $conf"

workdir="$(conf_get "$name" WORKDIR)"
start="$(conf_get "$name" START)"
port="$(conf_get "$name" PORT)"
data="$(conf_get "$name" DATA_DIR)"
env_file="$(conf_get "$name" ENV_FILE)"

[ -d "$workdir" ] || die "checkout missing: $workdir"
[ -n "$start" ]   || die "no START command configured for '$name'"

cd "$workdir"

# Bind to loopback only: the reverse proxy is the single public entry point, so
# an app must never be reachable directly from outside.
export HOST=127.0.0.1
export PORT="$port"
export NODE_ENV="${NODE_ENV:-production}"
[ -n "$data" ] && export DATA_DIR="$data" SPLAT_DATA_DIR="${SPLAT_DATA_DIR:-$data}"

if [ -n "$env_file" ] && [ -f "$env_file" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$env_file"
  set +a
fi

exec /bin/bash -lc "$start"
