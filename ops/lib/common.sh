# Shared helpers for the claude-projects host tooling.
#
# Every path is overridable so the whole thing can run against a throwaway root
# in tests, and so a host can relocate state without editing scripts.

set -euo pipefail

: "${OPS_ROOT:=/etc/claude-projects}"      # project registry
: "${OPS_SRV:=/srv/claude-projects}"       # checkouts and per-project data
: "${OPS_USER:=claudeapps}"                # unprivileged service account
: "${OPS_CADDY_FILE:=/etc/caddy/Caddyfile}"
: "${OPS_SYSTEMD_DIR:=/etc/systemd/system}"
: "${OPS_PORT_BASE:=8801}"                 # first auto-assigned port

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; DIM=$'\033[2m'; RESET=$'\033[0m'
[ -t 1 ] || { RED=; GREEN=; YELLOW=; DIM=; RESET=; }

log()  { printf '%s\n' "$*"; }
info() { printf '%s==>%s %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '%s warn%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die()  { printf '%serror%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

need_root() {
  [ "$(id -u)" -eq 0 ] || die "this needs root; re-run with sudo"
}

# Project names become systemd unit names, directory names and Caddy labels, so
# keep them to something all three accept without quoting.
valid_name() {
  case "$1" in
    ''|*[!a-z0-9-]*) return 1 ;;
    -*|*-)           return 1 ;;
    *)               [ ${#1} -le 40 ] ;;
  esac
}

project_conf() { printf '%s/%s.conf\n' "$OPS_ROOT" "$1"; }

project_exists() { [ -f "$(project_conf "$1")" ]; }

list_projects() {
  [ -d "$OPS_ROOT" ] || return 0
  for f in "$OPS_ROOT"/*.conf; do
    [ -e "$f" ] || continue
    basename "$f" .conf
  done
}

# Read one key from a project's conf without sourcing it, so a stray line in a
# config file cannot execute as shell.
conf_get() {
  local name="$1" key="$2" default="${3:-}" file value
  file="$(project_conf "$name")"
  [ -f "$file" ] || { printf '%s\n' "$default"; return 0; }
  value="$(grep -m1 -E "^${key}=" "$file" 2>/dev/null || true)"
  value="${value#*=}"
  # Strip one layer of surrounding quotes if present.
  value="${value%\"}"; value="${value#\"}"
  value="${value%\'}"; value="${value#\'}"
  [ -n "$value" ] && printf '%s\n' "$value" || printf '%s\n' "$default"
}

# Lowest free port at or above OPS_PORT_BASE that no project has claimed.
next_free_port() {
  local port="$OPS_PORT_BASE" taken name
  taken="$(for name in $(list_projects); do conf_get "$name" PORT; done)"
  while printf '%s\n' "$taken" | grep -qx "$port"; do
    port=$((port + 1))
  done
  printf '%s\n' "$port"
}

# Guess how to run a checkout. Explicit config always wins; this only fills the
# blank at `add` time so the common cases need no flags.
detect_kind() {
  local dir="$1"
  if [ -f "$dir/docker-compose.yml" ] || [ -f "$dir/compose.yaml" ]; then
    printf 'docker\n'
  elif [ -f "$dir/package.json" ]; then
    printf 'node\n'
  elif [ -f "$dir/pyproject.toml" ] || [ -f "$dir/requirements.txt" ]; then
    printf 'python\n'
  elif [ -f "$dir/index.html" ] || [ -d "$dir/dist" ] || [ -d "$dir/public" ]; then
    printf 'static\n'
  else
    printf 'unknown\n'
  fi
}

default_start_command() {
  local kind="$1" dir="$2"
  case "$kind" in
    node)
      if [ -f "$dir/package.json" ] && grep -q '"start"' "$dir/package.json" 2>/dev/null; then
        printf 'npm start\n'
      else
        printf 'node server/index.js\n'
      fi
      ;;
    python) printf '.venv/bin/python -m app\n' ;;
    docker) printf 'docker compose up\n' ;;
    static) printf '\n' ;;   # served by the proxy, no process
    *)      printf '\n' ;;
  esac
}
