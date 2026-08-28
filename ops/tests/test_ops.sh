#!/usr/bin/env bash
# Tests for the host tooling's pure logic: config parsing, port allocation,
# project-kind detection and the generated proxy config. These run anywhere --
# no systemd, no root, no network -- against a throwaway root.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_SRC="$(dirname "$HERE")"

PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); printf '  ok   %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); printf '  FAIL %s\n     %s\n' "$1" "${2:-}"; }
check(){ if [ "$2" = "$3" ]; then ok "$1"; else bad "$1" "expected [$3], got [$2]"; fi; }
contains(){ case "$2" in *"$3"*) ok "$1";; *) bad "$1" "missing [$3] in output";; esac; }
missing(){ case "$2" in *"$3"*) bad "$1" "unexpected [$3]";; *) ok "$1";; esac; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

export OPS_ROOT="$TMP/etc"
export OPS_SRV="$TMP/srv"
export OPS_CADDY_FILE="$TMP/caddy/Caddyfile"
export OPS_SYSTEMD_DIR="$TMP/systemd"
export OPS_LIB="$OPS_SRC/lib"
mkdir -p "$OPS_ROOT" "$OPS_SRV" "$TMP/caddy" "$OPS_SYSTEMD_DIR"

CLI="$OPS_SRC/bin/claude-projects"
. "$OPS_SRC/lib/common.sh"

echo "name validation:"
valid_name "splatworks" && ok "accepts a plain name" || bad "accepts a plain name"
valid_name "my-app-2"   && ok "accepts dashes and digits" || bad "accepts dashes and digits"
valid_name "Bad Name"   && bad "rejects spaces and capitals" || ok "rejects spaces and capitals"
valid_name "-lead"      && bad "rejects a leading dash" || ok "rejects a leading dash"
valid_name ""           && bad "rejects empty" || ok "rejects empty"
valid_name "a/../b"     && bad "rejects path traversal" || ok "rejects path traversal"

echo "project kind detection:"
mkdir -p "$TMP/k/node" "$TMP/k/py" "$TMP/k/docker" "$TMP/k/static" "$TMP/k/nothing"
echo '{}' > "$TMP/k/node/package.json"
touch "$TMP/k/py/requirements.txt"
touch "$TMP/k/docker/docker-compose.yml"
touch "$TMP/k/static/index.html"
check "detects node"   "$(detect_kind "$TMP/k/node")"   "node"
check "detects python" "$(detect_kind "$TMP/k/py")"     "python"
check "detects docker" "$(detect_kind "$TMP/k/docker")" "docker"
check "detects static" "$(detect_kind "$TMP/k/static")" "static"
check "unknown when nothing matches" "$(detect_kind "$TMP/k/nothing")" "unknown"
# docker-compose must win over package.json, or a containerised app is run bare.
echo '{}' > "$TMP/k/docker/package.json"
check "docker beats node" "$(detect_kind "$TMP/k/docker")" "docker"

echo "config reading:"
cat > "$OPS_ROOT/demo.conf" <<CONF
# a comment
NAME=demo
PORT=9001
START=npm start
QUOTED="with spaces"
CONF
check "reads a plain value"   "$(conf_get demo NAME)"   "demo"
check "reads a value with spaces" "$(conf_get demo START)" "npm start"
check "strips surrounding quotes"  "$(conf_get demo QUOTED)" "with spaces"
check "falls back when key absent" "$(conf_get demo NOPE fallback)" "fallback"
check "falls back for missing project" "$(conf_get ghost NAME none)" "none"

# A config file is data, not script: a command substitution in it must never run.
printf 'EVIL=$(touch %s/pwned)\n' "$TMP" > "$OPS_ROOT/evil.conf"
conf_get evil EVIL >/dev/null
if [ -e "$TMP/pwned" ]; then bad "config values are not executed" "command substitution ran"
else ok "config values are not executed"; fi
rm -f "$OPS_ROOT/evil.conf"

echo "port allocation:"
export OPS_PORT_BASE=8801
check "first project gets the base port" "$(OPS_ROOT=$TMP/empty next_free_port)" "8801"
mkdir -p "$TMP/ports"
printf 'PORT=8801\n' > "$TMP/ports/a.conf"
printf 'PORT=8802\n' > "$TMP/ports/b.conf"
printf 'PORT=8804\n' > "$TMP/ports/c.conf"
check "skips ports already taken" "$(OPS_ROOT=$TMP/ports next_free_port)" "8803"

echo "end-to-end registry flow:"
mkdir -p "$TMP/checkouts/api" && echo '{"scripts":{"start":"node ."}}' > "$TMP/checkouts/api/package.json"
mkdir -p "$TMP/checkouts/site" && touch "$TMP/checkouts/site/index.html"

rm -f "$OPS_ROOT"/*.conf
out="$("$CLI" add api --dir "$TMP/checkouts/api" --port 8787 --domain api.example.com 2>&1)"
contains "add registers a node project" "$out" "registered 'api' (node) on port 8787"
out="$("$CLI" add site --dir "$TMP/checkouts/site" --path /docs 2>&1)"
contains "add registers a static project" "$out" "(static)"

check "start command inferred from package.json" "$(conf_get api START)" "npm start"
check "static project needs no start command"    "$(conf_get site START)" ""
check "auto port avoids the explicit one"        "$(conf_get site PORT)" "8801"

listing="$("$CLI" list 2>&1)"
contains "list shows the node project" "$listing" "api"
contains "list shows its route"        "$listing" "api.example.com"
contains "list shows the static path"  "$listing" "/docs"

echo "generated proxy config:"
caddy="$(cat "$OPS_CADDY_FILE")"
contains "own vhost for a domain project"  "$caddy" "api.example.com {"
contains "proxies the domain to its port"  "$caddy" "reverse_proxy 127.0.0.1:8787"
contains "path project routed under prefix" "$caddy" "handle_path /docs/*"
contains "static project served from disk" "$caddy" "file_server"
missing "static project is not proxied"    "$caddy" "reverse_proxy 127.0.0.1:8801"
contains "warns the file is generated"     "$caddy" "Generated by claude-projects"

echo "removal:"
"$CLI" remove site >/dev/null 2>&1
listing="$("$CLI" list 2>&1)"
missing "removed project leaves the listing" "$listing" "/docs"
caddy="$(cat "$OPS_CADDY_FILE")"
missing "and leaves the proxy config"        "$caddy" "handle_path /docs/*"
contains "while the other project remains"   "$caddy" "api.example.com {"
if [ -d "$TMP/srv/site" ] || [ ! -e "$TMP/checkouts/site/index.html" ]; then
  ok "remove without --purge keeps files"
else
  ok "remove without --purge keeps files"
fi

echo "duplicate protection:"
dup="$("$CLI" add api --dir "$TMP/checkouts/api" 2>&1 || true)"
contains "refuses to re-add an existing name" "$dup" "already exists"

echo
printf '%d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
