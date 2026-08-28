#!/usr/bin/env bash
# Provision a Linux host to run Claude Code projects.
#
# Idempotent: safe to re-run to pick up changes to the tooling.
#
#   sudo ./ops/install.sh
#
# Afterwards, add a project:
#   sudo claude-projects add splatworks --repo https://github.com/you/repo --port 8787
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_LIB="$SRC_DIR/lib"
. "$OPS_LIB/common.sh"

need_root

INSTALL_LIB=/usr/local/lib/claude-projects
INSTALL_BIN=/usr/local/bin/claude-projects

info "installing claude-projects host tooling"

# --- packages ---------------------------------------------------------------
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq git curl ca-certificates ufw >/dev/null
else
  warn "no apt-get; install git, curl and a firewall yourself"
fi

# --- reverse proxy ----------------------------------------------------------
# Caddy gets automatic HTTPS from Let's Encrypt with no extra configuration,
# which is the single biggest saving when hosting several projects.
if ! command -v caddy >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    info "installing caddy"
    apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https >/dev/null
    curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key \
      | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt \
      > /etc/apt/sources.list.d/caddy-stable.list
    apt-get update -qq
    apt-get install -y -qq caddy >/dev/null
  else
    warn "install caddy manually: https://caddyserver.com/docs/install"
  fi
fi

# --- service account --------------------------------------------------------
if ! id "$OPS_USER" >/dev/null 2>&1; then
  info "creating service user $OPS_USER"
  useradd --system --create-home --home-dir "/var/lib/$OPS_USER" --shell /usr/sbin/nologin "$OPS_USER"
fi

# --- layout -----------------------------------------------------------------
mkdir -p "$OPS_ROOT" "$OPS_SRV" "$INSTALL_LIB"
chown -R "$OPS_USER:$OPS_USER" "$OPS_SRV"
chmod 0755 "$OPS_ROOT" "$OPS_SRV"

install -m 0644 "$SRC_DIR/lib/common.sh"        "$INSTALL_LIB/common.sh"
install -m 0755 "$SRC_DIR/templates/run.sh"     "$INSTALL_LIB/run.sh"
install -m 0755 "$SRC_DIR/bin/claude-projects"  "$INSTALL_BIN"
install -m 0644 "$SRC_DIR/templates/claude-project@.service" \
  "$OPS_SYSTEMD_DIR/claude-project@.service"

# The CLI resolves its library relative to itself; installed separately, point it
# at the installed copy.
sed -i "s|^LIB_DIR=.*|LIB_DIR=\"$INSTALL_LIB\"|" "$INSTALL_BIN"

# --- runtimes ---------------------------------------------------------------
if ! command -v node >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
  info "installing node 22"
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null 2>&1 || \
    warn "nodesource setup failed; install node yourself"
  apt-get install -y -qq nodejs >/dev/null 2>&1 || warn "node install failed"
fi
if ! command -v python3 >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
  apt-get install -y -qq python3 python3-venv python3-pip >/dev/null
fi

# --- firewall ---------------------------------------------------------------
# Only the proxy is exposed; project ports stay on loopback.
if command -v ufw >/dev/null 2>&1; then
  ufw allow OpenSSH >/dev/null 2>&1 || true
  ufw allow 80/tcp  >/dev/null 2>&1 || true
  ufw allow 443/tcp >/dev/null 2>&1 || true
  ufw --force enable >/dev/null 2>&1 || warn "could not enable ufw"
fi

systemctl daemon-reload 2>/dev/null || true
"$INSTALL_BIN" render || warn "initial render failed"

info "done"
cat <<'NEXT'

Next:
  claude-projects doctor                 check the host
  claude-projects add <name> --repo URL  register a project
  claude-projects list                   see what is running

Projects listen on loopback only; Caddy is the single public entry point.
Give a project its own hostname with --domain app.example.com and Caddy
obtains a certificate automatically.
NEXT
