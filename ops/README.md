# Running Claude Code projects on one host

Turns a Linux box into a server that runs several projects side by side, each
with its own service, logs, restarts and HTTPS route.

```bash
sudo ./ops/install.sh
sudo claude-projects add splatworks \
  --repo https://github.com/kcimedia7/kcimedia7 \
  --branch claude/gaussian-splat-converter-pnx7mg \
  --domain splats.example.com
```

That is the whole setup. The project is cloned, registered, started under
systemd, and published over HTTPS with a certificate Caddy obtains itself.

## The idea

One project is one small config file in `/etc/claude-projects`. Everything else
is generated from those files, so there are no per-project unit files or proxy
blocks to hand-maintain:

```
/etc/claude-projects/<name>.conf   the registry -- the only thing you edit
/srv/claude-projects/<name>/src    the checkout
/srv/claude-projects/<name>/data   persistent data, kept across deploys
```

A single systemd template unit serves every project (`claude-project@<name>`),
and the Caddyfile is regenerated whenever the registry changes.

## Commands

| | |
|---|---|
| `claude-projects add <name>` | register and start a project |
| `claude-projects list` | name, kind, port, route, state |
| `claude-projects status` | live health check of every project |
| `claude-projects deploy <name>` | pull, install, build, restart |
| `claude-projects logs <name> -f` | follow the journal |
| `claude-projects restart <name>` | restart one project |
| `claude-projects remove <name> [--purge]` | unregister, optionally delete files |
| `claude-projects render` | regenerate units and proxy config |
| `claude-projects doctor` | check the host is set up correctly |

`add` detects the project type from the checkout -- `package.json` means node,
`requirements.txt` means python, a compose file means docker, a bare
`index.html` means static files served straight from disk. Pass `--kind` or
`--start "…"` when the guess is wrong.

## Routing

Two ways to publish a project, and you can mix them:

- `--domain app.example.com` gives it its own hostname, with automatic TLS.
  Point the DNS record at the host first, or the certificate request fails.
- `--path /app` serves it under the host's default site.

## Security posture

- **Projects listen on loopback only.** The runner forces `HOST=127.0.0.1`, so
  the reverse proxy is the single public entry point and no project can be
  reached directly. The firewall opens only SSH, 80 and 443.
- **Projects run as an unprivileged user** (`claudeapps`), never root.
- **The unit is confined**: `ProtectSystem=strict`, `PrivateTmp`,
  `NoNewPrivileges`, no device access, writable only under
  `/srv/claude-projects`. A project that genuinely needs more gets a drop-in
  via `systemctl edit claude-project@<name>` rather than a loosened template.
- **Config files are data.** Values are read with `grep`, never sourced, so a
  line in a `.conf` cannot execute as shell. There is a test for this.

There is deliberately **no authentication** in front of the projects
themselves. If a project should not be public, put Caddy basic auth or an
identity proxy in front of it -- do not rely on the URL being unguessed.

## Tests

```bash
./ops/tests/test_ops.sh
```

39 checks over the parts that are pure logic: name validation, project-type
detection, port allocation, config parsing (including that values are not
executed), the add/list/remove lifecycle, and the generated proxy config. They
need no root, no systemd and no network, so they run in CI.

What they do **not** cover is the installer itself -- package installation,
user creation and systemd behaviour need a real host. Run `claude-projects
doctor` there.

## Requirements

A Linux host with systemd (Ubuntu 22.04/24.04 or Debian 12 are the tested
shapes), a public IP if you want TLS, and root. Node and Python are installed
by the installer if missing.
