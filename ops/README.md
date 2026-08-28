# Running Claude Code projects

Two modes, because a laptop and a public server want opposite things.

| | `claude-local` | `claude-projects` |
|---|---|---|
| For | the computer you work on | a server with a public IP |
| Needs | Node. That is all. | root, systemd, a domain |
| Root | no | yes |
| Ports | each project on its own, opened directly | one proxy on 80/443 |
| TLS | none needed (loopback) | automatic, via Caddy |
| OS | macOS, Linux, WSL | Linux with systemd |

**On your own machine, use `claude-local`** -- it is one Node file with no
dependencies and asks for no root, no domain and no certificates:

```bash
node ops/local/claude-local.mjs doctor          # what this machine can do
node ops/local/claude-local.mjs add splatworks --dir . --health /api/health
node ops/local/claude-local.mjs up              # dashboard on :7777
```

`up` starts every registered project, restarts anything that crashes (with
backoff), and serves a dashboard listing them with live health. `down` stops
the lot. `autostart` writes a systemd user unit on Linux or a launchd agent on
macOS so it comes back after a reboot.

Each project keeps **its own port and is opened directly** rather than being
proxied under a path prefix, because prefix-proxying breaks any app that uses
absolute URLs -- which includes this one. Everything binds to `127.0.0.1`, so
nothing is exposed to the network you happen to be on.

Logs land in `~/.claude-projects/logs/<name>.log`; state is a single
`projects.json` beside them.

## Windows

`claude-local` runs natively on Windows -- no WSL required. It uses
`taskkill /T` instead of POSIX process groups, follows logs in Node rather than
shelling out to `tail`, and `autostart` writes a Startup-folder script.

In PowerShell, from the repository:

```powershell
node ops\local\claude-local.mjs doctor
node ops\local\claude-local.mjs add splatworks --dir . --health /api/health
node ops\local\claude-local.mjs up
node ops\local\claude-local.mjs autostart   # start at login, no admin needed
```

`autostart` writes `claude-local.vbs` into your Startup folder, which launches
it hidden at login. Delete that file to turn it off.

For the trainer, `pip install torch` on Windows already gives the CPU build --
the CUDA wheels only come from PyTorch's own index -- so:

```powershell
pip install -r trainer\requirements.txt
```

**Use WSL2 instead if** a `pycolmap` wheel is not published for your Python
version, or you want the server-mode bash tooling. Inside WSL2 everything in
this repository behaves exactly as it does on Linux.

## Hosting from a Windows machine (e.g. Admin-Server)

To have one machine serve the apps and reach them from other desktops:

```powershell
# on the host, from the repository
node ops\local\claude-local.mjs add splatworks --dir . --health /api/health
node ops\local\claude-local.mjs windows-setup --password "<a secret>"
```

`windows-setup` writes a boot-time scheduled task, three desktop shortcuts
(Start / Stop / Open), and prints the two Administrator commands to run: one
`netsh` firewall rule, and the `schtasks` import. Then from any machine on the
network:

```
http://Admin-Server:7777
```

### How it publishes

Each project keeps its own port. The host binds `<lan-ip>:PORT` for each one and
forwards to `127.0.0.1:PORT`, which does not collide because they are different
addresses. Paths are forwarded untouched, so every absolute `/api/...` URL
inside an app still resolves -- a path-prefix proxy would break them.

`up` refuses `--lan` unless you set a password or pass `--allow-anonymous`.
These projects accept uploads and can delete a library, so publishing them to a
network is a decision rather than a default.

**The password is stored as a SHA-256 digest** in `~/.claude-projects/password.sha256`
(mode 600). It is deliberately not passed as an argument: a boot task and a
desktop shortcut are long-lived plaintext files, and a command line is visible
in `ps`. Requests are checked with a constant-time comparison of digests.

Connections from the host itself stay unauthenticated -- someone sitting at
Admin-Server already has it.

### Running at boot vs at login

`autostart` starts it when you log in. `windows-setup` builds a **boot** task
that runs with no one logged in, which is what a server wants. The task runs at
LeastPrivilege, not as SYSTEM. Windows prompts for the account password during
`schtasks /create` and stores it in the Task Scheduler's own credential store,
so it never passes through this tool.

## What your GPU needs to be

Having an NVIDIA card is not the question; the generation is.

| Compute capability | Example cards | GPU training |
|---|---|---|
| 7.5 and up | RTX 20xx, 30xx, 40xx, A100 | yes |
| 6.x (Pascal) | GTX 10xx | no |
| 5.x (Maxwell) | GTX 9xx | no |
| 3.x (Kepler) | GTX 6xx, 7xx | no |
| 2.x (Fermi) | GTX 4xx, 5xx | no |

Current PyTorch wheels require 7.5+, and the reference 3DGS CUDA rasterizer
requires 7.0+. CUDA itself dropped Fermi after 8.0, in 2017. On anything below
the line there is no driver combination that works -- `claude-local doctor`
says so outright rather than reporting "GPU found", because the alternative is
someone losing an evening to driver installs for a dead end.

Below the line, the bundled CPU trainer is the path. It is the same algorithm,
not a cheaper approximation; it just takes longer.

---

# The server mode

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
./ops/tests/test_ops.sh    # server mode
npm test                   # includes the claude-local tests
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
