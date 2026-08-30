#!/usr/bin/env node
/**
 * claude-local -- run your Claude Code projects on your own computer.
 *
 * This is the personal-machine counterpart to ops/install.sh. That one assumes
 * a public Linux server: root, systemd, a domain, TLS. None of which apply to
 * the laptop you are typing on, so this asks for none of them.
 *
 *   - no root, no systemd, no reverse proxy, no certificates
 *   - one Node file with zero dependencies, identical on macOS, Linux and WSL
 *   - each project keeps its own port and is opened directly, so nothing
 *     rewrites URLs and no app breaks on a path prefix
 *   - a dashboard lists everything with links, health and logs
 *
 * State lives in ~/.claude-projects.
 */

import fs from 'node:fs';
import fsp from 'node:fs/promises';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawn, execFile } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

const HOME = process.env.CLAUDE_LOCAL_HOME || path.join(os.homedir(), '.claude-projects');
const STATE_FILE = path.join(HOME, 'projects.json');
const PID_FILE = path.join(HOME, 'supervisor.pid');
const LOG_DIR = path.join(HOME, 'logs');
const DASHBOARD_PORT = Number(process.env.CLAUDE_LOCAL_PORT || 7777);
const PORT_BASE = 8801;

const IS_WINDOWS = process.platform === 'win32';
const colour = process.stdout.isTTY;
const c = {
  dim: (s) => (colour ? `\x1b[2m${s}\x1b[0m` : s),
  green: (s) => (colour ? `\x1b[32m${s}\x1b[0m` : s),
  red: (s) => (colour ? `\x1b[31m${s}\x1b[0m` : s),
  yellow: (s) => (colour ? `\x1b[33m${s}\x1b[0m` : s),
  bold: (s) => (colour ? `\x1b[1m${s}\x1b[0m` : s),
};

// ------------------------------------------------------------------- state

async function loadState() {
  try {
    const raw = await fsp.readFile(STATE_FILE, 'utf8');
    const parsed = JSON.parse(raw);
    return { projects: [], ...parsed };
  } catch (err) {
    if (err.code === 'ENOENT') return { projects: [] };
    throw new Error(`${STATE_FILE} is not readable JSON: ${err.message}`);
  }
}

async function saveState(state) {
  await fsp.mkdir(HOME, { recursive: true });
  const tmp = `${STATE_FILE}.tmp`;
  await fsp.writeFile(tmp, JSON.stringify(state, null, 2));
  await fsp.rename(tmp, STATE_FILE);
}

function findProject(state, name) {
  return state.projects.find((p) => p.name === name);
}

function nextPort(state) {
  const taken = new Set(state.projects.map((p) => p.port));
  let port = PORT_BASE;
  while (taken.has(port)) port += 1;
  return port;
}

// --------------------------------------------------------------- detection

export function detectKind(dir) {
  const has = (f) => fs.existsSync(path.join(dir, f));
  // Compose first: a containerised project usually also has a package.json,
  // and running it bare would bypass its services.
  if (has('docker-compose.yml') || has('compose.yaml')) return 'docker';
  if (has('package.json')) return 'node';
  if (has('pyproject.toml') || has('requirements.txt')) return 'python';
  if (has('index.html') || has('dist') || has('public')) return 'static';
  return 'unknown';
}

export function defaultStart(kind, dir) {
  switch (kind) {
    case 'node': {
      try {
        const pkg = JSON.parse(fs.readFileSync(path.join(dir, 'package.json'), 'utf8'));
        if (pkg.scripts?.start) return 'npm start';
      } catch { /* fall through */ }
      return 'node server/index.js';
    }
    case 'python': return 'python3 -m app';
    case 'docker': return 'docker compose up';
    // Served by the dashboard's own file server, so no process to run.
    case 'static': return '';
    default: return '';
  }
}

export function validName(name) {
  return typeof name === 'string' && /^[a-z0-9][a-z0-9-]{0,39}$/.test(name) && !name.endsWith('-');
}

/**
 * Decide whether a GPU can actually run gaussian-splat training.
 *
 * "Has an NVIDIA GPU" is not the question -- current PyTorch wheels need
 * compute capability 7.5+, and CUDA dropped Fermi (2.x) after 8.0. A GTX 560 Ti
 * reports as a perfectly healthy CUDA device and still cannot run any of this,
 * so saying "GPU found" would send someone off installing drivers for an
 * evening to reach a dead end.
 */
export function classifyGpu(name = '', computeCap = null) {
  const MIN_TORCH_CAP = 7.5;      // default PyTorch wheels
  const MIN_3DGS_CAP = 7.0;       // the reference CUDA rasterizer

  let cap = computeCap === null ? null : Number(computeCap);
  if (cap !== null && !Number.isFinite(cap)) cap = null;

  // Old drivers have no compute_cap query -- a Fermi card with the last R390
  // driver cannot report one -- so fall back to the model number. This is a
  // conservative floor for the usable/not-usable decision, not an exact figure:
  // an RTX 3090 is really 8.6 but 7.5 is enough to clear the bar.
  if (cap === null) {
    const m = /\b(?:GTX|GT|RTX)\s*(\d{3,4})\b/i.exec(name);
    if (m) {
      const model = Number(m[1]);
      if (model >= 2000) cap = 7.5;            // RTX 20xx and up
      else if (model >= 1600) cap = 7.5;       // GTX 16xx
      else if (model >= 1000) cap = 6.1;       // GTX 10xx, Pascal
      else if (model >= 900) cap = 5.2;        // GTX 9xx, Maxwell
      else if (model >= 600) cap = 3.0;        // GTX 6xx/7xx, Kepler
      else cap = 2.1;                          // GTX 4xx/5xx, Fermi
    }
  }

  if (cap === null) {
    return { usable: false, cap: null, reason: 'could not determine compute capability' };
  }
  if (cap >= MIN_TORCH_CAP) {
    // Blackwell (sm_120) is newer than the architectures the stable PyTorch
    // wheels are compiled for, which stop at sm_90. Installing the default
    // wheel on a 50-series card gets you "no kernel image is available for
    // execution on the device" -- a confusing failure that no driver update
    // fixes, so name the build that does work.
    const wheel = cap >= 12 ? 'cu128' : 'cu121';
    return { usable: true, cap, wheel, reason: null };
  }
  const era = cap < 3 ? 'Fermi' : cap < 3.5 ? 'Kepler' : cap < 5.3 ? 'Maxwell'
    : cap < 7 ? 'Pascal' : 'Volta';
  return {
    usable: false,
    cap,
    reason: `compute capability ${cap} (${era}) is below what current PyTorch wheels `
      + `(${MIN_TORCH_CAP}+) and the 3DGS CUDA rasterizer (${MIN_3DGS_CAP}+) require`,
  };
}

// --------------------------------------------------------------- supervisor

/**
 * Runs every enabled project as a child process and restarts it if it dies.
 * Deliberately simple: this is a personal machine, not a cluster.
 */
class Supervisor {
  constructor(state) {
    this.state = state;
    this.running = new Map();   // name -> { child, restarts, startedAt }
    this.stopping = false;
  }

  async startAll() {
    await fsp.mkdir(LOG_DIR, { recursive: true });
    for (const project of this.state.projects) {
      if (project.enabled === false || project.kind === 'static') continue;
      this.startOne(project);
    }
  }

  startOne(project) {
    if (this.running.has(project.name)) return;
    const logPath = path.join(LOG_DIR, `${project.name}.log`);
    const out = fs.createWriteStream(logPath, { flags: 'a' });
    out.write(`\n=== ${new Date().toISOString()} starting ${project.name} ===\n`);

    const child = spawn(project.start, {
      cwd: project.dir,
      shell: true,          // start commands are written as shell lines
      env: {
        ...process.env,
        // Bind to loopback: this is a personal machine, and a project should
        // not become reachable from the coffee-shop wifi by accident.
        HOST: '127.0.0.1',
        PORT: String(project.port),
        NODE_ENV: process.env.NODE_ENV || 'production',
        DATA_DIR: project.dataDir,
        SPLAT_DATA_DIR: project.dataDir,
        ...(project.env || {}),
      },
      stdio: ['ignore', 'pipe', 'pipe'],
      // `shell: true` means the direct child is a shell, so killing that alone
      // orphans whatever it started -- npm -> node keeps serving and the port
      // stays bound. The two platforms need opposite things to prevent that:
      // POSIX gets its own process group so the group can be signalled, while
      // Windows has no process groups at all (see signalGroup) and `detached`
      // there would only pop a console window and outlive us.
      ...(IS_WINDOWS ? { windowsHide: true } : { detached: true }),
    });
    child.stdout.pipe(out);
    child.stderr.pipe(out);

    const entry = { child, restarts: 0, startedAt: Date.now(), logPath };
    this.running.set(project.name, entry);

    child.on('exit', (code, signal) => {
      out.write(`=== exited code=${code} signal=${signal} ===\n`);
      this.running.delete(project.name);
      if (this.stopping) return;
      // Restart, but back off so a project that cannot start does not spin.
      const delay = Math.min(30_000, 1000 * 2 ** Math.min(entry.restarts, 5));
      entry.restarts += 1;
      if (entry.restarts <= 10) {
        setTimeout(() => {
          if (!this.stopping) this.startOne({ ...project });
        }, delay);
      } else {
        out.write('=== giving up after 10 restarts ===\n');
      }
    });
  }

  /** Signal every project's process group, then wait for them to actually go. */
  async stopAll(graceMs = 8000) {
    this.stopping = true;
    for (const [, entry] of [...this.running]) this.signalGroup(entry, 'SIGTERM');

    const deadline = Date.now() + graceMs;
    while (this.running.size && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 100));
    }
    // Anything still alive after the grace period gets no further say.
    for (const [name, entry] of [...this.running]) {
      this.signalGroup(entry, 'SIGKILL');
      this.running.delete(name);
    }
  }

  signalGroup(entry, signal) {
    const pid = entry.child.pid;
    if (!pid) return;

    if (IS_WINDOWS) {
      // Negative PIDs are a POSIX process-group idea and simply throw here.
      // taskkill /T walks the child tree, which is the equivalent guarantee.
      const args = ['/PID', String(pid), '/T'];
      if (signal === 'SIGKILL') args.push('/F');
      try {
        spawn('taskkill', args, { stdio: 'ignore', windowsHide: true })
          .on('error', () => { try { entry.child.kill(); } catch { /* gone */ } });
      } catch {
        try { entry.child.kill(); } catch { /* already gone */ }
      }
      return;
    }

    try {
      process.kill(-pid, signal);      // the whole group: shell, npm, node
    } catch {
      try { entry.child.kill(signal); } catch { /* already gone */ }
    }
  }
}

// -------------------------------------------------------------- lan gateway

/** First non-internal IPv4 address, i.e. how other machines reach this one. */
const VIRTUAL_ADAPTER = /^(veth|docker|br-|vEthernet|VMware|VirtualBox|Hyper-V|WSL|Loopback|TAP|Tailscale|ZeroTier)/i;
const WIRED_ADAPTER = /^(eth|en[a-z0-9]|Ethernet)/i;
const WIRELESS_ADAPTER = /^(wl|Wi-?Fi|Wireless)/i;

/**
 * Every usable IPv4 address on this machine, best candidate first.
 *
 * Ranking matters on a laptop, where several adapters are up at once: picking
 * whichever one the OS happened to list first can hand out an address that is
 * on the wrong network, or a 169.254.x.x from a port with no DHCP lease, and
 * the failure looks like a firewall problem rather than a wrong address.
 * Wired beats wireless because it is the steadier host, and anything virtual
 * (VM bridges, VPNs, WSL) loses to anything physical.
 */
export function lanCandidates(interfaces = os.networkInterfaces()) {
  const found = [];
  for (const [name, addrs] of Object.entries(interfaces)) {
    for (const addr of addrs || []) {
      if (addr.family !== 'IPv4' && addr.family !== 4) continue;
      if (addr.internal) continue;

      // 169.254.x.x means the adapter never got a lease -- nothing can reach it.
      const selfAssigned = addr.address.startsWith('169.254.');
      const virtual = VIRTUAL_ADAPTER.test(name);
      const wired = !virtual && WIRED_ADAPTER.test(name);
      const wireless = !virtual && WIRELESS_ADAPTER.test(name);

      let score = 1;                       // physical but unrecognised
      if (wired) score = 3;
      else if (wireless) score = 2;
      if (virtual) score = 0;
      if (selfAssigned) score = -1;        // never choose this on its own

      found.push({ name, address: addr.address, score, virtual, wired, wireless, selfAssigned });
    }
  }
  return found.sort((a, b) => b.score - a.score);
}

export function lanAddress(interfaces = os.networkInterfaces()) {
  const best = lanCandidates(interfaces).find((c) => c.score > 0);
  return best?.address || null;
}

export function hashPassword(password) {
  return crypto.createHash('sha256').update(String(password)).digest('hex');
}

/** @param passwordHash hex sha256, or empty/null for an open service. */
function checkBasicAuth(header, passwordHash) {
  if (!passwordHash) return true;
  if (!header || !header.startsWith('Basic ')) return false;
  let decoded;
  try {
    decoded = Buffer.from(header.slice(6), 'base64').toString('utf8');
  } catch {
    return false;
  }
  const supplied = decoded.slice(decoded.indexOf(':') + 1);
  // Fixed-width digests, so the comparison leaks neither length nor content.
  const a = Buffer.from(hashPassword(supplied), 'hex');
  const b = Buffer.from(passwordHash, 'hex');
  if (a.length !== b.length) return false;
  return crypto.timingSafeEqual(a, b);
}

const PASSWORD_FILE = path.join(HOME, 'password.sha256');

/**
 * Where the network password comes from, most explicit first.
 *
 * The stored form is a digest, and it is stored rather than passed as an
 * argument because a boot task and a desktop shortcut are both long-lived
 * plaintext files: putting --password in either would leave the secret sitting
 * on disk and in `ps` output for anyone on the machine.
 */
export function resolvePasswordHash(argv = {}, env = process.env, readFile = defaultReadPasswordFile) {
  if (typeof argv.password === 'string' && argv.password) return hashPassword(argv.password);
  if (env.CLAUDE_LOCAL_PASSWORD) return hashPassword(env.CLAUDE_LOCAL_PASSWORD);
  const stored = readFile();
  return stored || null;
}

function defaultReadPasswordFile() {
  try {
    const value = fs.readFileSync(PASSWORD_FILE, 'utf8').trim();
    return /^[0-9a-f]{64}$/.test(value) ? value : null;
  } catch {
    return null;
  }
}

async function savePasswordHash(hash) {
  await fsp.mkdir(HOME, { recursive: true });
  await fsp.writeFile(PASSWORD_FILE, `${hash}\n`, { mode: 0o600 });
  try { await fsp.chmod(PASSWORD_FILE, 0o600); } catch { /* Windows uses profile ACLs */ }
  return PASSWORD_FILE;
}

/**
 * Publish one loopback service on the LAN.
 *
 * Binding <lan-ip>:PORT does not collide with 127.0.0.1:PORT -- they are
 * different addresses -- so each project keeps its own port number and every
 * absolute URL inside it still resolves. That is why this forwards rather than
 * rewriting paths: a path prefix would break every "/api/..." the app emits.
 */
export function createGateway({ bindHost, port, targetPort, passwordHash, onError }) {
  const server = http.createServer((req, res) => {
    if (!checkBasicAuth(req.headers.authorization, passwordHash)) {
      res.writeHead(401, {
        'WWW-Authenticate': 'Basic realm="Claude Code projects", charset="UTF-8"',
        'content-type': 'text/plain',
      });
      res.end('authentication required\n');
      return;
    }
    const upstream = http.request(
      { host: '127.0.0.1', port: targetPort, path: req.url, method: req.method, headers: req.headers },
      (upstreamRes) => {
        res.writeHead(upstreamRes.statusCode, upstreamRes.headers);
        // Piped, never buffered, so server-sent events keep streaming.
        upstreamRes.pipe(res);
      },
    );
    upstream.on('error', (err) => {
      onError?.(err);
      if (!res.headersSent) res.writeHead(502, { 'content-type': 'text/plain' });
      res.end('project is not responding\n');
    });
    req.pipe(upstream);
  });

  // Carry websocket upgrades through as well, for projects that use them.
  server.on('upgrade', (req, socket, head) => {
    if (!checkBasicAuth(req.headers.authorization, passwordHash)) {
      socket.end('HTTP/1.1 401 Unauthorized\r\n\r\n');
      return;
    }
    const upstream = http.request({
      host: '127.0.0.1', port: targetPort, path: req.url, method: req.method,
      headers: req.headers,
    });
    upstream.on('upgrade', (upstreamRes, upstreamSocket, upstreamHead) => {
      const lines = Object.entries(upstreamRes.headers)
        .map(([k, v]) => `${k}: ${v}`).join('\r\n');
      socket.write(`HTTP/1.1 101 Switching Protocols\r\n${lines}\r\n\r\n`);
      if (upstreamHead?.length) socket.unshift(upstreamHead);
      upstreamSocket.pipe(socket).pipe(upstreamSocket);
    });
    upstream.on('error', () => socket.destroy());
    if (head?.length) upstream.write(head);
    upstream.end();
  });

  server.listen(port, bindHost);
  return server;
}

// --------------------------------------------------------------- dashboard

function dashboardHtml(state, health, viewHost = '127.0.0.1') {
  const rows = state.projects.map((p) => {
    const status = health[p.name] || 'unknown';
    const badge = status === 'up' ? 'up' : status === 'static' ? 'static' : 'down';
    // Build links from the host the browser actually used, or every link on
    // the dashboard points at the viewer's own machine instead of the server.
    const href = p.kind === 'static'
      ? `/static/${encodeURIComponent(p.name)}/`
      : `http://${viewHost}:${p.port}/`;
    return `
      <a class="card ${badge}" href="${href}"${p.kind === 'static' ? '' : ' target="_blank" rel="noreferrer"'}>
        <span class="dot"></span>
        <span class="body">
          <strong>${escapeHtml(p.name)}</strong>
          <em>${escapeHtml(p.kind)} &middot; port ${p.port}</em>
          <small>${escapeHtml(p.dir)}</small>
        </span>
      </a>`;
  }).join('');

  return `<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Claude Code projects</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0b0d12;--card:#141824;--line:#242b3c;--ink:#e8ebf4;--dim:#98a1b8;--up:#4ade80;--down:#ff6b6b;--stat:#f0b445}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;padding:40px 20px}
.wrap{max-width:760px;margin:0 auto}
h1{font-size:20px;font-weight:600;letter-spacing:-.02em;margin:0 0 4px}
p.sub{color:var(--dim);font-size:13.5px;margin:0 0 28px}
.grid{display:grid;gap:10px}
.card{display:flex;align-items:center;gap:14px;padding:14px 16px;background:var(--card);border:1px solid var(--line);border-radius:10px;text-decoration:none;color:inherit;transition:border-color .15s,transform .15s}
.card:hover{border-color:#39415c;transform:translateY(-1px)}
.dot{width:9px;height:9px;border-radius:50%;flex:none;background:var(--down)}
.card.up .dot{background:var(--up);box-shadow:0 0 8px #4ade8080}
.card.static .dot{background:var(--stat)}
.body{display:flex;flex-direction:column;gap:2px;min-width:0}
.body strong{font-weight:600;font-size:14.5px}
.body em{font-style:normal;color:var(--dim);font-size:12px;font-variant-numeric:tabular-nums}
.body small{color:#687189;font-size:11.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.empty{padding:40px;text-align:center;color:var(--dim);border:1px dashed var(--line);border-radius:10px}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#0f1320;border:1px solid var(--line);border-radius:4px;padding:1px 6px;font-size:12px}
footer{margin-top:28px;color:#687189;font-size:12px}
</style></head><body><div class="wrap">
<h1>Claude Code projects</h1>
<p class="sub">Running on this machine &middot; refreshed ${new Date().toLocaleTimeString()}</p>
<div class="grid">${rows || '<div class="empty">No projects yet. Add one with <code>claude-local add &lt;name&gt; --dir .</code></div>'}</div>
<footer><code>claude-local status</code> &middot; <code>claude-local logs &lt;name&gt;</code> &middot; <code>claude-local down</code></footer>
</div>
<script>setTimeout(()=>location.reload(),5000)</script>
</body></html>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
}

async function probe(project) {
  if (project.kind === 'static') return 'static';
  return new Promise((resolve) => {
    const req = http.get(
      { host: '127.0.0.1', port: project.port, path: project.health || '/', timeout: 1500 },
      (res) => { res.resume(); resolve(res.statusCode < 500 ? 'up' : 'down'); },
    );
    req.on('error', () => resolve('down'));
    req.on('timeout', () => { req.destroy(); resolve('down'); });
  });
}

function serveStatic(project, urlPath, res) {
  const root = path.resolve(project.dir);
  const rel = decodeURIComponent(urlPath) || '/';
  let file = path.join(root, rel === '/' ? 'index.html' : rel);
  if (!file.startsWith(root)) { res.writeHead(403).end('forbidden'); return; }
  if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, 'index.html');
  if (!fs.existsSync(file)) { res.writeHead(404).end('not found'); return; }
  const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css',
    '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml' };
  res.writeHead(200, { 'content-type': types[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
}

function startDashboard(state) {
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');

    const staticMatch = /^\/static\/([^/]+)(\/.*)?$/.exec(url.pathname);
    if (staticMatch) {
      const project = findProject(state, decodeURIComponent(staticMatch[1]));
      if (!project) { res.writeHead(404).end('no such project'); return; }
      serveStatic(project, staticMatch[2] || '/', res);
      return;
    }

    if (url.pathname === '/api/status') {
      const health = {};
      await Promise.all(state.projects.map(async (p) => { health[p.name] = await probe(p); }));
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ projects: state.projects, health }, null, 2));
      return;
    }

    const health = {};
    await Promise.all(state.projects.map(async (p) => { health[p.name] = await probe(p); }));
    const hostHeader = req.headers.host || '127.0.0.1';
    const viewHost = hostHeader.replace(/:\d+$/, '').replace(/^\[|\]$/g, '') || '127.0.0.1';
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' });
    res.end(dashboardHtml(state, health, viewHost));
  });
  // Loopback only, deliberately.
  server.listen(DASHBOARD_PORT, '127.0.0.1');
  return server;
}

// ---------------------------------------------------------------- commands

async function cmdAdd(argv) {
  const name = argv._[0];
  if (!validName(name)) throw new Error('name must be lowercase letters, digits and dashes');
  const state = await loadState();
  if (findProject(state, name)) throw new Error(`project '${name}' already exists`);

  const dir = path.resolve(argv.dir || process.cwd());
  if (!fs.existsSync(dir)) throw new Error(`no such directory: ${dir}`);

  const kind = argv.kind || detectKind(dir);
  const start = argv.start ?? defaultStart(kind, dir);
  if (kind !== 'static' && !start) {
    throw new Error(`could not work out how to start '${name}'; pass --start "..."`);
  }

  // An explicit port has to be checked. Auto-allocation avoids collisions by
  // construction, but --port did not, so two projects could claim the same one
  // and the second would fail to bind long after registration looked fine.
  let port;
  if (argv.port !== undefined && argv.port !== true) {
    port = Number(argv.port);
    if (!Number.isInteger(port) || port < 1 || port > 65535) {
      throw new Error(`--port must be a number between 1 and 65535, got '${argv.port}'`);
    }
    const clash = state.projects.find((p) => p.port === port);
    if (clash) throw new Error(`port ${port} is already used by '${clash.name}'`);
  } else {
    port = nextPort(state);
  }

  const project = {
    name,
    dir,
    kind,
    start,
    port,
    health: argv.health || '/',
    dataDir: path.join(HOME, 'data', name),
    enabled: true,
  };
  await fsp.mkdir(project.dataDir, { recursive: true });
  state.projects.push(project);
  await saveState(state);

  console.log(`${c.green('added')} ${name} (${kind}) on port ${project.port}`);
  console.log(c.dim(`  ${dir}`));
  console.log(c.dim(`  start: ${start || '(served as static files)'}`));
  console.log(`\nRun ${c.bold('claude-local up')} to start everything.`);
}

async function cmdList() {
  const state = await loadState();
  if (!state.projects.length) { console.log(c.dim('no projects yet')); return; }
  console.log(`${'NAME'.padEnd(20)}${'KIND'.padEnd(9)}${'PORT'.padEnd(7)}DIRECTORY`);
  for (const p of state.projects) {
    console.log(`${p.name.padEnd(20)}${p.kind.padEnd(9)}${String(p.port).padEnd(7)}${c.dim(p.dir)}`);
  }
}

async function cmdStatus() {
  const state = await loadState();
  const supervisor = readPid();
  console.log(supervisor ? `${c.green('supervisor running')} (pid ${supervisor})`
                         : c.yellow('supervisor not running'));
  console.log(`dashboard: http://127.0.0.1:${DASHBOARD_PORT}\n`);
  if (!state.projects.length) { console.log(c.dim('no projects yet')); return; }
  console.log(`${'NAME'.padEnd(20)}${'PORT'.padEnd(7)}HEALTH`);
  for (const p of state.projects) {
    const status = await probe(p);
    const mark = status === 'up' ? c.green('up') : status === 'static' ? c.yellow('static') : c.red('down');
    console.log(`${p.name.padEnd(20)}${String(p.port).padEnd(7)}${mark}`);
  }
}

async function cmdRemove(argv) {
  const name = argv._[0];
  const state = await loadState();
  const index = state.projects.findIndex((p) => p.name === name);
  if (index === -1) throw new Error(`no such project: ${name}`);
  state.projects.splice(index, 1);
  await saveState(state);
  console.log(`${c.green('removed')} ${name}`);
}

async function cmdUp(argv = { _: [] }) {
  if (readPid()) throw new Error('already running; use `claude-local down` first');
  const state = await loadState();
  await fsp.mkdir(LOG_DIR, { recursive: true });

  const wantsLan = Boolean(argv.lan || argv.host);
  const passwordHash = resolvePasswordHash(argv);
  let bindHost = null;

  if (wantsLan) {
    bindHost = typeof argv.host === 'string' ? argv.host : lanAddress();
    if (!bindHost) {
      throw new Error('could not find a LAN address on this machine; pass --host <ip>');
    }
    // These projects accept uploads and can delete a library. Publishing them
    // to a network without a password should be a decision, not a default.
    if (!passwordHash && !argv['allow-anonymous']) {
      throw new Error(
        'refusing to publish on the network without a password.\n'
        + '  Add --password <secret>, or --allow-anonymous if this network is trusted\n'
        + '  and you accept that anyone on it can use and delete these projects.',
      );
    }
  }

  const supervisor = new Supervisor(state);
  await supervisor.startAll();
  const server = startDashboard(state);

  const gateways = [];
  if (bindHost) {
    const publish = (port, targetPort) => {
      const gw = createGateway({
        bindHost, port, targetPort, passwordHash,
        onError: () => {},
      });
      gw.on('error', (err) => console.error(`${c.red('gateway')} ${port}: ${err.message}`));
      gateways.push(gw);
    };
    publish(DASHBOARD_PORT, DASHBOARD_PORT);
    for (const p of state.projects) {
      if (p.kind !== 'static') publish(p.port, p.port);
    }
  }

  fs.writeFileSync(PID_FILE, String(process.pid));

  let shuttingDown = false;
  const shutdown = async () => {
    if (shuttingDown) return;
    shuttingDown = true;
    server.close();
    for (const gw of gateways) { try { gw.close(); } catch { /* already closed */ } }
    // Wait for the children to die before the supervisor does, or `down`
    // reports success while projects still hold their ports.
    await supervisor.stopAll();
    try { fs.unlinkSync(PID_FILE); } catch { /* already gone */ }
    process.exit(0);
  };
  process.on('SIGINT', () => { void shutdown(); });
  process.on('SIGTERM', () => { void shutdown(); });

  const shown = bindHost || '127.0.0.1';
  console.log(`${c.green('up')} — dashboard at ${c.bold(`http://${shown}:${DASHBOARD_PORT}`)}`);
  for (const p of state.projects) {
    if (p.kind === 'static') console.log(`  ${p.name}  ${c.dim('served on the dashboard')}`);
    else console.log(`  ${p.name}  http://${shown}:${p.port}`);
  }
  if (bindHost) {
    console.log(`\n${c.green('published on the network')} as ${c.bold(os.hostname())} (${bindHost})`);
    console.log(c.dim(`  from another machine: http://${os.hostname()}:${DASHBOARD_PORT}`));
    if (passwordHash) console.log(c.dim('  a password is required'));
    else console.log(c.yellow('  no password: anyone on this network can use and delete these projects'));
  }
  console.log(c.dim('\nCtrl-C to stop, or run `claude-local down` from another shell.'));
}

function readPid() {
  try {
    const pid = Number(fs.readFileSync(PID_FILE, 'utf8').trim());
    process.kill(pid, 0);      // throws when the process is gone
    return pid;
  } catch {
    return null;
  }
}

async function cmdDown() {
  const pid = readPid();
  if (!pid) { console.log(c.dim('not running')); return; }

  if (IS_WINDOWS) {
    // Windows has no real signals: Node maps process.kill(pid, 'SIGTERM') onto
    // an unconditional terminate, so the supervisor dies before its shutdown
    // handler can stop the servers it started. They survive, still holding
    // their ports, and the next `up` fails on an address already in use.
    // taskkill /T ends the whole tree instead, which is the guarantee SIGTERM
    // gives on POSIX.
    await new Promise((resolve) => {
      const kill = spawn('taskkill', ['/PID', String(pid), '/T', '/F'],
        { stdio: 'ignore', windowsHide: true });
      kill.on('error', () => { try { process.kill(pid); } catch { /* gone */ } resolve(); });
      kill.on('exit', resolve);
    });
  } else {
    process.kill(pid, 'SIGTERM');
  }
  console.log(`${c.green('stopped')} supervisor (pid ${pid}) and the servers it started`);
}

async function cmdLogs(argv) {
  const name = argv._[0];
  const file = path.join(LOG_DIR, `${name}.log`);
  if (!fs.existsSync(file)) throw new Error(`no log yet for '${name}'`);
  if (argv.f || argv.follow) {
    // Polling in Node rather than shelling out to `tail`, which Windows has
    // no equivalent of, and which fs.watch reports unreliably across platforms.
    let position = 0;
    const initial = await fsp.readFile(file, 'utf8');
    process.stdout.write(initial.split('\n').slice(-100).join('\n'));
    position = Buffer.byteLength(initial);

    await new Promise((resolve) => {
      process.on('SIGINT', resolve);
      const timer = setInterval(async () => {
        try {
          const { size } = await fsp.stat(file);
          if (size < position) position = 0;          // truncated or rotated
          if (size > position) {
            const chunk = fs.createReadStream(file, { start: position, end: size - 1 });
            for await (const part of chunk) process.stdout.write(part);
            position = size;
          }
        } catch { /* file briefly missing during a restart */ }
      }, 500);
      timer.unref?.();
    });
  } else {
    const text = await fsp.readFile(file, 'utf8');
    console.log(text.split('\n').slice(-200).join('\n'));
  }
}

async function cmdDoctor() {
  const checks = [];
  const probeCmd = async (label, cmd, args) => {
    try {
      const { stdout } = await execFileAsync(cmd, args, { timeout: 15_000 });
      checks.push([true, label, stdout.trim().split('\n')[0].slice(0, 60)]);
    } catch {
      checks.push([false, label, 'not found']);
    }
  };

  console.log(c.bold(`host: ${os.type()} ${os.release()} (${os.arch()}), ${os.cpus().length} cores, `
    + `${Math.round(os.totalmem() / 1e9)} GB RAM\n`));

  await probeCmd('node', 'node', ['--version']);
  await probeCmd('git', 'git', ['--version']);
  await probeCmd('python3', 'python3', ['--version']);
  await probeCmd('docker', 'docker', ['--version']);

  // The GPU question decides whether real gaussian training is minutes or hours
  // -- or, for an old card, impossible.
  let gpu = null;
  let verdict = null;
  for (const query of ['name,memory.total,compute_cap', 'name,memory.total']) {
    try {
      const { stdout } = await execFileAsync('nvidia-smi',
        [`--query-gpu=${query}`, '--format=csv,noheader'], { timeout: 15_000 });
      const [name, memory, cap] = stdout.trim().split('\n')[0].split(',').map((v) => v.trim());
      gpu = { name, memory, cap: cap ?? null };
      verdict = classifyGpu(name, cap ?? null);
      break;
    } catch { /* try the simpler query, then give up */ }
  }
  if (gpu) {
    checks.push([verdict.usable, 'nvidia gpu', `${gpu.name}, ${gpu.memory}`]);
  } else {
    checks.push([false, 'nvidia gpu', os.type() === 'Darwin' ? 'none (Apple silicon)' : 'none']);
  }

  for (const [ok, label, detail] of checks) {
    console.log(`  ${ok ? c.green('ok ') : c.red('no ')} ${label.padEnd(12)} ${c.dim(detail)}`);
  }

  const adapters = lanCandidates();
  if (adapters.length) {
    console.log('\n' + c.bold('network addresses') + c.dim('  (used by --lan; override with --host)'));
    for (const a of adapters) {
      const kind = a.virtual ? 'virtual' : a.wired ? 'wired' : a.wireless ? 'wi-fi' : 'other';
      const note = a.selfAssigned ? c.yellow('  no DHCP lease, unreachable') : '';
      const mark = a.score > 0 && a === adapters[0] ? c.green(' <- chosen') : '';
      console.log(`  ${a.address.padEnd(16)} ${kind.padEnd(8)} ${c.dim(a.name)}${mark}${note}`);
    }
  }

  console.log('\n' + c.bold('gaussian splat training on this machine:'));
  if (gpu && verdict.usable) {
    const wheel = verdict.wheel || 'cu121';
    console.log(`  ${c.green(gpu.name)} can run CUDA training`
      + (verdict.cap ? c.dim(`  (compute capability ${verdict.cap})`) : ''));
    console.log('');
    console.log('  The bundled trainer is plain PyTorch, so it runs on this GPU directly.');
    console.log('  Install a CUDA build of PyTorch -- the default wheel is CPU-only on');
    console.log(`  Windows, and on this card you need the ${c.bold(wheel)} build specifically:`);
    console.log(c.dim(`    pip install torch --index-url https://download.pytorch.org/whl/${wheel}`));
    console.log(c.dim('    pip install pycolmap numpy pillow'));
    console.log('');
    console.log('  Check it took:');
    console.log(c.dim('    python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"'));
    console.log('');
    console.log('  Then convert with the GPU instead of the CPU:');
    // Printing a POSIX assignment on Windows would be advice that silently
    // does nothing -- cmd and PowerShell each spell this a third way.
    console.log(c.dim(IS_WINDOWS
      ? '    $env:SPLAT_TRAIN_DEVICE = "cuda"     (PowerShell, this window only)'
      : '    export SPLAT_TRAIN_DEVICE=cuda'));
    if (IS_WINDOWS) {
      console.log(c.dim('    setx SPLAT_TRAIN_DEVICE cuda          (permanent, new windows only)'));
    }
    if (verdict.cap >= 12) {
      console.log(c.yellow('\n  This is a Blackwell card. The stable PyTorch wheels stop at sm_90, so'));
      console.log(c.yellow('  the default install fails with "no kernel image is available for'));
      console.log(c.yellow(`  execution on the device". The ${wheel} build is not optional here.`));
    }
  } else if (gpu) {
    console.log(`  ${c.yellow(gpu.name)} cannot run it: ${verdict.reason}.`);
    console.log('  This is a hard limit, not a slow path -- no combination of drivers');
    console.log('  makes it work. Use the bundled CPU trainer, which is the same');
    console.log('  algorithm at lower speed.');
  } else if (os.type() === 'Darwin') {
    console.log('  No CUDA GPU. The bundled trainer runs on CPU here and is real but slow;');
    console.log('  PyTorch\'s Metal (mps) backend is not yet wired into the rasterizer.');
  } else {
    console.log('  No CUDA GPU detected, so the bundled trainer runs on CPU: real');
    console.log('  reconstruction, but minutes-to-hours rather than minutes.');
  }
  console.log(c.dim('\n  Either way: pip install -r trainer/requirements.txt'));
}

async function cmdAutostart() {
  const type = os.type();
  const script = path.resolve(process.argv[1]);
  const node = process.execPath;

  if (type === 'Linux') {
    const unitDir = path.join(os.homedir(), '.config', 'systemd', 'user');
    await fsp.mkdir(unitDir, { recursive: true });
    const unit = `[Unit]
Description=Claude Code projects (local)
After=network-online.target

[Service]
Type=simple
ExecStart=${node} ${script} up
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
`;
    const unitPath = path.join(unitDir, 'claude-local.service');
    await fsp.writeFile(unitPath, unit);
    console.log(`wrote ${unitPath}\n`);
    console.log('Enable it with:');
    console.log(c.bold('  systemctl --user daemon-reload'));
    console.log(c.bold('  systemctl --user enable --now claude-local'));
    console.log(c.dim('  loginctl enable-linger $USER   # keep running after logout'));
  } else if (type === 'Darwin') {
    const plistDir = path.join(os.homedir(), 'Library', 'LaunchAgents');
    await fsp.mkdir(plistDir, { recursive: true });
    const plist = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.claude.local</string>
  <key>ProgramArguments</key><array>
    <string>${node}</string><string>${script}</string><string>up</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>${path.join(LOG_DIR, 'supervisor.log')}</string>
  <key>StandardErrorPath</key><string>${path.join(LOG_DIR, 'supervisor.log')}</string>
</dict></plist>
`;
    const plistPath = path.join(plistDir, 'com.claude.local.plist');
    await fsp.mkdir(LOG_DIR, { recursive: true });
    await fsp.writeFile(plistPath, plist);
    console.log(`wrote ${plistPath}\n`);
    console.log('Enable it with:');
    console.log(c.bold(`  launchctl load -w ${plistPath}`));
  } else if (IS_WINDOWS) {
    // The Startup folder needs no admin rights and is easy to undo -- better
    // than a scheduled task for something the user should be able to remove.
    const startup = path.join(os.homedir(), 'AppData', 'Roaming', 'Microsoft',
      'Windows', 'Start Menu', 'Programs', 'Startup');
    await fsp.mkdir(startup, { recursive: true });
    // A VBScript wrapper launches it with no console window flashing up.
    const q = (v) => `""${v}""`;
    const vbs = [
      "' Starts Claude Code projects at login. Delete this file to stop.",
      'Set sh = CreateObject("WScript.Shell")',
      `sh.Run "${q(node)} ${q(script)} up", 0, False`,
      '',
    ].join('\r\n');
    const vbsPath = path.join(startup, 'claude-local.vbs');
    await fsp.writeFile(vbsPath, vbs);
    console.log(`wrote ${vbsPath}\n`);
    console.log('It will start at your next login. To start it now:');
    console.log(c.bold(`  wscript "${vbsPath}"`));
    console.log(c.dim('To disable, delete that file.'));
  } else {
    console.log(`Unrecognised platform (${type}). Run this at login:`);
    console.log(c.bold(`  ${node} ${script} up`));
  }
}

/**
 * Set up a Windows machine to host these projects: a boot-time scheduled task,
 * a firewall rule, and desktop shortcuts.
 *
 * Deliberately writes files and prints commands rather than running the
 * privileged ones itself -- creating a task that runs without login needs the
 * account password, and the Task Scheduler import prompts for it securely
 * instead of leaving it in a command line and shell history.
 */
async function cmdWindowsSetup(argv) {
  if (!IS_WINDOWS && !argv.force) {
    console.log(c.yellow('This prepares a Windows host; run it there (or pass --force to generate the files anyway).'));
    if (!argv.force) return;
  }

  const state = await loadState();
  const outDir = path.join(HOME, 'windows');
  await fsp.mkdir(outDir, { recursive: true });

  const node = process.execPath;
  const script = path.resolve(process.argv[1]);
  // Store the digest; never write the secret into the task or the shortcuts.
  let storedAt = null;
  if (typeof argv.password === 'string' && argv.password) {
    storedAt = await savePasswordHash(hashPassword(argv.password));
  }
  const passwordHash = resolvePasswordHash({});     // file/env only, not argv
  const args = ['up', '--lan'];
  if (!passwordHash) args.push('--allow-anonymous');

  // A scheduled task at startup, so it serves without anyone logging in.
  const taskXml = `<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Hosts Claude Code projects on this machine.</Description>
  </RegistrationInfo>
  <Triggers>
    <BootTrigger><Enabled>true</Enabled></BootTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>Password</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <RestartOnFailure><Interval>PT1M</Interval><Count>3</Count></RestartOnFailure>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>${node}</Command>
      <Arguments>"${script}" ${args.map((a) => (a.startsWith('--') ? a : `"${a}"`)).join(' ')}</Arguments>
      <WorkingDirectory>${path.dirname(script)}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
`;
  const xmlPath = path.join(outDir, 'claude-local-task.xml');
  // Task Scheduler wants UTF-16 with a BOM; UTF-8 is rejected on import.
  await fsp.writeFile(xmlPath, '\ufeff' + taskXml, 'utf16le');

  // Desktop shortcuts, as .cmd so they are readable and easy to delete.
  const desktop = path.join(os.homedir(), 'Desktop');
  const shortcuts = {
    'Start projects.cmd': `@echo off\r\n"${node}" "${script}" ${args.join(' ')}\r\npause\r\n`,
    'Stop projects.cmd': `@echo off\r\n"${node}" "${script}" down\r\npause\r\n`,
    'Open projects.cmd': `@echo off\r\nstart http://localhost:${DASHBOARD_PORT}\r\n`,
  };
  const written = [];
  for (const [file, body] of Object.entries(shortcuts)) {
    const target = path.join(fs.existsSync(desktop) ? desktop : outDir, file);
    await fsp.writeFile(target, body);
    written.push(target);
  }

  const ports = [DASHBOARD_PORT, ...state.projects.filter((p) => p.kind !== 'static').map((p) => p.port)];
  const firewall = `netsh advfirewall firewall add rule name="Claude Code projects" `
    + `dir=in action=allow protocol=TCP localport=${ports.join(',')} profile=private`;

  console.log(c.bold('Files written'));
  console.log(`  ${xmlPath}`);
  for (const f of written) console.log(`  ${f}`);

  console.log('\n' + c.bold('1. Allow the ports through the firewall') + c.dim('  (Administrator prompt)'));
  console.log(`   ${firewall}`);
  console.log(c.dim('   profile=private keeps it off public networks. Change to =domain on a work domain.'));

  console.log('\n' + c.bold('2. Run at boot, without anyone logged in') + c.dim('  (Administrator prompt)'));
  console.log(`   schtasks /create /tn "Claude Code projects" /xml "${xmlPath}" /ru "%USERNAME%"`);
  console.log(c.dim('   Windows will prompt for your password. It is stored by the Task'));
  console.log(c.dim('   Scheduler, not by this tool, and never appears in your command history.'));
  console.log(c.dim(`   Start it now:  schtasks /run /tn "Claude Code projects"`));

  console.log('\n' + c.bold('3. Reach it from another machine'));
  const lan = lanAddress();
  console.log(`   http://${os.hostname()}:${DASHBOARD_PORT}${lan ? `   or   http://${lan}:${DASHBOARD_PORT}` : ''}`);
  if (storedAt) {
    console.log(c.dim(`\n   Password stored as a hash in ${storedAt}; the task and shortcuts`));
    console.log(c.dim('   contain no secret.'));
  } else if (!passwordHash) {
    console.log(c.yellow('\n   No password set. Anyone on this network can use and delete these'));
    console.log(c.yellow('   projects. Re-run with --password <secret> to require one.'));
  }
}

// -------------------------------------------------------------------- main

function parseArgv(args) {
  const out = { _: [] };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a.startsWith('--')) {
      const key = a.slice(2);
      const next = args[i + 1];
      if (next === undefined || next.startsWith('--')) out[key] = true;
      else { out[key] = next; i += 1; }
    } else if (a.startsWith('-') && a.length > 1) {
      out[a.slice(1)] = true;
    } else {
      out._.push(a);
    }
  }
  return out;
}

const USAGE = `claude-local -- run your Claude Code projects on this computer

  add <name> [--dir PATH] [--port N] [--kind K] [--start "CMD"] [--health PATH]
  list                     registered projects
  status                   supervisor and per-project health
  up                       start everything, with a dashboard
  down                     stop everything
  logs <name> [-f]         output from one project
  remove <name>            unregister
  doctor                   what this machine can do (including GPU)
  autostart                start at login (this OS)
  windows-setup            host on Windows: boot task, firewall, shortcuts

up options:
  --lan                    publish on the local network, not just this machine
  --host IP                bind a specific address (default: detected LAN address)
  --password SECRET        require a password over the network
  --allow-anonymous        publish with no password (say so deliberately)

State lives in ~/.claude-projects. Projects bind to 127.0.0.1 unless you pass
--lan, which also publishes them on this machine's network address.`;

const COMMANDS = {
  add: cmdAdd, list: cmdList, ls: cmdList, status: cmdStatus, up: cmdUp,
  down: cmdDown, logs: cmdLogs, remove: cmdRemove, rm: cmdRemove,
  doctor: cmdDoctor, autostart: cmdAutostart, 'windows-setup': cmdWindowsSetup,
};

async function main() {
  const [, , cmd, ...rest] = process.argv;
  if (!cmd || cmd === 'help' || cmd === '--help' || cmd === '-h') { console.log(USAGE); return; }
  const handler = COMMANDS[cmd];
  if (!handler) { console.log(USAGE); process.exitCode = 1; return; }
  try {
    await handler(parseArgv(rest));
  } catch (err) {
    console.error(`${c.red('error')} ${err.message}`);
    process.exitCode = 1;
  }
}

/**
 * Is this module the program the user actually launched?
 *
 * `new URL(import.meta.url).pathname` looks like the obvious answer and is
 * wrong on Windows: it yields "/C:/Apps/..." with a leading slash, which never
 * equals argv[1]'s "C:\\Apps\\...". The comparison silently fails, main()
 * never runs, and the command exits 0 having printed nothing -- which is
 * exactly how this presented. fileURLToPath understands drive letters.
 *
 * `windows` is a seam so the Windows behaviour can be tested from any platform.
 */
export function isMainModule(argv1, metaUrl, windows = process.platform === 'win32') {
  if (!argv1) return false;
  const impl = windows ? path.win32 : path.posix;
  try {
    return impl.resolve(argv1) === impl.resolve(fileURLToPath(metaUrl, { windows }));
  } catch {
    return false;
  }
}

if (isMainModule(process.argv[1], import.meta.url)) main();
