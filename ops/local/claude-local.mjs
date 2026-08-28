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
import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

const HOME = process.env.CLAUDE_LOCAL_HOME || path.join(os.homedir(), '.claude-projects');
const STATE_FILE = path.join(HOME, 'projects.json');
const PID_FILE = path.join(HOME, 'supervisor.pid');
const LOG_DIR = path.join(HOME, 'logs');
const DASHBOARD_PORT = Number(process.env.CLAUDE_LOCAL_PORT || 7777);
const PORT_BASE = 8801;

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
      // Its own process group. `shell: true` means the direct child is a shell,
      // so killing that alone orphans whatever it started -- npm -> node keeps
      // serving and the port stays bound. Signalling the group gets all of it.
      detached: true,
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
    try {
      process.kill(-pid, signal);      // the whole group: shell, npm, node
    } catch {
      try { entry.child.kill(signal); } catch { /* already gone */ }
    }
  }
}

// --------------------------------------------------------------- dashboard

function dashboardHtml(state, health) {
  const rows = state.projects.map((p) => {
    const status = health[p.name] || 'unknown';
    const badge = status === 'up' ? 'up' : status === 'static' ? 'static' : 'down';
    const href = p.kind === 'static'
      ? `/static/${encodeURIComponent(p.name)}/`
      : `http://127.0.0.1:${p.port}/`;
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
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' });
    res.end(dashboardHtml(state, health));
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

async function cmdUp() {
  if (readPid()) throw new Error('already running; use `claude-local down` first');
  const state = await loadState();
  await fsp.mkdir(LOG_DIR, { recursive: true });

  const supervisor = new Supervisor(state);
  await supervisor.startAll();
  const server = startDashboard(state);
  fs.writeFileSync(PID_FILE, String(process.pid));

  let shuttingDown = false;
  const shutdown = async () => {
    if (shuttingDown) return;
    shuttingDown = true;
    server.close();
    // Wait for the children to die before the supervisor does, or `down`
    // reports success while projects still hold their ports.
    await supervisor.stopAll();
    try { fs.unlinkSync(PID_FILE); } catch { /* already gone */ }
    process.exit(0);
  };
  process.on('SIGINT', () => { void shutdown(); });
  process.on('SIGTERM', () => { void shutdown(); });

  console.log(`${c.green('up')} — dashboard at ${c.bold(`http://127.0.0.1:${DASHBOARD_PORT}`)}`);
  for (const p of state.projects) {
    if (p.kind === 'static') console.log(`  ${p.name}  ${c.dim('static')}`);
    else console.log(`  ${p.name}  http://127.0.0.1:${p.port}`);
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
  process.kill(pid, 'SIGTERM');
  console.log(`${c.green('stopped')} supervisor (pid ${pid})`);
}

async function cmdLogs(argv) {
  const name = argv._[0];
  const file = path.join(LOG_DIR, `${name}.log`);
  if (!fs.existsSync(file)) throw new Error(`no log yet for '${name}'`);
  if (argv.f || argv.follow) {
    const tail = spawn('tail', ['-f', '-n', '100', file], { stdio: 'inherit' });
    await new Promise((r) => tail.on('exit', r));
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

  // The GPU question decides whether real gaussian training is minutes or hours.
  let gpu = null;
  try {
    const { stdout } = await execFileAsync('nvidia-smi',
      ['--query-gpu=name,memory.total', '--format=csv,noheader'], { timeout: 15_000 });
    gpu = stdout.trim();
    checks.push([true, 'nvidia gpu', gpu]);
  } catch {
    checks.push([false, 'nvidia gpu', os.type() === 'Darwin' ? 'none (Apple silicon: see below)' : 'none']);
  }

  for (const [ok, label, detail] of checks) {
    console.log(`  ${ok ? c.green('ok ') : c.red('no ')} ${label.padEnd(12)} ${c.dim(detail)}`);
  }

  console.log('\n' + c.bold('gaussian splat training on this machine:'));
  if (gpu) {
    console.log('  A CUDA GPU is present. For fast, full-quality reconstruction install a');
    console.log('  GPU trainer and point the app at it:');
    console.log(c.dim('    SPLAT_TRAINER_CMD="python /opt/gaussian-splatting/train.py \\'));
    console.log(c.dim('      -s {source} -m {output} --iterations {iterations}"'));
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
  } else {
    console.log('On Windows, run this inside WSL2 and use the Linux instructions,');
    console.log('or create a Task Scheduler entry running:');
    console.log(c.bold(`  ${node} ${script} up`));
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
  autostart                install a login/boot service for this OS

State lives in ~/.claude-projects. Everything binds to 127.0.0.1.`;

const COMMANDS = {
  add: cmdAdd, list: cmdList, ls: cmdList, status: cmdStatus, up: cmdUp,
  down: cmdDown, logs: cmdLogs, remove: cmdRemove, rm: cmdRemove,
  doctor: cmdDoctor, autostart: cmdAutostart,
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

const invokedDirectly = process.argv[1]
  && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname);
if (invokedDirectly) main();
