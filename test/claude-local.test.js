import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { execFile, spawn } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

import { detectKind, defaultStart, validName } from '../ops/local/claude-local.mjs';

const execFileAsync = promisify(execFile);
const CLI = fileURLToPath(new URL('../ops/local/claude-local.mjs', import.meta.url));

async function tempHome() {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), 'claude-local-'));
  return dir;
}

function run(home, args, extraEnv = {}) {
  return execFileAsync(process.execPath, [CLI, ...args], {
    env: { ...process.env, CLAUDE_LOCAL_HOME: home, ...extraEnv },
    encoding: 'utf8',
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function reachable(port, pathname = '/') {
  try {
    const res = await fetch(`http://127.0.0.1:${port}${pathname}`, {
      signal: AbortSignal.timeout(1500),
    });
    return res.status;
  } catch {
    return null;
  }
}

test('project names are validated', () => {
  assert.ok(validName('splatworks'));
  assert.ok(validName('my-app-2'));
  assert.ok(!validName('Bad Name'));
  assert.ok(!validName('-leading'));
  assert.ok(!validName('trailing-'));
  assert.ok(!validName('../escape'));
  assert.ok(!validName(''));
});

test('project kind is detected from the checkout', async () => {
  const dir = await tempHome();
  const make = async (name, file) => {
    const sub = path.join(dir, name);
    await fsp.mkdir(sub, { recursive: true });
    await fsp.writeFile(path.join(sub, file), '{}');
    return sub;
  };
  assert.equal(detectKind(await make('n', 'package.json')), 'node');
  assert.equal(detectKind(await make('p', 'requirements.txt')), 'python');
  assert.equal(detectKind(await make('s', 'index.html')), 'static');
  assert.equal(detectKind(await make('u', 'README')), 'unknown');

  // A compose file must win, or a containerised app gets run bare.
  const both = await make('d', 'docker-compose.yml');
  await fsp.writeFile(path.join(both, 'package.json'), '{}');
  assert.equal(detectKind(both), 'docker');
});

test('start command comes from the project type', async () => {
  const dir = await tempHome();
  const withStart = path.join(dir, 'a');
  await fsp.mkdir(withStart, { recursive: true });
  await fsp.writeFile(path.join(withStart, 'package.json'),
    JSON.stringify({ scripts: { start: 'node .' } }));
  assert.equal(defaultStart('node', withStart), 'npm start');

  const noStart = path.join(dir, 'b');
  await fsp.mkdir(noStart, { recursive: true });
  await fsp.writeFile(path.join(noStart, 'package.json'), '{}');
  assert.equal(defaultStart('node', noStart), 'node server/index.js');

  assert.equal(defaultStart('docker', dir), 'docker compose up');
  assert.equal(defaultStart('static', dir), '', 'static projects have no process');
});

test('projects register, list and unregister', async () => {
  const home = await tempHome();
  const src = path.join(home, 'src');
  await fsp.mkdir(src, { recursive: true });
  await fsp.writeFile(path.join(src, 'index.html'), '<h1>hi</h1>');

  const added = await run(home, ['add', 'notes', '--dir', src]);
  assert.match(added.stdout, /added notes \(static\)/);

  const listed = await run(home, ['list']);
  assert.match(listed.stdout, /notes/);
  assert.match(listed.stdout, /static/);

  // Ports are allocated around what is already taken, and an explicit port
  // that is already claimed is refused rather than silently duplicated.
  await fsp.writeFile(path.join(src, 'package.json'), '{}');
  const taken = JSON.parse(await fsp.readFile(path.join(home, 'projects.json'), 'utf8'))
    .projects[0].port;
  await assert.rejects(
    run(home, ['add', 'clash', '--dir', src, '--port', String(taken)]),
    /already used by 'notes'/,
  );
  await assert.rejects(run(home, ['add', 'weird', '--dir', src, '--port', 'abc']), /--port must be/);

  await run(home, ['add', 'api', '--dir', src, '--port', '8901']);
  await run(home, ['add', 'other', '--dir', src]);
  const state = JSON.parse(await fsp.readFile(path.join(home, 'projects.json'), 'utf8'));
  const ports = state.projects.map((p) => p.port);
  assert.equal(new Set(ports).size, ports.length, `ports collided: ${ports}`);

  await assert.rejects(run(home, ['add', 'notes', '--dir', src]), /already exists/);
  await assert.rejects(run(home, ['add', 'Bad Name', '--dir', src]), /lowercase/);

  await run(home, ['remove', 'notes']);
  const after = await run(home, ['list']);
  assert.ok(!/^notes/m.test(after.stdout), 'removed project should be gone');
});

test('state survives a corrupt-free round trip and reports bad JSON clearly', async () => {
  const home = await tempHome();
  await fsp.writeFile(path.join(home, 'projects.json'), '{ not json');
  await assert.rejects(run(home, ['list']), /not readable JSON/);
});

test('up starts a project and down leaves nothing running', async () => {
  const home = await tempHome();
  const src = path.join(home, 'app');
  await fsp.mkdir(src, { recursive: true });
  // A tiny server that exits only when signalled, so shutdown is really tested.
  await fsp.writeFile(path.join(src, 'server.js'), `
    const http = require('node:http');
    http.createServer((_, res) => { res.writeHead(200); res.end('ok'); })
        .listen(process.env.PORT, '127.0.0.1');
  `);
  await fsp.writeFile(path.join(src, 'package.json'),
    JSON.stringify({ scripts: { start: 'node server.js' } }));

  const port = 8931;
  const dashPort = 7931;
  await run(home, ['add', 'tiny', '--dir', src, '--port', String(port)]);

  const up = spawn(process.execPath, [CLI, 'up'], {
    env: { ...process.env, CLAUDE_LOCAL_HOME: home, CLAUDE_LOCAL_PORT: String(dashPort) },
    stdio: 'ignore',
    detached: true,
  });

  try {
    let status = null;
    for (let i = 0; i < 40 && status === null; i++) {
      await sleep(250);
      status = await reachable(port);
    }
    assert.equal(status, 200, 'the project should be serving after `up`');
    assert.equal(await reachable(dashPort), 200, 'the dashboard should be serving');

    await run(home, ['down'], { CLAUDE_LOCAL_PORT: String(dashPort) });

    // The bug this guards: `shell: true` without `detached` kills only the
    // wrapping shell, leaving npm/node holding the port after `down` reports
    // success. Poll until both are actually gone.
    let projectGone = false;
    for (let i = 0; i < 40 && !projectGone; i++) {
      await sleep(250);
      projectGone = (await reachable(port)) === null;
    }
    assert.ok(projectGone, 'the project must not survive `down`');
    assert.equal(await reachable(dashPort), null, 'the dashboard must stop too');
  } finally {
    // Negative PIDs are POSIX-only and throw on Windows, which would leave the
    // supervisor running and holding its ports for every later test run.
    if (process.platform === 'win32') {
      try {
        const { execFileSync } = await import('node:child_process');
        execFileSync('taskkill', ['/PID', String(up.pid), '/T', '/F'], { stdio: 'ignore' });
      } catch { /* already gone */ }
    } else {
      try { process.kill(-up.pid, 'SIGKILL'); } catch { /* already gone */ }
    }
    try { up.kill('SIGKILL'); } catch { /* already gone */ }
  }
});

test('GPU capability is judged, not just presence', async () => {
  const { classifyGpu } = await import('../ops/local/claude-local.mjs');

  // The card that prompted this: reports fine to nvidia-smi, runs none of it.
  for (const cap of [null, '2.1']) {
    const fermi = classifyGpu('NVIDIA GeForce GTX 560 Ti', cap);
    assert.equal(fermi.usable, false);
    assert.equal(fermi.cap, 2.1);
    assert.match(fermi.reason, /Fermi/);
  }

  // Pascal and Maxwell are also below current PyTorch wheels.
  assert.equal(classifyGpu('NVIDIA GeForce GTX 1080 Ti').usable, false);
  assert.equal(classifyGpu('NVIDIA GeForce GTX 970').usable, false);

  // Turing and newer clear the bar.
  assert.equal(classifyGpu('NVIDIA GeForce RTX 3090').usable, true);
  assert.equal(classifyGpu('NVIDIA GeForce RTX 4090', '8.9').usable, true);
  assert.equal(classifyGpu('NVIDIA A100-SXM4-40GB', '8.0').usable, true);

  // An explicit capability from the driver wins over the name heuristic.
  assert.equal(classifyGpu('Mystery Accelerator', '9.0').usable, true);

  // Unknown hardware is reported as unknown rather than assumed good.
  const unknown = classifyGpu('Some Unknown Card', null);
  assert.equal(unknown.usable, false);
  assert.equal(unknown.cap, null);
  assert.match(unknown.reason, /could not determine/);
});

test('the LAN address avoids virtual adapters', async () => {
  const { lanAddress } = await import('../ops/local/claude-local.mjs');
  assert.equal(lanAddress({ lo: [{ family: 'IPv4', address: '127.0.0.1', internal: true }] }), null);

  // A Windows host is full of Hyper-V/WSL/Docker adapters; the real NIC wins.
  const mixed = {
    'vEthernet (WSL)': [{ family: 'IPv4', address: '172.19.0.1', internal: false }],
    'docker0': [{ family: 'IPv4', address: '172.17.0.1', internal: false }],
    'Ethernet': [{ family: 'IPv4', address: '192.168.1.50', internal: false }],
  };
  assert.equal(lanAddress(mixed), '192.168.1.50');
});

test('a laptop with several live adapters picks the right one', async () => {
  const { lanAddress, lanCandidates } = await import('../ops/local/claude-local.mjs');

  // What a gaming laptop actually looks like: Wi-Fi carrying the traffic, an
  // Ethernet port with no cable (so a self-assigned 169.254 address), and the
  // usual pile of virtual adapters. Listing order is deliberately unhelpful.
  const laptop = {
    'vEthernet (Default Switch)': [{ family: 'IPv4', address: '172.28.80.1', internal: false }],
    'Ethernet': [{ family: 'IPv4', address: '169.254.11.4', internal: false }],
    'Tailscale': [{ family: 'IPv4', address: '100.94.3.7', internal: false }],
    'Wi-Fi': [{ family: 'IPv4', address: '192.168.1.42', internal: false }],
    'Loopback Pseudo-Interface 1': [{ family: 'IPv4', address: '127.0.0.1', internal: true }],
  };
  // Wi-Fi wins: the wired port has no DHCP lease, so nothing could reach it.
  assert.equal(lanAddress(laptop), '192.168.1.42');

  const ranked = lanCandidates(laptop);
  assert.equal(ranked[0].address, '192.168.1.42');
  assert.ok(ranked.find((c) => c.address === '169.254.11.4').selfAssigned);
  assert.ok(ranked.find((c) => c.address === '100.94.3.7').virtual, 'a VPN is not the LAN');

  // With the cable plugged in, wired should take over from Wi-Fi.
  const docked = { ...laptop, Ethernet: [{ family: 'IPv4', address: '192.168.1.77', internal: false }] };
  assert.equal(lanAddress(docked), '192.168.1.77');
});

test('an adapter with only a self-assigned address is never chosen', async () => {
  const { lanAddress } = await import('../ops/local/claude-local.mjs');
  assert.equal(lanAddress({ 'Wi-Fi': [{ family: 'IPv4', address: '169.254.9.9', internal: false }] }), null);
});

test('the network password is taken from flag, env, then stored hash', async () => {
  const { resolvePasswordHash, hashPassword } = await import('../ops/local/claude-local.mjs');
  const stored = hashPassword('from-file');

  assert.equal(resolvePasswordHash({ password: 'flag' }, {}, () => stored), hashPassword('flag'));
  assert.equal(resolvePasswordHash({}, { CLAUDE_LOCAL_PASSWORD: 'env' }, () => stored), hashPassword('env'));
  assert.equal(resolvePasswordHash({}, {}, () => stored), stored);
  assert.equal(resolvePasswordHash({}, {}, () => null), null);
});

test('the LAN gateway demands the password and proxies what it allows', async () => {
  const http = await import('node:http');
  const { createGateway, hashPassword } = await import('../ops/local/claude-local.mjs');

  const upstream = http.createServer((req, res) => {
    res.writeHead(200, { 'content-type': 'text/plain' });
    res.end(`upstream saw ${req.url}`);
  });
  await new Promise((r) => upstream.listen(0, '127.0.0.1', r));
  const targetPort = upstream.address().port;

  const gateway = createGateway({
    bindHost: '127.0.0.1', port: 0, targetPort, passwordHash: hashPassword('open-sesame'),
  });
  await new Promise((r) => gateway.on('listening', r));
  const port = gateway.address().port;
  const call = (headers) => fetch(`http://127.0.0.1:${port}/api/thing`, { headers });

  try {
    const anonymous = await call({});
    assert.equal(anonymous.status, 401);
    assert.match(anonymous.headers.get('www-authenticate') || '', /Basic realm/);

    const wrong = await call({ authorization: `Basic ${Buffer.from('u:nope').toString('base64')}` });
    assert.equal(wrong.status, 401);

    const right = await call({ authorization: `Basic ${Buffer.from('u:open-sesame').toString('base64')}` });
    assert.equal(right.status, 200);
    // The path is forwarded untouched -- absolute URLs inside an app depend on it.
    assert.equal(await right.text(), 'upstream saw /api/thing');

    const malformed = await call({ authorization: 'Basic not-base64!!' });
    assert.equal(malformed.status, 401);
  } finally {
    gateway.close();
    upstream.close();
  }
});

test('an open gateway needs no credentials', async () => {
  const http = await import('node:http');
  const { createGateway } = await import('../ops/local/claude-local.mjs');
  const upstream = http.createServer((_, res) => { res.writeHead(200); res.end('ok'); });
  await new Promise((r) => upstream.listen(0, '127.0.0.1', r));
  const gateway = createGateway({
    bindHost: '127.0.0.1', port: 0, targetPort: upstream.address().port, passwordHash: null,
  });
  await new Promise((r) => gateway.on('listening', r));
  try {
    assert.equal((await fetch(`http://127.0.0.1:${gateway.address().port}/`)).status, 200);
  } finally {
    gateway.close();
    upstream.close();
  }
});

test('the gateway reports an unreachable project rather than hanging', async () => {
  const { createGateway } = await import('../ops/local/claude-local.mjs');
  // Port 1 has nothing on it, standing in for a project that failed to start.
  const gateway = createGateway({ bindHost: '127.0.0.1', port: 0, targetPort: 1, passwordHash: null });
  await new Promise((r) => gateway.on('listening', r));
  try {
    const res = await fetch(`http://127.0.0.1:${gateway.address().port}/`);
    assert.equal(res.status, 502);
    assert.match(await res.text(), /not responding/);
  } finally {
    gateway.close();
  }
});

test('windows-setup never writes the password into the task or shortcuts', async () => {
  // Regression guard: the first version put --password on the task's command
  // line, leaving the secret in a long-lived XML file and in `ps` output.
  const home = await tempHome();
  const src = path.join(home, 'app');
  await fsp.mkdir(src, { recursive: true });
  await fsp.writeFile(path.join(src, 'package.json'), '{}');
  await run(home, ['add', 'demo', '--dir', src, '--port', '8787']);

  const secret = 'correct-horse-battery-staple';
  await run(home, ['windows-setup', '--force', '--password', secret]);

  const winDir = path.join(home, 'windows');
  for (const entry of await fsp.readdir(winDir)) {
    const raw = await fsp.readFile(path.join(winDir, entry));
    for (const encoding of ['utf8', 'utf16le']) {
      assert.ok(!raw.toString(encoding).includes(secret),
        `${entry} contains the password (as ${encoding})`);
    }
  }

  // What is stored is a digest, and only a digest.
  const stored = (await fsp.readFile(path.join(home, 'password.sha256'), 'utf8')).trim();
  assert.match(stored, /^[0-9a-f]{64}$/);
  assert.notEqual(stored, secret);

  // The boot task must still be a boot task, and unprivileged.
  const xml = (await fsp.readFile(path.join(winDir, 'claude-local-task.xml'))).toString('utf16le');
  assert.match(xml, /<BootTrigger>/);
  assert.match(xml, /LeastPrivilege/);
  assert.match(xml, /"up" --lan/);
});

test('the direct-invocation guard works on Windows paths', async () => {
  const { isMainModule } = await import('../ops/local/claude-local.mjs');

  // The exact shapes Windows produces. Comparing import.meta.url against a
  // hand-built `file://${argv[1]}` (or using new URL().pathname, which yields a
  // leading-slash "/C:/...") never matches here, so main() never ran and the
  // command exited 0 having printed nothing. Observed on a real Windows host.
  const winUrl = 'file:///C:/Apps/splatworks/ops/local/claude-local.mjs';
  const winArgv = 'C:\\Apps\\splatworks\\ops\\local\\claude-local.mjs';
  assert.equal(isMainModule(winArgv, winUrl, true), true, 'must run when launched directly on Windows');

  // The old broken comparison, kept here so the regression is explicit.
  assert.notEqual(
    new URL(winUrl).pathname, winArgv,
    'this mismatch is the bug: .pathname keeps a leading slash and forward slashes',
  );

  // Case and separator differences must not defeat it.
  assert.equal(isMainModule('C:/Apps/splatworks/ops/local/claude-local.mjs', winUrl, true), true);

  // A different file is still not the entrypoint.
  assert.equal(isMainModule('C:\\Apps\\splatworks\\other.mjs', winUrl, true), false);

  // POSIX keeps working.
  const posixUrl = 'file:///home/user/app/ops/local/claude-local.mjs';
  assert.equal(isMainModule('/home/user/app/ops/local/claude-local.mjs', posixUrl, false), true);
  assert.equal(isMainModule('/home/user/app/other.mjs', posixUrl, false), false);

  // Imported rather than launched: argv[1] is some other program.
  assert.equal(isMainModule(undefined, winUrl, true), false);
});
