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
    try { process.kill(-up.pid, 'SIGKILL'); } catch { /* already gone */ }
    try { up.kill('SIGKILL'); } catch { /* already gone */ }
  }
});
