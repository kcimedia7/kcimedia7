import { spawn } from 'node:child_process';

/** Run a command, streaming each output line to `onLine`. Rejects on non-zero exit. */
export function run(cmd, args, { cwd, env, onLine, signal, timeoutMs } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, {
      cwd,
      env: { ...process.env, ...env },
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let tail = '';
    const pump = (stream) => {
      let buf = '';
      stream.setEncoding('utf8');
      stream.on('data', (chunk) => {
        buf += chunk;
        let nl;
        while ((nl = buf.indexOf('\n')) !== -1) {
          const line = buf.slice(0, nl).trimEnd();
          buf = buf.slice(nl + 1);
          if (line) {
            tail = line;
            onLine?.(line);
          }
        }
      });
      stream.on('end', () => {
        if (buf.trim()) { tail = buf.trim(); onLine?.(buf.trim()); }
      });
    };
    pump(child.stdout);
    pump(child.stderr);

    const timer = timeoutMs ? setTimeout(() => child.kill('SIGKILL'), timeoutMs) : null;
    const onAbort = () => child.kill('SIGTERM');
    signal?.addEventListener('abort', onAbort, { once: true });

    const cleanup = () => {
      if (timer) clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
    };

    child.on('error', (err) => { cleanup(); reject(err); });
    child.on('close', (code) => {
      cleanup();
      if (code === 0) resolve({ code });
      else reject(new Error(`${cmd} exited with code ${code}${tail ? `: ${tail}` : ''}`));
    });
  });
}

/** True when `cmd` exists and answers a version probe. */
export async function commandExists(cmd, probeArgs = ['--version']) {
  try {
    await run(cmd, probeArgs, { timeoutMs: 10_000 });
    return true;
  } catch (err) {
    // A tool that exists but exits non-zero on the probe still counts as present.
    return err.code !== 'ENOENT' && !/ENOENT/.test(err.message);
  }
}
