import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, extname, join } from 'node:path';
import { spawn } from 'node:child_process';

const root = process.cwd();
const port = Number(process.env.PORT || 8791);
const types = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.bin': 'application/octet-stream',
  '.json': 'application/json; charset=utf-8',
  '.jpg': 'image/jpeg',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
};

const server = createServer(async (request, response) => {
  const urlPath = decodeURIComponent((request.url || '/').split('?')[0]);
  const file = resolve(root, urlPath === '/' ? 'index.html' : `.${urlPath}`);
  if (!file.startsWith(root)) {
    response.writeHead(403).end('forbidden');
    return;
  }
  try {
    const data = await readFile(file);
    response.writeHead(200, {
      'Content-Type': types[extname(file)] || 'application/octet-stream',
      'Cache-Control': 'no-store',
    }).end(data);
  } catch {
    response.writeHead(404).end('not found');
  }
});

await new Promise((done) => server.listen(port, '127.0.0.1', done));
const child = spawn(process.execPath, [join(root, 'tools/verify-browser.mjs')], {
  cwd: root,
  env: { ...process.env, VERIFY_URL: `http://127.0.0.1:${port}/` },
  stdio: 'inherit',
});
const code = await new Promise((done) => child.on('exit', done));
server.close();
process.exitCode = code || 0;
