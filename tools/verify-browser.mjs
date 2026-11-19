import { chromium } from 'playwright-core';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';

const baseURL = process.env.VERIFY_URL || 'http://127.0.0.1:8791/';
const outputDir = resolve(process.env.VERIFY_OUTPUT || 'verification/screenshots');
const chromePath = process.env.CHROME_PATH || 'C:\\Users\\vipshao\\AppData\\Local\\Google\\Chrome\\Application\\chrome.exe';
const times = [0, 1, 2.5, 4, 5];
const canvasWidth = 752;
const canvasHeight = 352;
await mkdir(outputDir, { recursive: true });

const browser = await chromium.launch({
  executablePath: chromePath,
  headless: true,
  args: [
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-webgl',
    '--ignore-gpu-blocklist',
    '--disable-gpu-sandbox',
    '--disable-frame-rate-limit',
    '--run-all-compositor-stages-before-draw',
    '--disable-features=CalculateNativeWinOcclusion',
  ],
});
const context = await browser.newContext({ viewport: { width: 810, height: 620 }, deviceScaleFactor: 1 });
const page = await context.newPage();
const consoleErrors = [];
const shaderErrors = [];
page.on('console', (message) => {
  const text = message.text();
  if (message.type() === 'error' || /shader|webgl|glsl/i.test(text)) {
    consoleErrors.push(`${message.type()}: ${text}`);
  }
  if (/shader|compile|glsl/i.test(text)) shaderErrors.push(text);
});
page.on('pageerror', (error) => consoleErrors.push(`pageerror: ${error.stack || error.message}`));

await page.goto(baseURL, { waitUntil: 'domcontentloaded' });
await page.waitForFunction(() => document.body.dataset.ready === 'true', { timeout: 90000 });
  await page.evaluate(() => { window.player.playing = false; });
  const runtime = await page.evaluate(() => ({
  meta: window.player.meta,
  strokeCount: window.player.strokeCount,
  modelCounts: Array.from(window.player.modelCounts),
  gl: {
    version: window.player.renderer.getContext().getParameter(window.player.renderer.getContext().VERSION),
    renderer: window.player.renderer.getContext().getParameter(window.player.renderer.getContext().RENDERER),
    error: window.player.renderer.getContext().getError(),
  },
  videoElements: document.querySelectorAll('video').length,
  errors: window.errors,
}));

const screenshots = [];
const overlayFrames = [];
for (const seconds of times) {
  const name = `render_${String(seconds).replace('.', '_')}s.png`;
  const path = resolve(outputDir, name);
  // Save an exact logical-resolution crop for ROI comparison with reference frames.
  const pixels = await page.evaluate(async (t) => {
    const source = document.getElementById('canvas');
    const target = document.createElement('canvas');
    target.width = 752;
    target.height = 352;
    const context = target.getContext('2d', { willReadFrequently: true });
    window.player.setTime(t);
    return new Promise((done) => requestAnimationFrame(() => {
      window.player.setTime(t);
      const capture = () => target.toDataURL('image/png').split(',')[1];
      context.clearRect(0, 0, 752, 352);
      context.drawImage(source, 0, 0, 752, 352);
      const webglPng = capture();
      const overlay = document.querySelector('canvas.dynamic-overlay');
      if (overlay) context.drawImage(overlay, 0, 0, 752, 352);
      requestAnimationFrame(() => done({
        png: target.toDataURL('image/png').split(',')[1],
        webglPng,
        checksum: Array.from(context.getImageData(0, 0, 752, 352).data).reduce((sum, value, index) => (sum + value * ((index % 17) + 1)) | 0, 0),
      }));
    }));
  }, seconds);
  const renderedTime = seconds;
  // canvas.toDataURL reads the preserved WebGL buffer directly. It avoids a
  // compositor screenshot path that can stall on some SwiftShader builds.
  await writeFile(path, Buffer.from(pixels.png, 'base64'));
  const logicalPath = resolve(outputDir, `logical_${String(seconds).replace('.', '_')}s.png`);
  await writeFile(logicalPath, Buffer.from(pixels.png, 'base64'));
  const webglPath = outputDir === resolve('verification/screenshots')
    ? resolve(outputDir, `webgl_${String(seconds).replace('.', '_')}s.png`)
    : null;
  if (webglPath) await writeFile(webglPath, Buffer.from(pixels.webglPng, 'base64'));
  const glErrorAfterDraw = await page.evaluate(() => window.player.renderer.getContext().getError());
  const frameUniforms = await page.evaluate(() => ({
    time: window.player.mesh.material.uniforms.uTime.value,
    frame: window.player.mesh.material.uniforms.uFrame.value,
  }));
  const overlayCoverage = await page.evaluate(() => {
    const overlay = document.querySelector('canvas.dynamic-overlay');
    if (!overlay) return null;
    const probe = document.createElement('canvas');
    probe.width = 752;
    probe.height = 352;
    const context = probe.getContext('2d', { willReadFrequently: true });
    context.drawImage(overlay, 0, 0, 752, 352);
    const pixels = context.getImageData(0, 0, 752, 352).data;
    const boxes = {
      hair: [300, 15, 445, 145], chest: [330, 95, 415, 165],
      wing_left: [55, 180, 265, 330], wing_right: [485, 95, 745, 320],
    };
    const coverage = {};
    for (const [name, [x0, y0, x1, y1]] of Object.entries(boxes)) {
      let colored = 0;
      let opaque = 0;
      let saturation = 0;
      for (let y = y0; y < y1; y += 1) {
        for (let x = x0; x < x1; x += 1) {
          const r = pixels[(y * 752 + x) * 4];
          const g = pixels[(y * 752 + x) * 4 + 1];
          const b = pixels[(y * 752 + x) * 4 + 2];
          const alpha = pixels[(y * 752 + x) * 4 + 3] / 255;
          if (alpha > 0.15) colored += 1;
          if (alpha > 0.75) opaque += 1;
          saturation += (Math.max(r, g, b) - Math.min(r, g, b)) * alpha;
        }
      }
      const total = (x1 - x0) * (y1 - y0);
      coverage[name] = {
        colored: colored / total,
        opaque: opaque / total,
        mean_saturation: saturation / total,
      };
    }
    return coverage;
  });
  const overlayPng = await page.evaluate(() => {
    const overlay = document.querySelector('canvas.dynamic-overlay');
    return overlay ? overlay.toDataURL('image/png').split(',')[1] : null;
  });
  const overlayFrame = await page.evaluate(() => window.player.overlay?.lastFrame);
  overlayFrames.push(overlayFrame);
  const overlayPath = outputDir === resolve('verification/screenshots')
    ? resolve(outputDir, `overlay_${String(seconds).replace('.', '_')}s.png`)
    : null;
  if (overlayPng && overlayPath) await writeFile(overlayPath, Buffer.from(overlayPng, 'base64'));
  screenshots.push({
    seconds,
    renderedTime,
    path,
    logicalPath,
    width: canvasWidth,
    height: canvasHeight,
    logicalSize: [canvasWidth, canvasHeight],
    glErrorAfterDraw,
    frameUniforms,
    overlayCoverage,
    pixelChecksum: pixels.checksum,
    overlayFrame,
    webglPath,
    overlayPath,
  });
}

const expectedFrames = times.map((seconds) => Math.round(seconds * 24));
const frameSequenceMatches = overlayFrames.every((frame, index) => frame === expectedFrames[index]);
const coloredDynamicRegions = screenshots.every(({ overlayCoverage }) => (
  overlayCoverage && Object.values(overlayCoverage).every(({ colored, opaque }) => (
    colored > 0.25 && opaque > 0.18
  ))
));
const imageRequests = await page.evaluate(() => performance.getEntriesByType('resource')
  .filter((entry) => /\.(?:jpe?g|png|webp)(?:\?|$)/i.test(entry.name))
  .map((entry) => entry.name));
const baseImageRequests = imageRequests.filter((name) => /\/base\.jpg(?:\?|$)/.test(name));

const diagnostics = await page.evaluate(() => ({
  errors: window.errors,
  diagnosticVisible: !document.getElementById('diagnostic').hidden,
  diagnosticText: document.getElementById('diagnostic').textContent,
  status: document.getElementById('status').textContent,
  glError: window.player.renderer.getContext().getError(),
}));
  const shaderLinkErrors = {
    programCount: await page.evaluate(() => window.player.renderer.info.programs?.length || 0),
    noReportedShaderErrors: shaderErrors.length === 0,
  };
const report = {
  baseURL,
  chromePath,
  times,
  screenshots,
  expectedFrames,
  overlayFrames,
  imageRequests,
  runtime,
  consoleErrors,
  shaderErrors,
  shaderLinkErrors,
  diagnostics,
  checks: {
    loaded: runtime.meta?.format === 'VSTK0004',
    videoElements: runtime.videoElements === 0,
    noPageErrors: runtime.errors.length === 0 && diagnostics.errors.length === 0,
    noConsoleErrors: consoleErrors.length === 0,
    noShaderErrors: shaderErrors.length === 0,
    noShaderLinkFailures: !diagnostics.diagnosticText.includes('shader 编译失败'),
    noWebGLError: diagnostics.glError === 0,
    fiveScreenshots: screenshots.length === 5,
    frameSequenceMatches,
    coloredDynamicRegions,
    oneFixedBaseImage: baseImageRequests.length === 1,
    noPerFrameImages: !imageRequests.some((name) => /frame_\d+\.(?:jpe?g|png|webp)/i.test(name)),
  },
};
await writeFile(resolve(outputDir, 'browser_report.json'), JSON.stringify(report, null, 2), 'utf8');
console.log(JSON.stringify(report, null, 2));
await browser.close();
if (Object.values(report.checks).some((value) => !value)) process.exitCode = 1;
