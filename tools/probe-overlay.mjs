import { chromium } from 'playwright-core';
import { writeFile } from 'node:fs/promises';

const browser = await chromium.launch({
  executablePath: 'C:\\Users\\vipshao\\AppData\\Local\\Google\\Chrome\\Application\\chrome.exe',
  headless: true,
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage({ viewport: { width: 810, height: 620 } });
page.on('console', (message) => console.log('console', message.type(), message.text()));
page.on('pageerror', (error) => console.log('pageerror', error.message));
await page.goto(process.env.VERIFY_URL || 'http://127.0.0.1:8792/', { waitUntil: 'domcontentloaded' });
await page.waitForFunction(() => document.body.dataset.ready === 'true', { timeout: 90000 });
for (const seconds of [0, 1, 2.5, 4, 5]) {
  const result = await page.evaluate(async (t) => {
    window.player.playing = false;
    window.player.setTime(t);
    return new Promise((done) => requestAnimationFrame(() => {
      window.player.setTime(t);
      const target = document.createElement('canvas');
      target.width = 752;
      target.height = 352;
      const context = target.getContext('2d');
      context.drawImage(document.getElementById('canvas'), 0, 0, 752, 352);
      context.drawImage(document.querySelector('canvas.dynamic-overlay'), 0, 0, 752, 352);
      done({
        composite: target.toDataURL('image/png').split(',')[1],
        overlay: document.querySelector('canvas.dynamic-overlay').toDataURL('image/png').split(',')[1],
        frame: window.player.overlay.lastFrame,
      });
    }));
  }, seconds);
  const label = String(seconds).replace('.', '_');
  await writeFile(`verification/probe_composite_${label}.png`, Buffer.from(result.composite, 'base64'));
  await writeFile(`verification/probe_overlay_${label}.png`, Buffer.from(result.overlay, 'base64'));
  console.log(seconds, result.frame);
}
await browser.close();
