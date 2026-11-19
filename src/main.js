import { StrokePlayer } from './player.js?v=12';
import { REGIONS, MOTION_MODELS } from './instructions.js?v=12';

const canvas = document.getElementById('canvas');
const status = document.getElementById('status');
const loading = document.getElementById('loading');
const playButton = document.getElementById('play');
const restartButton = document.getElementById('restart');
const timeRange = document.getElementById('time');
const timeLabel = document.getElementById('timeLabel');
const stats = document.getElementById('stats');
const diagnostic = document.getElementById('diagnostic');

function reportError(title, detail) {
  const text = typeof detail === 'string' ? detail : JSON.stringify(detail, null, 2);
  const message = `${title}\n${text}`;
  window.errors.push(message);
  diagnostic.hidden = false;
  diagnostic.textContent = message;
  setStatus(title);
  loading.hidden = true;
}

function setStatus(text) {
  status.textContent = text;
  status.setAttribute('aria-label', text);
}

function setStats(info) {
  if (!info.temporal) return;
  const rows = [
    ['固定基底', `${info.background.path}（1 张）`],
    ['固定 WebGL 静态笔触', info.stroke_count.toLocaleString('zh-CN')],
    ['时序帧组', `${info.temporal.frame_count} 帧 / ${info.temporal.fps} fps`],
    ['逐帧轮廓笔触', info.temporal.strokes.total.toLocaleString('zh-CN')],
    ['逐帧全彩时序笔刷', info.temporal.wash_patches.total.toLocaleString('zh-CN')],
    ['动态区域', info.temporal.regions.map((region) => region.name).join(' / ')],
    ['逻辑画布', `${info.width} × ${info.height}`],
    ['时长 / 帧率', `${info.duration.toFixed(3)} s / ${info.fps} fps`],
    ['指令体积', `${((info.bytes + info.temporal.bytes) / 1048576).toFixed(2)} MiB`],
    ['静态相机', '固定（位移 / 缩放 / 旋转 = 0）'],
  ];
  stats.replaceChildren(...rows.flatMap(([key, value]) => {
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = key;
    dd.textContent = value;
    return [dt, dd];
  }));
}

let player;
window.player = null;
window.errors = [];
window.REGIONS = REGIONS;
window.MOTION_MODELS = MOTION_MODELS;
window.addEventListener('error', (event) => {
  reportError('页面脚本错误', `${event.message} | ${event.filename}:${event.lineno}:${event.colno}`);
});
window.addEventListener('unhandledrejection', (event) => {
  reportError('未处理的 Promise 错误', event.reason?.stack || String(event.reason));
});

try {
  player = new StrokePlayer(canvas, { onStatus: setStatus, onStats: setStats, onError: reportError });
} catch (error) {
  reportError('WebGL 初始化失败', error.stack || error.message);
  throw error;
}
window.player = player;

playButton.addEventListener('click', () => {
  const playing = player.toggle();
  playButton.textContent = playing ? '暂停' : '播放';
});
restartButton.addEventListener('click', () => player.setTime(0));
timeRange.addEventListener('input', () => {
  player.playing = false;
  playButton.textContent = '播放';
  player.setTime(Number(timeRange.value));
});

function syncTime() {
  if (player.loaded) {
    timeRange.max = player.meta.duration.toFixed(6);
    timeRange.value = player.elapsed.toFixed(3);
    timeLabel.value = `${player.elapsed.toFixed(3)} s`;
  }
  requestAnimationFrame(syncTime);
}
requestAnimationFrame(syncTime);

player.load('assets/strokes.bin?v=12').then(() => {
  loading.classList.add('hidden');
  playButton.disabled = false;
  playButton.textContent = '暂停';
  document.body.dataset.ready = 'true';
}).catch((error) => {
  reportError('绘图指令启动失败', JSON.stringify({
    error: error.stack || error.message,
    buffer: player.lastBufferInfo || null,
    errors: window.errors,
  }, null, 2));
});
