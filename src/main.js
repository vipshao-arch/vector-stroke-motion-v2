import { StrokePlayer } from './player.js?v=6';
import { REGIONS, MOTION_MODELS } from './instructions.js?v=6';

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
  const rows = [
    ['总笔触数', info.stroke_count.toLocaleString('zh-CN')],
    ['图元类型', Object.values(info.counts).map((n) => n.toLocaleString('zh-CN')).join(' / ')],
    ['时间关键帧', `${info.keyframe_count} 组 / 笔触`],
    ['静态锁定笔触', info.motion_model_counts.static_lock.toLocaleString('zh-CN')],
    ['呼吸 / 头发 / 双翼', [
      info.motion_model_counts.breathing,
      info.motion_model_counts.hair_wave,
      info.motion_model_counts.wing_float_left + info.motion_model_counts.wing_float_right,
    ].map((n) => n.toLocaleString('zh-CN')).join(' / ')],
    ['花瓣 / 闪动', `${info.motion_model_counts.petal_fall.toLocaleString('zh-CN')} / ${info.motion_model_counts.shimmer.toLocaleString('zh-CN')}`],
    ['逻辑画布', `${info.width} × ${info.height}`],
    ['时长 / 帧率', `${info.duration.toFixed(3)} s / ${info.fps} fps`],
    ['指令体积', `${(info.bytes / 1048576).toFixed(2)} MiB`],
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

player.load('assets/strokes.bin?v=6').then(() => {
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
