const MAGIC = 'VTSK0002';
const HEADER_BYTES = 42;
const TABLE_ENTRY_BYTES = 10;
const STROKE_BYTES = 14;
const WASH_BYTES = 8;

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

/**
 * Canvas 2D foreground driven by frame-indexed vector strokes.
 *
 * The WebGL scene draws exactly one fixed low-frequency base. Each time step
 * selects its reference-frame instruction group: short contour strokes and
 * nearly opaque full-color brushes repaint the moving details. No video,
 * per-frame reference bitmap, coverage mask, or second base image is loaded.
 */
export class DynamicOverlay {
  static async load(binaryUrl, manifestUrl = 'assets/temporal_runtime.manifest.json') {
    const [response, manifestResponse] = await Promise.all([
      fetch(binaryUrl, { cache: 'no-store' }),
      fetch(manifestUrl, { cache: 'no-store' }),
    ]);
    if (!response.ok) throw new Error(`无法加载 ${binaryUrl}: HTTP ${response.status}`);
    if (!manifestResponse.ok) throw new Error('无法加载时序笔触清单');
    return new DynamicOverlay(await response.arrayBuffer(), await manifestResponse.json());
  }

  constructor(buffer, manifest) {
    this.buffer = buffer;
    this.manifest = manifest;
    const view = new DataView(buffer);
    if (new TextDecoder().decode(new Uint8Array(buffer, 0, 8)) !== MAGIC) {
      throw new Error('时序动态笔触文件标识无效');
    }
    const version = view.getUint16(8, true);
    this.width = view.getUint16(10, true);
    this.height = view.getUint16(12, true);
    this.frameCount = view.getUint16(14, true);
    this.regionCount = view.getUint16(16, true);
    this.fps = view.getUint16(18, true);
    this.strokeBytes = view.getUint16(20, true);
    this.washBytes = view.getUint16(22, true);
    this.gridWidth = view.getUint16(24, true);
    this.gridHeight = view.getUint16(26, true);
    this.cell = view.getUint16(28, true);
    this.strokeTableOffset = view.getUint32(30, true);
    this.washTableOffset = view.getUint32(34, true);
    this.dataOffset = view.getUint32(38, true);
    if (
      version !== 2 || this.width !== 752 || this.height !== 352
      || this.frameCount !== 124 || this.fps !== 24
      || this.strokeBytes !== STROKE_BYTES || this.washBytes !== WASH_BYTES
      || this.gridWidth !== 188 || this.gridHeight !== 88 || this.cell !== 4
      || this.regionCount !== manifest.regions.length
    ) {
      throw new Error('时序笔触版本、尺寸或区域定义不兼容');
    }
    if (
      this.strokeTableOffset !== HEADER_BYTES
      || this.dataOffset !== manifest.data_offset
      || buffer.byteLength !== manifest.bytes
    ) {
      throw new Error('时序动态笔触文件长度或偏移异常');
    }
    this.regions = manifest.regions.map((region, index) => ({ ...region, index }));
    this.view = view;
    this.lastFrame = -1;
  }

  bind(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d', { alpha: true, willReadFrequently: true });
    this.resize();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement || canvas);
  }

  resize() {
    if (!this.canvas) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(this.width * ratio);
    this.canvas.height = Math.round(this.height * ratio);
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.lastFrame = -1;
  }

  table(tableOffset, frame, region) {
    const offset = tableOffset + (frame * this.regionCount + region) * TABLE_ENTRY_BYTES;
    const view = new DataView(this.buffer, offset, TABLE_ENTRY_BYTES);
    return {
      count: view.getUint16(0, true),
      offset: Number(view.getBigUint64(2, true)),
    };
  }

  draw(seconds) {
    if (!this.context) return;
    const frame = clamp(Math.round(seconds * this.fps), 0, this.frameCount - 1);
    if (frame === this.lastFrame) return;
    this.lastFrame = frame;
    const ctx = this.context;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.restore();

    ctx.globalCompositeOperation = 'source-over';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (const region of this.regions) {
      const strokes = this.table(this.strokeTableOffset, frame, region.index);
      const washes = this.table(this.washTableOffset, frame, region.index);
      this.drawWashes(ctx, washes.offset, washes.count);
      this.drawStrokes(ctx, strokes.offset, strokes.count, frame);
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
  }

  drawWashes(ctx, offset, count) {
    const view = this.view;
    let previousColor = '';
    const feather = Math.max(1, Math.round(this.cell * 0.25));
    for (let item = 0; item < count; item += 1) {
      const r = offset + item * WASH_BYTES;
      const index = view.getUint32(r, true);
      const red = view.getUint8(r + 4);
      const green = view.getUint8(r + 5);
      const blue = view.getUint8(r + 6);
      const alpha = view.getUint8(r + 7) / 255;
      const color = `${red},${green},${blue}`;
      if (color !== previousColor) {
        ctx.fillStyle = `rgb(${color})`;
        previousColor = color;
      }
      const gx = index % this.gridWidth;
      const gy = Math.floor(index / this.gridWidth);
      const x = gx * this.cell;
      const y = gy * this.cell;
      ctx.globalAlpha = alpha;
      ctx.fillRect(x - feather, y - feather, this.cell + feather * 2, this.cell + feather * 2);
      ctx.globalAlpha = alpha;
      ctx.fillRect(x, y, this.cell, this.cell);
    }
  }

  drawStrokes(ctx, offset, count, frame) {
    const view = this.view;
    let previousColor = '';
    for (let stroke = 0; stroke < count; stroke += 1) {
      const r = offset + stroke * STROKE_BYTES;
      const x = view.getUint16(r, true) / 16;
      const y = view.getUint16(r + 2, true) / 16;
      const angle = view.getUint16(r + 4, true) / 65535 * Math.PI * 2 - Math.PI;
      const tx = Math.cos(angle);
      const ty = Math.sin(angle);
      const half = view.getUint8(r + 6) / 4;
      const lineWidth = view.getUint8(r + 7) / 4;
      const red = view.getUint8(r + 8);
      const green = view.getUint8(r + 9);
      const blue = view.getUint8(r + 10);
      const alpha = view.getUint8(r + 11) / 255;
      const isPetal = view.getUint8(r + 12) === 1;
      const color = `${red},${green},${blue}`;
      if (color !== previousColor) {
        ctx.strokeStyle = `rgb(${color})`;
        ctx.fillStyle = `rgb(${color})`;
        previousColor = color;
      }

      if (isPetal) {
        const spin = angle + frame * 0.031 + x * 0.017 + y * 0.013;
        const radiusX = clamp(half * 0.92, 1.7, 5.8);
        const radiusY = clamp(half * 0.42, 0.75, 2.8);
        ctx.globalAlpha = clamp(alpha * 0.86, 0, 0.9);
        this.drawPetal(ctx, x, y, radiusX, radiusY, spin);
        continue;
      }

      ctx.globalAlpha = clamp(alpha, 0, 0.78);
      ctx.lineWidth = lineWidth;
      const x0 = x - tx * half;
      const y0 = y - ty * half;
      const x2 = x + tx * half;
      const y2 = y + ty * half;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x2, y2);
      ctx.stroke();

      if (alpha > 0.34 && half > 4) {
        ctx.globalAlpha = clamp((alpha - 0.34) * 0.42, 0, 0.18);
        ctx.beginPath();
        ctx.arc(x, y, Math.min(1.4, half * 0.16), 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  drawPetal(ctx, x, y, rx, ry, angle) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(-rx, 0);
    ctx.quadraticCurveTo(0, -ry, rx, 0);
    ctx.quadraticCurveTo(0, ry * 0.72, -rx, 0);
    ctx.fill();
    ctx.restore();
  }
}
