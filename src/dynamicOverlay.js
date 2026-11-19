import { KEYFRAMES, KEYFRAME_FIELDS, MOTION_FIELDS } from './instructions.js?v=6';

const PETAL_COLORS = [
  [255, 210, 236], [214, 232, 255], [250, 245, 210], [236, 219, 255],
];
const HAIR_BOX = [300, 8, 450, 152];
const CHEST_BOX = [326, 88, 420, 172];
const WING_LEFT_BOX = [40, 165, 285, 340];
const WING_RIGHT_BOX = [470, 70, 748, 328];

function rgba(color, alpha) {
  const a = Math.max(0, Math.min(1, alpha)).toFixed(3);
  return `rgba(${color[0] | 0},${color[1] | 0},${color[2] | 0},${a})`;
}

function contains(box, x, y) {
  return x >= box[0] && x <= box[2] && y >= box[1] && y <= box[3];
}

function sampleKeys(records, id, normalizedTime) {
  const position = Math.max(0, Math.min(1, normalizedTime)) * (KEYFRAMES - 1);
  const a = Math.floor(position);
  const b = Math.min(a + 1, KEYFRAMES - 1);
  let t = position - a;
  t = t * t * (3 - 2 * t);
  const base = id * KEYFRAMES * KEYFRAME_FIELDS;
  const values = new Float32Array(KEYFRAME_FIELDS);
  for (let field = 0; field < KEYFRAME_FIELDS; field += 1) {
    values[field] = records.keys[base + a * KEYFRAME_FIELDS + field] * (1 - t)
      + records.keys[base + b * KEYFRAME_FIELDS + field] * t;
  }
  return values;
}

/**
 * Deterministic Canvas 2D foreground for region-aware motion.
 *
 * It reads the same VSTK0004 records as the WebGL player. The intact WebGL
 * reconstruction stays fixed; only explicit detail strokes are redrawn here
 * with their keyframe displacement and procedural motion. This keeps motion
 * visible on GPU drivers that do not refresh large float-texture vertex draws.
 */
export class DynamicOverlay {
  constructor(canvas, records, meta) {
    this.canvas = canvas;
    this.records = records;
    this.meta = meta;
    this.context = canvas.getContext('2d', { alpha: true });
    this.width = meta.width;
    this.height = meta.height;
    this.resize();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement || canvas);
  }

  resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(this.width * ratio);
    this.canvas.height = Math.round(this.height * ratio);
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  drawPetal(ctx, x, y, rx, ry, angle, alpha, color, bend) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.globalAlpha = alpha;
    ctx.fillStyle = rgba(color, 1);
    ctx.beginPath();
    ctx.moveTo(-rx, 0);
    ctx.quadraticCurveTo(-rx * 0.2, -ry * (1 + bend), rx, 0);
    ctx.quadraticCurveTo(-rx * 0.2, ry * (1 - bend * 0.25), -rx, 0);
    ctx.fill();
    ctx.strokeStyle = rgba([255, 255, 255], 0.45);
    ctx.lineWidth = 0.45;
    ctx.beginPath();
    ctx.moveTo(-rx * 0.86, 0);
    ctx.quadraticCurveTo(0, -ry * 0.12, rx * 0.82, 0);
    ctx.stroke();
    ctx.restore();
  }

  drawCurve(ctx, x0, y0, x1, y1, x2, y2, width, alpha, color) {
    if (width <= 0 || alpha <= 0) return;
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = rgba(color, 1);
    ctx.lineWidth = width;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.quadraticCurveTo(x1, y1, x2, y2);
    ctx.stroke();
  }

  draw(seconds) {
    const ctx = this.context;
    const { records, meta, width, height } = this;
    const normalized = Math.max(0, Math.min(1, seconds / meta.duration));
    const frame = normalized * (meta.frame_count - 1);

    // Clear in device coordinates so high-DPI transforms cannot leave residues.
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.restore();
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;

    // Breathing: a clipped halo and a few chest detail strokes scale around
    // their common anchor. The underlying WebGL reconstruction remains fixed.
    const breath = Math.sin(seconds * Math.PI * 2 / 2.7);
    ctx.save();
    ctx.beginPath();
    ctx.rect(326, 88, 94, 84);
    ctx.clip();
    const haloRadius = 52 + breath * 4;
    const halo = ctx.createRadialGradient(373, 132, 4, 373, 132, haloRadius);
    halo.addColorStop(0, `rgba(255,238,255,${(0.04 + breath * 0.022).toFixed(3)})`);
    halo.addColorStop(1, 'rgba(255,238,255,0)');
    ctx.fillStyle = halo;
    ctx.fillRect(326, 88, 94, 84);
    ctx.translate(373, 132);
    ctx.scale(1 + breath * 0.018, 1 + breath * 0.011);
    ctx.translate(-373, -132);
    let breathDrawn = 0;
    for (let id = 0; id < records.models.length && breathDrawn < 38; id += 1) {
      if (records.models[id] !== 1 || records.kinds[id] !== 2) continue;
      if (id % 3 !== 0) continue;
      const p = id * 12;
      const c = id * 3;
      if (!contains(CHEST_BOX, records.geometry[p], records.geometry[p + 1])) continue;
      const key = sampleKeys(records, id, normalized);
      this.drawCurve(
        ctx,
        records.geometry[p] + key[0], records.geometry[p + 1] + key[1],
        records.geometry[p + 2], records.geometry[p + 3],
        records.geometry[p + 4], records.geometry[p + 5],
        records.geometry[p + 6] * 0.65,
        0.02 + breath * 0.012 + Math.max(0, key[4]) * 0.04,
        [records.colors[c], records.colors[c + 1], records.colors[c + 2]],
      );
      breathDrawn += 1;
    }
    ctx.restore();

    const detailStride = { 2: 6, 3: 9, 4: 9, 6: 6 };
    const dynamicStart = records.dynamicStart;
    const dynamicCount = records.dynamicCount;
    for (let slot = 0; slot < dynamicCount; slot += 1) {
      // Texture slots are static-first/dynamic-second, while source records
      // retain their interleaved order. Resolve every slot before reading.
      const id = records.ids[dynamicStart + slot];
      const model = records.models[id];
      const kind = records.kinds[id];
      if (model === 0) continue;

      // Explicit motion primitives are curved strands, petals and particles.
      // General reconstruction strokes belong to the fixed WebGL base.
      const isDetail = (model === 2 && kind === 3)
        || ((model === 3 || model === 4) && kind === 2)
        || model === 5 || model === 6;
      if (!isDetail) continue;
      if (model === 5 && id % 6 !== 0) continue;
      if (model !== 5 && slot % detailStride[model] !== 0) continue;

      const p = id * 12;
      const c = id * 3;
      const m = id * MOTION_FIELDS;
      const key = sampleKeys(records, id, normalized);
      const color = [records.colors[c], records.colors[c + 1], records.colors[c + 2]];

      if (model === 5) {
        // A looping per-stroke life keeps the visible density constant while
        // each petal falls, sways, spins and fades near its wrap point.
        const life = (normalized * records.motion[m + 4] + records.motion[m + 2]) % 1;
        const envelope = Math.min(1, life / 0.12) * Math.min(1, (1 - life) / 0.18);
        if (envelope <= 0) continue;
        const travel = height + 150;
        const spawn = records.geometry[p + 1] + records.motion[m + 3] * 0.35;
        const fall = life * travel;
        const y = (spawn + fall) % travel - 60;
        const sway = Math.sin(frame * records.motion[m + 5] * 1.35 + records.motion[m])
          * records.motion[m + 6] * 2.1;
        const x = records.geometry[p] + sway + key[0] * 0.25;
        if (y < -12 || y > height + 12) continue;
        if (x > 240 && x < 500 && y > 245) continue;
        const angle = records.geometry[p + 4] + key[3]
          + records.motion[m + 7] * life * 5.2
          + Math.sin(frame * records.motion[m + 5] * 0.5 + records.motion[m]) * 0.65;
        const alpha = Math.min(0.48, records.alpha[id] * envelope * (0.34 + key[4] * 0.22));
        this.drawPetal(
          ctx, x, y,
          Math.max(2, records.geometry[p + 2] * 1.15),
          Math.max(1, records.geometry[p + 3] * 1.2),
          angle, alpha, PETAL_COLORS[id % PETAL_COLORS.length],
          records.geometry[p + 3] / Math.max(records.geometry[p + 2], 0.01),
        );
      } else if (model === 6) {
        // Independent shimmer particles.
        const sx = records.geometry[p];
        const sy = records.geometry[p + 1];
        if (sx > 240 && sx < 500 && sy > 245) continue;
        ctx.globalAlpha = Math.max(0, records.alpha[id] + key[4])
          * (0.12 + 0.18 * Math.abs(key[7]));
        ctx.fillStyle = rgba([235, 245, 255], 1);
        ctx.beginPath();
        ctx.arc(sx, sy, Math.max(0.45, records.geometry[p + 2] * (1 + key[3])), 0, Math.PI * 2);
        ctx.fill();
      } else if (model === 2) {
        // Hair: phase travels from each strand root toward its tip.
        if (!contains(HAIR_BOX, records.geometry[p], records.geometry[p + 1])) continue;
        const phase = frame * records.motion[m + 1] + records.motion[m]
          + records.motion[m + 3] * Math.hypot(
            records.geometry[p + 2] - records.motion[m + 4],
            records.geometry[p + 3] - records.motion[m + 5],
          );
        const bend = Math.sin(phase) * records.motion[m + 2] * 2.4;
        const x0 = records.geometry[p] + key[0] * 1.6;
        const y0 = records.geometry[p + 1] + key[1] * 1.6;
        const x2 = records.geometry[p + 4] + key[0] * 1.6 + bend * 0.8;
        const y2 = records.geometry[p + 5] + key[1] * 1.6 + bend * 0.25;
        const x1 = records.geometry[p + 2] + key[0] * 1.6 + bend;
        const y1 = records.geometry[p + 3] + key[1] * 1.6 + bend * 0.4;
        this.drawCurve(
          ctx, x0, y0, x1, y1, x2, y2,
          Math.max(0.7, records.geometry[p + 6] * 0.85),
          Math.min(0.3, records.alpha[id] * 0.72),
          color,
        );
      } else if (model === 3 || model === 4) {
        // Wings use separate anchors, periods and phases; tips move further.
        const sign = model === 3 ? -1 : 1;
        const box = model === 3 ? WING_LEFT_BOX : WING_RIGHT_BOX;
        if (!contains(box, records.geometry[p], records.geometry[p + 1])) continue;
        const anchorX = records.motion[m];
        const anchorY = records.motion[m + 1];
        const distance = Math.hypot(records.geometry[p] - anchorX, records.geometry[p + 1] - anchorY) / 210;
        const angle = frame * records.motion[m + 2] + records.motion[m + 3] + distance * 1.8;
        const wave = Math.sin(angle);
        const dx = sign * (1 + distance * 2.4) * wave * 2 + key[0] * 1.2;
        const dy = (0.8 + distance * 3.2)
          * Math.cos(frame * records.motion[m + 2] + records.motion[m + 3]) * 1.5
          + key[1] * 1.2;
        const x0 = records.geometry[p] + dx;
        const y0 = records.geometry[p + 1] + dy;
        const x2 = records.geometry[p + 4] + dx * 1.14;
        const y2 = records.geometry[p + 5] + dy * 1.18;
        const x1 = records.geometry[p + 2] + dx + wave * records.motion[m + 4] * 1.5;
        const y1 = records.geometry[p + 3] + dy * 0.6;
        const alpha = Math.min(0.25, Math.max(0.025, records.alpha[id] * 0.42 + key[4] * 0.6));
        this.drawCurve(
          ctx, x0, y0, x1, y1, x2, y2,
          Math.max(0.7, records.geometry[p + 6] * 0.85),
          alpha, color,
        );
      }
    }
    ctx.globalAlpha = 1;
  }
}
