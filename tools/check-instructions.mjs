import { readFile } from 'node:fs/promises';
import {
  MAGIC, KEYFRAMES, STROKE_BYTES, parseInstructions, REGIONS, MOTION_MODELS,
} from '../src/instructions.js';

const file = process.argv[2] || 'assets/strokes.bin';
const buffer = await readFile(file);
const typed = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
const result = parseInstructions(typed);
const finite = (array) => Array.from(array).every(Number.isFinite);
const expectedCamera = [1, 0, 0, 0, 1, 0];
let cameraIsIdentity = true;
for (let k = 0; k < KEYFRAMES; k += 1) {
  for (let i = 0; i < 6; i += 1) {
    if (Math.abs(result.camera[k * 6 + i] - expectedCamera[i]) > 1e-6) cameraIsIdentity = false;
  }
}
let staticMotionMax = 0;
let dynamicMotionMax = 0;
let scaleMin = Infinity;
let scaleMax = -Infinity;
let rotateMin = Infinity;
let rotateMax = -Infinity;
const stride = KEYFRAMES * 8;
for (let stroke = 0; stroke < result.strokeCount; stroke += 1) {
  const model = Math.round(result.params[stroke * 16 + 2] * 255);
  for (let k = 0; k < KEYFRAMES; k += 1) {
    const base = stroke * stride + k * 8;
    const scale = result.keys[base + 2];
    const rotate = result.keys[base + 3];
    scaleMin = Math.min(scaleMin, scale);
    scaleMax = Math.max(scaleMax, scale);
    rotateMin = Math.min(rotateMin, rotate);
    rotateMax = Math.max(rotateMax, rotate);
    const magnitude = Math.max(
      Math.abs(result.keys[base]), Math.abs(result.keys[base + 1]),
      Math.abs(scale - 1), Math.abs(rotate),
      Math.abs(result.keys[base + 4]), Math.abs(result.keys[base + 5] - 1),
    );
    if (model === 0) staticMotionMax = Math.max(staticMotionMax, magnitude);
    else dynamicMotionMax = Math.max(dynamicMotionMax, magnitude);
  }
}
const checks = {
  magic: buffer.subarray(0, 8).toString() === MAGIC,
  version: result.meta.version === 4,
  keyframes: result.meta.keyframe_count === KEYFRAMES,
  strokeCountMatches: result.meta.stroke_count === result.strokeCount,
  stageMatchesMeta: result.layerCounts.reduce((a, b) => a + b, 0) === result.strokeCount,
  regionTable: result.regionCounts.reduce((a, b) => a + b, 0) === result.strokeCount,
  modelTable: result.modelCounts.reduce((a, b) => a + b, 0) === result.strokeCount,
  staticRegionsPresent: REGIONS.slice(0, 5).every((_, i) => result.regionCounts[i] > 0),
  dynamicsPresent: [1, 2, 3, 4, 5, 6].every((i) => result.modelCounts[i] > 0),
  cameraIsIdentity,
  staticMotionMaxIsZero: staticMotionMax === 0,
  dynamicMotionIsNonzero: dynamicMotionMax > 0.1,
  allFinite: finite(result.params) && finite(result.colors) && finite(result.motion) && finite(result.keys) && finite(result.camera),
  bounds: scaleMin > 0.7 && scaleMax < 1.3 && rotateMin > -2 && rotateMax < 2,
};
const output = {
  file,
  bytes: result.bytes,
  strokeBytes: STROKE_BYTES,
  strokes: result.strokeCount,
  layers: result.layerCounts,
  regions: Object.fromEntries(REGIONS.map((name, i) => [name, result.regionCounts[i]])),
  models: Object.fromEntries(MOTION_MODELS.map((name, i) => [name, result.modelCounts[i]])),
  duration: result.meta.duration,
  keyframeCount: result.meta.keyframe_count,
  staticMotionMax,
  dynamicMotionMax,
  scaleRange: [scaleMin, scaleMax],
  rotateRange: [rotateMin, rotateMax],
  cameraIsIdentity,
  checks,
};
console.log(JSON.stringify(output, null, 2));
if (Object.values(checks).some((value) => !value)) process.exitCode = 1;
