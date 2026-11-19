export const MAGIC = 'VSTK0004';
export const KEYFRAMES = 31;
export const PARAMS_PER_STROKE = 12;
export const COLOR_COMPONENTS = 3;
export const KEYFRAME_FIELDS = 8;
export const MOTION_FIELDS = 8;
export const STATIC_LAYER_INDEX = 0;
export const DYNAMIC_LAYER_INDEX = 1;
export const STROKE_BYTES =
  8 + // kind, region, motion model, flags, alpha
  PARAMS_PER_STROKE * 4 +
  COLOR_COMPONENTS * 4 +
  MOTION_FIELDS * 4 +
  KEYFRAMES * KEYFRAME_FIELDS * 4;

export const REGIONS = [
  'background', 'arch', 'statue_left', 'statue_right', 'seat',
  'torso', 'chest', 'hair', 'wing_left', 'wing_right', 'petals',
];

export const MOTION_MODELS = [
  'static_lock', 'breathing', 'hair_wave', 'wing_float_left',
  'wing_float_right', 'petal_fall', 'shimmer',
];

// Offline static anchor compensation only. Runtime camera matrices are always
// identity and are kept in the format for explicit auditability.
export const CAMERA_FLOATS = KEYFRAMES * 6;

export function parseInstructions(buffer) {
  const view = new DataView(buffer);
  if (buffer.byteLength < 20) throw new Error('绘图指令文件过短');
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 8));
  if (magic !== MAGIC) throw new Error(`绘图指令文件标识无效：${magic}`);
  let offset = 8;
  const version = view.getUint16(offset, true); offset += 2;
  const keyCount = view.getUint16(offset, true); offset += 2;
  const jsonLength = view.getUint32(offset, true); offset += 4;
  const strokeCount = view.getUint32(offset, true); offset += 4;
  if (version !== 4 || keyCount !== KEYFRAMES) {
    throw new Error(`绘图指令版本不兼容：version=${version}, keyframes=${keyCount}`);
  }
  const metadataEnd = offset + jsonLength + CAMERA_FLOATS * 4;
  if (metadataEnd > buffer.byteLength) {
    throw new Error(`指令元信息越界：${metadataEnd} > ${buffer.byteLength}`);
  }
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, offset, jsonLength)));
  offset += jsonLength;
  const camera = new Float32Array(CAMERA_FLOATS);
  new Uint8Array(camera.buffer).set(new Uint8Array(buffer, offset, CAMERA_FLOATS * 4));
  offset += CAMERA_FLOATS * 4;
  if (offset + strokeCount * STROKE_BYTES !== buffer.byteLength) {
    throw new Error(`指令长度异常：${offset} + ${strokeCount} * ${STROKE_BYTES} != ${buffer.byteLength}`);
  }

  const staticIds = [];
  const dynamicIds = [];
  const sourceGeometry = new Float32Array(strokeCount * 12);
  const sourceColors = new Float32Array(strokeCount * 3);
  const sourceMotion = new Float32Array(strokeCount * MOTION_FIELDS);
  const sourceKeys = new Float32Array(strokeCount * KEYFRAMES * KEYFRAME_FIELDS);
  const sourceKinds = new Uint8Array(strokeCount);
  const sourceRegions = new Uint8Array(strokeCount);
  const sourceModels = new Uint8Array(strokeCount);
  const sourceAlpha = new Float32Array(strokeCount);
  const strokeOffset = offset;
  for (let i = 0; i < strokeCount; i += 1) {
    const record = offset + i * STROKE_BYTES;
    (view.getUint8(record + 2) === 0 ? staticIds : dynamicIds).push(i);
    sourceKinds[i] = view.getUint8(record);
    sourceRegions[i] = view.getUint8(record + 1);
    sourceModels[i] = view.getUint8(record + 2);
    sourceAlpha[i] = view.getFloat32(record + 4, true);
    for (let j = 0; j < 12; j += 1) sourceGeometry[i * 12 + j] = view.getFloat32(record + 8 + j * 4, true);
    for (let j = 0; j < 3; j += 1) sourceColors[i * 3 + j] = view.getFloat32(record + 56 + j * 4, true);
    for (let j = 0; j < MOTION_FIELDS; j += 1) sourceMotion[i * MOTION_FIELDS + j] = view.getFloat32(record + 68 + j * 4, true);
    for (let j = 0; j < KEYFRAMES * KEYFRAME_FIELDS; j += 1) {
      sourceKeys[i * KEYFRAMES * KEYFRAME_FIELDS + j] = view.getFloat32(record + 100 + j * 4, true);
    }
  }
  const orderedIds = [...staticIds, ...dynamicIds];
  const params = new Float32Array(strokeCount * 16);
  const colors = new Float32Array(strokeCount * 4);
  const motion = new Float32Array(strokeCount * MOTION_FIELDS);
  const keys = new Float32Array(strokeCount * KEYFRAMES * KEYFRAME_FIELDS);
  const layerCounts = [0, 0, 0, 0, 0, 0];
  const regionCounts = new Uint32Array(REGIONS.length);
  const modelCounts = new Uint32Array(MOTION_MODELS.length);
  const chunk = (length) => {
    const out = new Float32Array(length);
    new Uint8Array(out.buffer).set(new Uint8Array(buffer, offset, length * 4));
    offset += length * 4;
    return out;
  };
  // Seek to the exact source record of each ordered slot.
  for (let slot = 0; slot < strokeCount; slot += 1) {
    const i = orderedIds[slot];
    offset = strokeOffset + i * STROKE_BYTES;
    const kind = view.getUint8(offset);
    const region = view.getUint8(offset + 1);
    const model = view.getUint8(offset + 2);
    const flags = view.getUint8(offset + 3);
    const alpha = view.getFloat32(offset + 4, true);
    offset += 8;
    if (kind >= layerCounts.length) throw new Error(`笔触 ${i} 的图元类型越界：${kind}`);
    if (region >= REGIONS.length) throw new Error(`笔触 ${i} 的区域越界：${region}`);
    if (model >= MOTION_MODELS.length) throw new Error(`笔触 ${i} 的运动模型越界：${model}`);
    if (flags !== 0) throw new Error(`笔触 ${i} 存在未知标志位：${flags}`);
    layerCounts[kind] += 1;
    regionCounts[region] += 1;
    modelCounts[model] += 1;
    const p = chunk(PARAMS_PER_STROKE);
    const c = chunk(COLOR_COMPONENTS);
    const m = chunk(MOTION_FIELDS);
    const k = chunk(KEYFRAMES * KEYFRAME_FIELDS);
    // params.rgba = kind/255, region/255, model/255, alpha. The three unused
    // fields in p8..p11 carry generator parameters that the shader reads from
    // the dedicated motion block, so only four vec4 slots are needed here.
    // Texture slot order is static first, dynamic second. aIndex addresses
    // these slots; sourceRecords.ids maps each slot back to its source stroke.
    params.set([kind / 255, region / 255, model / 255, alpha, ...p], slot * 16);
    colors.set([c[0] / 255, c[1] / 255, c[2] / 255, 1], slot * 4);
    motion.set(m, slot * MOTION_FIELDS);
    keys.set(k, slot * KEYFRAMES * KEYFRAME_FIELDS);
  }
  // The final ordered slot may point to any source record, so validate the
  // consumed source range instead of the cursor left by the last iteration.
  offset = strokeOffset + strokeCount * STROKE_BYTES;
  if (offset !== buffer.byteLength) throw new Error(`解析后偏移异常：${offset} != ${buffer.byteLength}`);
  return {
    meta, camera, params, colors, motion, keys, layerCounts,
    regionCounts, modelCounts, strokeCount, bytes: buffer.byteLength,
    staticCount: staticIds.length,
    dynamicCount: dynamicIds.length,
    dynamicStart: staticIds.length,
    sourceRecords: {
      kinds: sourceKinds,
      regions: sourceRegions,
      models: sourceModels,
      alpha: sourceAlpha,
      geometry: sourceGeometry,
      colors: sourceColors,
      motion: sourceMotion,
      keys: sourceKeys,
      ids: orderedIds,
      dynamicStart: staticIds.length,
      dynamicCount: dynamicIds.length,
    },
  };
}
