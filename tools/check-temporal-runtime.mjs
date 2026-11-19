import { readFile } from 'node:fs/promises';

const binaryPath = process.argv[2] || 'assets/temporal_runtime.bin';
const manifestPath = process.argv[3] || 'assets/temporal_runtime.manifest.json';
const [buffer, manifestText] = await Promise.all([readFile(binaryPath), readFile(manifestPath, 'utf8')]);
const manifest = JSON.parse(manifestText);
const bytes = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
const view = new DataView(bytes);
const HEADER = 42;
const ENTRY = 10;
const STROKE = 14;
const WASH = 8;
const regions = manifest.regions.length;
const frames = manifest.frame_count;

const readEntry = (tableOffset, frame, region) => {
  const at = tableOffset + (frame * regions + region) * ENTRY;
  return {
    count: view.getUint16(at, true),
    offset: Number(view.getBigUint64(at + 2, true)),
  };
};

const values = [];
for (let frame = 0; frame < frames; frame += 1) {
  for (let region = 0; region < regions; region += 1) {
    for (const [table, size] of [
      [manifest.stroke_table_offset, STROKE],
      [manifest.wash_table_offset, WASH],
    ]) {
      const { count, offset } = readEntry(table, frame, region);
      if (count > 65535 || offset < manifest.data_offset || offset + count * size > buffer.length) {
        throw new Error(`invalid table entry frame=${frame} region=${region} count=${count} offset=${offset}`);
      }
    }
  }
}
for (const frame of [0, 24, 60, 96, 120]) {
  values.push({
    frame,
    strokes: Array.from({ length: regions }, (_, region) => readEntry(manifest.stroke_table_offset, frame, region).count),
    washes: Array.from({ length: regions }, (_, region) => readEntry(manifest.wash_table_offset, frame, region).count),
  });
}

const checks = {
  magic: buffer.subarray(0, 8).toString() === manifest.magic,
  version: view.getUint16(8, true) === manifest.version,
  dimensions: view.getUint16(10, true) === manifest.width && view.getUint16(12, true) === manifest.height,
  frameCount: view.getUint16(14, true) === frames,
  fps: view.getUint16(18, true) === manifest.fps,
  recordSizes: view.getUint16(20, true) === STROKE && view.getUint16(22, true) === WASH,
  tableOffsets: view.getUint32(30, true) === HEADER
    && view.getUint32(34, true) === manifest.wash_table_offset
    && view.getUint32(38, true) === manifest.data_offset,
  fileLength: buffer.length === manifest.bytes,
  tableBounds: true,
  fiveFrameGroupsHaveRecords: values.every((item) => item.strokes.some(Boolean) && item.washes.some(Boolean)),
};
const result = { binaryPath, bytes: buffer.length, frames, regions, checks, fiveFrameGroups: values };
console.log(JSON.stringify(result, null, 2));
if (Object.values(checks).some((value) => !value)) process.exitCode = 1;
