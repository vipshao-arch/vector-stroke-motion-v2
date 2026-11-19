#!/usr/bin/env python3
"""Clean one fixed base with a local low-frequency patch and sync metadata.

The sharp reconstruction is kept outside moving envelopes. Only residual
frozen silhouettes inside those envelopes receive a bounded low-frequency
plate. The runtime still loads exactly one image, assets/base.jpg.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import cv2
import numpy as np


WIDTH, HEIGHT = 752, 352
ENVELOPES = (
    ((20, 352), (22, 235), (60, 175), (195, 145), (310, 165), (310, 232),
     (165, 242), (100, 305), (58, 352), (260, 352), (310, 352)),
    ((442, 78), (752, 138), (752, 335), (702, 352), (525, 352),
     (495, 285), (544, 218), (635, 145), (446, 115)),
    ((238, 0), (355, 0), (360, 182), (332, 216), (258, 212), (232, 125)),
    ((305, 70), (437, 70), (437, 190), (305, 190)),
)
FRAMES_PATH = Path(r"C:\Users\vipshao\Documents\MagicLomix\2026-11-19\1\analysis\frames_bgr24.raw")
SOURCE_PATH = Path(r"C:\Users\vipshao\Documents\MagicLomix\2026-11-19\2\inspect_current\base_head.jpg")
FRAME_COUNT = 124


def envelope_map() -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    for points in ENVELOPES:
        cv2.fillPoly(mask, [np.array(points, np.int32)], 255)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5) / 10.0
    feather = np.clip(cv2.GaussianBlur(distance, (0, 0), 2), 0, 1)
    return feather * feather * (3.0 - 2.0 * feather)


def background_plate(frames: np.ndarray) -> np.ndarray:
    gray_stack = np.stack([
        cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
        for frame in frames
    ])
    rank = int(round((len(frames) - 1) * 0.88))
    index = np.argsort(gray_stack, axis=0)[rank][None, :, :, None]
    bright = np.take_along_axis(frames, np.repeat(index, 3, axis=3), axis=0)[0]
    plate = cv2.GaussianBlur(cv2.medianBlur(bright, 5), (0, 0), 1.2).astype(np.float32)
    continuous = cv2.GaussianBlur(plate, (0, 0), 12).astype(np.float32)
    return np.clip(continuous + (plate - continuous) * 0.28, 0, 255)


def make_base(frames: np.ndarray, source: np.ndarray) -> np.ndarray:
    gray_stack = np.stack([
        cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
        for frame in frames
    ])
    activity = cv2.GaussianBlur(np.clip((gray_stack.std(axis=0) - 3.0) / 10.0, 0, 1), (0, 0), 1.2)
    plate = background_plate(frames)
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)
    plate_gray = cv2.cvtColor(np.uint8(np.clip(plate, 0, 255)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    trace = np.clip((plate_gray - source_gray - 5.0) / 35.0, 0, 1)
    trace = cv2.GaussianBlur(trace, (0, 0), 1.5) * np.clip((activity + 0.22) / 0.7, 0, 1)
    # A bounded, low-frequency patch removes frozen traces without replacing
    # the sharp central subject or turning the whole swept area into white fog.
    weight = np.clip(envelope_map() * trace, 0, 0.12)[:, :, None]
    return np.uint8(np.clip(source.astype(np.float32) * (1 - weight) + plate * weight, 0, 255))


def replace_embedded_json(binary: Path, base: Path, manifest: Path) -> None:
    data = bytearray(binary.read_bytes())
    json_length = struct.unpack_from("<I", data, 12)[0]
    start = 20
    end = start + json_length
    meta = json.loads(bytes(data[start:end]).decode("utf-8"))
    meta["background"]["bytes"] = base.stat().st_size
    meta["background"]["description"] = "唯一的固定静态底板；运动包络内的冻结轮廓已做局部低频修补"
    encoded = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > json_length:
        raise RuntimeError(f"embedded JSON grew from {json_length} to {len(encoded)} bytes")
    padded = encoded[:-1] + b" " * (json_length - len(encoded)) + b"}"
    data[start:end] = padded
    binary.write_bytes(bytes(data))

    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["meta"]["background"]["bytes"] = base.stat().st_size
    document["meta"]["background"]["description"] = meta["background"]["description"]
    manifest.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    frames = np.fromfile(FRAMES_PATH, dtype=np.uint8).reshape(FRAME_COUNT, HEIGHT, WIDTH, 3)
    source = cv2.imread(str(SOURCE_PATH), cv2.IMREAD_COLOR)
    if source is None:
        raise SystemExit(f"Cannot read {SOURCE_PATH}")
    base = make_base(frames, source)
    output = Path("assets/base.jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), base, [
        cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_OPTIMIZE, 1,
    ])
    binary = Path("assets/strokes.bin")
    manifest = Path("assets/strokes.manifest.json")
    replace_embedded_json(binary, output, manifest)
    print(json.dumps({
        "base": output.as_posix(),
        "bytes": output.stat().st_size,
        "description": json.loads(manifest.read_text(encoding="utf-8"))["meta"]["background"]["description"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
