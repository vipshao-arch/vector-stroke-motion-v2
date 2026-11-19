#!/usr/bin/env python3
"""Rebuild the single fixed base and synchronize its metadata.

The base is a temporal percentile plate with dark frozen outlines suppressed
only inside the moving-object envelopes. Static architecture and statues are
preserved. The runtime still loads exactly one image, assets/base.jpg.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np


WIDTH, HEIGHT = 752, 352
FRAME_COUNT = 124
ENVELOPES = (
    ((20, 352), (22, 235), (60, 175), (195, 145), (310, 165), (310, 232),
     (165, 242), (100, 305), (58, 352), (260, 352), (310, 352)),
    ((442, 78), (752, 138), (752, 335), (702, 352), (525, 352),
     (495, 285), (544, 218), (635, 145), (446, 115)),
    ((238, 0), (355, 0), (360, 182), (332, 216), (258, 212), (232, 125)),
    ((305, 70), (437, 70), (437, 190), (305, 190)),
)
STATIC_ANCHORS = (
    (615, 5, 705, 105), (58, 190, 160, 288),
    (565, 150, 665, 285), (285, 270, 470, 348),
)


def make_base(frames: np.ndarray, center_source: Path | None = None) -> np.ndarray:
    percentile = np.percentile(frames, 70, axis=0).astype(np.uint8)
    plate = cv2.GaussianBlur(percentile, (0, 0), 16).astype(np.float32)
    gray = cv2.cvtColor(percentile, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_plate = cv2.GaussianBlur(gray, (0, 0), 16)
    envelope = np.zeros((HEIGHT, WIDTH), np.uint8)
    for points in ENVELOPES:
        cv2.fillPoly(envelope, [np.array(points, np.int32)], 255)
    anchors = np.zeros((HEIGHT, WIDTH), np.uint8)
    for x0, y0, x1, y1 in STATIC_ANCHORS:
        anchors[y0:y1, x0:x1] = 255
    anchors = cv2.GaussianBlur(anchors.astype(np.float32) / 255, (0, 0), 8)
    soft = cv2.distanceTransform(envelope, cv2.DIST_L2, 5) / 8.0
    soft = np.clip(cv2.GaussianBlur(soft, (0, 0), 2), 0, 1)
    dark = np.clip((gray_plate - gray - 2.0) / 24.0, 0, 1)
    replacement = np.clip(dark * soft * (1 - anchors * 0.98) * 1.15, 0, 1)[:, :, None]
    result = percentile.astype(np.float32) * (1 - replacement) + plate * replacement
    result = np.clip(result, 0, 255).astype(np.uint8)
    if center_source is not None:
        # The WebGL static reconstruction is sharper for the subject and seat.
        # It is baked into this one offline JPEG so runtime still loads one image.
        center = cv2.imread(str(center_source), cv2.IMREAD_COLOR)
        if center is None:
            raise SystemExit(f"Cannot read {center_source}")
        weight = np.zeros((HEIGHT, WIDTH), np.float32)
        weight[:, 235:455] = 1
        weight = cv2.GaussianBlur(weight, (0, 0), 9)[:, :, None]
        result = (result.astype(np.float32) * (1 - weight)
                  + center.astype(np.float32) * weight)
        result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def replace_embedded_json(binary: Path, base: Path, manifest: Path) -> None:
    data = bytearray(binary.read_bytes())
    json_length = struct.unpack_from("<I", data, 12)[0]
    start = 20
    end = start + json_length
    meta = json.loads(bytes(data[start:end]).decode("utf-8"))
    meta["background"]["bytes"] = base.stat().st_size
    meta["background"]["description"] = "唯一的固定静态底板；动态物体的冻结轮廓已按包络清理"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--center-source", type=Path, default=None)
    parser.add_argument("--base", type=Path, default=Path("assets/base.jpg"))
    parser.add_argument("--binary", type=Path, default=Path("assets/strokes.bin"))
    parser.add_argument("--manifest", type=Path, default=Path("assets/strokes.manifest.json"))
    args = parser.parse_args()
    frames = np.fromfile(args.frames, dtype=np.uint8).reshape(FRAME_COUNT, HEIGHT, WIDTH, 3)
    base = make_base(frames, args.center_source)
    args.base.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.base), base, [cv2.IMWRITE_JPEG_QUALITY, 95])
    replace_embedded_json(args.binary, args.base, args.manifest)
    print(json.dumps({
        "base": args.base.as_posix(),
        "bytes": args.base.stat().st_size,
        "description": json.loads(args.manifest.read_text(encoding="utf-8"))["meta"]["background"]["description"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
