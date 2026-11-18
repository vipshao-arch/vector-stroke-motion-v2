#!/usr/bin/env python3
"""Render VSTK0004 frames on the CPU for deterministic ROI verification.

This uses the exact binary drawing instructions loaded by the browser. It is a
measurement instrument, not a runtime asset: the released web page still draws
every primitive in Three.js/WebGL and never loads video frames.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import cv2
import numpy as np


W, H = 752, 352
K, FIELDS = 31, 8
STROKE_BYTES = 8 + 12 * 4 + 3 * 4 + 8 * 4 + K * FIELDS * 4
TIME_LABELS = ("0s", "1s", "2_5s", "4s", "5s")


def read_strokes(path: Path) -> tuple[dict, np.ndarray]:
    raw = path.read_bytes()
    magic = raw[:8].decode()
    version, key_count, json_len, count = struct.unpack_from("<HHII", raw, 8)
    if magic != "VSTK0004" or version != 4 or key_count != K:
        raise SystemExit(f"Unexpected instruction header: {magic} v{version} k{key_count}")
    offset = 20
    meta = json.loads(raw[offset:offset + json_len].decode("utf-8"))
    offset += json_len + K * 6 * 4
    dtype = np.dtype([
        ("kind", "u1"), ("region", "u1"), ("model", "u1"), ("flags", "u1"),
        ("alpha", "<f4"), ("params", "<f4", (12,)), ("color", "<f4", (3,)),
        ("motion", "<f4", (8,)), ("keys", "<f4", (K, FIELDS)),
    ])
    strokes = np.frombuffer(raw, dtype=dtype, count=count, offset=offset)
    return meta, strokes


def interpolate_keys(strokes: np.ndarray, normalized_time: float) -> np.ndarray:
    position = float(np.clip(normalized_time, 0, 1)) * (K - 1)
    index = int(math.floor(position))
    t = position - index
    t = t * t * (3 - 2 * t)
    a = strokes["keys"][:, index, :].astype(np.float64)
    b = strokes["keys"][:, min(index + 1, K - 1), :].astype(np.float64)
    return a * (1 - t) + b * t


def blend(canvas: np.ndarray, color: np.ndarray, alpha: np.ndarray) -> None:
    a = np.clip(alpha * 0.55, 0, 1)[..., None]
    canvas *= 1 - a
    canvas += np.asarray(color, np.float32)[None, None, :] * a


def render(strokes: np.ndarray, base: np.ndarray, seconds: float, duration: float) -> np.ndarray:
    canvas = base.copy()
    keys = interpolate_keys(strokes, seconds / duration)
    sample = strokes
    # The packed list is ordered from broad ellipses to foreground primitives.
    for i in range(len(sample)):
        s = sample[i]
        kind, model = int(s["kind"]), int(s["model"])
        p, color, alpha = s["params"].astype(np.float64), s["color"].astype(np.float32), float(s["alpha"])
        m = s["motion"].astype(np.float64)
        if model == 0:
            dx = dy = da = 0.0
            scale, rotate, brightness = 1.0, 0.0, 1.0
        else:
            dx, dy = keys[i, 0], keys[i, 1]
            scale, rotate, da, brightness = keys[i, 2], keys[i, 3], keys[i, 4], keys[i, 5]
        ink = np.clip(color * brightness, 0, 255)
        opacity = np.clip(alpha + da, 0, 1)
        if opacity < 0.004:
            continue
        if kind == 0:
            cx, cy, rx, ry, angle = p[:5]
            cx += dx
            cy += dy
            rx, ry = max(rx * scale, 0.7), max(ry * scale, 0.7)
            angle += rotate
            center = (float(cx), float(cy))
            axes = (max(1, int(rx)), max(1, int(ry)))
            patch = np.zeros((axes[1] * 2 + 3, axes[0] * 2 + 3), np.float32)
            cv2.ellipse(patch, (axes[0] + 1, axes[1] + 1), axes, math.degrees(angle), 0, 360, 1, -1, cv2.LINE_AA)
            x0, y0 = int(center[0] - axes[0] - 1), int(center[1] - axes[1] - 1)
            xs, xe = max(0, x0), min(W, x0 + patch.shape[1])
            ys, ye = max(0, y0), min(H, y0 + patch.shape[0])
            if xe > xs and ye > ys:
                blend(canvas[ys:ye, xs:xe], ink, patch[ys - y0:ye - y0, xs - x0:xe - x0] * opacity)
        elif kind in (1, 2, 3):
            p0, p1, p2 = p[0:2], p[2:4], p[4:6]
            center = (p0 + p1 + p2) / 3
            turn = np.array([
                [math.cos(rotate), -math.sin(rotate)],
                [math.sin(rotate), math.cos(rotate)],
            ])
            points = [(turn @ ((q - center) * scale) + center + np.array([dx, dy])) for q in (p0, p1, p2)]
            pts = [np.round(q).astype(np.int32) for q in points]
            width = max(1, int(round(max(p[6] * scale, 0.65))))
            patch = np.zeros((H, W), np.float32)
            if kind == 1:
                cv2.line(patch, tuple(pts[0]), tuple(pts[2]), 1, width * 2 + 1, cv2.LINE_AA)
            elif kind == 2:
                cv2.polylines(patch, [np.array([pts[0], pts[1], pts[2]], np.int32)], False, 1, width * 2 + 1, cv2.LINE_AA)
            else:
                curve = []
                for t in np.linspace(0, 1, 12):
                    curve.append((1 - t) ** 2 * pts[0] + 2 * (1 - t) * t * pts[1] + t * t * pts[2])
                cv2.polylines(patch, [np.round(curve).astype(np.int32)], False, 1, width * 2 + 1, cv2.LINE_AA)
            mask = patch > 0
            if mask.any():
                alpha_patch = np.zeros((H, W), np.float32)
                alpha_patch[mask] = opacity
                blend(canvas[mask], ink, np.full(int(mask.sum()), opacity, np.float32)) if False else None
                # ROI-safe direct blend on the sparse mask.
                a = np.clip(opacity * 0.55, 0, 1)
                canvas[mask] = canvas[mask] * (1 - a) + ink * a
        elif kind == 4:
            # Petal position is the shader's exact parametric fall model.
            frame = seconds / duration * 123.0
            life = math.modf(seconds / duration * m[4] + m[2])[0]
            fall = frame * m[1]
            sway = math.sin(frame * m[5] + m[0]) * m[6]
            wrap = math.floor(max(fall - m[3] + 352, 0) / 520) * 520
            cx = p[0] + sway + dx
            cy = p[1] + fall - m[3] - wrap + dy
            angle = p[4] + rotate + m[7] * life + math.sin(frame * m[5] * 0.5 + m[0]) * 0.35
            envelope = min(max(life / 0.09, 0), 1) * min(max((1 - life) / 0.28, 0), 1)
            rx, ry = max(p[2] * scale, 0.8), max(p[3] * scale, 0.8)
            patch = np.zeros((int(ry * 2) + 3, int(rx * 2) + 3), np.float32)
            cv2.ellipse(patch, (int(rx) + 1, int(ry) + 1), (max(1, int(rx)), max(1, int(ry))), math.degrees(angle), 0, 360, 1, -1, cv2.LINE_AA)
            x0, y0 = int(cx - rx - 1), int(cy - ry - 1)
            xs, xe, ys, ye = max(0, x0), min(W, x0 + patch.shape[1]), max(0, y0), min(H, y0 + patch.shape[0])
            if xe > xs and ye > ys:
                blend(canvas[ys:ye, xs:xe], ink, patch[ys - y0:ye - y0, xs - x0:xe - x0] * opacity * envelope)
        else:
            cx, cy, radius = p[0] + dx, p[1] + dy, max(p[2] * scale, 0.5)
            patch = np.zeros((int(radius * 2) + 3, int(radius * 2) + 3), np.float32)
            cv2.circle(patch, (int(radius) + 1, int(radius) + 1), max(1, int(radius)), 1, -1, cv2.LINE_AA)
            x0, y0 = int(cx - radius - 1), int(cy - radius - 1)
            xs, xe, ys, ye = max(0, x0), min(W, x0 + patch.shape[1]), max(0, y0), min(H, y0 + patch.shape[0])
            if xe > xs and ye > ys:
                blend(canvas[ys:ye, xs:xe], ink, patch[ys - y0:ye - y0, xs - x0:xe - x0] * opacity)
    return np.clip(canvas, 0, 255).astype(np.uint8)[..., ::-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instructions", type=Path, default=Path("assets/strokes.bin"))
    parser.add_argument("--base", type=Path, default=Path("assets/base.jpg"))
    parser.add_argument("--output", type=Path, default=Path("verification/projection"))
    args = parser.parse_args()
    meta, strokes = read_strokes(args.instructions)
    base = cv2.imread(str(args.base))[..., ::-1].astype(np.float32)
    args.output.mkdir(parents=True, exist_ok=True)
    times = [0, 1, 2.5, 4, 5]
    outputs = []
    for seconds, label in zip(times, TIME_LABELS):
        frame = render(strokes, base, seconds, meta["duration"])
        path = args.output / f"logical_{label}.png"
        cv2.imwrite(str(path), frame)
        outputs.append({"seconds": seconds, "path": str(path)})
    print(json.dumps({"meta": meta["format"], "frames": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
