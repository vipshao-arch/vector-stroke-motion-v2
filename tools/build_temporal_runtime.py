#!/usr/bin/env python3
"""Build frame-indexed vector strokes and residual brushes (VTSK0002).

Reference frames are consumed only by this offline generator. The runtime
loads one fixed clean static base plus these compact instructions. Every
instruction describes a short stroke or a small colored brush sampled from a
single frame; no frame bitmap or coverage image is stored.
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
FPS = 24
MAGIC = b"VTSK0002"
REGION_NAMES = ("hair", "chest", "wing_left", "wing_right", "petals")
REGIONS = {
    "hair": (255, 0, 445, 215),
    "chest": (305, 80, 435, 190),
    "wing_left": (20, 140, 315, 352),
    "wing_right": (440, 70, 752, 352),
    "petals": (0, 0, WIDTH, HEIGHT),
}
MASK_POLYGONS = {
    "hair": [
        (332, 5), (382, 5), (410, 32), (438, 52), (432, 70),
        (408, 82), (398, 112), (382, 207), (340, 207),
        (272, 175), (282, 132), (315, 92), (315, 40),
    ],
    "chest": [(318, 88), (428, 88), (430, 180), (312, 180)],
    "wing_left": [
        (24, 352), (26, 238), (65, 178), (195, 148), (307, 168),
        (307, 230), (165, 240), (103, 302), (62, 352),
        (255, 352), (307, 352),
    ],
    "wing_right": [
        (446, 82), (752, 142), (752, 332), (698, 352),
        (528, 352), (498, 288), (546, 222), (632, 148), (450, 118),
    ],
}
LIMITS = {
    "hair": 1100, "chest": 380, "wing_left": 1500,
    "wing_right": 2100, "petals": 900,
}
CELL = 4
GRID_W, GRID_H = WIDTH // CELL, HEIGHT // CELL
STROKE_BYTES = 14  # <HHH8B
WASH_BYTES = 8     # <IBBBB
HEADER_BYTES = 42  # <8s11H3I
TABLE_ENTRY_BYTES = 10
STROKE_TABLE_OFFSET = HEADER_BYTES
WASH_TABLE_OFFSET = STROKE_TABLE_OFFSET + FRAME_COUNT * len(REGION_NAMES) * TABLE_ENTRY_BYTES
DATA_OFFSET = WASH_TABLE_OFFSET + FRAME_COUNT * len(REGION_NAMES) * TABLE_ENTRY_BYTES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("assets/base.jpg"))
    parser.add_argument("--output", type=Path, default=Path("assets/temporal_runtime.bin"))
    parser.add_argument("--manifest", type=Path, default=Path("assets/temporal_runtime.manifest.json"))
    return parser.parse_args()


def region_mask(name: str, box: tuple[int, int, int, int]) -> np.ndarray:
    """Soft membership inside a moving region, eroded away from hard edges."""
    x0, y0, x1, y1 = box
    local = np.zeros((y1 - y0, x1 - x0), np.uint8)
    if name in MASK_POLYGONS:
        points = np.array([[round(x - x0), round(y - y0)] for x, y in MASK_POLYGONS[name]], np.int32)
        cv2.fillPoly(local, [points], 255)
    else:
        local[:] = 255
    distance = cv2.distanceTransform(local, cv2.DIST_L2, 5)
    interior = np.clip((distance - 2.0) / 4.0, 0, 1)
    return cv2.GaussianBlur(interior.astype(np.float32), (0, 0), 1.0)


def static_guard() -> np.ndarray:
    """Feathered suppression across the annotated static architecture."""
    guard = np.ones((HEIGHT, WIDTH), np.float32)
    for x0, y0, x1, y1 in (
        (615, 5, 705, 105), (58, 190, 160, 288),
        (565, 150, 665, 285), (285, 270, 470, 348),
    ):
        guard[y0:y1, x0:x1] = 0.08
    return cv2.GaussianBlur(guard, (0, 0), 12)


def face_guard() -> np.ndarray:
    """Keep the face sharp while the surrounding hair and chest repaint."""
    face = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.ellipse(face, (374, 57), (28, 33), 0, 0, 360, 255, -1)
    face = cv2.GaussianBlur(face.astype(np.float32) / 255, (0, 0), 5)
    return np.clip(1.0 - face * 0.90, 0, 1)


def petal_exclusion() -> np.ndarray:
    exclusion = np.ones((HEIGHT, WIDTH), np.float32)
    cv2.rectangle(exclusion, (245, 25), (505, HEIGHT), 0.07, -1)
    cv2.ellipse(exclusion, (190, 275), (190, 105), -18, 0, 360, 0.12, -1)
    cv2.ellipse(exclusion, (625, 210), (165, 150), -5, 0, 360, 0.12, -1)
    return cv2.GaussianBlur(exclusion, (0, 0), 3)


def petal_transients(frame: np.ndarray, median_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compact bright blobs only, excluding broad horizontal video scanlines."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    values = gray.astype(np.float32)
    local = values - cv2.GaussianBlur(values, (0, 0), 5)
    temporal = values - median_gray
    candidate = ((temporal > 12) & (local > 6) & (values > 138)).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    compact = np.zeros_like(candidate)
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if 4 <= area <= 260 and w <= 30 and h <= 30 and max(w, h) / max(1, min(w, h)) < 4.2:
            compact[labels == label] = 255
    score = np.clip(local, 0, 80) * np.clip(temporal, 0, 80) / 400
    score *= (compact > 0) * petal_exclusion()
    wash = cv2.dilate(compact.astype(np.float32) / 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    wash = cv2.GaussianBlur(wash, (0, 0), 1.4) * petal_exclusion()
    return cv2.GaussianBlur(score.astype(np.float32), (0, 0), 0.6), wash


def local_residual(
    frame: np.ndarray,
    base: np.ndarray,
    box: tuple[int, int, int, int],
    membership: np.ndarray,
    global_activity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Signed color residual and its confidence in a dynamic region."""
    x0, y0, x1, y1 = box
    current = frame[y0:y1, x0:x1].astype(np.float32)[:, :, ::-1]
    fixed = base[y0:y1, x0:x1].astype(np.float32)[:, :, ::-1]
    delta = current - fixed
    magnitude = np.abs(delta).mean(axis=2)
    # Temporal activity gating keeps the fixed architecture and statues out of
    # broad moving-object polygons, while brightness changes from the video's
    # global shimmer are suppressed by the local high-pass below.
    high_pass = magnitude - cv2.GaussianBlur(magnitude, (0, 0), 6)
    activity = global_activity[y0:y1, x0:x1]
    confidence = (
        np.clip((magnitude - 2.0) / 20.0, 0, 1) * 0.68
        + np.clip((np.abs(high_pass) - 0.8) / 16.0, 0, 1) * 0.20
        + np.clip(activity / 28.0, 0, 1) * 0.12
    ) * membership * np.clip(activity / 5.5, 0, 1)
    confidence = cv2.GaussianBlur(confidence, (0, 0), 0.7)
    return delta, confidence


def sample_strokes(
    frame: np.ndarray,
    residual: np.ndarray,
    confidence: np.ndarray,
    box: tuple[int, int, int, int],
    limit: int,
    rng: np.random.Generator,
) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    score = (0.65 * np.sqrt(magnitude) + 0.35 * np.sqrt(np.abs(residual).mean(axis=2) + 1)) * confidence
    score = cv2.GaussianBlur(score, (0, 0), 0.55)
    cell = 3
    h, w = score.shape
    usable_h, usable_w = h // cell * cell, w // cell * cell
    pooled = score[:usable_h, :usable_w].reshape(usable_h // cell, cell, usable_w // cell, cell)
    candidates = pooled.max(axis=(1, 3))
    flat = candidates.ravel()
    if flat.sum() <= 1e-6:
        return np.zeros((0, 9), np.float32)
    count = min(limit, int(np.count_nonzero(flat)))
    chosen = rng.choice(len(flat), size=count, replace=False, p=flat / flat.sum())
    cell_y, cell_x = np.divmod(chosen, usable_w // cell)
    offsets = np.argmax(pooled[cell_y, :, cell_x, :].reshape(count, cell * cell), axis=1)
    oy, ox = np.divmod(offsets, cell)
    ys, xs = cell_y * cell + oy, cell_x * cell + ox
    tx, ty = -gy[ys, xs], gx[ys, xs]
    norm = np.maximum(np.hypot(tx, ty), 1e-4)
    tx, ty = tx / norm, ty / norm
    colors = np.clip(crop[ys, xs].astype(np.float32)[:, ::-1] * 1.08 + 8, 0, 255)
    strength = np.clip(magnitude[ys, xs] / 105 + confidence[ys, xs], 0.1, 1)
    half = np.clip(2.2 + np.sqrt(magnitude[ys, xs]) * 0.075, 2.3, 7.2)
    alpha = np.clip(0.10 + strength * 0.40, 0.10, 0.50)
    return np.stack([
        xs + x0 + rng.uniform(-0.3, 0.3, count),
        ys + y0 + rng.uniform(-0.3, 0.3, count),
        tx, ty, half, colors[:, 0], colors[:, 1], colors[:, 2], alpha,
    ], axis=1).astype(np.float32)


def sample_petal_strokes(frame: np.ndarray, score: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    flat = score.ravel()
    if flat.sum() <= 0:
        return np.zeros((0, 9), np.float32)
    count = min(limit, int(np.count_nonzero(flat)))
    chosen = rng.choice(len(flat), size=count, replace=False, p=flat / flat.sum())
    ys, xs = np.divmod(chosen, WIDTH)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)[ys, xs]
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)[ys, xs]
    norm = np.maximum(np.hypot(gx, gy), 1e-4)
    tx, ty = -gy / norm, gx / norm
    colors = np.clip(frame[ys, xs].astype(np.float32)[:, ::-1] * 1.16 + 22, 0, 255)
    half = np.clip(1.8 + np.sqrt(score[ys, xs]) * 0.16, 1.8, 6.2)
    alpha = np.clip(0.38 + score[ys, xs] / 170, 0.38, 0.9)
    return np.stack([
        xs + rng.uniform(-0.3, 0.3, count), ys + rng.uniform(-0.3, 0.3, count),
        tx, ty, half, colors[:, 0], colors[:, 1], colors[:, 2], alpha,
    ], axis=1).astype(np.float32)


def quantize_stroke(record: np.ndarray, is_petal: bool) -> bytes:
    x, y, tx, ty, half, r, g, b, alpha = record
    angle = (np.arctan2(ty, tx) + np.pi) / (2 * np.pi)
    return struct.pack(
        "<HHH8B",
        round(float(np.clip(x, 0, WIDTH - 0.01)) * 16),
        round(float(np.clip(y, 0, HEIGHT - 0.01)) * 16),
        round(float(angle) * 65535),
        round(float(np.clip(half, 0, 63.75)) * 4),
        round(float(np.clip(1.0 + half * 0.09, 0, 63.75)) * 4),
        round(float(np.clip(r, 0, 255))), round(float(np.clip(g, 0, 255))),
        round(float(np.clip(b, 0, 255))), round(float(np.clip(alpha, 0, 1)) * 255),
        1 if is_petal else 0, 0,
    )


def quantize_wash(index: int, rgb: np.ndarray, alpha: float) -> bytes:
    return struct.pack(
        "<IBBBB", index,
        round(float(np.clip(rgb[0], 0, 255))), round(float(np.clip(rgb[1], 0, 255))),
        round(float(np.clip(rgb[2], 0, 255))), round(float(np.clip(alpha, 0, 1)) * 255),
    )


def wash_instructions(
    residual: np.ndarray,
    confidence: np.ndarray,
    box: tuple[int, int, int, int],
    claimed: np.ndarray,
    base_cells: np.ndarray,
    frame_cells: np.ndarray,
    face_guard_grid: np.ndarray,
) -> tuple[list[bytes], np.ndarray]:
    x0, y0, x1, y1 = box
    gx0, gy0 = x0 // CELL, y0 // CELL
    gx1, gy1 = min(GRID_W, (x1 + CELL - 1) // CELL), min(GRID_H, (y1 + CELL - 1) // CELL)
    membership = cv2.resize(confidence, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
    available = cv2.resize(claimed, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
    strength = np.clip(membership * (1.0 - np.clip(available, 0, 1)), 0, 1) * face_guard_grid
    strength = np.where(strength > 0.16, 0.995, np.clip(strength * 5.8, 0, 1))
    indices = np.argwhere(strength[gy0:gy1, gx0:gx1] > 0.02)
    output = []
    for local_y, local_x in indices:
        gy, gx = gy0 + int(local_y), gx0 + int(local_x)
        alpha = float(strength[gy, gx])
        if alpha <= 0:
            continue
        # Full current-frame color makes moving wings and hair opaque. The
        # record is still only a position, three color bytes and an alpha.
        output.append(quantize_wash(int(gy * GRID_W + gx), frame_cells[gy, gx], alpha))
    claim = np.zeros((HEIGHT, WIDTH), np.float32)
    claim[y0:y1, x0:x1] = confidence
    return output, np.maximum(claimed, claim)


def main() -> None:
    args = parse_args()
    raw = np.fromfile(args.frames, dtype=np.uint8)
    frames = raw.reshape(FRAME_COUNT, HEIGHT, WIDTH, 3)
    median = np.median(frames, axis=0).astype(np.uint8)
    median_gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) for frame in frames])
    global_activity = gray.std(axis=0)
    base = cv2.imread(str(args.base))
    if base is None:
        raise SystemExit(f"Cannot read {args.base}")
    blurred_base = cv2.GaussianBlur(base, (0, 0), 3.0)
    base_cells = cv2.resize(blurred_base, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32)
    guard = static_guard()
    face_guard_grid = cv2.resize(face_guard(), (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
    memberships = {}
    for name in REGION_NAMES:
        x0, y0, x1, y1 = REGIONS[name]
        membership = region_mask(name, REGIONS[name]) * guard[y0:y1, x0:x1]
        if name in ("hair", "chest"):
            membership *= face_guard()[y0:y1, x0:x1]
        memberships[name] = membership

    rng = np.random.default_rng(240518)
    total_entries = FRAME_COUNT * len(REGION_NAMES)
    stroke_counts: list[int] = [0] * total_entries
    stroke_offsets: list[int] = [0] * total_entries
    wash_counts: list[int] = [0] * total_entries
    wash_offsets: list[int] = [0] * total_entries
    frame_payloads: list[bytearray] = []

    for frame_index, frame in enumerate(frames):
        frame_payload = bytearray()
        petal_score, petal_wash = petal_transients(frame, median_gray)
        frame_cells = cv2.resize(frame, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32)
        claimed = np.zeros((HEIGHT, WIDTH), np.float32)
        for region_index, name in enumerate(REGION_NAMES):
            entry_index = frame_index * len(REGION_NAMES) + region_index
            box = REGIONS[name]
            if name == "petals":
                records = sample_petal_strokes(frame, petal_score, LIMITS[name], rng)
                washes = [
                    quantize_wash(int(gy * GRID_W + gx),
                                  cv2.resize(frame, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)[gy, gx, ::-1],
                                  float(min(petal_wash[min(gy * CELL, HEIGHT - 1), min(gx * CELL, WIDTH - 1)] * 0.65, 0.62)))
                    for gy, gx in np.argwhere(
                        cv2.resize(petal_wash, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA) > 0.15
                    )
                ]
            else:
                delta, confidence = local_residual(
                    frame, blurred_base, box, memberships[name], global_activity,
                )
                records = sample_strokes(frame, delta, confidence, box, LIMITS[name], rng)
                washes, claimed = wash_instructions(
                    delta, confidence, box, claimed, base_cells,
                    frame_cells, face_guard_grid,
                )
            stroke_counts[entry_index] = len(records)
            stroke_offsets[entry_index] = DATA_OFFSET + sum(len(payload) for payload in frame_payloads) + len(frame_payload)
            for record in records:
                frame_payload.extend(quantize_stroke(record, name == "petals"))
            wash_counts[entry_index] = len(washes)
            wash_offsets[entry_index] = DATA_OFFSET + sum(len(payload) for payload in frame_payloads) + len(frame_payload)
            for wash in washes:
                frame_payload.extend(wash)
        frame_payloads.append(frame_payload)

    stroke_table = b"".join(
        struct.pack("<HQ", count, offset)
        for count, offset in zip(stroke_counts, stroke_offsets)
    )
    wash_table = b"".join(
        struct.pack("<HQ", count, offset)
        for count, offset in zip(wash_counts, wash_offsets)
    )
    header = struct.pack(
        "<8sHHHHHHHHHHHIII", MAGIC, 2, WIDTH, HEIGHT, FRAME_COUNT,
        len(REGION_NAMES), FPS, STROKE_BYTES, WASH_BYTES,
        GRID_W, GRID_H, CELL, STROKE_TABLE_OFFSET, WASH_TABLE_OFFSET, DATA_OFFSET,
    )
    if len(header) != HEADER_BYTES:
        raise RuntimeError(f"unexpected header size: {len(header)}")
    binary = header + stroke_table + wash_table + b"".join(bytes(payload) for payload in frame_payloads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(binary)

    manifest = {
        "magic": MAGIC.decode(), "version": 2,
        "width": WIDTH, "height": HEIGHT,
        "frame_count": FRAME_COUNT, "fps": FPS,
        "regions": [{"name": name, "box": list(REGIONS[name]), "limit": LIMITS[name]} for name in REGION_NAMES],
        "file": args.output.as_posix(),
        "base": args.base.as_posix(), "bytes": len(binary),
        "header_bytes": HEADER_BYTES,
        "stroke_record_bytes": STROKE_BYTES, "wash_record_bytes": WASH_BYTES,
        "stroke_table_offset": STROKE_TABLE_OFFSET,
        "wash_table_offset": WASH_TABLE_OFFSET, "data_offset": DATA_OFFSET,
        "grid": {"width": GRID_W, "height": GRID_H, "cell_px": CELL},
        "strokes": {
            "total": sum(stroke_counts),
            "mean_per_frame": round(sum(stroke_counts) / FRAME_COUNT, 1),
            "min_per_frame": min(sum(stroke_counts[f * 5:(f + 1) * 5]) for f in range(FRAME_COUNT)),
            "max_per_frame": max(sum(stroke_counts[f * 5:(f + 1) * 5]) for f in range(FRAME_COUNT)),
        },
        "wash_patches": {
            "total": sum(wash_counts),
            "mean_per_frame": round(sum(wash_counts) / FRAME_COUNT, 1),
        },
        "source": args.frames.as_posix(),
        "description": "frame-indexed contour strokes and full-color dynamic brushes; one fixed clean static base at runtime",
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
