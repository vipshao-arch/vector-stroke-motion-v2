#!/usr/bin/env python3
"""Build VTSK0002 vector instructions that repaint moving silhouettes.

The offline generator uses reference frames only. Each frame's moving object
is represented by compact silhouette-constrained full-color brushes and short
contour strokes. No video frame, per-frame bitmap, or image tile is shipped to
the browser.
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
ENVELOPES = {
    "hair": ((238, 0), (355, 0), (360, 182), (332, 216), (258, 212), (232, 125)),
    "chest": ((305, 70), (437, 70), (437, 190), (305, 190)),
    "wing_left": (
        (20, 352), (22, 235), (60, 175), (195, 145), (310, 165), (310, 232),
        (165, 242), (100, 305), (58, 352), (260, 352), (310, 352),
    ),
    "wing_right": (
        (442, 78), (752, 138), (752, 335), (702, 352), (525, 352),
        (495, 285), (544, 218), (635, 145), (446, 115),
    ),
}
STATIC_ANCHORS = (
    (615, 5, 705, 105), (58, 190, 160, 288),
    (565, 150, 665, 285), (285, 270, 470, 348),
)
WASH_BYTES = 8    # <IBBBB
STROKE_BYTES = 14  # <HHH8B
HEADER_BYTES = 42
TABLE_ENTRY_BYTES = 10
STROKE_TABLE_OFFSET = HEADER_BYTES
WASH_TABLE_OFFSET = STROKE_TABLE_OFFSET + FRAME_COUNT * len(REGION_NAMES) * TABLE_ENTRY_BYTES
DATA_OFFSET = WASH_TABLE_OFFSET + FRAME_COUNT * len(REGION_NAMES) * TABLE_ENTRY_BYTES
CELL = 4
GRID_W, GRID_H = WIDTH // CELL, HEIGHT // CELL
BRUSH_LIMITS = {"hair": 4500, "chest": 2200, "wing_left": 8500, "wing_right": 12000}
CONTOUR_LIMITS = {"hair": 950, "chest": 380, "wing_left": 1300, "wing_right": 1800, "petals": 900}


def envelope(name: str) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    if name in ENVELOPES:
        cv2.fillPoly(mask, [np.array(ENVELOPES[name], np.int32)], 255)
    else:
        mask[:] = 255
    return mask


def static_guard() -> np.ndarray:
    """Suppress fixed stone anchors only where no swept foreground passes."""
    guard = np.zeros((HEIGHT, WIDTH), np.float32)
    for x0, y0, x1, y1 in STATIC_ANCHORS:
        guard[y0:y1, x0:x1] = 1.0
    return cv2.GaussianBlur(guard, (0, 0), 8)


def face_guard() -> np.ndarray:
    face = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.ellipse(face, (374, 57), (28, 33), 0, 0, 360, 255, -1)
    return np.clip(1.0 - cv2.GaussianBlur(face.astype(np.float32) / 255, (0, 0), 5) * 0.90, 0, 1)


def petal_exclusion() -> np.ndarray:
    exclusion = np.ones((HEIGHT, WIDTH), np.float32)
    cv2.rectangle(exclusion, (245, 25), (505, HEIGHT), 0.07, -1)
    cv2.ellipse(exclusion, (190, 275), (190, 105), -18, 0, 360, 0.12, -1)
    cv2.ellipse(exclusion, (625, 210), (165, 150), -5, 0, 360, 0.12, -1)
    return cv2.GaussianBlur(exclusion, (0, 0), 3)


def motion_maps(gray_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Swept foreground coverage and direct evidence of changing pixels."""
    low = np.percentile(gray_stack, 15, axis=0).astype(np.float32)
    high = np.percentile(gray_stack, 75, axis=0).astype(np.float32)
    darkness = np.clip((high - low - 8.0) / 35.0, 0, 1)
    darkness *= np.clip((high - gray_stack.min(axis=0) - 10.0) / 35.0, 0, 1)
    darkness = cv2.GaussianBlur(darkness, (0, 0), 1.0)
    dark_union = cv2.dilate(
        (darkness > 0.18).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    )
    dark_union = cv2.GaussianBlur(dark_union.astype(np.float32), (0, 0), 3.0)
    activity = np.clip((gray_stack.std(axis=0) - 1.6) / 4.5, 0, 1)
    active_union = cv2.dilate(
        (activity > 0.12).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    active_union = cv2.GaussianBlur(active_union.astype(np.float32), (0, 0), 4.0)
    envelope_map = np.zeros((HEIGHT, WIDTH), np.float32)
    for name in ("hair", "chest", "wing_left", "wing_right"):
        envelope_map += envelope(name).astype(np.float32) / 255.0
    envelope_map = cv2.GaussianBlur(np.clip(envelope_map, 0, 1), (0, 0), 4)
    # Both the dark silhouette and the bright gaps it exposes must use the
    # current frame's color. The union closes holes inside the swept envelope
    # while low-activity architecture remains on the fixed base.
    # The full swept envelope is the foreground coverage. A moving wing must
    # carry the current frame color in both dark blades and bright gaps; using
    # only high-variance pixels leaves translucent-looking holes in the result.
    coverage = np.maximum(np.maximum(dark_union * 1.15, active_union * envelope_map), envelope_map)
    swept = coverage
    anchors = np.zeros((HEIGHT, WIDTH), np.uint8)
    for x0, y0, x1, y1 in STATIC_ANCHORS:
        anchors[y0:y1, x0:x1] = 255
    anchors = cv2.GaussianBlur(anchors.astype(np.float32) / 255, (0, 0), 8)
    anchor_release = np.clip((swept - 0.12) * 1.7, 0, 1)
    coverage *= 1.0 - anchors * (1.0 - anchor_release) * 0.30
    return np.clip(coverage, 0, 1), np.clip(swept, 0, 1)


def frame_activity(gray_stack: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(gray_stack.std(axis=0), (0, 0), 0.8)


def petal_transients(frame: np.ndarray, median_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local = gray - cv2.GaussianBlur(gray, (0, 0), 5)
    temporal = gray - median_gray
    candidate = ((temporal > 12) & (local > 6) & (gray > 138)).astype(np.uint8)
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


def membership_map(name: str, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return envelope(name)[y0:y1, x0:x1].astype(np.float32) / 255.0


def moving_alpha(
    box: tuple[int, int, int, int],
    membership: np.ndarray,
    coverage: np.ndarray,
    activity: np.ndarray,
    motion_evidence: np.ndarray,
) -> np.ndarray:
    x0, y0, x1, y1 = box
    swept = coverage[y0:y1, x0:x1]
    # The swept area is an opaque, frame-colored region. Semi-transparent
    # brushes reveal the cleaned static plate through moving wings and hair,
    # leaving pale transparent-looking patches in the final composition.
    # Every cell inside the swept silhouette is a solid, frame-colored brush.
    # Only the envelope boundary and fixed stone anchors are feathered.
    alpha = membership * np.clip((swept - 0.025) * 2.0, 0, 1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7)
    # A moving silhouette carries its current-frame color even when it crosses
    # a static stone box. Only the boundary and the face are feathered.
    anchors = static_guard()[y0:y1, x0:x1]
    anchor_release = np.clip((swept - 0.12) * 1.7, 0, 1)
    protection = 1.0 - anchors * (1.0 - anchor_release) * 0.45
    return np.clip(alpha * 1.02, 0, 1) * face_guard()[y0:y1, x0:x1] * protection


def brush_instructions(
    alpha: np.ndarray,
    box: tuple[int, int, int, int],
    limit: int,
    frame_cells: np.ndarray,
) -> list[bytes]:
    """Frame-colored 4 px cells confined to the swept moving silhouette."""
    x0, y0, x1, y1 = box
    full_alpha = np.zeros((HEIGHT, WIDTH), np.float32)
    full_alpha[y0:y0 + alpha.shape[0], x0:x0 + alpha.shape[1]] = alpha
    grid_alpha = cv2.resize(full_alpha, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
    gx0, gy0 = max(0, x0 // CELL), max(0, y0 // CELL)
    gx1, gy1 = min(GRID_W, (x1 + CELL - 1) // CELL), min(GRID_H, (y1 + CELL - 1) // CELL)
    output: list[bytes] = []
    for gy in range(gy0, gy1):
        for gx in range(gx0, gx1):
            if grid_alpha[gy, gx] <= 0.02:
                continue
            strength = float(grid_alpha[gy, gx])
            # A covered cell is fully opaque. Feathering is confined to the
            # narrow boundary of the swept envelope.
            if strength > 0.16:
                strength = 1.0
            else:
                strength = min(1.0, strength * 6.0)
            if strength <= 0:
                continue
            output.append(quantize_wash(int(gy * GRID_W + gx), frame_cells[gy, gx], strength))
    if len(output) > limit:
        # Never thin the interior with an ordered subset; that creates regular
        # transparent gaps. A limit overflow is a build error instead.
        raise RuntimeError(f"brush limit {limit} is below full coverage {len(output)}")
    return output


def opacity_byte(alpha: float) -> int:
    return round(float(np.clip(alpha, 0, 1)) * 255)


def contour_instructions(
    frame: np.ndarray,
    base: np.ndarray,
    alpha: np.ndarray,
    box: tuple[int, int, int, int],
    limit: int,
    rng: np.random.Generator,
) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    fixed = base[y0:y1, x0:x1].astype(np.float32)
    score = (
        0.7 * np.sqrt(magnitude)
        + 0.3 * np.sqrt(np.abs(crop.astype(np.float32) - fixed).mean(axis=2) + 1)
    ) * alpha
    score = cv2.GaussianBlur(score, (0, 0), 0.55)
    cell = 3
    h, w = score.shape
    usable_h, usable_w = h // cell * cell, w // cell * cell
    pooled = score[:usable_h, :usable_w].reshape(usable_h // cell, cell, usable_w // cell, cell)
    candidates = pooled.max(axis=(1, 3)).ravel()
    if candidates.sum() <= 1e-6:
        return np.zeros((0, 9), np.float32)
    count = min(limit, int(np.count_nonzero(candidates)))
    chosen = rng.choice(len(candidates), size=count, replace=False, p=candidates / candidates.sum())
    cell_y, cell_x = np.divmod(chosen, usable_w // cell)
    offsets = np.argmax(pooled[cell_y, :, cell_x, :].reshape(count, cell * cell), axis=1)
    oy, ox = np.divmod(offsets, cell)
    ys, xs = cell_y * cell + oy, cell_x * cell + ox
    tx, ty = -gy[ys, xs], gx[ys, xs]
    norm = np.maximum(np.hypot(tx, ty), 1e-4)
    colors = np.clip(crop[ys, xs].astype(np.float32)[:, ::-1] * 1.08 + 8, 0, 255)
    strength = np.clip(magnitude[ys, xs] / 105 + alpha[ys, xs], 0.1, 1)
    half = np.clip(2.2 + np.sqrt(magnitude[ys, xs]) * 0.075, 2.3, 7.2)
    stroke_alpha = np.clip(0.10 + strength * 0.40, 0.10, 0.50)
    return np.stack([
        xs + x0, ys + y0, tx / norm, ty / norm, half,
        colors[:, 0], colors[:, 1], colors[:, 2], stroke_alpha,
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
    colors = np.clip(frame[ys, xs].astype(np.float32)[:, ::-1] * 1.16 + 22, 0, 255)
    half = np.clip(1.8 + np.sqrt(score[ys, xs]) * 0.16, 1.8, 6.2)
    alpha = np.clip(0.38 + score[ys, xs] / 170, 0.38, 0.9)
    return np.stack([
        xs, ys, -gy / norm, gx / norm, half,
        colors[:, 0], colors[:, 1], colors[:, 2], alpha,
    ], axis=1).astype(np.float32)


def quantize_wash(index: int, rgb: np.ndarray, alpha: float) -> bytes:
    return struct.pack(
        "<IBBBB", index,
        round(float(np.clip(rgb[0], 0, 255))), round(float(np.clip(rgb[1], 0, 255))),
        round(float(np.clip(rgb[2], 0, 255))), opacity_byte(alpha),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("assets/base.jpg"))
    parser.add_argument("--output", type=Path, default=Path("assets/temporal_runtime.bin"))
    parser.add_argument("--manifest", type=Path, default=Path("assets/temporal_runtime.manifest.json"))
    args = parser.parse_args()
    frames = np.fromfile(args.frames, dtype=np.uint8).reshape(FRAME_COUNT, HEIGHT, WIDTH, 3)
    base = cv2.imread(str(args.base))
    if base is None:
        raise SystemExit(f"Missing fixed base: {args.base}")
    median_gray = cv2.cvtColor(np.median(frames, axis=0).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_stack = np.stack([
        cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), 0.8)
        for frame in frames
    ])
    activity = frame_activity(gray_stack)
    coverage, motion_evidence = motion_maps(gray_stack)
    rng = np.random.default_rng(240518)
    entries = FRAME_COUNT * len(REGION_NAMES)
    stroke_counts = [0] * entries
    stroke_offsets = [0] * entries
    wash_counts = [0] * entries
    wash_offsets = [0] * entries
    payloads: list[bytearray] = []
    for frame_index, frame in enumerate(frames):
        payload = bytearray()
        petal_score, petal_wash = petal_transients(frame, median_gray)
        frame_cells = cv2.resize(frame, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32)
        for region_index, name in enumerate(REGION_NAMES):
            entry = frame_index * len(REGION_NAMES) + region_index
            box = REGIONS[name]
            if name == "petals":
                records = sample_petal_strokes(frame, petal_score, CONTOUR_LIMITS[name], rng)
                washes = [
                    quantize_wash(
                        int(gy * GRID_W + gx), frame_cells[gy, gx],
                        float(min(petal_wash[min(gy * CELL, HEIGHT - 1), min(gx * CELL, WIDTH - 1)] * 0.65, 0.62)),
                    )
                    for gy, gx in np.argwhere(
                        cv2.resize(petal_wash, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA) > 0.15
                    )
                ]
            else:
                alpha = moving_alpha(box, membership_map(name, box), coverage, activity, motion_evidence)
                records = contour_instructions(frame, base, alpha, box, CONTOUR_LIMITS[name], rng)
                washes = brush_instructions(alpha, box, BRUSH_LIMITS[name], frame_cells)
            stroke_counts[entry] = len(records)
            stroke_offsets[entry] = DATA_OFFSET + sum(len(item) for item in payloads) + len(payload)
            for record in records:
                payload.extend(quantize_stroke(record, name == "petals"))
            wash_counts[entry] = len(washes)
            wash_offsets[entry] = DATA_OFFSET + sum(len(item) for item in payloads) + len(payload)
            for wash in washes:
                payload.extend(wash)
        payloads.append(payload)
    stroke_table = b"".join(struct.pack("<HQ", count, offset) for count, offset in zip(stroke_counts, stroke_offsets))
    wash_table = b"".join(struct.pack("<HQ", count, offset) for count, offset in zip(wash_counts, wash_offsets))
    header = struct.pack(
        "<8sHHHHHHHHHHHIII", MAGIC, 2, WIDTH, HEIGHT, FRAME_COUNT,
        len(REGION_NAMES), FPS, STROKE_BYTES, WASH_BYTES,
        GRID_W, GRID_H, CELL, STROKE_TABLE_OFFSET, WASH_TABLE_OFFSET, DATA_OFFSET,
    )
    binary = header + stroke_table + wash_table + b"".join(bytes(item) for item in payloads)
    args.output.write_bytes(binary)
    manifest = {
        "magic": MAGIC.decode(), "version": 2,
        "width": WIDTH, "height": HEIGHT,
        "frame_count": FRAME_COUNT, "fps": FPS,
        "regions": [{"name": name, "box": list(REGIONS[name]), "limit": BRUSH_LIMITS.get(name, CONTOUR_LIMITS[name])} for name in REGION_NAMES],
        "file": args.output.as_posix(),
        "base": args.base.as_posix(), "bytes": len(binary),
        "header_bytes": HEADER_BYTES,
        "stroke_record_bytes": STROKE_BYTES, "wash_record_bytes": WASH_BYTES,
        "stroke_table_offset": STROKE_TABLE_OFFSET,
        "wash_table_offset": WASH_TABLE_OFFSET, "data_offset": DATA_OFFSET,
        "grid": {"width": GRID_W, "height": GRID_H, "cell_px": CELL},
        "strokes": {"total": sum(stroke_counts), "mean_per_frame": round(sum(stroke_counts) / FRAME_COUNT, 1)},
        "wash_patches": {"total": sum(wash_counts), "mean_per_frame": round(sum(wash_counts) / FRAME_COUNT, 1)},
        "source": "offline BGR24 reference frames; 124 frames at 752x352 and 24 fps",
        "description": "frame-indexed full-color moving silhouette brushes, contours and petal highlights; one fixed clean static base at runtime",
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
