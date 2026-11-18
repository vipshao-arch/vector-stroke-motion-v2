#!/usr/bin/env python3
"""Build a VSTK0004 region-aware procedural drawing program.

Reference video pixels are consumed only by this offline script. The web player
loads a fixed low-frequency base plus binary drawing instructions and renders
all primitives in Three.js/WebGL. There is no video element and no per-frame
video texture in the released page.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


MAGIC = b"VSTK0004"
VERSION = 4
K = 31
WORK_W, WORK_H = 752, 352
FPS = 24
MAX_STROKES = 65535
REGIONS = [
    "background", "arch", "statue_left", "statue_right", "seat",
    "torso", "chest", "hair", "wing_left", "wing_right", "petals",
]
MODELS = [
    "static_lock", "breathing", "hair_wave", "wing_float_left",
    "wing_float_right", "petal_fall", "shimmer",
]
KINDS = ["ellipse", "line", "quadratic_bezier", "segment_stroke", "petal", "soft_brush"]
CHEST_CENTER = np.array([373.0, 132.0], np.float32)
WING_ANCHORS = {
    "wing_left": np.array([176.0, 248.0], np.float32),
    "wing_right": np.array([570.0, 165.0], np.float32),
}


def load_region_map(analysis_script: Path) -> np.ndarray:
    spec = importlib.util.spec_from_file_location("analyze_motion", analysis_script)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load {analysis_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.region_map()


def read_frames(cache: Path) -> np.ndarray:
    size = WORK_H * WORK_W * 3
    data = np.fromfile(cache, dtype=np.uint8)
    if data.size % size:
        raise SystemExit(f"Unexpected frame cache size: {cache}")
    return data.reshape(-1, WORK_H, WORK_W, 3)


def fixed_base(frames: np.ndarray, output: Path) -> tuple[np.ndarray, dict[str, object]]:
    """Temporal median in fixed coordinates, then edge-preserving smoothing."""
    painting = np.median(frames, axis=0).astype(np.uint8)
    low = cv2.bilateralFilter(painting, 5, 24, 7)
    # Remove moving bright particles from the static base by taking a darker
    # percentile in sparse high-variance outlier pixels. The particles are
    # generated as explicit runtime petals instead.
    gray_stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    transient = np.percentile(gray_stack, 97, axis=0) - np.percentile(gray_stack, 35, axis=0)
    mask = cv2.GaussianBlur((transient > 58).astype(np.float32), (0, 0), 0.8)
    darker = np.percentile(frames, 28, axis=0).astype(np.uint8)
    low = (low * (1 - mask[..., None]) + darker * mask[..., None]).astype(np.uint8)
    cv2.imwrite(str(output), low, [cv2.IMWRITE_JPEG_QUALITY, 93, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    return low[..., ::-1].astype(np.float32), {
        "path": "assets/base.jpg",
        "mime": "image/jpeg",
        "bytes": output.stat().st_size,
        "description": "固定坐标的时域中值低频基底；运行时不位移、不缩放、不旋转",
    }


def ellipse_geometry(painting: np.ndarray, target: int, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Color/luma-aware grid superpixels fitted as oriented ellipses."""
    rgb = painting[..., ::-1].astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    cell = 3
    gh, gw = WORK_H // cell, WORK_W // cell
    blocks = lab[: gh * cell, : gw * cell].reshape(gh, cell, gw, cell, 3)
    means = blocks.mean(axis=(1, 3))
    edge_blocks = edge[: gh * cell, : gw * cell].reshape(gh, cell, gw, cell).mean(axis=(1, 3))
    # A deterministic offset grid yields compact, evenly distributed patches.
    yy, xx = np.indices((gh, gw), dtype=np.float32)
    rng = np.random.default_rng(2401)
    jitter = rng.uniform(-0.65, 0.65, (gh, gw, 2)).astype(np.float32)
    cx = xx * cell + cell * 0.5 + jitter[..., 0]
    cy = yy * cell + cell * 0.5 + jitter[..., 1]
    rx = np.full((gh, gw), cell * 1.15, np.float32)
    ry = np.full((gh, gw), cell * 0.82, np.float32)
    ry = np.where(edge_blocks > 35, cell * 1.05, ry)
    angle = 0.5 * np.arctan2(gy[::cell, ::cell][:gh, :gw], gx[::cell, ::cell][:gh, :gw] + 1e-5)
    flat = np.stack([cx, cy, rx, ry, angle], axis=-1).reshape(-1, 5)
    colors = means.reshape(-1, 3)
    # Use RGB mean colors, with a slight darkening so translucent overdraw
    # reconstructs the underlying painting.
    colors_rgb = cv2.cvtColor(colors.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB).reshape(-1, 3).astype(np.float32)
    colors_rgb *= 0.94
    if len(flat) > target:
        ids = np.linspace(0, len(flat) - 1, target).astype(np.int32)
        flat, colors_rgb = flat[ids], colors_rgb[ids]
    return flat.astype(np.float32), np.clip(colors_rgb, 0, 255)


def edge_strokes(painting: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rgb = painting[..., ::-1].astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 0.6)
    rng = np.random.default_rng(seed)
    cell = 6
    gh, gw = WORK_H // cell, WORK_W // cell
    blocks = magnitude[: gh * cell, : gw * cell].reshape(gh, cell, gw, cell)
    strength = blocks.max(axis=(1, 3)).ravel()
    probability = np.clip((strength - np.percentile(strength, 30)) / max(np.percentile(strength, 99), 1), 0.002, 1)
    probability /= probability.sum()
    chosen = rng.choice(len(strength), size=min(count, len(strength)), replace=False, p=probability)
    lines, beziers = [], []
    for j, block in enumerate(chosen):
        by, bx = divmod(int(block), gw)
        local = blocks[by, :, bx, :]
        iy, ix = np.unravel_index(int(np.argmax(local)), local.shape)
        x, y = bx * cell + ix, by * cell + iy
        gvx, gvy = float(gx[y, x]), float(gy[y, x])
        norm = max(math.hypot(gvx, gvy), 1e-6)
        tangent = np.array([-gvy / norm, gvx / norm], np.float32)
        normal = np.array([-tangent[1], tangent[0]], np.float32)
        length = float(rng.uniform(4.0, 13.0))
        curvature = float(rng.uniform(-0.24, 0.24)) * length
        p0 = np.array([x, y], np.float32) - tangent * length * 0.5
        p2 = np.array([x, y], np.float32) + tangent * length * 0.5
        p1 = np.array([x, y], np.float32) + normal * curvature
        width = float(np.clip(2.2 - norm / 650, 0.6, 2.1))
        ink = np.clip(rgb[y, x].astype(np.float32) * rng.uniform(0.78, 0.98), 0, 255)
        if j % 7 == 0:
            lines.append([*p0, *(p0 + p2) * 0.5, *p2, width, *ink])
        else:
            beziers.append([*p0, *p1, *p2, width, *ink])
    return np.asarray(lines, np.float32), np.asarray(beziers, np.float32)


def residual_brushes(painting: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rgb = painting[..., ::-1].astype(np.float32)
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    texture = cv2.GaussianBlur(np.abs(gray.astype(np.float32) - cv2.GaussianBlur(gray, (0, 0), 3).astype(np.float32)), (0, 0), 0.8)
    probability = (texture.ravel() + 0.25)
    probability /= probability.sum()
    count = min(count, probability.size)
    ids = rng.choice(probability.size, size=count, replace=False, p=probability)
    output = np.zeros((count, 7), np.float32)
    for j, idx in enumerate(ids):
        y, x = divmod(int(idx), WORK_W)
        radius = rng.uniform(1.2, 3.2)
        color = np.clip(rgb[y, x] * rng.uniform(0.82, 1.16), 0, 255)
        output[j] = [x, y, radius, 0, *color]
    return output


@dataclass
class Primitive:
    kind: int
    region: int
    model: int
    params: np.ndarray  # 12 floats
    color: np.ndarray  # 3 floats RGB
    alpha: float
    motion: np.ndarray  # 8 floats
    keyframes: np.ndarray  # K x 8


def classify(center: np.ndarray, region_map: np.ndarray, model_hint: str | None = None) -> tuple[int, str]:
    x = int(np.clip(center[0], 0, WORK_W - 1))
    y = int(np.clip(center[1], 0, WORK_H - 1))
    if model_hint == "petals":
        return REGIONS.index("petals"), "petal_fall"
    if model_hint == "hair":
        return REGIONS.index("hair"), "hair_wave"
    if model_hint in ("wing_left", "wing_right"):
        return REGIONS.index(model_hint), f"wing_float_{model_hint.split('_')[1]}"
    if model_hint == "shimmer":
        return REGIONS.index("petals"), "shimmer"
    name = REGIONS[int(region_map[y, x])]
    if name in ("background", "arch", "statue_left", "statue_right", "seat"):
        return REGIONS.index(name), "static_lock"
    if name == "chest":
        return REGIONS.index(name), "breathing"
    if name == "hair":
        return REGIONS.index(name), "hair_wave"
    if name == "wing_left":
        return REGIONS.index(name), "wing_float_left"
    if name == "wing_right":
        return REGIONS.index(name), "wing_float_right"
    if name == "torso":
        return REGIONS.index(name), "breathing"
    return REGIONS.index("background"), "static_lock"


def motion_parameters(model: str, center: np.ndarray, rng: np.random.Generator, index: int) -> np.ndarray:
    """Return anchor, phase, speed and per-stroke generator parameters."""
    out = np.zeros(8, np.float32)
    if model == "breathing":
        out[:2] = CHEST_CENTER
        out[2] = 2 * math.pi / (2.7 * FPS)  # radians per frame
        out[3] = rng.uniform(-0.35, 0.35)
        out[4] = 120.0  # falloff radius
    elif model == "hair_wave":
        out[0] = rng.uniform(0, 2 * math.pi)
        out[1] = rng.uniform(0.065, 0.115)
        out[2] = rng.uniform(1.2, 4.6) * (1.0 - np.clip(center[1] / 175, 0, 0.75))
        out[3] = rng.uniform(0.018, 0.048)  # phase along individual strand
        out[4], out[5] = 346.0, 18.0  # strand root anchor
    elif model.startswith("wing_float"):
        side = "left" if model.endswith("left") else "right"
        out[:2] = WING_ANCHORS[f"wing_{side}"]
        out[2] = 2 * math.pi / ((3.25 if side == "left" else 2.85) * FPS)
        out[3] = rng.uniform(0, 2 * math.pi) + (0.45 if side == "left" else 1.6)
        out[4] = rng.uniform(1.5, 5.5)
        out[5], out[6] = -1.0 if side == "left" else 1.0, rng.uniform(0, 1)
    elif model == "petal_fall":
        out[0] = rng.uniform(0, 2 * math.pi)
        out[1] = rng.uniform(0.19, 0.55)  # px per frame
        out[2] = rng.uniform(0.12, 0.46)  # life cycles across clip
        out[3] = center[1] + rng.uniform(0, 170)  # initial vertical offset
        out[4] = rng.uniform(0.16, 0.36)  # life speed
        out[5] = rng.uniform(0.045, 0.11)  # sway angular speed
        out[6] = rng.uniform(3.0, 13.0)  # sway amplitude
        out[7] = rng.uniform(-1.3, 1.3)  # spin
    elif model == "shimmer":
        out[0], out[1] = center
        out[2] = rng.uniform(0, 2 * math.pi)
        out[3] = rng.uniform(0.08, 0.24)
    return out


def sample_stroke_keyframes(
    model: str,
    center: np.ndarray,
    motion: np.ndarray,
    sample_ids: np.ndarray,
    frame_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """31 smooth keyframes with genuine per-stroke parameters."""
    values = np.zeros((K, 8), np.float32)
    frames = sample_ids.astype(np.float32)
    u = frames / max(frame_count - 1, 1)
    values[:, 2] = 1.0
    values[:, 5] = 1.0
    values[:, 6] = u
    if model == "static_lock":
        return values
    phase = motion[0]
    speed = motion[1] if motion[1] > 0.005 else 0.08
    if model == "breathing":
        breath = np.sin(frames * speed + phase + motion[3])
        radial = center - CHEST_CENTER
        falloff = math.exp(-float(radial @ radial) / (motion[4] * motion[4]))
        values[:, 0] = radial[0] * 0.006 * breath
        values[:, 1] = radial[1] * 0.006 * breath
        values[:, 2] = 1.0 + 0.006 * breath * falloff
        values[:, 5] = 1.0 + 0.035 * breath * falloff
        values[:, 7] = breath * 0.2
    elif model == "hair_wave":
        wave = np.sin(frames * speed + phase + np.linalg.norm(center - motion[4:6]) * motion[3])
        values[:, 0] = wave * motion[2] * 0.55
        values[:, 1] = np.cos(frames * speed * 0.83 + phase) * motion[2] * 0.22
        values[:, 3] = wave * 0.025
        values[:, 5] = 1.0 + 0.018 * wave
        values[:, 7] = wave
    elif model.startswith("wing_float"):
        side_sign = -1.0 if model.endswith("left") else 1.0
        anchor = WING_ANCHORS["wing_left" if side_sign < 0 else "wing_right"]
        radial = center - anchor
        distance = np.linalg.norm(radial) / 210.0
        flap = np.sin(frames * speed + phase + distance * 1.8)
        secondary = np.sin(frames * speed * 1.93 + phase * 1.7)
        values[:, 0] = side_sign * (1.0 + distance * 2.4) * flap + secondary * 0.45
        values[:, 1] = (0.8 + distance * 3.2) * np.cos(frames * speed + phase) - 0.5
        values[:, 2] = 1.0 + 0.012 * flap * distance
        values[:, 3] = side_sign * (0.006 + 0.016 * distance) * flap
        values[:, 4] = 0.035 * flap - 0.02 * secondary
        values[:, 5] = 1.0 + 0.022 * flap
        values[:, 7] = flap
    elif model == "petal_fall":
        life = np.mod(u * motion[4] + (center[0] * 0.013 + phase * 0.16) % 1.0, 1.0)
        values[:, 0] = np.sin(frames * motion[5] + phase) * motion[6] * 0.45
        # Keyframes encode only the small residual around the shader's exact
        # linear fall. The shader owns spawn time, velocity, sway and spin.
        values[:, 1] = np.sin(frames * motion[5] * 0.7 + phase) * 1.8
        values[:, 3] = life * motion[7] + np.sin(frames * motion[5] * 0.5 + phase) * 0.3
        envelope = np.clip(life / 0.09, 0, 1) * np.clip((1 - life) / 0.28, 0, 1)
        values[:, 4] = (envelope - 0.65) * 0.4
        values[:, 5] = 1.0 + 0.06 * np.sin(frames * 0.18 + phase)
        values[:, 7] = np.sin(frames * motion[5] + phase)
    elif model == "shimmer":
        values[:, 4] = 0.18 * np.sin(frames * speed + phase)
        values[:, 5] = 1.0 + 0.12 * np.sin(frames * speed * 1.7 + phase)
        values[:, 7] = np.sin(frames * speed + phase)
    # Catmull-like temporal smoothing for packed keyframes; interpolation in
    # the vertex shader uses smoothstep between these samples.
    for field in (0, 1, 3, 4, 5, 7):
        raw = values[:, field]
        smooth = 0.5 * raw + 0.25 * (np.roll(raw, 1) + np.roll(raw, -1))
        smooth[0], smooth[-1] = raw[0], raw[-1]
        values[:, field] = np.where(np.abs(raw) > 1e-6, smooth, 0.0)
    return values


def add_primitive(
    primitives: list[Primitive],
    kind: int,
    params: np.ndarray,
    color: np.ndarray,
    alpha: float,
    center: np.ndarray,
    region_map: np.ndarray,
    sample_ids: np.ndarray,
    frame_count: int,
    rng: np.random.Generator,
    model_hint: str | None = None,
) -> None:
    region, model = classify(center, region_map, model_hint)
    motion = motion_parameters(model, center, rng, len(primitives))
    keyframes = sample_stroke_keyframes(model, center, motion, sample_ids, frame_count, rng)
    p = np.zeros(12, np.float32)
    p[:min(len(params), 12)] = params[:12]
    primitives.append(Primitive(
        kind, region, MODELS.index(model), p,
        np.asarray(color, np.float32), float(alpha), motion, keyframes,
    ))


def dynamic_detail_strokes(
    frames: np.ndarray,
    painting: np.ndarray,
    region_map: np.ndarray,
    sample_ids: np.ndarray,
    rng: np.random.Generator,
    primitives: list[Primitive],
    hair_count: int,
    wing_count: int,
    petal_count: int,
    shimmer_count: int,
) -> None:
    rgb = painting[..., ::-1].astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Explicit curved hair strands follow the two long hair silhouettes. Each
    # strand has a distinct phase from its root, so wave propagation is visible.
    for j in range(hair_count):
        side = -1.0 if j % 2 else 1.0
        root_x = 356.0 + rng.uniform(-22, 22)
        root_y = rng.uniform(18, 70)
        x = root_x + side * rng.uniform(20, 62)
        y = root_y + rng.uniform(35, 110)
        tangent = np.array([side * 0.35, 1.0], np.float32)
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]], np.float32)
        length = rng.uniform(12, 32)
        p0 = np.array([x, y], np.float32) - tangent * length * 0.5
        p2 = np.array([x, y], np.float32) + tangent * length * 0.5
        p1 = np.array([x, y], np.float32) + normal * rng.uniform(-7, 7)
        ink = np.clip(rgb[int(np.clip(y, 0, 351)), int(np.clip(x, 0, 751))].astype(np.float32) * rng.uniform(1.04, 1.35), 0, 255)
        params = [*p0, *p1, *p2, rng.uniform(0.8, 1.7), p2[0] - p0[0], p2[1] - p0[1], 0, 0]
        add_primitive(primitives, KINDS.index("segment_stroke"), params, ink, rng.uniform(0.18, 0.38), np.array([x, y]), region_map, sample_ids, len(frames), rng, "hair")

    # Wing contour strokes use quadratic controls with different left/right
    # anchors and phases; their deformation increases toward the wing tips.
    for j in range(wing_count):
        side = -1.0 if j % 2 else 1.0
        anchor = WING_ANCHORS["wing_left" if side < 0 else "wing_right"]
        center = anchor + np.array([side * rng.uniform(0, 205), rng.uniform(-65, 105)], np.float32)
        center[0] = np.clip(center[0], 3, WORK_W - 3)
        center[1] = np.clip(center[1], 3, WORK_H - 3)
        gvx, gvy = float(gx[int(center[1]), int(center[0])]), float(gy[int(center[1]), int(center[0])])
        norm = max(math.hypot(gvx, gvy), 1e-6)
        tangent = np.array([-gvy / norm, gvx / norm], np.float32)
        normal = np.array([-tangent[1], tangent[0]], np.float32)
        length = rng.uniform(8, 25)
        p0 = center - tangent * length * 0.5
        p2 = center + tangent * length * 0.5
        p1 = center + normal * rng.uniform(-5, 5)
        ink = np.clip(rgb[int(center[1]), int(center[0])].astype(np.float32) * rng.uniform(0.8, 1.2), 0, 255)
        params = [*p0, *p1, *p2, rng.uniform(0.75, 2.1), *anchor, 0, 0]
        hint = "wing_left" if side < 0 else "wing_right"
        add_primitive(primitives, KINDS.index("quadratic_bezier"), params, ink, rng.uniform(0.13, 0.34), center, region_map, sample_ids, len(frames), rng, hint)

    # Petals are real procedural primitives. Spawn positions and velocities are
    # sampled from high temporal-transient pixels, never copied video frames.
    gray_stack = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])
    transient = np.percentile(gray_stack, 96, axis=0) - np.percentile(gray_stack, 30, axis=0)
    transient[220:352, 275:500] *= 0.25
    probability = np.clip(transient.ravel() - 18, 0, None) + 0.012
    probability /= probability.sum()
    chosen = rng.choice(probability.size, size=petal_count, replace=False, p=probability)
    for idx in chosen:
        y, x = divmod(int(idx), WORK_W)
        rx, ry = rng.uniform(2.2, 6.0), rng.uniform(0.9, 2.8)
        angle = rng.uniform(-0.8, 0.8)
        ink = np.clip(rgb[y, x].astype(np.float32) * rng.uniform(1.15, 1.8) + rng.uniform(0, 30), 0, 255)
        params = [x, y, rx, ry, angle, rng.uniform(0.15, 0.55), 0, 0, 0, 0, 0, 0]
        # Force the petal model while retaining the petal region code.
        add_primitive(primitives, KINDS.index("petal"), params, ink, rng.uniform(0.3, 0.78), np.array([x, y]), region_map, sample_ids, len(frames), rng, "petals")

    # Tiny shimmer particles are soft brushes with independent flicker keys.
    for _ in range(shimmer_count):
        x, y = rng.uniform(0, WORK_W), rng.uniform(0, WORK_H)
        radius = rng.uniform(0.7, 2.1)
        color = np.array([220, 245, 255], np.float32) * rng.uniform(0.8, 1.15)
        params = [x, y, radius, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        add_primitive(primitives, KINDS.index("soft_brush"), params, color, rng.uniform(0.08, 0.3), np.array([x, y]), region_map, sample_ids, len(frames), rng, "shimmer")


def write_binary(
    output: Path,
    primitives: list[Primitive],
    meta: dict[str, object],
    camera: np.ndarray,
    manifest: dict[str, object],
) -> None:
    encoded = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((4 - len(encoded) % 4) % 4)
    with output.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<HHII", VERSION, K, len(encoded), len(primitives)))
        f.write(encoded)
        # All runtime matrices are identity. Offline anchor measurements are in
        # analysis/motion_report.json, not in the player's runtime transform.
        f.write(camera.astype(np.float32).tobytes())
        for prim in primitives:
            f.write(struct.pack("<BBBBf", prim.kind, prim.region, prim.model, 0, prim.alpha))
            f.write(prim.params.astype(np.float32).tobytes())
            f.write(prim.color.astype(np.float32).tobytes())
            f.write(prim.motion.astype(np.float32).tobytes())
            f.write(prim.keyframes.astype(np.float32).tobytes())
    (output.parent / "strokes.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, default=Path("analysis/motion_report.json"))
    parser.add_argument("--analysis-script", type=Path, default=Path("tools/analyze_motion.py"))
    parser.add_argument("--output", type=Path, default=Path("assets/strokes.bin"))
    parser.add_argument("--ellipses", type=int, default=16000)
    parser.add_argument("--edges", type=int, default=7000)
    parser.add_argument("--brushes", type=int, default=6500)
    parser.add_argument("--hair", type=int, default=1400)
    parser.add_argument("--wings", type=int, default=1800)
    parser.add_argument("--petals", type=int, default=2600)
    parser.add_argument("--shimmer", type=int, default=1000)
    args = parser.parse_args()

    frames = read_frames(args.cache)
    region_map = load_region_map(args.analysis_script)
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    base_bgr, background = fixed_base(frames, args.output.parent / "base.jpg")
    painting = base_bgr[..., ::-1].astype(np.float32)
    rng = np.random.default_rng(90124)
    sample_ids = np.linspace(0, len(frames) - 1, K).round().astype(np.int32)
    primitives: list[Primitive] = []

    geometry, colors = ellipse_geometry(painting, args.ellipses, region_map)
    for params, color in zip(geometry, colors):
        add_primitive(primitives, 0, params, color, 0.95, params[:2], region_map, sample_ids, len(frames), rng)
    lines, beziers = edge_strokes(painting, args.edges, 1301)
    for row in lines:
        add_primitive(primitives, 1, row, row[7:10], 0.5, (row[:2] + row[4:6]) * 0.5, region_map, sample_ids, len(frames), rng)
    for row in beziers:
        add_primitive(primitives, 2, row, row[7:10], 0.5, row[2:4], region_map, sample_ids, len(frames), rng)
    brushes = residual_brushes(painting, args.brushes, 7907)
    for row in brushes:
        add_primitive(primitives, 5, row, row[4:7], 0.32, row[:2], region_map, sample_ids, len(frames), rng)
    dynamic_detail_strokes(
        frames, painting, region_map, sample_ids, rng, primitives,
        args.hair, args.wings, args.petals, args.shimmer,
    )
    if len(primitives) > MAX_STROKES:
        raise SystemExit(f"Too many primitives: {len(primitives)}")

    counts = {name: int(np.sum([p.kind == i for p in primitives])) for i, name in enumerate(KINDS)}
    region_counts = {name: int(np.sum([p.region == i for p in primitives])) for i, name in enumerate(REGIONS)}
    model_counts = {name: int(np.sum([p.model == i for p in primitives])) for i, name in enumerate(MODELS)}
    identity = np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), (K, 1))
    meta = {
        "version": VERSION,
        "format": "VSTK0004",
        "width": WORK_W,
        "height": WORK_H,
        "fps": FPS,
        "frame_count": int(len(frames)),
        "duration": round(len(frames) / FPS, 6),
        "keyframe_count": K,
        "stroke_count": len(primitives),
        "counts": counts,
        "regions": REGIONS,
        "region_counts": region_counts,
        "motion_models": MODELS,
        "motion_model_counts": model_counts,
        "source": args.video.name,
        "background": background,
        "runtime_camera": "identity; static layers are locked in unified coordinates",
        "analysis": {
            "report": str(args.analysis).replace("\\", "/"),
            "camera_max_translation_px": analysis["summary"]["camera_max_translation_px"],
            "camera_frames_rejected": analysis["summary"]["camera_frames_rejected"],
            "anchor_method": analysis["method"]["static_anchors"],
        },
    }
    manifest = {
        "meta": meta,
        "camera_keyframes": identity.round(5).tolist(),
        "instruction_format": {
            "header": "magic(8), version u16, keyframes u16, json_length u32, stroke_count u32",
            "camera": "31 * 6 float32 identity affine matrices; offline anchor compensation only",
            "stroke": (
                "kind u8, region u8, motion_model u8, flags u8, alpha f32, "
                "params 12 f32, color 3 f32, motion 8 f32, keyframes 31*8 f32"
            ),
            "kinds": KINDS,
            "regions": REGIONS,
            "motion_models": MODELS,
            "motion_fields": [
                "anchor_or_phase_x", "anchor_or_phase_y", "phase_or_speed", "fall_or_phase",
                "radius_or_life_speed", "anchor_or_sway_speed", "sway_or_distance", "spin",
            ],
            "keyframe_fields": [
                "dx", "dy", "scale", "rotate", "delta_alpha", "brightness", "time", "wobble",
            ],
        },
    }
    write_binary(args.output, primitives, meta, identity, manifest)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1048576:.2f} MiB)")


if __name__ == "__main__":
    main()
