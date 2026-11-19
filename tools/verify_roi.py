#!/usr/bin/env python3
"""Quantify static ROI drift and dynamic ROI change from browser screenshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROI = {
    "background_arch": (615, 5, 705, 105),
    "statue_left": (58, 190, 160, 288),
    "statue_right": (565, 150, 665, 285),
    "seat": (285, 270, 470, 348),
    "petals": (0, 0, 752, 352),
    "hair": (300, 15, 445, 145),
    "chest": (330, 95, 415, 165),
    "wing_left": (55, 180, 265, 330),
    "wing_right": (485, 95, 745, 320),
}
STATIC = ("background_arch", "statue_left", "statue_right", "seat")
DYNAMIC = ("petals", "hair", "chest", "wing_left", "wing_right")
TIME_LABELS = ("0s", "1s", "2_5s", "4s", "5s")
REFERENCE_FRAMES = (0, 24, 60, 96, 120)
ALIGNED_ROI = {
    "hair": (292, 12, 452, 168),
    "chest": (324, 86, 422, 178),
    "wing_left": (30, 154, 292, 348),
    "wing_right": (462, 64, 750, 344),
}


def load_images(folder: Path, prefix: str) -> list[np.ndarray]:
    return [cv2.imread(str(folder / f"{prefix}_{label}.png")) for label in TIME_LABELS]


def estimate_shift(reference: np.ndarray, current: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float, float]:
    """Subpixel translation from masked ECC; return magnitude and x/y."""
    a = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    b = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    mask = np.ones(a.shape, np.float32) if mask is None else mask.astype(np.float32)
    if float(mask.mean()) < 0.08:
        return 0.0, 0.0, 0.0
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        correlation, warp = cv2.findTransformECC(
            a, b, warp, cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-7),
            mask, 5,
        )
        if correlation < 0.999:
            warp[:, 2] = 0
    except cv2.error:
        warp[:, 2] = 0
    dx, dy = float(warp[0, 2]), float(warp[1, 2])
    return float(np.hypot(dx, dy)), dx, dy


def temporal_change(images: list[np.ndarray], box: tuple[int, int, int, int]) -> dict[str, object]:
    x0, y0, x1, y1 = box
    crops = [image[y0:y1, x0:x1].astype(np.float32) for image in images]
    baseline = crops[0]
    mean_abs = [float(np.abs(crop - baseline).mean()) for crop in crops]
    p95_abs = [float(np.percentile(np.abs(crop - baseline), 95)) for crop in crops]
    temporal_std = float(np.stack(crops).std(axis=0).mean())
    return {
        "mean_abs_vs_t0": [round(value, 4) for value in mean_abs],
        "p95_abs_vs_t0": [round(value, 4) for value in p95_abs],
        "max_mean_abs_vs_t0": round(max(mean_abs), 4),
        "max_p95_abs_vs_t0": round(max(p95_abs), 4),
        "temporal_std_mean": round(temporal_std, 4),
    }


def measure(images: list[np.ndarray]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, box in ROI.items():
        x0, y0, x1, y1 = box
        crops = [image[y0:y1, x0:x1] for image in images]
        if name in STATIC:
            # Only low temporal-difference pixels are static anchors. This
            # excludes petals and shimmer that pass in front of stone regions.
            differences = [np.abs(crop.astype(np.float32) - crops[0].astype(np.float32)).mean(axis=2) for crop in crops]
            stable = (np.stack(differences).max(axis=0) < 1.5).astype(np.float32)
            stable = cv2.erode(stable, np.ones((3, 3), np.uint8), iterations=1)
            anchor_fraction = float(stable.mean())
            shifts = [estimate_shift(crops[0], crop, stable) for crop in crops]
            exact_equal = [
                bool(np.array_equal(crops[0][stable > 0], crop[stable > 0]))
                for crop in crops
            ]
            result[name] = {
                "type": "static",
                "drift_px": [round(item[0], 5) for item in shifts],
                "max_drift_px": round(max(item[0] for item in shifts), 5),
                "mean_drift_px": round(float(np.mean([item[0] for item in shifts])), 5),
                "dx_dy": [[round(item[1], 5), round(item[2], 5)] for item in shifts],
                "anchor_fraction": round(anchor_fraction, 4),
                "pixel_identical_to_t0": exact_equal,
                **temporal_change(images, box),
            }
        else:
            result[name] = {"type": "dynamic", **temporal_change(images, box)}
    return result


def temporal_alignment(images: list[np.ndarray], reference: list[np.ndarray]) -> dict[str, object]:
    """Mean per-frame pixel error to the reference, before and after motion."""
    result = {}
    for name, (x0, y0, x1, y1) in ALIGNED_ROI.items():
        result[name] = {
            "reference_mae": [
                round(float(np.abs(image[y0:y1, x0:x1].astype(np.float32) - frame[y0:y1, x0:x1]).mean()), 4)
                for image, frame in zip(images, reference)
            ],
            "reference_mae_mean": round(float(np.mean([
                np.abs(image[y0:y1, x0:x1].astype(np.float32) - frame[y0:y1, x0:x1]).mean()
                for image, frame in zip(images, reference)
            ])), 4),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshots", type=Path, required=True)
    parser.add_argument("--reference-frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("assets/base.jpg"))
    args = parser.parse_args()

    rendered = load_images(args.screenshots, "logical")
    reference = []
    cache = args.reference_frames
    data = np.fromfile(cache, dtype=np.uint8)
    frames = data.reshape(-1, 352, 752, 3)
    for index in REFERENCE_FRAMES:
        reference.append(frames[index])
    if any(image is None for image in rendered):
        raise SystemExit("Missing browser screenshot")
    rendered_metrics = measure(rendered)
    aligned = temporal_alignment(rendered, reference)
    reference_metrics = measure(reference)
    fixed_base = cv2.imread(str(args.base))
    if fixed_base is None:
        raise SystemExit(f"Missing fixed base: {args.base}")
    same_frame_alignment = {}
    for name, (x0, y0, x1, y1) in ALIGNED_ROI.items():
        errors = []
        baseline_errors = []
        for image, frame in zip(rendered, reference):
            crop = image[y0:y1, x0:x1].astype(np.float32)
            target = frame[y0:y1, x0:x1].astype(np.float32)
            fixed = fixed_base[y0:y1, x0:x1].astype(np.float32)
            errors.append(float(np.abs(crop - target).mean()))
            baseline_errors.append(float(np.abs(fixed - target).mean()))
        # Absolute pixel MAE is dominated by the shared fixed base, so the
        # temporal-order check compares each frame's change from its own t0.
        # A correctly aligned reconstruction is closest to the matching
        # reference change even when the underlying painting is stylized.
        rendered_change = [image.astype(np.float32) - rendered[0].astype(np.float32) for image in rendered]
        reference_change = [frame.astype(np.float32) - reference[0].astype(np.float32) for frame in reference]
        change_pairs = [
            [
                round(float(np.abs(render[y0:y1, x0:x1] - target[y0:y1, x0:x1]).mean()), 4)
                for target in reference_change
            ]
            for render in rendered_change
        ]
        correlation = float(np.mean([
            np.corrcoef(
                (image.astype(np.float32) - rendered[0].astype(np.float32))[y0:y1, x0:x1].ravel(),
                (frame.astype(np.float32) - reference[0].astype(np.float32))[y0:y1, x0:x1].ravel(),
            )[0, 1]
            for image, frame in zip(rendered[1:], reference[1:])
        ]))
        same_frame_alignment[name] = {
            "rendered_mae": [round(value, 4) for value in errors],
            "fixed_base_mae": [round(value, 4) for value in baseline_errors],
            "rendered_mae_mean": round(float(np.mean(errors)), 4),
            "fixed_base_mae_mean": round(float(np.mean(baseline_errors)), 4),
            "improvement": round(float(np.mean(baseline_errors) - np.mean(errors)), 4),
            "every_frame_improves": all(a < b for a, b in zip(errors, baseline_errors)),
            "same_frame_mae_mean": round(float(np.mean(errors)), 4),
            "mismatched_frame_mae_mean": round(float(np.mean([
                np.abs(image[y0:y1, x0:x1].astype(np.float32) - frame[y0:y1, x0:x1]).mean()
                for image_index, image in enumerate(rendered)
                for frame_index, frame in enumerate(reference)
                if image_index != frame_index
            ])), 4),
            "change_from_t0_mae": change_pairs,
            "change_from_t0_correlation_mean": round(correlation, 4),
            "best_matching_reference_frame_absolute_mae": [
                int(np.argmin([
                    float(np.abs(image[y0:y1, x0:x1].astype(np.float32) - candidate[y0:y1, x0:x1]).mean())
                    for candidate in reference
                ]))
                for image in rendered
            ],
            "best_matching_motion_change": [int(np.argmin(row)) for row in change_pairs],
        }
    static_max = max(rendered_metrics[name]["max_drift_px"] for name in STATIC)
    dynamic_min = min(rendered_metrics[name]["temporal_std_mean"] for name in DYNAMIC)
    checks = {
        "static_drift_below_0_2px": static_max < 0.2,
        "all_dynamic_rois_change": all(rendered_metrics[name]["max_mean_abs_vs_t0"] > 0.15 for name in DYNAMIC),
        "petals_change": rendered_metrics["petals"]["max_mean_abs_vs_t0"] > 0.15,
        "hair_change": rendered_metrics["hair"]["max_mean_abs_vs_t0"] > 0.15,
        "chest_change": rendered_metrics["chest"]["max_mean_abs_vs_t0"] > 0.15,
        "wings_change": min(
            rendered_metrics["wing_left"]["max_mean_abs_vs_t0"],
            rendered_metrics["wing_right"]["max_mean_abs_vs_t0"],
        ) > 0.15,
        "same_frame_mae_improves": all(
            same_frame_alignment[name]["improvement"] > 0.8
            for name in ALIGNED_ROI
        ),
        "same_frame_better_than_mismatched": all(
            metrics["same_frame_mae_mean"] < metrics["mismatched_frame_mae_mean"]
            for metrics in same_frame_alignment.values()
        ),
        "motion_change_correlation": all(
            metrics["change_from_t0_correlation_mean"] > 0.35
            for metrics in same_frame_alignment.values()
        ),
    }
    report = {
        "times_seconds": [0, 1, 2.5, 4, 5],
        "reference_frame_indices": list(REFERENCE_FRAMES),
        "roi": {name: list(box) for name, box in ROI.items()},
        "static_rois": list(STATIC),
        "dynamic_rois": list(DYNAMIC),
        "rendered": rendered_metrics,
        "reference": reference_metrics,
        "temporal_alignment": aligned,
        "same_frame_alignment": same_frame_alignment,
        "summary": {
            "rendered_static_max_drift_px": static_max,
            "rendered_dynamic_min_temporal_std": dynamic_min,
            "reference_static_max_drift_px": max(reference_metrics[name]["max_drift_px"] for name in STATIC),
            "aligned_region_reference_mae_mean": {
                name: aligned[name]["reference_mae_mean"] for name in ALIGNED_ROI
            },
        },
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
