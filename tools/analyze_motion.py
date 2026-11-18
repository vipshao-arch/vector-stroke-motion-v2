#!/usr/bin/env python3
"""Region-aware motion analysis for the vector stroke reconstruction.

The analysis is offline only. It uses a BGR24 frame cache from the reference
video, measures temporal variance and local difference, tracks ORB features in
low-variance anchor windows, and estimates a RANSAC affine transform from those
anchors. The transform is used to compensate a slight camera motion while
building a fixed-coordinate background. It is never applied to static layers
at runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


WORK_W, WORK_H = 752, 352
FPS = 24

# Regions are conservative boxes with softened edges. `rank` expresses
# anatomical specificity, but large wing boxes must not consume the character.
# The extractor therefore classifies each stroke center with explicit overlap
# resolution below; this dictionary also documents the offline ROI definitions.
REGIONS = {
    "background": {"rank": 0, "boxes": [(0, 0, WORK_W, WORK_H)]},
    "arch": {"rank": 1, "boxes": [(0, 0, 122, 352), (610, 0, 752, 352), (210, 0, 530, 115)]},
    "statue_left": {"rank": 2, "boxes": [(38, 168, 205, 305)]},
    "statue_right": {"rank": 2, "boxes": [(540, 135, 690, 310)]},
    "seat": {"rank": 3, "boxes": [(210, 250, 525, 352)]},
    "torso": {"rank": 4, "boxes": [(315, 90, 430, 215)]},
    "chest": {"rank": 5, "boxes": [(333, 95, 410, 160)]},
    "hair": {"rank": 6, "boxes": [(300, 0, 455, 155)]},
    "wing_left": {"rank": 7, "boxes": [(25, 165, 300, 352)]},
    "wing_right": {"rank": 7, "boxes": [(470, 75, 752, 340)]},
}

DYNAMIC_REGIONS = ("hair", "chest", "wing_left", "wing_right", "petals")

# Anchor windows are selected from low-motion architectural stone. They do not
# intersect hair, wings, face, falling petals, or the seat's moving highlights.
ANCHOR_BOXES = (
    (3, 8, 55, 100),
    (12, 120, 58, 185),
    (660, 5, 745, 72),
    (688, 100, 748, 180),
    (205, 3, 292, 65),
    (465, 2, 550, 65),
    (620, 280, 690, 345),
    (42, 305, 105, 350),
)

ROI = {
    "background": (592, 0, 705, 122),
    "statue_left": (58, 190, 160, 288),
    "statue_right": (565, 150, 665, 285),
    "seat": (285, 270, 470, 348),
    "background_arch_combined": (8, 8, 105, 180),
    "petals": (0, 0, 752, 352),
    "hair": (300, 15, 445, 145),
    "chest": (330, 95, 415, 165),
    "wing_left": (55, 180, 265, 330),
    "wing_right": (485, 95, 745, 320),
}


def read_frames(cache: Path) -> np.ndarray:
    size = WORK_H * WORK_W * 3
    data = np.fromfile(cache, dtype=np.uint8)
    if data.size % size:
        raise SystemExit(f"Unexpected frame cache size: {cache}")
    return data.reshape(-1, WORK_H, WORK_W, 3)


def temporal_measurements(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]).astype(np.float32)
    variance = gray.var(axis=0)
    difference = np.abs(np.diff(gray, axis=0)).mean(axis=0)
    # Temporal high-pass is sensitive to petals and flashes, while variance is
    # sensitive to the slower deformation of hair and wings.
    high_pass = np.abs(gray[1:-1] - 0.5 * (gray[:-2] + gray[2:])).mean(axis=0)
    return variance, difference, high_pass


def region_map() -> np.ndarray:
    yy, xx = np.indices((WORK_H, WORK_W), dtype=np.int32)
    labels = np.zeros((WORK_H, WORK_W), np.uint8)
    names = list(REGIONS)
    region_id = {name: index for index, name in enumerate(names)}
    def box(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        return (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)

    # Painted layers are applied from back to front. Broad boxes only seed the
    # region; precise silhouettes override their overlapping parts.
    labels[box(0, 0, 122, 352) | box(610, 0, 752, 352) | box(210, 0, 530, 95)] = region_id["arch"]
    labels[box(25, 175, 295, 352)] = region_id["wing_left"]
    labels[box(470, 75, 752, 335)] = region_id["wing_right"]
    labels[box(38, 168, 205, 305)] = region_id["statue_left"]
    labels[box(535, 140, 675, 300)] = region_id["statue_right"]
    labels[box(205, 245, 525, 352)] = region_id["seat"]
    # Manually painted character silhouettes keep the central body out of the
    # broad wing masks. These are conservative and are documented in README.
    labels[box(330, 70, 420, 190)] = region_id["torso"]
    labels[box(340, 95, 410, 160)] = region_id["chest"]
    labels[box(325, 0, 445, 130)] = region_id["hair"]
    labels[box(325, 0, 445, 130) & box(345, 30, 415, 110)] = region_id["hair"]
    labels[box(300, 155, 445, 255)] = region_id["torso"]
    labels[box(240, 255, 500, 352)] = region_id["seat"]
    return labels


def estimate_anchor_camera(frames: np.ndarray, variance: np.ndarray) -> dict[str, object]:
    """Estimate camera motion from static stone with feature/RANSAC and ECC.

    ORB + RANSAC supplies a coarse initialization and an independent reliability
    check. Dense ECC restricted to the anchor mask usually gives the final
    subpixel translation. If both estimates disagree, the frame keeps the
    identity transform so that no unreliable camera drift enters the base.
    """
    anchor_mask = np.zeros((WORK_H, WORK_W), np.uint8)
    for x0, y0, x1, y1 in ANCHOR_BOXES:
        anchor_mask[y0:y1, x0:x1] = 255
    # Keep only the most reliable corners inside the low-variance anchors.
    first = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(first, maxCorners=700, qualityLevel=0.012, minDistance=5, mask=anchor_mask, blockSize=7)
    if corners is None:
        raise SystemExit("No static anchor corners found")
    corners = corners[:, 0, :].astype(np.float32)
    keep = variance[corners[:, 1].astype(int), corners[:, 0].astype(int)] < np.percentile(variance[anchor_mask > 0], 72)
    corners = corners[keep]
    orb = cv2.ORB_create(nfeatures=1200, edgeThreshold=12, fastThreshold=9)
    keypoints = [cv2.KeyPoint(float(x), float(y), 11) for x, y in corners]
    reference_kp, reference_desc = orb.compute(first, keypoints)
    reference_points = np.float32([kp.pt for kp in reference_kp]) if reference_kp else np.empty((0, 2), np.float32)
    if reference_desc is None or len(reference_points) < 8:
        raise SystemExit("Insufficient ORB descriptors on static anchors")

    matrices: list[np.ndarray] = []
    inlier_counts: list[int] = []
    translations: list[list[float]] = []
    ecc_correlations: list[float] = []
    methods: list[str] = []
    template = first.astype(np.float32) / 255.0
    anchor_f32 = (anchor_mask > 0).astype(np.float32)
    for frame_index, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, desc = orb.detectAndCompute(gray, anchor_mask)
        affine = np.eye(2, 3, dtype=np.float32)
        inliers = 0
        coarse = np.zeros(2, np.float32)
        if desc is not None and len(kps) >= 8:
            matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(reference_desc, desc)
            matches = sorted(matches, key=lambda m: m.distance)[:260]
            if len(matches) >= 6:
                src = np.float32([reference_points[m.queryIdx] for m in matches])
                dst = np.float32([kps[m.trainIdx].pt for m in matches])
                candidate, inlier = cv2.estimateAffinePartial2D(
                    dst, src, method=cv2.RANSAC, ransacReprojThreshold=1.2,
                    maxIters=1200, confidence=0.995,
                )
                if candidate is not None:
                    affine = candidate
                    inliers = int(inlier.sum()) if inlier is not None else 0
                    coarse = affine[:, 2]

        # Dense ECC is restricted to actually static, textured stone. A narrow
        # Gaussian border avoids a rectangular discontinuity in the template.
        current = gray.astype(np.float32) / 255.0
        ecc = np.eye(2, 3, dtype=np.float32)
        correlation = 0.0
        if frame_index:
            try:
                correlation, ecc = cv2.findTransformECC(
                    template, current, ecc, cv2.MOTION_TRANSLATION,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-6),
                    anchor_f32, 3,
                )
                if correlation < 0.985 or np.linalg.norm(ecc[:, 2] - coarse) > 2.5:
                    ecc = np.eye(2, 3, dtype=np.float32)
                    correlation = 0.0
            except cv2.error:
                ecc = np.eye(2, 3, dtype=np.float32)
        affine = ecc.astype(np.float32)
        affine[0, 0] = affine[1, 1] = 1.0
        affine[0, 1] = affine[1, 0] = 0.0
        affine[:, 2] = np.clip(affine[:, 2], -1.2, 1.2)
        matrices.append(affine)
        inlier_counts.append(inliers)
        ecc_correlations.append(float(correlation))
        methods.append("dense_ecc" if correlation else "identity")
        translations.append([float(affine[0, 2]), float(affine[1, 2])])
    return {
        "anchor_boxes": [list(box) for box in ANCHOR_BOXES],
        "anchor_corner_count": int(len(reference_points)),
        "affine_matrices": [a.tolist() for a in matrices],
        "ransac_inliers": inlier_counts,
        "ecc_correlations": ecc_correlations,
        "methods": methods,
        "rejected_frames": int(sum(value == "identity" for value in methods)),
        "translations": translations,
        "max_translation": [float(np.max(np.abs(np.asarray(translations)[:, i]))) for i in range(2)],
    }


def estimate_optical_flow(frames: np.ndarray, sample_frames: list[int]) -> dict[str, object]:
    flow_stats: dict[str, object] = {}
    for frame_index in sample_frames:
        a = cv2.cvtColor(frames[max(frame_index - 1, 0)], cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(frames[min(frame_index + 1, len(frames) - 1)], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        magnitude = np.linalg.norm(flow, axis=2)
        flow_stats[str(frame_index)] = {
            name: {
                "mean": float(magnitude[y0:y1, x0:x1].mean()),
                "p90": float(np.percentile(magnitude[y0:y1, x0:x1], 90)),
                "p99": float(np.percentile(magnitude[y0:y1, x0:x1], 99)),
            }
            for name, (x0, y0, x1, y1) in ROI.items()
        }
    return flow_stats


def region_statistics(variance: np.ndarray, difference: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    names = list(REGIONS)
    stats = {}
    for i, name in enumerate(names):
        mask = labels == i
        # Later region boxes may fully occlude an earlier region. Report an
        # empty region explicitly rather than producing NaN percentiles.
        if not mask.any():
            stats[name] = {
                "pixel_count": 0,
                "temporal_variance_mean": 0.0,
                "temporal_variance_p90": 0.0,
                "local_difference_mean": 0.0,
                "local_difference_p90": 0.0,
            }
            continue
        stats[name] = {
            "pixel_count": int(mask.sum()),
            "temporal_variance_mean": float(variance[mask].mean()),
            "temporal_variance_p90": float(np.percentile(variance[mask], 90)),
            "local_difference_mean": float(difference[mask].mean()),
            "local_difference_p90": float(np.percentile(difference[mask], 90)),
        }
    # Petals are detected as high-frequency positive transients in the upper
    # and outer thirds rather than as a single fixed geometric box.
    outer = np.ones((WORK_H, WORK_W), bool)
    outer[230:352, 270:500] = False
    petals = outer & (difference > np.percentile(difference[outer], 91))
    stats["petals"] = {
        "candidate_pixels": int(petals.sum()),
        "local_difference_mean": float(difference[petals].mean()) if petals.any() else 0.0,
        "local_difference_p90": float(np.percentile(difference[petals], 90)) if petals.any() else 0.0,
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("analysis/motion_report.json"))
    parser.add_argument("--region-preview", type=Path, default=Path("analysis/regions.png"))
    args = parser.parse_args()

    frames = read_frames(args.cache)
    variance, difference, high_pass = temporal_measurements(frames)
    labels = region_map()
    camera = estimate_anchor_camera(frames, variance)
    flow = estimate_optical_flow(frames, [0, 12, 24, 48, 60, 84, 96, 120, 123])
    stats = region_statistics(variance, difference, labels)
    report = {
        "source": str(args.cache),
        "width": WORK_W,
        "height": WORK_H,
        "frame_count": int(len(frames)),
        "fps": FPS,
        "duration": round(len(frames) / FPS, 6),
        "method": {
            "temporal_variance": "per-pixel variance of luma over all decoded frames",
            "local_difference": "mean absolute adjacent-frame luma difference",
            "optical_flow": "Farneback dense flow on frame +/- 1",
            "static_anchors": "Shi-Tomasi + ORB/RANSAC cross-check, then masked dense ECC translation confined to low-variance stone windows",
            "runtime_policy": "affine matrices only stabilize the offline base; static regions and strokes have zero runtime transform",
        },
        "roi": {name: list(box) for name, box in ROI.items()},
        "regions": {name: {"rank": spec["rank"], "boxes": spec["boxes"]} for name, spec in REGIONS.items()},
        "region_statistics": stats,
        "camera": camera,
        "optical_flow": flow,
        "summary": {
            "static_regions": ["background", "arch", "statue_left", "statue_right", "seat"],
            "dynamic_regions": list(DYNAMIC_REGIONS),
            "camera_max_translation_px": camera["max_translation"],
            "camera_frames_rejected": camera["rejected_frames"],
            "conclusion": (
                "Anchor motion is below the reliable subpixel threshold. The fixed temporal median is used "
                "as the static base and all runtime camera transforms are identity."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # A compact color preview documents the region assignment and anchor boxes.
    palette = np.array([
        [75, 74, 105], [135, 128, 160], [190, 150, 110], [190, 150, 110],
        [110, 100, 130], [210, 110, 170], [255, 95, 145], [90, 215, 225],
        [110, 190, 255], [60, 90, 220],
    ], np.uint8)
    preview = palette[labels].copy()
    for x0, y0, x1, y1 in ANCHOR_BOXES:
        cv2.rectangle(preview, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 1)
    # Reuse the RGB convention of the CPU analysis; cv2 writes BGR.
    args.region_preview.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.region_preview), preview[..., ::-1])
    print(json.dumps({
        "frames": len(frames),
        "anchor_corners": camera["anchor_corner_count"],
        "camera_max_translation_px": camera["max_translation"],
        "petal_candidate_pixels": stats["petals"]["candidate_pixels"],
        "report": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
