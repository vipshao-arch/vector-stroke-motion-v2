#!/usr/bin/env python3
"""Build the reference/browser contact sheet from cached frames and runtime screenshots."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TIMES = ("0s", "1s", "2_5s", "4s", "5s")
LABELS = ("0s", "1s", "2.5s", "4s", "5s")
REFERENCE_INDICES = (0, 24, 60, 96, 120)
ROOT = Path(__file__).resolve().parents[1]
FRAMES = Path(r"C:\Users\vipshao\Documents\MagicLomix\2026-11-19\1\analysis\frames_bgr24.raw")


def main() -> None:
    data = np.fromfile(FRAMES, dtype=np.uint8)
    frames = data.reshape(-1, 352, 752, 3)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    output = Image.new("RGB", (1504, 1960), (14, 14, 14))
    for row, (label, index, time_name) in enumerate(zip(LABELS, REFERENCE_INDICES, TIMES)):
        reference = Image.fromarray(cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB))
        browser = cv2.imread(str(ROOT / "verification" / "screenshots" / f"logical_{time_name}.png"))
        if browser is None:
            raise SystemExit(f"Missing browser screenshot for {time_name}")
        browser = Image.fromarray(cv2.cvtColor(browser, cv2.COLOR_BGR2RGB))
        strip = Image.new("RGB", (1504, 382), (0, 0, 0))
        strip.paste(reference, (0, 30))
        strip.paste(browser, (752, 30))
        draw = ImageDraw.Draw(strip)
        draw.text((8, 6), f"{label} reference", fill=(255, 255, 0), font=font)
        draw.text((760, 6), f"{label} browser composition", fill=(255, 255, 0), font=font)
        output.paste(strip, (0, row * 392))
    output.save(ROOT / "verification" / "comparison.png")
    print(f"Saved {ROOT / 'verification' / 'comparison.png'}")


if __name__ == "__main__":
    main()
