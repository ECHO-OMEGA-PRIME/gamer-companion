"""Auto ROI — Self-discovering region of interest detector for unknown games.

When a game profile doesn't define screen regions, AutoROI learns them
by analyzing frame variance: static pixels = UI elements, changing pixels = gameplay.
After ~30 frames it can identify health bars, minimaps, ammo counters, etc.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional, Tuple
from loguru import logger

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class AutoROI:
    """Automatically detect and track game UI regions.

    Technique: UI elements are (usually) static across frames while
    gameplay content changes. By comparing many frames:
    1. Edges/corners that NEVER move = UI elements
    2. Regions with high text density = info displays
    3. Regions in screen corners/edges = HUD elements
    4. Bars (horizontal rectangles with fill) = health/mana/XP

    After ~30 frames of gameplay, AutoROI can identify most UI regions
    without any game-specific knowledge.
    """

    def __init__(self):
        self._frame_buffer: List[np.ndarray] = []
        self._static_mask: Optional[np.ndarray] = None
        self._detected_regions: Dict[str, dict] = {}
        self._calibrated = False

    def feed_frame(self, frame: np.ndarray):
        """Feed a frame for calibration. Need ~30 frames (2 seconds at 15fps)."""
        if not HAS_CV2 or frame is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._frame_buffer.append(gray)

        if len(self._frame_buffer) >= 30 and not self._calibrated:
            self._calibrate()

    def _calibrate(self):
        """Analyze buffered frames to find static UI regions."""
        if not HAS_CV2:
            return

        frames = self._frame_buffer[-30:]

        # Compute pixel variance across frames
        stack = np.stack(frames, axis=0).astype(np.float32)
        variance = np.var(stack, axis=0)

        # Low variance = static (UI). High variance = dynamic (gameplay)
        static_mask = (variance < 5.0).astype(np.uint8) * 255

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_CLOSE, kernel)
        static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_OPEN, kernel)

        self._static_mask = static_mask

        # Find contours of static regions
        contours, _ = cv2.findContours(
            static_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h, w = frames[0].shape[:2]
        for c in contours:
            area = cv2.contourArea(c)
            if area < w * h * 0.001:  # Too small
                continue
            if area > w * h * 0.3:  # Too large (probably background)
                continue

            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bw / max(bh, 1)

            region_type = self._classify_region(x, y, bw, bh, w, h, aspect)
            if region_type:
                self._detected_regions[region_type] = {
                    "x": x, "y": y, "w": bw, "h": bh,
                    "x_pct": x / w, "y_pct": y / h,
                    "w_pct": bw / w, "h_pct": bh / h,
                }

        self._calibrated = True
        logger.info(
            f"AutoROI calibrated: found {len(self._detected_regions)} UI regions: "
            f"{list(self._detected_regions.keys())}"
        )

    def _classify_region(
        self, x: int, y: int, bw: int, bh: int,
        fw: int, fh: int, aspect: float,
    ) -> Optional[str]:
        """Classify a detected static region by position and shape."""
        cx, cy = x + bw / 2, y + bh / 2
        rel_x, rel_y = cx / fw, cy / fh

        # Bottom-left corner: usually health
        if rel_y > 0.85 and rel_x < 0.25:
            if 2 < aspect < 15:
                return "health_bar"
            return "health_region"

        # Bottom-right corner: usually ammo
        if rel_y > 0.85 and rel_x > 0.75:
            return "ammo_region"

        # Top-left corner: usually minimap/radar
        if rel_y < 0.35 and rel_x < 0.25:
            return "minimap"

        # Top-right corner: usually kill feed or score
        if rel_y < 0.15 and rel_x > 0.65:
            return "kill_feed"

        # Top-center: usually scoreboard/timer
        if rel_y < 0.1 and 0.3 < rel_x < 0.7:
            return "scoreboard"

        # Bottom-center: usually ability bar or item bar
        if rel_y > 0.85 and 0.3 < rel_x < 0.7:
            if aspect > 3:
                return "ability_bar"
            return "item_bar"

        return None

    @property
    def regions(self) -> Dict[str, dict]:
        return self._detected_regions

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated
