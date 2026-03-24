"""Hybrid Perception Pipeline — GPU + Cloud + OCR perception.

Combines:
1. Local GPU detection via YOLO-NAS/ONNX (120fps, ~8ms on RTX 4060)
2. OCR for numeric readouts (health, ammo, money)
3. Frame differencing for motion detection
4. Periodic LLM vision for strategic understanding (every 3s)
"""

from __future__ import annotations
import asyncio
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from loguru import logger

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from easyocr import Reader as EasyOCRReader
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False


@dataclass
class Detection:
    """A detected object in the game frame."""
    class_name: str          # "enemy", "ally", "weapon", "item", "projectile"
    confidence: float        # 0.0 - 1.0
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]  # (cx, cy)
    distance_est: str = "unknown"  # "near", "medium", "far"
    quadrant: str = "C"      # "TL", "T", "TR", "L", "C", "R", "BL", "B", "BR"
    metadata: dict = field(default_factory=dict)


@dataclass
class PerceptionResult:
    """Complete perception output for one frame."""
    timestamp: float
    frame_id: int

    # From YOLO/local detection
    detections: List[Detection] = field(default_factory=list)
    enemies_visible: int = 0
    allies_visible: int = 0
    crosshair_on_enemy: bool = False
    nearest_enemy: Optional[Detection] = None

    # From OCR/ROI
    health: Optional[int] = None
    armor: Optional[int] = None
    ammo_clip: Optional[int] = None
    ammo_reserve: Optional[int] = None
    money: Optional[int] = None
    round_time: Optional[float] = None

    # From minimap parser
    minimap_positions: List[dict] = field(default_factory=list)

    # From kill feed parser
    recent_kills: List[dict] = field(default_factory=list)

    # From frame differ
    motion_events: List[dict] = field(default_factory=list)

    # From LLM (when available, may be stale)
    strategic_context: Optional[str] = None
    strategic_timestamp: float = 0

    # Derived
    game_phase: str = "unknown"
    threat_level: str = "low"  # "none", "low", "medium", "high", "critical"


class LocalDetector:
    """GPU-accelerated object detection using YOLO-NAS / ONNX Runtime.

    Models are small (15-40MB), run at 120+ fps on RTX 4060.
    Falls back to CPU if no GPU available.
    """

    MODELS = {
        "general": "models/gamer_yolo_general.onnx",
        "fps": "models/gamer_yolo_fps.onnx",
        "moba": "models/gamer_yolo_moba.onnx",
        "radar_parser": "models/minimap_cnn.onnx",
    }
    CONFIDENCE_THRESHOLD = 0.45
    NMS_THRESHOLD = 0.5
    INPUT_SIZE = (640, 640)

    def __init__(self, model_id: str = "general", device: str = "cuda"):
        self._session: Optional[ort.InferenceSession] = None
        self._model_id = model_id
        self._device = device
        self._class_names: List[str] = []
        self._initialized = False

    def initialize(self):
        if not HAS_ONNX:
            logger.warning("onnxruntime not installed. Local detection disabled.")
            return
        model_path = Path(self.MODELS.get(self._model_id, self.MODELS["general"]))
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}. Will use LLM fallback only.")
            return

        providers = []
        if self._device == "cuda":
            providers.append(("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
            }))
        providers.append("CPUExecutionProvider")

        self._session = ort.InferenceSession(str(model_path), providers=providers)
        names_file = model_path.with_suffix(".names")
        if names_file.exists():
            self._class_names = names_file.read_text().strip().split("\n")
        else:
            self._class_names = ["enemy", "ally", "weapon", "item", "projectile", "ui_element"]
        self._initialized = True
        active_provider = self._session.get_providers()[0]
        logger.info(f"Local detector initialized: {model_path.name} on {active_provider}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a frame."""
        if not self._initialized or self._session is None:
            return []
        if not HAS_CV2:
            return []

        h, w = frame.shape[:2]
        input_img = cv2.resize(frame, self.INPUT_SIZE)
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = np.expand_dims(input_img, axis=0)

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: input_img})

        detections = []
        if len(outputs) > 0:
            preds = outputs[0]
            if len(preds.shape) == 3:
                preds = preds[0]
            for pred in preds:
                if len(pred) < 6:
                    continue
                conf = float(pred[4])
                if conf < self.CONFIDENCE_THRESHOLD:
                    continue
                class_scores = pred[5:]
                class_id = int(np.argmax(class_scores))
                class_conf = float(class_scores[class_id]) * conf
                if class_conf < self.CONFIDENCE_THRESHOLD:
                    continue

                cx, cy, bw, bh = pred[:4]
                x1 = int((cx - bw / 2) * w / self.INPUT_SIZE[0])
                y1 = int((cy - bh / 2) * h / self.INPUT_SIZE[1])
                x2 = int((cx + bw / 2) * w / self.INPUT_SIZE[0])
                y2 = int((cy + bh / 2) * h / self.INPUT_SIZE[1])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                class_name = (self._class_names[class_id]
                              if class_id < len(self._class_names)
                              else f"class_{class_id}")

                bbox_area = (x2 - x1) * (y2 - y1)
                frame_area = w * h
                area_ratio = bbox_area / frame_area
                if area_ratio > 0.05:
                    distance = "near"
                elif area_ratio > 0.01:
                    distance = "medium"
                else:
                    distance = "far"

                qx = "L" if center[0] < w * 0.33 else ("R" if center[0] > w * 0.67 else "")
                qy = "T" if center[1] < h * 0.33 else ("B" if center[1] > h * 0.67 else "")
                quadrant = (qy + qx) or "C"

                detections.append(Detection(
                    class_name=class_name,
                    confidence=round(class_conf, 3),
                    bbox=(x1, y1, x2, y2),
                    center=center,
                    distance_est=distance,
                    quadrant=quadrant,
                ))

        if len(detections) > 1:
            detections = self._nms(detections)
        return detections

    def _nms(self, dets: List[Detection]) -> List[Detection]:
        dets.sort(key=lambda d: d.confidence, reverse=True)
        keep = []
        for d in dets:
            overlap = False
            for k in keep:
                iou = self._iou(d.bbox, k.bbox)
                if iou > self.NMS_THRESHOLD and d.class_name == k.class_name:
                    overlap = True
                    break
            if not overlap:
                keep.append(d)
        return keep

    @staticmethod
    def _iou(box1, box2) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0


class ROIExtractor:
    """Extract and process screen regions defined in the game profile."""

    def __init__(self, profile):
        self.profile = profile
        self._ocr = None
        if HAS_EASYOCR:
            try:
                self._ocr = EasyOCRReader(["en"], gpu=True, verbose=False)
            except Exception:
                pass

    def extract(self, frame: np.ndarray) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        results = {}

        for name, region in self.profile.regions.items():
            x1 = int(region.x_pct * w)
            y1 = int(region.y_pct * h)
            x2 = int((region.x_pct + region.w_pct) * w)
            y2 = int((region.y_pct + region.h_pct) * h)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if region.ocr_enabled and self._ocr:
                text_results = self._ocr.readtext(crop, detail=0)
                text = " ".join(text_results).strip()
                numbers = []
                for c in text.replace("O", "0").replace("o", "0").split():
                    cleaned = "".join(ch for ch in c if ch.isdigit())
                    if cleaned:
                        numbers.append(int(cleaned))
                results[name] = {
                    "text": text,
                    "value": numbers[0] if numbers else None,
                    "raw_crop": crop,
                }
            else:
                results[name] = {"raw_crop": crop}

        return results


class FrameDiffer:
    """Detect changes between consecutive frames for motion detection."""

    def __init__(self, threshold: int = 25, min_area: int = 500):
        self.threshold = threshold
        self.min_area = min_area
        self._prev_gray = None

    def diff(self, frame: np.ndarray) -> List[dict]:
        if not HAS_CV2:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return []

        delta = cv2.absdiff(self._prev_gray, gray)
        thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        events = []
        h, w = frame.shape[:2]
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            cx, cy = x + bw // 2, y + bh // 2
            qx = "L" if cx < w * 0.33 else ("R" if cx > w * 0.67 else "")
            qy = "T" if cy < h * 0.33 else ("B" if cy > h * 0.67 else "")
            events.append({
                "bbox": (x, y, x + bw, y + bh),
                "center": (cx, cy),
                "area": area,
                "magnitude": area / (w * h),
                "quadrant": (qy + qx) or "C",
            })

        self._prev_gray = gray
        return events


class PerceptionPipeline:
    """Orchestrates all perception subsystems.

    Combines local GPU detection, OCR, frame differencing, and periodic
    LLM vision analysis into a unified perception result per frame.
    """

    def __init__(self, profile, vision_engine=None):
        self.profile = profile
        self._detector = LocalDetector(
            model_id="fps" if "fps" in profile.genre else "general"
        )
        self._roi = ROIExtractor(profile)
        self._differ = FrameDiffer()
        self._vision_engine = vision_engine
        self._frame_count = 0
        self._last_strategic_time = 0
        self._strategic_cache: Optional[str] = None

    def initialize(self):
        self._detector.initialize()
        logger.info(f"Perception pipeline initialized for {self.profile.display_name}")

    async def perceive(self, frame: np.ndarray) -> PerceptionResult:
        t0 = time.monotonic()
        result = PerceptionResult(timestamp=time.time(), frame_id=self._frame_count)

        # 1. Local detection (GPU)
        detections = self._detector.detect(frame)
        result.detections = detections
        result.enemies_visible = sum(1 for d in detections if d.class_name == "enemy")
        result.allies_visible = sum(1 for d in detections if d.class_name == "ally")

        enemies = [d for d in detections if d.class_name == "enemy"]
        if enemies:
            result.nearest_enemy = min(enemies, key=lambda d: d.distance_est != "near")
            screen_center = (frame.shape[1] // 2, frame.shape[0] // 2)
            for e in enemies:
                dist = ((e.center[0] - screen_center[0]) ** 2 +
                        (e.center[1] - screen_center[1]) ** 2) ** 0.5
                if dist < 50:
                    result.crosshair_on_enemy = True
                    break

        # 2. ROI extraction (OCR)
        roi_data = self._roi.extract(frame)
        for key in ("health", "armor", "ammo", "money", "round_timer"):
            mapped = {"health": "health", "armor": "armor", "ammo": "ammo_clip",
                      "money": "money", "round_timer": "round_time"}
            field_name = mapped.get(key, key)
            # Try exact key or key with _bar suffix
            for roi_key in (key, f"{key}_bar"):
                if roi_key in roi_data and roi_data[roi_key].get("value") is not None:
                    setattr(result, field_name, roi_data[roi_key]["value"])
                    break

        # 3. Frame differencing (CPU)
        result.motion_events = self._differ.diff(frame)

        # 4. Threat level
        if result.crosshair_on_enemy:
            result.threat_level = "critical"
        elif result.enemies_visible >= 3:
            result.threat_level = "high"
        elif result.enemies_visible >= 1:
            result.threat_level = "medium"
        elif result.motion_events:
            result.threat_level = "low"
        else:
            result.threat_level = "none"

        # 5. Periodic LLM strategic analysis
        now = time.time()
        if (self._vision_engine and
                now - self._last_strategic_time > self.profile.strategic_interval_s):
            self._last_strategic_time = now
            asyncio.create_task(self._update_strategic(frame))
        result.strategic_context = self._strategic_cache

        self._frame_count += 1
        if self._frame_count % 300 == 0:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.debug(
                f"Perception: {elapsed_ms:.1f}ms | enemies={result.enemies_visible} "
                f"hp={result.health} threat={result.threat_level}"
            )

        return result

    async def _update_strategic(self, frame: np.ndarray):
        try:
            result = await asyncio.to_thread(
                self._vision_engine.analyze,
                frame=frame,
                prompt="Analyze this game screenshot. What's the tactical situation? "
                       "What should the player do next?",
                mode="game",
            )
            self._strategic_cache = result.get("analysis", str(result))
        except Exception as e:
            logger.debug(f"Strategic LLM update failed: {e}")
