"""Perception Loop — Continuous hybrid perception pipeline.

Fuses multiple perception sources into a unified game state:
- Screen capture (primary visual)
- Audio classification (footsteps, gunshots, abilities)
- OCR text extraction (health, ammo, score, timer)
- Object detection (enemies, items, UI elements)
- Minimap analysis (positions, zones)

Runs at configurable FPS, outputs GameState for the cognition engine.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
from loguru import logger


@dataclass
class PerceptionSource:
    """A registered perception data source."""
    name: str
    source_type: str      # "visual", "audio", "ocr", "detection", "minimap"
    callback: Optional[Callable] = None
    enabled: bool = True
    priority: int = 0     # Lower = processed first
    last_result: Any = None
    last_run_ms: float = 0
    error_count: int = 0
    max_errors: int = 10  # Disable after this many consecutive errors


@dataclass
class PerceptionFrame:
    """A single frame of perception data from all sources."""
    frame_id: int = 0
    timestamp: float = field(default_factory=time.time)
    visual: Dict[str, Any] = field(default_factory=dict)
    audio: List[dict] = field(default_factory=list)
    ocr: Dict[str, str] = field(default_factory=dict)
    detections: List[dict] = field(default_factory=list)
    minimap: Dict[str, Any] = field(default_factory=dict)
    game_phase: str = "unknown"
    processing_ms: float = 0
    source_count: int = 0


class PerceptionLoop:
    """Continuous perception pipeline that fuses multiple data sources.

    Architecture:
        Sources register callbacks that return typed data.
        Each tick, all enabled sources are polled in priority order.
        Results are fused into a single PerceptionFrame.
        The cognition engine consumes frames from the output buffer.

    Graceful degradation: if a source fails repeatedly, it auto-disables
    rather than blocking the entire pipeline.
    """

    def __init__(self, target_fps: float = 30.0):
        self._sources: Dict[str, PerceptionSource] = {}
        self._target_fps = target_fps
        self._frame_counter = 0
        self._history: List[PerceptionFrame] = []
        self._max_history = 60  # ~2 seconds at 30fps
        self._last_frame: Optional[PerceptionFrame] = None
        self._running = False

        # Fusion weights per source type
        self._weights: Dict[str, float] = {
            "visual": 1.0,
            "audio": 0.8,
            "ocr": 0.9,
            "detection": 1.0,
            "minimap": 0.7,
        }

    def register_source(
        self,
        name: str,
        source_type: str,
        callback: Callable,
        priority: int = 0,
    ) -> bool:
        """Register a perception data source."""
        if source_type not in self._weights:
            logger.warning(f"Unknown source type: {source_type}")
            return False

        self._sources[name] = PerceptionSource(
            name=name,
            source_type=source_type,
            callback=callback,
            priority=priority,
        )
        return True

    def unregister_source(self, name: str) -> bool:
        if name in self._sources:
            del self._sources[name]
            return True
        return False

    def enable_source(self, name: str) -> bool:
        src = self._sources.get(name)
        if src:
            src.enabled = True
            src.error_count = 0
            return True
        return False

    def disable_source(self, name: str) -> bool:
        src = self._sources.get(name)
        if src:
            src.enabled = False
            return True
        return False

    def tick(self) -> PerceptionFrame:
        """Run one perception cycle across all sources.

        Returns a fused PerceptionFrame combining all source data.
        """
        start = time.perf_counter()
        self._frame_counter += 1

        frame = PerceptionFrame(
            frame_id=self._frame_counter,
            timestamp=time.time(),
        )

        # Poll sources in priority order
        active_sources = sorted(
            [s for s in self._sources.values() if s.enabled],
            key=lambda s: s.priority,
        )

        for source in active_sources:
            if not source.callback:
                continue

            src_start = time.perf_counter()
            try:
                result = source.callback()
                source.last_result = result
                source.last_run_ms = (time.perf_counter() - src_start) * 1000
                source.error_count = 0

                # Fuse result into frame by type
                self._fuse(frame, source.source_type, result)
                frame.source_count += 1

            except Exception as e:
                source.error_count += 1
                source.last_run_ms = (time.perf_counter() - src_start) * 1000
                if source.error_count >= source.max_errors:
                    source.enabled = False
                    logger.warning(
                        f"Source '{source.name}' disabled after "
                        f"{source.error_count} errors: {e}"
                    )

        frame.processing_ms = round((time.perf_counter() - start) * 1000, 2)

        # Buffer management
        self._last_frame = frame
        self._history.append(frame)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return frame

    def _fuse(self, frame: PerceptionFrame, source_type: str, result: Any):
        """Fuse source data into the perception frame."""
        if result is None:
            return

        if source_type == "visual":
            if isinstance(result, dict):
                frame.visual.update(result)
        elif source_type == "audio":
            if isinstance(result, list):
                frame.audio.extend(result)
            elif isinstance(result, dict):
                frame.audio.append(result)
        elif source_type == "ocr":
            if isinstance(result, dict):
                frame.ocr.update(result)
        elif source_type == "detection":
            if isinstance(result, list):
                frame.detections.extend(result)
        elif source_type == "minimap":
            if isinstance(result, dict):
                frame.minimap.update(result)

    def get_recent_audio(self, seconds: float = 2.0) -> List[dict]:
        """Get audio events from recent frames."""
        cutoff = time.time() - seconds
        events = []
        for frame in reversed(self._history):
            if frame.timestamp < cutoff:
                break
            events.extend(frame.audio)
        return events

    def get_detection_trend(self, frames: int = 10) -> Dict[str, int]:
        """Count detection types over recent frames."""
        counts: Dict[str, int] = {}
        for frame in self._history[-frames:]:
            for det in frame.detections:
                label = det.get("label", "unknown")
                counts[label] = counts.get(label, 0) + 1
        return counts

    @property
    def last_frame(self) -> Optional[PerceptionFrame]:
        return self._last_frame

    @property
    def fps(self) -> float:
        if len(self._history) < 2:
            return 0
        dt = self._history[-1].timestamp - self._history[0].timestamp
        if dt <= 0:
            return 0
        return round(len(self._history) / dt, 1)

    def get_stats(self) -> dict:
        source_stats = {}
        for name, src in self._sources.items():
            source_stats[name] = {
                "type": src.source_type,
                "enabled": src.enabled,
                "last_run_ms": round(src.last_run_ms, 2),
                "errors": src.error_count,
            }
        return {
            "sources": source_stats,
            "total_frames": self._frame_counter,
            "buffered_frames": len(self._history),
            "measured_fps": self.fps,
            "target_fps": self._target_fps,
        }
