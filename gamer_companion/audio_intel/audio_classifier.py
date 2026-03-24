"""Audio Intelligence Engine — WASAPI capture + PANNs classification + spatial audio.

Complete audio intelligence pipeline:
1. WASAPI loopback capture (game audio output)
2. Adaptive noise gate (filter non-game audio)
3. PANNs audio event classifier (527 classes, fine-tuned for gaming)
4. Spatial audio analyzer (stereo -> 3D direction)
5. Game-specific sound matching (weapon identification)
6. Event dispatcher (callbacks for each event type)
"""

from __future__ import annotations
import numpy as np
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable
from loguru import logger

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


@dataclass
class AudioEvent:
    """A detected audio event."""
    event_type: str  # "gunshot", "footstep", "explosion", "ability", "reload"
    confidence: float
    direction: str  # "left", "right", "front", "behind"
    direction_degrees: float  # 0-360
    distance: str  # "close", "medium", "far"
    energy: float
    timestamp: float
    zone: Optional[str] = None
    weapon_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class AudioIntelligenceEngine:
    """Complete audio intelligence pipeline.

    The PANNs model classifies 527 types of sounds. We fine-tune the
    last layer on gaming audio: gunshots (by weapon), footsteps (by surface),
    abilities, reloads, explosions, voice lines, etc.

    Combined with stereo analysis for direction and amplitude for distance,
    this gives the AI "ears" that rival a pro player with $500 headphones.
    """

    SAMPLE_RATE = 44100
    CHANNELS = 2
    BLOCK_SIZE = 2048
    DTYPE = "float32"

    EVENT_CLASSES = {
        0: "gunshot_rifle", 1: "gunshot_pistol", 2: "gunshot_sniper",
        3: "gunshot_smg", 4: "gunshot_shotgun", 5: "explosion_grenade",
        6: "explosion_molotov", 7: "footstep_run", 8: "footstep_walk",
        9: "footstep_crouch", 10: "reload", 11: "weapon_switch",
        12: "ability_cast", 13: "ability_impact", 14: "defuse_start",
        15: "plant_start", 16: "flash_pop", 17: "smoke_deploy",
        18: "door_open", 19: "glass_break", 20: "voice_callout",
        21: "ambient", 22: "music", 23: "ui_sound",
    }

    def __init__(self, model_path: str = "models/panns_audio.onnx"):
        self._model_path = model_path
        self._session: Optional[ort.InferenceSession] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: Dict[str, List[Callable]] = {}
        self._event_buffer: deque = deque(maxlen=500)
        self._noise_floor: float = 0.01
        self._noise_samples: deque = deque(maxlen=100)
        self._gunshot_detector = GunShotDetector()
        self._footstep_detector = FootstepDetector()

    def initialize(self):
        """Load PANNs model if available."""
        if HAS_ONNX:
            import os
            if os.path.exists(self._model_path):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self._session = ort.InferenceSession(
                    self._model_path, providers=providers
                )
                logger.info(f"PANNs audio classifier loaded: {self._model_path}")
            else:
                logger.warning(
                    f"PANNs model not found at {self._model_path}. "
                    f"Using heuristic detection only."
                )

    def start(self):
        """Start audio capture in background thread."""
        if not HAS_SOUNDDEVICE:
            logger.warning(
                "sounddevice not installed. Audio intelligence disabled."
            )
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Audio intelligence engine started")

    def stop(self):
        """Stop audio capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def on(self, event_type: str, callback: Callable[[AudioEvent], None]):
        """Register callback for specific audio event types."""
        self._callbacks.setdefault(event_type, []).append(callback)

    def on_any(self, callback: Callable[[AudioEvent], None]):
        """Register callback for ALL audio events."""
        self._callbacks.setdefault("*", []).append(callback)

    @property
    def recent_events(self) -> List[AudioEvent]:
        return list(self._event_buffer)

    def events_since(self, seconds_ago: float) -> List[AudioEvent]:
        cutoff = time.time() - seconds_ago
        return [e for e in self._event_buffer if e.timestamp >= cutoff]

    def _capture_loop(self):
        """WASAPI loopback capture loop."""
        try:
            devices = sd.query_devices()
            loopback_idx = None
            for i, d in enumerate(devices):
                name_lower = d["name"].lower()
                if "loopback" in name_lower or "stereo mix" in name_lower:
                    if d["max_input_channels"] >= 2:
                        loopback_idx = i
                        break

            if loopback_idx is None:
                loopback_idx = sd.default.device[1]
                logger.info(
                    f"No loopback device found, using default: {loopback_idx}"
                )

            def audio_callback(indata, frames, time_info, status):
                if status:
                    logger.debug(f"Audio status: {status}")
                self._process_audio(indata.copy())

            with sd.InputStream(
                device=loopback_idx,
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                blocksize=self.BLOCK_SIZE,
                dtype=self.DTYPE,
                callback=audio_callback,
            ):
                while self._running:
                    sd.sleep(50)

        except Exception as e:
            logger.error(f"Audio capture failed: {e}")
            self._running = False

    def _process_audio(self, audio_data: np.ndarray):
        """Process an audio chunk through the full pipeline."""
        now = time.time()

        # 1. Adaptive noise gate
        energy = float(np.sqrt(np.mean(audio_data ** 2)))
        self._noise_samples.append(energy)
        self._noise_floor = float(
            np.percentile(list(self._noise_samples), 20) * 1.5
        )

        if energy < self._noise_floor:
            return

        # 2. Spatial analysis
        direction, degrees = self._analyze_spatial(audio_data)

        # 3. Distance estimation
        if energy > 0.5:
            distance = "close"
        elif energy > 0.2:
            distance = "medium"
        else:
            distance = "far"

        # 4. PANNs classification (if model loaded)
        event_type = "unknown"
        confidence = 0.5
        if self._session is not None:
            event_type, confidence = self._classify_panns(audio_data)
        else:
            # Heuristic fallback
            gunshot = self._gunshot_detector.analyze(audio_data)
            if gunshot:
                self._dispatch(AudioEvent(
                    event_type=gunshot["type"],
                    confidence=gunshot.get("confidence", 0.7),
                    direction=gunshot["direction"],
                    direction_degrees=gunshot.get("degrees", 0),
                    distance=gunshot["distance"],
                    energy=gunshot["energy"],
                    timestamp=now,
                    weapon_id=gunshot.get("weapon"),
                ))
                return

            footstep = self._footstep_detector.analyze(audio_data)
            if footstep:
                self._dispatch(AudioEvent(
                    event_type=footstep["type"],
                    confidence=footstep.get("confidence", 0.6),
                    direction=footstep["direction"],
                    direction_degrees=footstep.get("degrees", 0),
                    distance="medium",
                    energy=footstep["energy"],
                    timestamp=now,
                ))
                return

        if confidence > 0.4 and event_type not in ("ambient", "music", "ui_sound"):
            self._dispatch(AudioEvent(
                event_type=event_type, confidence=confidence,
                direction=direction, direction_degrees=degrees,
                distance=distance, energy=round(energy, 4),
                timestamp=now,
            ))

    def _analyze_spatial(self, audio_data: np.ndarray):
        """Determine direction from stereo audio."""
        if len(audio_data.shape) == 2 and audio_data.shape[1] >= 2:
            left = audio_data[:, 0]
            right = audio_data[:, 1]
        else:
            return "front", 0.0

        left_e = float(np.sqrt(np.mean(left ** 2)))
        right_e = float(np.sqrt(np.mean(right ** 2)))
        total = left_e + right_e + 1e-10
        balance = (right_e - left_e) / total

        # Cross-correlation for front/back
        correlation = np.correlate(left[:256], right[:256], mode="full")
        peak_offset = int(np.argmax(correlation)) - 256

        if balance < -0.3:
            direction = "left"
            degrees = 270 + balance * 45
        elif balance < -0.1:
            direction = "front-left"
            degrees = 315 + balance * 45
        elif balance > 0.3:
            direction = "right"
            degrees = 90 - (balance - 0.3) * 45
        elif balance > 0.1:
            direction = "front-right"
            degrees = 45 - (balance - 0.1) * 45
        else:
            if peak_offset > 2:
                direction = "behind"
                degrees = 180
            else:
                direction = "front"
                degrees = 0

        return direction, round(degrees % 360, 1)

    def _classify_panns(self, audio_data: np.ndarray) -> tuple:
        """Run PANNs model for audio classification."""
        if len(audio_data.shape) == 2:
            mono = np.mean(audio_data, axis=1)
        else:
            mono = audio_data

        target_len = self.SAMPLE_RATE
        if len(mono) < target_len:
            mono = np.pad(mono, (0, target_len - len(mono)))
        else:
            mono = mono[:target_len]

        input_data = mono.reshape(1, -1).astype(np.float32)
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: input_data})

        probs = outputs[0][0]
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        event_type = self.EVENT_CLASSES.get(class_id, f"class_{class_id}")

        return event_type, confidence

    def _dispatch(self, event: AudioEvent):
        """Dispatch audio event to registered callbacks."""
        self._event_buffer.append(event)

        for cb in self._callbacks.get(event.event_type, []):
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Audio callback error: {e}")

        for cb in self._callbacks.get("*", []):
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Audio wildcard callback error: {e}")


class GunShotDetector:
    """Detect gunshots with weapon identification via transient analysis."""

    COOLDOWN_MS = 80
    ENERGY_THRESHOLD = 0.12

    def __init__(self):
        self._last_detection = 0
        self._burst_tracker: deque = deque(maxlen=30)

    def analyze(self, audio_data: np.ndarray) -> Optional[dict]:
        now = time.time()
        if now - self._last_detection < self.COOLDOWN_MS / 1000:
            return None

        if len(audio_data.shape) == 2:
            left, right = audio_data[:, 0], audio_data[:, 1]
        else:
            left = right = audio_data

        energy = float(np.sqrt(np.mean(left ** 2 + right ** 2)))
        if energy < self.ENERGY_THRESHOLD:
            return None

        # Check transient (sharp attack = gunshot)
        diff = np.abs(np.diff(left))
        peak_transient = float(np.max(diff))
        if peak_transient < 0.2:
            return None

        # Stereo direction
        left_e = float(np.sqrt(np.mean(left ** 2)))
        right_e = float(np.sqrt(np.mean(right ** 2)))
        total = left_e + right_e + 1e-10
        balance = (right_e - left_e) / total

        if balance < -0.3:
            direction, degrees = "left", 270
        elif balance < -0.1:
            direction, degrees = "front-left", 315
        elif balance > 0.3:
            direction, degrees = "right", 90
        elif balance > 0.1:
            direction, degrees = "front-right", 45
        else:
            direction, degrees = "front", 0

        distance = "close" if energy > 0.5 else ("medium" if energy > 0.25 else "far")

        weapon = "rifle"
        if peak_transient > 0.6 and energy > 0.4:
            weapon = "sniper"
        elif peak_transient < 0.25:
            weapon = "pistol"

        self._last_detection = now
        self._burst_tracker.append(now)

        recent = [t for t in self._burst_tracker if now - t < 1.0]
        fire_rate = len(recent)

        return {
            "type": "gunshot",
            "weapon": weapon,
            "direction": direction,
            "degrees": degrees,
            "distance": distance,
            "energy": round(energy, 3),
            "confidence": min(0.95, 0.5 + peak_transient * 0.3 + energy * 0.2),
            "fire_rate": fire_rate,
            "automatic": fire_rate > 3,
        }


class FootstepDetector:
    """Detect footsteps with speed estimation."""

    ENERGY_RANGE = (0.015, 0.12)
    RHYTHM_WINDOW = 3.0

    def __init__(self):
        self._events: deque = deque(maxlen=50)
        self._last_detection = 0

    def analyze(self, audio_data: np.ndarray) -> Optional[dict]:
        now = time.time()

        if len(audio_data.shape) == 2:
            left, right = audio_data[:, 0], audio_data[:, 1]
        else:
            left = right = audio_data

        energy = float(np.sqrt(np.mean(left ** 2 + right ** 2)))

        if not (self.ENERGY_RANGE[0] <= energy <= self.ENERGY_RANGE[1]):
            return None

        # Footsteps have rhythmic low-frequency impact pattern
        diff = np.abs(np.diff(left))
        peak_transient = float(np.max(diff))

        # Footsteps: moderate transient, NOT a gunshot
        if peak_transient > 0.2:
            return None  # Too sharp — probably gunshot
        if peak_transient < 0.01:
            return None  # Too quiet — ambient

        # Stereo direction
        left_e = float(np.sqrt(np.mean(left ** 2)))
        right_e = float(np.sqrt(np.mean(right ** 2)))
        total = left_e + right_e + 1e-10
        balance = (right_e - left_e) / total

        if balance < -0.2:
            direction, degrees = "left", 270
        elif balance > 0.2:
            direction, degrees = "right", 90
        else:
            direction, degrees = "front", 0

        # Track rhythm to determine running vs walking
        self._events.append(now)
        recent = [t for t in self._events if now - t < self.RHYTHM_WINDOW]
        if len(recent) >= 3:
            intervals = [
                recent[i + 1] - recent[i]
                for i in range(len(recent) - 1)
            ]
            avg_interval = sum(intervals) / len(intervals)
            if avg_interval < 0.3:
                step_type = "footstep_run"
            elif avg_interval < 0.6:
                step_type = "footstep_walk"
            else:
                step_type = "footstep_crouch"
        else:
            step_type = "footstep_walk"

        self._last_detection = now
        return {
            "type": step_type,
            "direction": direction,
            "degrees": degrees,
            "energy": round(energy, 4),
            "confidence": min(0.8, 0.4 + energy * 3),
        }
