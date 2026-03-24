"""Win32 Transparent Overlay — Real on-screen rendering over games.

Uses Win32 layered window with GDI+ drawing. Works on top of any game
including fullscreen borderless. Does NOT hook DirectX — uses a separate
transparent window, which is anti-cheat safe.
"""

from __future__ import annotations
import ctypes
import ctypes.wintypes as wintypes
import threading
import time
from typing import Optional, List
from loguru import logger

# Win32 constants
WS_EX_LAYERED = 0x80000
WS_EX_TRANSPARENT = 0x20
WS_EX_TOPMOST = 0x8
WS_EX_TOOLWINDOW = 0x80
WS_POPUP = 0x80000000
GWL_EXSTYLE = -20
LWA_COLORKEY = 1
LWA_ALPHA = 2
SW_SHOW = 5
HWND_TOPMOST = -1
SWP_NOMOVE = 2
SWP_NOSIZE = 1
PM_REMOVE = 1

try:
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    HAS_WIN32 = True
except AttributeError:
    HAS_WIN32 = False

try:
    gdiplus = ctypes.windll.gdiplus

    class GdiplusStartupInput(ctypes.Structure):
        _fields_ = [
            ("GdiplusVersion", ctypes.c_uint32),
            ("DebugEventCallback", ctypes.c_void_p),
            ("SuppressBackgroundThread", ctypes.c_bool),
            ("SuppressExternalCodecs", ctypes.c_bool),
        ]
    HAS_GDIPLUS = True
except Exception:
    HAS_GDIPLUS = False


class Win32Overlay:
    """Transparent always-on-top window for game overlay rendering.

    Features:
    - Enemy bounding boxes (when in debug/coaching mode)
    - Health/ammo/money readouts
    - Prediction arrows (where enemies likely are)
    - Threat level indicator
    - Coach callout text
    - Performance metrics (FPS, latency)

    All rendering runs in its own thread to not block the main loop.
    """

    TRANSPARENT_COLOR = 0x00FF00FF  # Magenta (color key)
    OVERLAY_CLASS = "GGI_Overlay_v1"

    def __init__(self, config=None):
        self._config = config
        self._hwnd: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._draw_commands: List[dict] = []
        self._lock = threading.Lock()
        self._gdiplus_token = ctypes.c_ulong(0)
        if HAS_WIN32:
            self._screen_w = user32.GetSystemMetrics(0)
            self._screen_h = user32.GetSystemMetrics(1)
        else:
            self._screen_w = 1920
            self._screen_h = 1080

    def start(self):
        """Start overlay in its own thread."""
        if not HAS_WIN32:
            logger.warning("Win32 not available (not on Windows). Overlay disabled.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._window_thread, daemon=True)
        self._thread.start()
        logger.info(f"Win32 overlay started ({self._screen_w}x{self._screen_h})")

    def stop(self):
        self._running = False
        if self._hwnd and HAS_WIN32:
            user32.PostMessageW(self._hwnd, 0x0010, 0, 0)  # WM_CLOSE

    def update(self, perception=None, decision=None, state=None):
        """Update overlay with latest game state. Called from main loop."""
        commands = []
        opacity = self._config.get("overlay.opacity", 0.8) if self._config else 0.8

        if perception and self._config and self._config.get("overlay.show_detections", True):
            for det in getattr(perception, "detections", []):
                if det.class_name == "enemy":
                    color = (255, 50, 50)
                    commands.append({
                        "type": "rect",
                        "x1": det.bbox[0], "y1": det.bbox[1],
                        "x2": det.bbox[2], "y2": det.bbox[3],
                        "color": color, "width": 2,
                    })
                    commands.append({
                        "type": "text",
                        "x": det.bbox[0], "y": det.bbox[1] - 18,
                        "text": f"{det.class_name} {det.confidence:.0%} {det.distance_est}",
                        "color": color, "size": 12,
                    })

        # Threat level indicator
        if state and perception:
            threat_colors = {
                "none": (100, 200, 100), "low": (200, 200, 100),
                "medium": (255, 165, 0), "high": (255, 80, 80),
                "critical": (255, 0, 0),
            }
            threat = getattr(perception, "threat_level", "unknown")
            color = threat_colors.get(threat, (200, 200, 200))
            commands.append({
                "type": "text",
                "x": self._screen_w // 2 - 60, "y": 8,
                "text": f"THREAT: {threat.upper()}",
                "color": color, "size": 16,
            })

        # FPS and mode
        if state:
            commands.append({
                "type": "text",
                "x": 10, "y": self._screen_h - 30,
                "text": (f"GGI {state.mode.upper()} | "
                         f"{state.perception_fps:.0f}fps | "
                         f"{state.cognition_latency_ms:.0f}ms"),
                "color": (180, 180, 180), "size": 11,
            })

        # Decision reasoning
        if decision and decision.get("reasoning"):
            commands.append({
                "type": "text",
                "x": 10, "y": 40,
                "text": f"AI: {decision['reasoning'][:60]}",
                "color": (200, 200, 255), "size": 13,
            })

        with self._lock:
            self._draw_commands = commands

    def _window_thread(self):
        """Win32 message loop thread — creates window and processes messages."""
        if not HAS_WIN32:
            return

        # Initialize GDI+
        if HAS_GDIPLUS:
            startup = GdiplusStartupInput()
            startup.GdiplusVersion = 1
            gdiplus.GdiplusStartup(
                ctypes.byref(self._gdiplus_token),
                ctypes.byref(startup),
                None,
            )

        # Register window class
        wc = wintypes.WNDCLASS()
        wc.lpfnWndProc = ctypes.WINFUNCTYPE(
            ctypes.c_long, ctypes.c_void_p, ctypes.c_uint,
            ctypes.c_void_p, ctypes.c_void_p,
        )(self._wnd_proc)
        wc.hInstance = ctypes.windll.kernel32.GetModuleHandleW(None)
        wc.lpszClassName = self.OVERLAY_CLASS
        wc.hbrBackground = gdi32.CreateSolidBrush(self.TRANSPARENT_COLOR)

        try:
            user32.RegisterClassW(ctypes.byref(wc))
        except Exception:
            pass

        # Create layered window
        ex_style = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW
        self._hwnd = user32.CreateWindowExW(
            ex_style,
            self.OVERLAY_CLASS,
            "GGI Overlay",
            WS_POPUP,
            0, 0, self._screen_w, self._screen_h,
            None, None, wc.hInstance, None,
        )

        # Set layered attributes
        user32.SetLayeredWindowAttributes(
            self._hwnd, self.TRANSPARENT_COLOR, 255, LWA_COLORKEY,
        )
        user32.ShowWindow(self._hwnd, SW_SHOW)

        # Message loop with periodic redraw
        msg = wintypes.MSG()
        last_draw = 0
        while self._running:
            while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                if msg.message == 0x0012:  # WM_QUIT
                    self._running = False
                    break
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))

            # Redraw at ~30fps
            now = time.time()
            if now - last_draw > 0.033:
                self._redraw()
                last_draw = now

            time.sleep(0.005)

        # Cleanup
        if self._hwnd:
            user32.DestroyWindow(self._hwnd)
        if HAS_GDIPLUS and self._gdiplus_token.value:
            gdiplus.GdiplusShutdown(self._gdiplus_token)

    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        if msg == 0x000F:  # WM_PAINT
            self._redraw()
            return 0
        if msg == 0x0010:  # WM_CLOSE
            self._running = False
            user32.PostQuitMessage(0)
            return 0
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _redraw(self):
        """Clear and redraw all commands using GDI."""
        if not self._hwnd or not HAS_WIN32:
            return

        hdc = user32.GetDC(self._hwnd)
        if not hdc:
            return

        # Clear with transparent color
        brush = gdi32.CreateSolidBrush(self.TRANSPARENT_COLOR)
        rect = wintypes.RECT(0, 0, self._screen_w, self._screen_h)
        user32.FillRect(hdc, ctypes.byref(rect), brush)
        gdi32.DeleteObject(brush)

        with self._lock:
            commands = self._draw_commands.copy()

        for cmd in commands:
            if cmd["type"] == "rect":
                self._draw_rect(hdc, cmd)
            elif cmd["type"] == "text":
                self._draw_text(hdc, cmd)

        user32.ReleaseDC(self._hwnd, hdc)

    def _draw_rect(self, hdc, cmd):
        """Draw a rectangle outline."""
        r, g, b = cmd.get("color", (255, 0, 0))
        width = cmd.get("width", 2)
        pen = gdi32.CreatePen(0, width, r | (g << 8) | (b << 16))
        old_pen = gdi32.SelectObject(hdc, pen)
        old_brush = gdi32.SelectObject(hdc, gdi32.GetStockObject(5))  # NULL_BRUSH
        gdi32.Rectangle(hdc, cmd["x1"], cmd["y1"], cmd["x2"], cmd["y2"])
        gdi32.SelectObject(hdc, old_pen)
        gdi32.SelectObject(hdc, old_brush)
        gdi32.DeleteObject(pen)

    def _draw_text(self, hdc, cmd):
        """Draw text on the overlay."""
        r, g, b = cmd.get("color", (255, 255, 255))
        size = cmd.get("size", 14)
        text = cmd.get("text", "")
        if not text:
            return

        font = gdi32.CreateFontW(
            size, 0, 0, 0, 700, 0, 0, 0, 0, 0, 0, 0, 0,
            "Consolas",
        )
        old_font = gdi32.SelectObject(hdc, font)
        gdi32.SetTextColor(hdc, r | (g << 8) | (b << 16))
        gdi32.SetBkMode(hdc, 1)  # TRANSPARENT
        gdi32.TextOutW(hdc, cmd["x"], cmd["y"], text, len(text))
        gdi32.SelectObject(hdc, old_font)
        gdi32.DeleteObject(font)
