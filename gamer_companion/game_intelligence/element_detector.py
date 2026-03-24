"""Element Detector — Detect clickable elements, buttons, menus.

Identifies interactive UI elements on screen for navigation:
- Buttons (rectangular, colored, with text)
- Menu items (lists of text options)
- Checkboxes / toggles
- Dropdowns / selects
- Text input fields
- Sliders
- Tabs

Works without game-specific knowledge using common UI patterns.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from loguru import logger


@dataclass
class DetectedElement:
    """A detected interactive UI element."""
    element_id: str
    element_type: str         # "button", "menu_item", "checkbox", "dropdown", "text_input", "slider", "tab"
    region: Tuple[int, int, int, int]  # (x1, y1, x2, y2) bounding box
    text: str = ""            # Detected text label
    state: str = "normal"     # "normal", "hover", "active", "disabled", "selected"
    confidence: float = 0.0
    clickable: bool = True
    center: Tuple[int, int] = (0, 0)

    def __post_init__(self):
        if self.center == (0, 0) and self.region != (0, 0, 0, 0):
            x1, y1, x2, y2 = self.region
            self.center = ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class MenuStructure:
    """A detected menu with items."""
    menu_id: str
    title: str = ""
    items: List[DetectedElement] = field(default_factory=list)
    region: Tuple[int, int, int, int] = (0, 0, 0, 0)
    orientation: str = "vertical"  # "vertical" or "horizontal"

    def get_item(self, text: str) -> Optional[DetectedElement]:
        """Find a menu item by text (case-insensitive partial match)."""
        text_lower = text.lower()
        for item in self.items:
            if text_lower in item.text.lower():
                return item
        return None


# Common button size ranges (width, height) in pixels
BUTTON_SIZE_HINTS = {
    "small": (60, 25, 120, 40),      # OK, Cancel, X
    "medium": (120, 30, 250, 50),     # Main menu buttons
    "large": (200, 40, 400, 70),      # PLAY, FIND MATCH
    "icon": (20, 20, 60, 60),         # Icon buttons (settings gear, etc.)
}


class ElementDetector:
    """Detect interactive UI elements on screen.

    Strategy:
    1. Analyze frame for rectangular regions with distinct borders
    2. Check for text within detected regions
    3. Classify element type from shape, position, and content
    4. Track element state changes (hover, click feedback)
    5. Build a clickable element map for NavigationEngine

    Designed to work WITHOUT game-specific templates.
    """

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self._width = screen_width
        self._height = screen_height
        self._elements: Dict[str, DetectedElement] = {}
        self._menus: Dict[str, MenuStructure] = {}
        self._element_counter = 0
        self._detection_history: List[Dict[str, DetectedElement]] = []

    def detect_elements(self, frame_data: Dict = None) -> List[DetectedElement]:
        """Detect all interactive elements in a frame.

        Args:
            frame_data: Simulated frame data dict for testing.
                       In production, this would be actual pixel data.
        """
        elements = []

        if frame_data and "elements" in frame_data:
            # Use provided element data (from real CV pipeline)
            for elem_data in frame_data["elements"]:
                self._element_counter += 1
                elem = DetectedElement(
                    element_id=f"elem_{self._element_counter}",
                    element_type=elem_data.get("type", "button"),
                    region=tuple(elem_data.get("region", (0, 0, 100, 40))),
                    text=elem_data.get("text", ""),
                    state=elem_data.get("state", "normal"),
                    confidence=elem_data.get("confidence", 0.8),
                )
                elements.append(elem)
                self._elements[elem.element_id] = elem

        return elements

    def detect_menu(self, region: Tuple[int, int, int, int] = None, items_data: List[Dict] = None) -> Optional[MenuStructure]:
        """Detect a menu structure in a region."""
        self._element_counter += 1
        menu_id = f"menu_{self._element_counter}"

        items = []
        if items_data:
            for item_data in items_data:
                self._element_counter += 1
                item = DetectedElement(
                    element_id=f"elem_{self._element_counter}",
                    element_type="menu_item",
                    region=tuple(item_data.get("region", (0, 0, 200, 30))),
                    text=item_data.get("text", ""),
                    confidence=item_data.get("confidence", 0.8),
                )
                items.append(item)
                self._elements[item.element_id] = item

        menu = MenuStructure(
            menu_id=menu_id,
            items=items,
            region=region or (0, 0, 0, 0),
        )
        self._menus[menu_id] = menu
        return menu

    def find_element_by_text(self, text: str) -> Optional[DetectedElement]:
        """Find an element by its text label."""
        text_lower = text.lower()
        best = None
        best_confidence = 0

        for elem in self._elements.values():
            if text_lower in elem.text.lower() and elem.confidence > best_confidence:
                best = elem
                best_confidence = elem.confidence

        return best

    def find_elements_by_type(self, element_type: str) -> List[DetectedElement]:
        """Find all elements of a specific type."""
        return [e for e in self._elements.values() if e.element_type == element_type]

    def find_nearest(self, x: int, y: int, element_type: str = None) -> Optional[DetectedElement]:
        """Find the nearest clickable element to a point."""
        import math
        best = None
        best_dist = float('inf')

        for elem in self._elements.values():
            if not elem.clickable:
                continue
            if element_type and elem.element_type != element_type:
                continue

            cx, cy = elem.center
            dist = math.sqrt((cx - x) ** 2 + (cy - y) ** 2)
            if dist < best_dist:
                best_dist = dist
                best = elem

        return best

    def update_element_state(self, element_id: str, state: str) -> bool:
        """Update an element's state."""
        elem = self._elements.get(element_id)
        if not elem:
            return False
        elem.state = state
        return True

    def clear(self):
        """Clear all detected elements."""
        if self._elements:
            self._detection_history.append(dict(self._elements))
            if len(self._detection_history) > 20:
                self._detection_history.pop(0)
        self._elements.clear()
        self._menus.clear()

    def get_clickable_map(self) -> List[dict]:
        """Get all clickable elements as a flat list."""
        return [
            {
                "id": e.element_id,
                "type": e.element_type,
                "text": e.text,
                "center": e.center,
                "region": e.region,
                "state": e.state,
            }
            for e in self._elements.values()
            if e.clickable
        ]

    def get_stats(self) -> dict:
        type_counts: Dict[str, int] = {}
        for e in self._elements.values():
            type_counts[e.element_type] = type_counts.get(e.element_type, 0) + 1
        return {
            "total_elements": len(self._elements),
            "menus": len(self._menus),
            "by_type": type_counts,
            "clickable": sum(1 for e in self._elements.values() if e.clickable),
        }
