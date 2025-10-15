"""Basic interaction manager for inspect mode overlays."""

from __future__ import annotations

import cv2
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

import config

Shape = Dict[str, Any]
SelectionCallback = Callable[[Optional[int], Shape], None]


def _ensure_hit_contour(shape: Shape) -> Optional[np.ndarray]:
    contour = shape.get("hit_contour")
    if contour is None:
        contour = shape.get("hit_cnt")
    if contour is None:
        contour = shape.get("hit_polygon")
    if contour is None:
        return None
    contour = np.asarray(contour)
    if contour.ndim == 2:
        contour = contour.reshape(-1, 1, 2)
    if contour.dtype != np.int32:
        contour = contour.astype(np.int32)
    return contour


class InteractionManager:
    """Lightweight interaction manager for automatic inspect mode."""

    def __init__(
        self,
        shapes: List[Shape],
        warped_image: np.ndarray,
        display_height: int = 800,
        hover_snap_distance_mm: float = 10.0,
        enable_performance_optimization: bool = True,
    ) -> None:
        self.shapes = shapes
        self.warped_image = warped_image
        self.display_height = max(1, display_height)
        self.enable_performance_optimization = enable_performance_optimization

        h, w = warped_image.shape[:2]
        if h <= 0 or w <= 0:
            self.display_scale = 1.0
        elif h <= self.display_height:
            self.display_scale = 1.0
        else:
            self.display_scale = self.display_height / float(h)
        self.display_size = (
            max(1, int(round(w * self.display_scale))),
            max(1, int(round(h * self.display_scale))),
        )

        self.window_name: Optional[str] = None
        self.selection_callback: Optional[SelectionCallback] = None
        self.hover_index: Optional[int] = None
        self.selected_index: Optional[int] = None

        # Convert hover snap distance from millimetres to pixels using config.PX_PER_MM
        px_per_mm = getattr(config, "PX_PER_MM", 1.0)
        self.hover_snap_distance_px = max(1.0, hover_snap_distance_mm * float(px_per_mm))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def setup_window(self, window_name: str) -> None:
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.display_size[0], self.display_size[1])
        cv2.setMouseCallback(window_name, self._on_mouse_event)

    def show_initial_render(self) -> None:
        if not self.window_name:
            return
        image = self.render_current_state()
        if image is not None:
            display = self._to_display(image)
            cv2.imshow(self.window_name, display)

    def cleanup(self) -> None:
        if self.window_name:
            cv2.setMouseCallback(self.window_name, lambda *args, **kwargs: None)

    def set_selection_callback(self, callback: SelectionCallback) -> None:
        self.selection_callback = callback

    def print_shape_summary(self) -> None:
        print(f"[INFO] Detected {len(self.shapes)} shape(s)")
        for idx, shape in enumerate(self.shapes, start=1):
            shape_type = shape.get("type", "shape")
            method = shape.get("detection_method", "automatic")
            detail = ""
            if shape_type == "circle" and shape.get("diameter_mm") is not None:
                detail = f"Ø {shape['diameter_mm']:.0f} mm"
            elif shape_type == "rectangle" and shape.get("width_mm") is not None:
                detail = (
                    f"{shape['width_mm']:.0f}×{shape.get('height_mm', 0):.0f} mm"
                )
            print(f"  {idx}. {shape_type} ({method}) {detail}")

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def handle_mouse_move(self, x: int, y: int) -> bool:
        idx = self._find_shape_at_point(x, y)
        if idx != self.hover_index:
            self.hover_index = idx
            return True
        return False

    def handle_mouse_click(self, x: int, y: int) -> bool:
        idx = self._find_shape_at_point(x, y)
        self.selected_index = idx
        if idx is not None and self.selection_callback:
            self.selection_callback(idx, self.shapes[idx])
        return True

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def render_current_state(self) -> Optional[np.ndarray]:
        if self.warped_image is None or self.warped_image.size == 0:
            return None

        result = self.warped_image.copy()
        for idx, shape in enumerate(self.shapes):
            if shape.get("inner"):
                color = (255, 200, 0)
            else:
                color = (0, 255, 0)

            thickness = 2
            if idx == self.selected_index:
                color = (0, 200, 0)
                thickness = 3
            elif idx == self.hover_index:
                color = (255, 255, 0)

            self._draw_shape(result, shape, color, thickness)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _on_mouse_event(self, event: int, x: int, y: int, flags: int, userdata: Any) -> None:
        if self.display_scale <= 0:
            return
        orig_x = int(x / self.display_scale)
        orig_y = int(y / self.display_scale)

        needs_render = False
        if event == cv2.EVENT_MOUSEMOVE:
            needs_render = self.handle_mouse_move(orig_x, orig_y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            needs_render = self.handle_mouse_click(orig_x, orig_y)

        if needs_render and self.window_name:
            image = self.render_current_state()
            if image is not None:
                display = self._to_display(image)
                cv2.imshow(self.window_name, display)
                cv2.waitKey(1)

    def _find_shape_at_point(self, x: int, y: int) -> Optional[int]:
        best_idx = None
        best_distance = float("inf")
        point = (x, y)

        for idx, shape in enumerate(self.shapes):
            contour = _ensure_hit_contour(shape)
            if contour is None:
                continue
            distance = cv2.pointPolygonTest(contour, point, True)
            if distance >= 0:
                return idx
            distance = abs(distance)
            if distance < best_distance and distance <= self.hover_snap_distance_px:
                best_distance = distance
                best_idx = idx
        return best_idx

    def _draw_shape(
        self,
        image: np.ndarray,
        shape: Shape,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        shape_type = shape.get("type")
        if shape_type == "circle":
            center = tuple(map(int, shape.get("center", (0, 0))))
            radius = int(round(shape.get("radius_px") or shape.get("radius", 0)))
            if radius > 0:
                cv2.circle(image, center, radius, color, thickness)
        elif shape_type == "rectangle":
            box = shape.get("box")
            if box is not None:
                contour = np.asarray(box, dtype=np.int32)
                cv2.drawContours(image, [contour], 0, color, thickness)
        else:
            contour = _ensure_hit_contour(shape)
            if contour is not None:
                cv2.drawContours(image, [contour], 0, color, thickness)

    def _to_display(self, image: np.ndarray) -> np.ndarray:
        if self.display_scale == 1.0:
            return image
        return cv2.resize(image, self.display_size)


def default_selection_callback(index: Optional[int], shape: Shape) -> None:
    shape_type = shape.get("type", "shape")
    method = shape.get("detection_method", "automatic")
    print(f"[INFO] Selected {shape_type} (method={method}) index={index}")


def validate_shapes_for_interaction(shapes: List[Shape]) -> List[Shape]:
    valid: List[Shape] = []
    for shape in shapes:
        contour = _ensure_hit_contour(shape)
        if contour is None:
            continue
        valid.append(shape)
    return valid
