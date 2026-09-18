"""Interactive subject selection using terminal graphics and pixel mouse events."""

from __future__ import annotations

import base64
import fcntl
import math
import os
import re
import select
import struct
import sys
import termios
import tty
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]
_MOUSE_RE = re.compile(rb"\x1b\[<(\d+);(\d+);(\d+)([Mm])")
_PIXEL_SIZE_RE = re.compile(rb"\x1b\[4;(\d+);(\d+)t")
_IMAGE_ID = 90421


class SelectionCancelled(RuntimeError):
    """Raised when the user quits the interactive picker."""


@dataclass(frozen=True)
class Panel:
    """Mapping between one source image and its contact-sheet rectangle."""

    index: int
    x: int
    y: int
    width: int
    height: int
    source_width: int
    source_height: int

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def to_source(self, x: int, y: int) -> Point:
        source_x = (x - self.x) * self.source_width / self.width
        source_y = (y - self.y) * self.source_height / self.height
        return (
            min(max(source_x, 0.0), self.source_width - 1.0),
            min(max(source_y, 0.0), self.source_height - 1.0),
        )

    def from_source(self, point: Point) -> Tuple[int, int]:
        x, y = point
        return (
            self.x + int(round(x * self.width / self.source_width)),
            self.y + int(round(y * self.height / self.source_height)),
        )


def track_subject_points(images: Sequence[np.ndarray], initial_point: Point) -> List[Point]:
    """Track an anchor using median motion from nearby optical-flow features."""
    if not images:
        return []

    points = [initial_point]
    previous_gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)

    for image in images[1:]:
        current_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = previous_gray.shape
        anchor = points[-1]
        radius = max(32, int(round(min(h, w) * 0.2)))
        feature_mask = np.zeros_like(previous_gray)
        cv2.circle(
            feature_mask,
            (int(round(anchor[0])), int(round(anchor[1]))),
            radius,
            255,
            thickness=-1,
        )
        previous_features = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=5,
            mask=feature_mask,
            blockSize=7,
        )

        if previous_features is None or len(previous_features) < 3:
            points.append(anchor)
            previous_gray = current_gray
            continue

        next_features, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            previous_features,
            None,
            winSize=(51, 51),
            maxLevel=4,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                40,
                0.01,
            ),
        )
        if next_features is None or forward_status is None:
            points.append(anchor)
            previous_gray = current_gray
            continue

        back_features, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            previous_gray,
            next_features,
            None,
            winSize=(51, 51),
            maxLevel=4,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                40,
                0.01,
            ),
        )
        if back_features is None or backward_status is None:
            points.append(anchor)
            previous_gray = current_gray
            continue

        forward_ok = forward_status.ravel().astype(bool)
        backward_ok = backward_status.ravel().astype(bool)
        finite = np.isfinite(next_features).all(axis=(1, 2))
        round_trip_error = np.linalg.norm(
            previous_features.reshape(-1, 2) - back_features.reshape(-1, 2),
            axis=1,
        )
        valid = forward_ok & backward_ok & finite & (round_trip_error < 2.0)

        if int(np.sum(valid)) >= 3:
            displacements = (
                next_features.reshape(-1, 2)[valid]
                - previous_features.reshape(-1, 2)[valid]
            )
            dx, dy = np.median(displacements, axis=0)
            current_h, current_w = current_gray.shape
            point = (
                min(max(anchor[0] + float(dx), 0.0), current_w - 1.0),
                min(max(anchor[1] + float(dy), 0.0), current_h - 1.0),
            )
        else:
            point = anchor

        points.append(point)
        previous_gray = current_gray

    return points


def create_subject_roi_masks(
    images: Sequence[np.ndarray],
    points: Sequence[Point],
    radius_fraction: float = 0.22,
) -> List[np.ndarray]:
    """Create elliptical local masks centered on selected subject anchors."""
    if len(images) != len(points):
        raise ValueError("images and points must have the same length")
    if not 0 < radius_fraction <= 0.5:
        raise ValueError("radius_fraction must be greater than 0 and at most 0.5")

    masks = []
    for image, point in zip(images, points):
        h, w = image.shape[:2]
        center = (
            int(round(min(max(point[0], 0), w - 1))),
            int(round(min(max(point[1], 0), h - 1))),
        )
        axes = (
            max(12, int(round(w * radius_fraction))),
            max(12, int(round(h * radius_fraction))),
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, thickness=-1)
        masks.append(mask)
    return masks


def build_contact_sheet(
    images: Sequence[np.ndarray],
    canvas_size: Tuple[int, int],
    points: Optional[Sequence[Point]] = None,
    initial: bool = False,
) -> Tuple[np.ndarray, List[Panel]]:
    """Build a two-column review sheet and the mappings used for mouse clicks."""
    if not images:
        raise ValueError("at least one image is required")

    canvas_width, canvas_height = canvas_size
    if canvas_width < 160 or canvas_height < 120:
        raise ValueError("terminal is too small for interactive subject selection")

    canvas = np.full((canvas_height, canvas_width, 3), 18, dtype=np.uint8)
    header_height = max(34, min(54, canvas_height // 10))
    footer_height = max(34, min(54, canvas_height // 10))
    gap = max(6, min(14, canvas_width // 100))
    columns = 2 if len(images) > 1 else 1
    rows = math.ceil(len(images) / columns)
    slot_width = (canvas_width - gap * (columns + 1)) // columns
    available_height = canvas_height - header_height - footer_height
    slot_height = (available_height - gap * (rows + 1)) // rows

    title = (
        "click the subject in frame 1"
        if initial
        else "review anchors - click any frame to correct"
    )
    cv2.putText(
        canvas,
        title,
        (gap, int(header_height * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.45, min(0.8, canvas_width / 1200)),
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    panels: List[Panel] = []
    for index, image in enumerate(images):
        source_height, source_width = image.shape[:2]
        row, column = divmod(index, columns)
        slot_x = gap + column * (slot_width + gap)
        slot_y = header_height + gap + row * (slot_height + gap)
        scale = min(slot_width / source_width, slot_height / source_height)
        width = max(1, int(round(source_width * scale)))
        height = max(1, int(round(source_height * scale)))
        x = slot_x + (slot_width - width) // 2
        y = slot_y + (slot_height - height) // 2

        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (width, height), interpolation=interpolation)
        canvas[y : y + height, x : x + width] = resized
        cv2.rectangle(canvas, (x, y), (x + width - 1, y + height - 1), (90, 90, 90), 1)
        cv2.putText(
            canvas,
            str(index + 1),
            (x + 8, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        panel = Panel(index, x, y, width, height, source_width, source_height)
        panels.append(panel)
        if points is not None and index < len(points):
            marker_x, marker_y = panel.from_source(points[index])
            marker_radius = max(7, min(14, min(width, height) // 30))
            cv2.drawMarker(
                canvas,
                (marker_x, marker_y),
                (70, 255, 120),
                cv2.MARKER_CROSS,
                marker_radius * 2,
                2,
                cv2.LINE_AA,
            )
            cv2.circle(canvas, (marker_x, marker_y), marker_radius, (70, 255, 120), 2)

    footer = "enter: accept   click: correct   q/esc: cancel"
    cv2.putText(
        canvas,
        footer,
        (gap, canvas_height - max(10, footer_height // 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.4, min(0.7, canvas_width / 1400)),
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return canvas, panels


def _panel_at(panels: Sequence[Panel], x: int, y: int) -> Optional[Panel]:
    return next((panel for panel in panels if panel.contains(x, y)), None)


def _terminal_geometry(fd: int) -> Tuple[int, int, int, int]:
    packed = fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\0" * 8)
    rows, columns, pixel_width, pixel_height = struct.unpack("HHHH", packed)
    columns = columns or 80
    rows = rows or 24
    if pixel_width and pixel_height:
        return columns, rows, pixel_width, pixel_height

    os.write(sys.stdout.fileno(), b"\x1b[14t")
    response = bytearray()
    deadline_reads = 4
    while deadline_reads > 0 and select.select([fd], [], [], 0.08)[0]:
        response.extend(os.read(fd, 64))
        match = _PIXEL_SIZE_RE.search(response)
        if match:
            height, width = (int(value) for value in match.groups())
            return columns, rows, width, height
        deadline_reads -= 1

    # Last-resort cell estimate. Pixel mouse coordinates still map consistently
    # enough for selection because the same dimensions are used for placement.
    return columns, rows, columns * 10, rows * 20


def _kitty_command(control: str, payload: bytes = b"") -> bytes:
    return b"\x1b_G" + control.encode("ascii") + b";" + payload + b"\x1b\\"


def _delete_image() -> None:
    os.write(sys.stdout.fileno(), _kitty_command(f"a=d,d=i,i={_IMAGE_ID},q=2"))


def _display_canvas(canvas: np.ndarray, columns: int, rows: int) -> None:
    encoded, png = cv2.imencode(".png", canvas)
    if not encoded:
        raise RuntimeError("failed to encode terminal preview")

    data = base64.b64encode(png.tobytes())
    chunks = [data[offset : offset + 4096] for offset in range(0, len(data), 4096)]
    output = sys.stdout.fileno()
    os.write(output, b"\x1b[?2026h")
    _delete_image()
    os.write(output, b"\x1b[1;1H")
    for index, chunk in enumerate(chunks):
        more = 1 if index < len(chunks) - 1 else 0
        if index == 0:
            control = (
                f"a=T,f=100,t=d,i={_IMAGE_ID},q=2,c={columns},r={rows},"
                f"C=1,m={more}"
            )
        else:
            control = f"q=2,m={more}"
        os.write(output, _kitty_command(control, chunk))
    os.write(output, b"\x1b[?2026l")


def _read_input(fd: int) -> Tuple[str, Optional[Tuple[int, int]]]:
    buffer = bytearray()
    while True:
        buffer.extend(os.read(fd, 1))
        mouse = _MOUSE_RE.search(buffer)
        if mouse:
            button, x, y, action = mouse.groups()
            if action == b"M" and int(button) & 3 == 0:
                return "click", (int(x) - 1, int(y) - 1)
            buffer.clear()
            continue

        if buffer in (b"\r", b"\n"):
            return "accept", None
        if buffer == b"\x1b":
            # Mouse and other terminal control sequences also begin with ESC.
            # Only treat it as the Escape key when no continuation arrives.
            if select.select([fd], [], [], 0.05)[0]:
                continue
            return "cancel", None
        if buffer in (b"q", b"Q", b"\x03"):
            return "cancel", None
        if len(buffer) > 64:
            buffer.clear()


def select_subject_points(images: Sequence[np.ndarray]) -> List[Point]:
    """Render images in the terminal, collect one click, then review tracked anchors."""
    if not images:
        raise ValueError("at least one image is required")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError("interactive subject selection requires an attached terminal")

    input_fd = sys.stdin.fileno()
    output_fd = sys.stdout.fileno()
    previous_settings = termios.tcgetattr(input_fd)
    try:
        tty.setraw(input_fd)
        columns, rows, pixel_width, pixel_height = _terminal_geometry(input_fd)
        os.write(
            output_fd,
            b"\x1b[?1049h\x1b[2J\x1b[H\x1b[?25l"
            b"\x1b[?1000h\x1b[?1006h\x1b[?1016h",
        )

        initial_canvas, initial_panels = build_contact_sheet(
            images,
            (pixel_width, pixel_height),
            initial=True,
        )
        _display_canvas(initial_canvas, columns, rows)

        initial_point: Optional[Point] = None
        while initial_point is None:
            event, location = _read_input(input_fd)
            if event == "cancel":
                raise SelectionCancelled("subject selection cancelled")
            if event == "click" and location is not None:
                panel = _panel_at(initial_panels, *location)
                if panel is not None and panel.index == 0:
                    initial_point = panel.to_source(*location)

        points = track_subject_points(images, initial_point)
        while True:
            canvas, panels = build_contact_sheet(
                images,
                (pixel_width, pixel_height),
                points=points,
            )
            _display_canvas(canvas, columns, rows)
            event, location = _read_input(input_fd)
            if event == "accept":
                return points
            if event == "cancel":
                raise SelectionCancelled("subject selection cancelled")
            if event == "click" and location is not None:
                panel = _panel_at(panels, *location)
                if panel is not None:
                    corrected = panel.to_source(*location)
                    if panel.index == 0:
                        points = track_subject_points(images, corrected)
                    else:
                        points[panel.index] = corrected
    finally:
        _delete_image()
        os.write(
            output_fd,
            b"\x1b[?1016l\x1b[?1006l\x1b[?1000l"
            b"\x1b[?25h\x1b[?1049l",
        )
        termios.tcsetattr(input_fd, termios.TCSADRAIN, previous_settings)
