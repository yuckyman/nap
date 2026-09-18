import os
import unittest

import cv2
import numpy as np

from nimslo_core.alignment import center_images_on_subject
from nimslo_core.terminal_picker import (
    _read_input,
    build_contact_sheet,
    create_subject_roi_masks,
    track_subject_points,
)


class TerminalPickerTests(unittest.TestCase):
    def test_mouse_sequence_is_not_mistaken_for_escape_key(self):
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b"\x1b[<0;359;495M")
            event, location = _read_input(read_fd)
        finally:
            os.close(read_fd)
            os.close(write_fd)

        self.assertEqual(event, "click")
        self.assertEqual(location, (358, 494))

    def test_contact_sheet_maps_points_both_directions(self):
        images = [
            np.zeros((240, 320, 3), dtype=np.uint8),
            np.zeros((240, 320, 3), dtype=np.uint8),
        ]

        _, panels = build_contact_sheet(images, (900, 700))
        source_point = (123.0, 87.0)
        canvas_point = panels[0].from_source(source_point)
        round_trip = panels[0].to_source(*canvas_point)

        self.assertAlmostEqual(round_trip[0], source_point[0], delta=1.0)
        self.assertAlmostEqual(round_trip[1], source_point[1], delta=1.0)

    def test_roi_masks_are_centered_on_selected_points(self):
        images = [
            np.zeros((200, 300, 3), dtype=np.uint8),
            np.zeros((200, 300, 3), dtype=np.uint8),
        ]
        points = [(100.0, 80.0), (120.0, 90.0)]

        masks = create_subject_roi_masks(images, points, radius_fraction=0.2)

        self.assertEqual(len(masks), 2)
        for mask, expected in zip(masks, points):
            moments = cv2.moments(mask)
            center = (
                moments["m10"] / moments["m00"],
                moments["m01"] / moments["m00"],
            )
            self.assertAlmostEqual(center[0], expected[0], delta=1.0)
            self.assertAlmostEqual(center[1], expected[1], delta=1.0)

    def test_optical_flow_tracks_adjacent_translations(self):
        rng = np.random.default_rng(12)
        base = np.zeros((280, 360, 3), dtype=np.uint8)
        for x, y in rng.integers([70, 50], [290, 230], size=(120, 2)):
            cv2.circle(base, (int(x), int(y)), 2, (255, 255, 255), -1)

        frames = [base]
        step_x, step_y = 9, -5
        for index in range(1, 4):
            transform = np.float32([[1, 0, step_x * index], [0, 1, step_y * index]])
            frames.append(cv2.warpAffine(base, transform, (360, 280)))

        points = track_subject_points(frames, (180.0, 140.0))

        for index, point in enumerate(points):
            self.assertAlmostEqual(point[0], 180 + step_x * index, delta=1.5)
            self.assertAlmostEqual(point[1], 140 + step_y * index, delta=1.5)

    def test_explicit_anchor_controls_centering_even_when_roi_is_clipped(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = create_subject_roi_masks([image], [(5.0, 10.0)])[0]

        _, _, transforms = center_images_on_subject(
            [image],
            [mask],
            subject_centers=[(5.0, 10.0)],
        )

        self.assertAlmostEqual(transforms[0][0, 2], 45.0)
        self.assertAlmostEqual(transforms[0][1, 2], 40.0)


if __name__ == "__main__":
    unittest.main()
