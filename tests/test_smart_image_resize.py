import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1]))

from image_sizing_and_resizing.smart_image_resize import (
    RESOLUTIONS,
    SmartImageResize,
    _closest_dimensions,
    _reported_aspect_ratio,
)


class SmartImageResizeSelectionTests(unittest.TestCase):
    def test_qwen_automatic_resize_returns_standard_ratio(self):
        result = SmartImageResize().resize(
            model="Qwen-Image-2.1",
            resolution_preset="1 MP ~ 1K (1024 × 1024)",
            selection_mode="automatic",
            dimensions="",
            width=1,
            height=1,
            upscale_method="nearest-exact",
            keep_proportion="pad_edge_pixel",
            pad_color="0, 0, 0",
            crop_position="center",
            image=torch.zeros((1, 144, 112, 3)),
        )

        self.assertEqual(result[1:4], (928, 1152, "4:5"))
        self.assertEqual(result[4], 1024)

    def test_resolution_outputs_are_integers(self):
        self.assertEqual(SmartImageResize.RETURN_TYPES[4], "INT")

        result = SmartImageResize().resize(
            model="FLUX.2 Klein",
            resolution_preset="1 MP ~ 1K (1024 × 1024)",
            selection_mode="automatic",
            dimensions="",
            width=1,
            height=1,
            upscale_method="nearest-exact",
            keep_proportion="stretch",
            pad_color="0, 0, 0",
            crop_position="center",
            image=torch.zeros((1, 16, 16, 3)),
            resolution=256,
        )

        self.assertEqual(result[4], 256)
        self.assertIsInstance(result[4], int)

    def test_automatic_selection_uses_nominal_aspect_ratio(self):
        source_width, source_height = 896, 1152

        for model in ("FLUX.2 Klein", "Qwen-Image-2.1"):
            items = next(iter(RESOLUTIONS[model].values()))
            selected = _closest_dimensions(items, source_width, source_height)
            self.assertEqual(selected["ratio"], "4:5")

    def test_automatic_target_canvas_reports_selected_standard_ratio(self):
        items = next(iter(RESOLUTIONS["Qwen-Image-2.1"].values()))
        selected = _closest_dimensions(items, 896, 1152)

        self.assertEqual(
            _reported_aspect_ratio(
                "automatic",
                "pad_edge_pixel",
                selected,
                selected["width"],
                selected["height"],
            ),
            "4:5",
        )

    def test_modes_that_preserve_source_shape_still_report_actual_ratio(self):
        selected = {"ratio": "4:5"}

        for keep_proportion in ("resize", "total_pixels"):
            self.assertEqual(
                _reported_aspect_ratio(
                    "automatic", keep_proportion, selected, 896, 1152
                ),
                "7:9",
            )

        self.assertEqual(
            _reported_aspect_ratio("manual", "pad_edge_pixel", selected, 928, 1152),
            "29:36",
        )
