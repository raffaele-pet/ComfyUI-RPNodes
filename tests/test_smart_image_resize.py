import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1]))

from image_sizing_and_resizing.smart_image_resize import (
    RESOLUTIONS,
    SmartImageResize,
    _closest_dimensions,
    _find_dimensions,
    _reported_aspect_ratio,
)
from image_sizing_and_resizing.smart_image_size import (
    SmartImageSize,
    dimension_text,
    resolution_output,
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

        self.assertEqual(result[1:4], (896, 1152, "7:9"))
        self.assertEqual(result[4], 1024)

    def test_qwen_automatic_four_five_uses_catalog_dimensions_and_ratio(self):
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
            image=torch.zeros((1, 1112, 880, 3)),
        )

        self.assertEqual(result[1:4], (928, 1152, "4:5"))
        self.assertEqual(tuple(result[0].shape[1:3]), (1152, 928))

    def test_both_nodes_resolve_every_preset_identically(self):
        size = SmartImageSize()
        for model, model_resolutions in RESOLUTIONS.items():
            for resolution, items in model_resolutions.items():
                for item in items:
                    with self.subTest(model=model, resolution=resolution, ratio=item["ratio"]):
                        dimensions = dimension_text(item)
                        _, resolved, found_items, selected = _find_dimensions(
                            model, resolution, dimensions
                        )
                        self.assertEqual(resolved, resolution)
                        self.assertIs(found_items, items)
                        self.assertIs(selected, item)
                        self.assertEqual(
                            size.get_resolution(model, resolution, dimensions),
                            (
                                item["width"],
                                item["height"],
                                item["ratio"],
                                resolution_output(resolution),
                            ),
                        )

    def test_saved_dimension_text_uses_the_updated_preset_in_both_nodes(self):
        model = "Z-Image-Turbo"
        resolution = "1.56 MP ~ 1.25K (1280 × 1280)"
        old_dimensions = "16:9 (Landscape Widescreen) - 1792 x 1008"
        size_result = SmartImageSize().get_resolution(model, resolution, old_dimensions)
        _, _, _, resize_item = _find_dimensions(model, resolution, old_dimensions)
        self.assertEqual(size_result[:3], (1536, 864, "16:9"))
        self.assertEqual((resize_item["width"], resize_item["height"]), (1536, 864))

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
            self.assertEqual(selected["ratio"], "7:9")

            selected = _closest_dimensions(items, source_height, source_width)
            self.assertEqual(selected["ratio"], "9:7")

    def test_automatic_target_canvas_reports_selected_standard_ratio(self):
        items = next(iter(RESOLUTIONS["Qwen-Image-2.1"].values()))
        selected = _closest_dimensions(items, 896, 1152)
        width, height = selected["width"], selected["height"]

        self.assertEqual(
            _reported_aspect_ratio(
                "automatic",
                "pad_edge_pixel",
                selected,
                width,
                height,
            ),
            "7:9",
        )

    def test_modes_that_preserve_source_shape_still_report_actual_ratio(self):
        selected = {"ratio": "4:5", "width": 928, "height": 1152}

        for keep_proportion in ("resize", "total_pixels"):
            self.assertEqual(
                _reported_aspect_ratio(
                    "automatic", keep_proportion, selected, 896, 1152
                ),
                "7:9",
            )

        self.assertEqual(
            _reported_aspect_ratio("manual", "pad_edge_pixel", selected, 928, 1152),
            "4:5",
        )
        self.assertEqual(
            _reported_aspect_ratio("manual", "pad_edge_pixel", selected, 960, 1152),
            "5:6",
        )
