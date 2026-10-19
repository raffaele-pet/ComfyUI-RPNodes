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
    RESIZE_MODES,
)
from image_sizing_and_resizing.smart_image_size import (
    SmartImageSize,
    actual_aspect_ratio,
    dimension_text,
    image_megapixels,
    resolution_output,
)


class SmartImageResizeSelectionTests(unittest.TestCase):
    def test_flux3_klein_is_the_default_in_both_nodes(self):
        for node in (SmartImageSize, SmartImageResize):
            models, options = node.INPUT_TYPES()["required"]["model"]
            self.assertEqual(options["default"], "FLUX.3 Klein")
            self.assertIn(options["default"], models)

    def test_flux3_automatic_resize_matches_size_node_outputs(self):
        for resolution in ("1 MP ~ 1K (1024 × 1024)", "4 MP ~ 2K (2048 × 2048)"):
            with self.subTest(resolution=resolution):
                item = next(item for item in RESOLUTIONS["FLUX.3 Klein"][resolution]
                            if item["ratio"] == "16:9")
                expected = SmartImageSize().get_resolution(
                    "FLUX.3 Klein", resolution, dimension_text(item)
                )
                result = SmartImageResize().resize(
                    model="FLUX.3 Klein", resolution_preset=resolution,
                    selection_mode="automatic", dimensions="", width=1, height=1,
                    upscale_method="nearest-exact", keep_proportion="pad_edge_pixel",
                    pad_color="0, 0, 0", crop_position="center",
                    image=torch.zeros((1, 9, 16, 3)),
                )
                self.assertEqual((*result[1:6], result[8]), expected)
                self.assertEqual(tuple(result[0].shape[1:3]), (item["height"], item["width"]))

    def test_no_pad_output_matches_resize_mode_for_every_primary_mode(self):
        image = torch.linspace(0, 1, 2 * 11 * 7 * 3).reshape(2, 11, 7, 3)
        mask = torch.ones((1, 11, 7))
        settings = dict(
            model="FLUX.2 Klein", resolution_preset="1 MP ~ 1K (1024 × 1024)",
            selection_mode="manual", dimensions="1:1 (Square) - 1024 x 1024",
            width=64, height=64, upscale_method="bilinear", pad_color="255, 0, 0",
            crop_position="center", image=image, mask=mask,
        )
        expected = SmartImageResize().resize(**settings, keep_proportion="resize")[0]
        self.assertEqual(tuple(expected.shape), (2, 64, 41, 3))
        for mode in RESIZE_MODES:
            with self.subTest(mode=mode):
                result = SmartImageResize().resize(**settings, keep_proportion=mode)
                self.assertEqual(len(result), 9)
                torch.testing.assert_close(result[7], expected, rtol=0, atol=0)
                self.assertEqual(result[8], result[1] * result[2] / (1024 * 1024))
        self.assertEqual(SmartImageResize.RETURN_TYPES[:7],
                         ("IMAGE", "INT", "INT", "STRING", "STRING", "INT", "MASK"))
        self.assertEqual(SmartImageResize.RETURN_NAMES[7], "resized_image_no_pad")

    def test_864_by_1080_resize_reports_exact_megapixels(self):
        result = SmartImageResize().resize(
            model="FLUX.3 Klein", resolution_preset="1 MP ~ 1K (1024 × 1024)",
            selection_mode="automatic", dimensions="", width=1, height=1,
            upscale_method="nearest-exact", keep_proportion="pad_edge_pixel",
            pad_color="0, 0, 0", crop_position="center",
            image=torch.zeros((1, 1080, 864, 3)),
        )
        self.assertEqual(result[1:6], (912, 1152, "4:5", "19:24", 1024))
        self.assertEqual(result[8], 1.001953125)
        self.assertIsInstance(result[8], float)
        self.assertEqual(SmartImageResize.RETURN_NAMES[8], "megapixels")
        self.assertEqual(SmartImageResize.RETURN_TYPES[8], "FLOAT")

    def test_no_pad_reuses_content_of_padded_result_for_every_alignment(self):
        image = torch.linspace(0, 1, 11 * 7 * 3).reshape(1, 11, 7, 3)
        for position in ("center", "left", "right", "top", "bottom"):
            with self.subTest(position=position):
                result = SmartImageResize().resize(
                    model="FLUX.2 Klein", resolution_preset="1 MP ~ 1K (1024 × 1024)",
                    selection_mode="manual", dimensions="", width=64, height=64,
                    upscale_method="nearest-exact", keep_proportion="pad_edge_pixel",
                    pad_color="0, 0, 0", crop_position=position, image=image,
                )
                self.assertEqual(result[1:3], (64, 64))
                self.assertEqual(tuple(result[6].shape), (1, 64, 64))
                left = 0 if position == "left" else 23 if position == "right" else 11
                torch.testing.assert_close(result[0][:, :, left:left + 41], result[7], rtol=0, atol=0)

    def test_no_pad_honors_automatic_preset_resolution_override_and_mask_only_input(self):
        settings = dict(
            model="FLUX.2 Klein", resolution_preset="1 MP ~ 1K (1024 × 1024)",
            selection_mode="automatic", dimensions="", width=1, height=1,
            upscale_method="nearest-exact", pad_color="0, 0, 0", crop_position="center",
            mask=torch.linspace(0, 1, 11 * 7).reshape(1, 11, 7), resolution=128,
        )
        padded = SmartImageResize().resize(**settings, keep_proportion="pad_edge_pixel")
        resized = SmartImageResize().resize(**settings, keep_proportion="resize")
        torch.testing.assert_close(padded[7], resized[0], rtol=0, atol=0)
        self.assertEqual(padded[5], 128)
        self.assertEqual(padded[7].shape[-1], 3)

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
        self.assertEqual(result[4], "7:9")
        self.assertEqual(result[5], 1024)

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
        self.assertEqual(result[4], "29:36")

    def test_real_ratio_differs_from_approximate_five_three(self):
        result = SmartImageResize().resize(
            model="Qwen-Image-2.1",
            resolution_preset="1 MP ~ 1K (1024 × 1024)",
            selection_mode="automatic",
            dimensions="",
            width=1,
            height=1,
            upscale_method="nearest-exact",
            keep_proportion="stretch",
            pad_color="0, 0, 0",
            crop_position="center",
            image=torch.zeros((1, 80, 131, 3)),
        )
        self.assertEqual(SmartImageResize.RETURN_NAMES[3], "aspect_ratio_approx")
        self.assertEqual(SmartImageResize.RETURN_NAMES[4], "aspect_ratio")
        self.assertEqual(result[1:4], (1312, 800, "5:3"))
        self.assertEqual(result[4], "41:25")

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
                                actual_aspect_ratio(item["width"], item["height"]),
                                resolution_output(resolution),
                                image_megapixels(item["width"], item["height"]),
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
        self.assertEqual(SmartImageResize.RETURN_TYPES[5], "INT")

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

        self.assertEqual(result[5], 256)
        self.assertIsInstance(result[5], int)

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
