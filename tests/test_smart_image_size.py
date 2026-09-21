import importlib.util
import math
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1] / "image_sizing_and_resizing" / "smart_image_size.py"
)
SPEC = importlib.util.spec_from_file_location("smart_image_size", MODULE_PATH)
smart_image_size = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smart_image_size)


QWEN_21_TIERS = {
    "1 MP ~ 1K (1024 × 1024)": 1024,
    "2 MP (1440 × 1440)": 1440,
    "3 MP (1760 × 1760)": 1760,
    "4 MP ~ 2K (2048 × 2048)": 2048,
}

ALL_RESOLUTION_LABELS = {
    "1 MP ~ 1K (1024 × 1024)": 1024,
    "1.56 MP ~ 1.25K (1280 × 1280)": 1280,
    "1.68 MP (1328 × 1328)": 1328,
    "2 MP (1440 × 1440)": 1440,
    "2.25 MP ~ 1.5K (1536 × 1536)": 1536,
    "3 MP (1760 × 1760)": 1760,
    "4 MP ~ 2K (2048 × 2048)": 2048,
}

QWEN_21_OFFICIAL_2K = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}


class SmartImageSizeTests(unittest.TestCase):
    def test_all_models_use_descriptive_resolution_labels(self):
        self.assertEqual(
            set(smart_image_size.unique_resolutions()), set(ALL_RESOLUTION_LABELS)
        )

        for model_resolutions in smart_image_size.RESOLUTIONS.values():
            for label, items in model_resolutions.items():
                square = items[0]
                self.assertEqual(square["ratio"], "1:1")
                self.assertEqual(square["width"], square["height"])
                self.assertEqual(ALL_RESOLUTION_LABELS[label], square["width"])
                self.assertEqual(
                    smart_image_size.resolution_output(label), square["width"]
                )

    def test_qwen_image_21_has_all_resolution_tiers_and_ratios(self):
        tiers = smart_image_size.RESOLUTIONS["Qwen-Image-2.1"]

        self.assertEqual(list(tiers), list(QWEN_21_TIERS))
        self.assertTrue(all(len(items) == 19 for items in tiers.values()))
        self.assertTrue(
            all(
                item["width"] % 32 == 0 and item["height"] % 32 == 0
                for items in tiers.values()
                for item in items
            )
        )

        ratios = [item["ratio"] for item in next(iter(tiers.values()))]
        self.assertTrue(
            all([item["ratio"] for item in items] == ratios for items in tiers.values())
        )

    def test_qwen_image_21_derived_sizes_use_pixel_budget_and_32_pixel_grid(self):
        tiers = smart_image_size.RESOLUTIONS["Qwen-Image-2.1"]

        for items in tiers.values():
            square_side = items[0]["width"]
            pixel_budget = square_side**2
            for item in items:
                if square_side == 2048 and item["ratio"] in QWEN_21_OFFICIAL_2K:
                    expected = QWEN_21_OFFICIAL_2K[item["ratio"]]
                else:
                    ratio_width, ratio_height = map(int, item["ratio"].split(":"))
                    expected = (
                        round(math.sqrt(pixel_budget * ratio_width / ratio_height) / 32)
                        * 32,
                        round(math.sqrt(pixel_budget * ratio_height / ratio_width) / 32)
                        * 32,
                    )
                self.assertEqual((item["width"], item["height"]), expected)

        two_mp = {item["ratio"]: item for item in tiers["2 MP (1440 × 1440)"]}
        self.assertEqual((two_mp["2:3"]["width"], two_mp["2:3"]["height"]), (1184, 1760))

    def test_qwen_image_21_square_sizes_and_resolution_outputs(self):
        node = smart_image_size.SmartImageSize()

        for tier, side in QWEN_21_TIERS.items():
            square = smart_image_size.RESOLUTIONS["Qwen-Image-2.1"][tier][0]
            self.assertEqual(square["ratio"], "1:1")
            self.assertEqual((square["width"], square["height"]), (side, side))
            self.assertEqual(smart_image_size.resolution_output(tier), side)
            self.assertEqual(
                node.get_resolution(
                    "Qwen-Image-2.1", tier, smart_image_size.dimension_text(square)
                ),
                (side, side, "1:1", side),
            )

        native_2k = {
            item["ratio"]: (item["width"], item["height"])
            for item in smart_image_size.RESOLUTIONS["Qwen-Image-2.1"][
                "4 MP ~ 2K (2048 × 2048)"
            ]
        }
        self.assertEqual(native_2k["4:3"], (2400, 1792))
        self.assertEqual(native_2k["3:4"], (1792, 2400))
        self.assertEqual(native_2k["16:9"], (2752, 1536))
        self.assertEqual(native_2k["9:16"], (1536, 2752))

    def test_legacy_resolution_values_are_still_supported(self):
        self.assertEqual(smart_image_size.SmartImageSize.RETURN_TYPES[3], "INT")
        self.assertEqual(smart_image_size.resolution_output("1K"), 1024)
        self.assertEqual(smart_image_size.resolution_output("1.5K"), 1536)
        self.assertEqual(smart_image_size.resolution_output("2K"), 2048)
        self.assertEqual(smart_image_size.resolution_output("1328"), 1328)

        flux_resolutions = smart_image_size.RESOLUTIONS["FLUX.2 Klein"]
        self.assertEqual(
            smart_image_size.resolve_resolution(flux_resolutions, "1536"),
            "2.25 MP ~ 1.5K (1536 × 1536)",
        )
        self.assertEqual(
            smart_image_size.resolve_resolution(flux_resolutions, "1.5K"),
            "2.25 MP ~ 1.5K (1536 × 1536)",
        )
