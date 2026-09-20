import importlib.util
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


class SmartImageSizeTests(unittest.TestCase):
    def test_qwen_image_21_has_all_resolution_tiers_and_ratios(self):
        tiers = smart_image_size.RESOLUTIONS["Qwen-Image-2.1"]

        self.assertEqual(list(tiers), list(QWEN_21_TIERS))
        self.assertTrue(all(len(items) == 19 for items in tiers.values()))
        self.assertTrue(
            all(
                item["width"] % 16 == 0 and item["height"] % 16 == 0
                for items in tiers.values()
                for item in items
            )
        )

        ratios = [item["ratio"] for item in next(iter(tiers.values()))]
        self.assertTrue(
            all([item["ratio"] for item in items] == ratios for items in tiers.values())
        )

    def test_qwen_image_21_square_sizes_and_resolution_outputs(self):
        node = smart_image_size.SmartImageSize()

        for tier, side in QWEN_21_TIERS.items():
            square = smart_image_size.RESOLUTIONS["Qwen-Image-2.1"][tier][0]
            self.assertEqual(square["ratio"], "1:1")
            self.assertEqual((square["width"], square["height"]), (side, side))
            self.assertEqual(smart_image_size.resolution_output(tier), str(side))
            self.assertEqual(
                node.get_resolution(
                    "Qwen-Image-2.1", tier, smart_image_size.dimension_text(square)
                ),
                (side, side, "1:1", str(side)),
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

    def test_existing_k_resolution_outputs_are_unchanged(self):
        self.assertEqual(smart_image_size.resolution_output("1K"), "1024")
        self.assertEqual(smart_image_size.resolution_output("1.5K"), "1536")
        self.assertEqual(smart_image_size.resolution_output("2K"), "2048")
