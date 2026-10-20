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
    "0.25 MP ~ 0.5K (512 × 512)": 512,
    "0.5625 MP ~ 0.75K (768 × 768)": 768,
    "1 MP ~ 1K (1024 × 1024)": 1024,
    "1.5 MP (1248 × 1248)": 1248,
    "1.56 MP ~ 1.25K (1280 × 1280)": 1280,
    "1.68 MP (1328 × 1328)": 1328,
    "2 MP (1440 × 1440)": 1440,
    "2 MP (1456 × 1456)": 1456,
    "2.25 MP ~ 1.5K (1536 × 1536)": 1536,
    "3 MP (1760 × 1760)": 1760,
    "4 MP ~ 2K (2048 × 2048)": 2048,
    "16 MP ~ 4K (4096 × 4096)": 4096,
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
    def test_moss_official_resolution_classes_and_examples(self):
        tiers = smart_image_size.RESOLUTIONS["MOSS-Image (Base / SFT)"]
        self.assertEqual(list(tiers), [
            "0.25 MP ~ 0.5K (512 × 512)",
            "1 MP ~ 1K (1024 × 1024)", "1.5 MP (1248 × 1248)",
            "2 MP (1456 × 1456)",
            "4 MP ~ 2K (2048 × 2048)",
        ])
        for tier, square_side, portrait, widescreen in (
            ("1 MP ~ 1K (1024 × 1024)", 1024, (912, 1152), (1360, 768)),
            ("1.5 MP (1248 × 1248)", 1248, (1120, 1408), (1680, 944)),
            ("2 MP (1456 × 1456)", 1456, (1296, 1616), (1936, 1088)),
            ("0.25 MP ~ 0.5K (512 × 512)", 512, (464, 576), (688, 384)),
            ("4 MP ~ 2K (2048 × 2048)", 2048, (1824, 2288), (2736, 1536)),
        ):
            with self.subTest(tier=tier):
                items = {item["ratio"]: item for item in tiers[tier]}
                self.assertEqual(len(items), 27)
                self.assertEqual((items["1:1"]["width"], items["1:1"]["height"]),
                                 (square_side, square_side))
                self.assertEqual((items["4:5"]["width"], items["4:5"]["height"]), portrait)
                self.assertEqual((items["16:9"]["width"], items["16:9"]["height"]), widescreen)
                self.assertTrue({"1:8", "1:4", "1:3", "3:1", "4:1", "8:1"} <= items.keys())
                for item in items.values():
                    result = smart_image_size.SmartImageSize().get_resolution(
                        "MOSS-Image (Base / SFT)", tier, smart_image_size.dimension_text(item)
                    )
                    self.assertEqual(result[5], item["width"] * item["height"] / 1024**2)

    def test_flux3_klein_resolution_tiers_and_dimension_limits(self):
        tiers = smart_image_size.RESOLUTIONS["FLUX.3 Klein"]
        self.assertEqual(list(tiers), [
            "0.5625 MP ~ 0.75K (768 × 768)",
            "1 MP ~ 1K (1024 × 1024)",
            "2.25 MP ~ 1.5K (1536 × 1536)",
            "4 MP ~ 2K (2048 × 2048)",
            "16 MP ~ 4K (4096 × 4096)",
        ])
        ratios = [item["ratio"] for item in next(iter(tiers.values()))]
        self.assertEqual(len(ratios), 21)
        for tier, items in tiers.items():
            with self.subTest(tier=tier):
                self.assertEqual([item["ratio"] for item in items], ratios)
                for item in items:
                    for side in (item["width"], item["height"]):
                        self.assertEqual(side % 16, 0)
                        self.assertGreaterEqual(side, 256)
                        self.assertLessEqual(side, 6256)

    def test_flux3_klein_reference_budget_dimensions(self):
        tiers = smart_image_size.RESOLUTIONS["FLUX.3 Klein"]
        one_k = {item["ratio"]: (item["width"], item["height"])
                 for item in tiers["1 MP ~ 1K (1024 × 1024)"]}
        for ratio, expected in {
            "1:1": (1024, 1024), "4:3": (1168, 880), "3:4": (880, 1168),
            "3:2": (1248, 832), "2:3": (832, 1248),
            "16:9": (1360, 768), "9:16": (768, 1360),
            "4:5": (912, 1136),
        }.items():
            self.assertEqual(one_k[ratio], expected)

        four_k = {item["ratio"]: (item["width"], item["height"])
                  for item in tiers["16 MP ~ 4K (4096 × 4096)"]}
        self.assertEqual(four_k["1:1"], (4096, 4096))
        self.assertEqual(four_k["16:9"], (5456, 3072))
        self.assertEqual(four_k["21:9"], (6256, 2672))
        self.assertEqual(four_k["3:7"], (2672, 6256))
        self.assertEqual(smart_image_size.resolve_resolution(tiers, "4K"),
                         "16 MP ~ 4K (4096 × 4096)")

    def test_exact_aspect_ratio_output_uses_pixel_dimensions(self):
        node = smart_image_size.SmartImageSize()
        item = next(
            item for item in smart_image_size.RESOLUTIONS["Qwen-Image-2.1"][
                "1 MP ~ 1K (1024 × 1024)"
            ] if item["ratio"] == "5:3"
        )
        self.assertEqual(node.RETURN_NAMES, (
            "width", "height", "aspect_ratio_approx", "aspect_ratio", "resolution", "megapixels"
        ))
        self.assertEqual(
            node.get_resolution(
                "Qwen-Image-2.1", "1 MP ~ 1K (1024 × 1024)",
                smart_image_size.dimension_text(item),
            ),
            (1312, 800, "5:3", "41:25", 1024, 1.0009765625),
        )

    def test_flux3_megapixels_returns_reference_target_budget(self):
        tiers = smart_image_size.RESOLUTIONS["FLUX.3 Klein"]
        item = next(item for item in tiers["1 MP ~ 1K (1024 × 1024)"]
                    if item["ratio"] == "4:5")
        result = smart_image_size.SmartImageSize().get_resolution(
            "FLUX.3 Klein", "1 MP ~ 1K (1024 × 1024)",
            smart_image_size.dimension_text(item),
        )
        self.assertEqual(result, (912, 1136, "4:5", "57:71", 1024, 1.0))
        self.assertEqual(smart_image_size.SmartImageSize.RETURN_TYPES[5], "FLOAT")
        self.assertIsInstance(result[5], float)

    def test_all_flux3_presets_keep_canvas_after_reference_resize_and_vae_crop(self):
        node = smart_image_size.SmartImageSize()
        tiers = smart_image_size.RESOLUTIONS["FLUX.3 Klein"]
        for tier, items in tiers.items():
            expected_mp = smart_image_size.resolution_output(tier) ** 2 / 1024**2
            for item in items:
                with self.subTest(tier=tier, ratio=item["ratio"]):
                    width, height, *_, megapixels = node.get_resolution(
                        "FLUX.3 Klein", tier, smart_image_size.dimension_text(item)
                    )
                    self.assertEqual(megapixels, expected_mp)
                    scale = math.sqrt(megapixels * 1024 * 1024 / (width * height))
                    self.assertEqual(
                        (round(width * scale) // 16 * 16, round(height * scale) // 16 * 16),
                        (width, height),
                    )

    def test_old_flux3_labels_resolve_updated_presets(self):
        self.assertEqual(
            smart_image_size.SmartImageSize().get_resolution(
                "FLUX.3 Klein", "0.59 MP ~ 0.75K (768 × 768)",
                "4:5 (Portrait Standard) - 688 x 864",
            ),
            (672, 848, "4:5", "42:53", 768, 0.5625),
        )

    def test_catalog_dimensions_are_valid_and_distinct(self):
        for model, model_resolutions in smart_image_size.RESOLUTIONS.items():
            for tier, items in model_resolutions.items():
                with self.subTest(model=model, tier=tier):
                    self.assertTrue(items)
                    self.assertEqual(len(items), len({item["ratio"] for item in items}))
                    for item in items:
                        self.assertGreater(item["width"], 0)
                        self.assertGreater(item["height"], 0)
                        self.assertEqual(item["width"] % 16, 0)
                        self.assertEqual(item["height"] % 16, 0)
                        ratio_width, ratio_height = map(int, item["ratio"].split(":"))
                        actual_ratio = item["width"] / item["height"]
                        # OpenMOSS's 0.25M 1:8 preset rounds its 181-pixel side to 176.
                        tolerance = 0.04 if model == "MOSS-Image (Base / SFT)" else 0.03
                        self.assertLess(abs(math.log(actual_ratio / (ratio_width / ratio_height))), tolerance)

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
        self.assertTrue(all(len(items) == 21 for items in tiers.values()))
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

    def test_all_models_include_seven_nine_orientations(self):
        for model_resolutions in smart_image_size.RESOLUTIONS.values():
            for items in model_resolutions.values():
                by_ratio = {item["ratio"]: item for item in items}
                portrait = by_ratio["7:9"]
                landscape = by_ratio["9:7"]
                self.assertEqual(portrait["label"], "Portrait Classic")
                self.assertEqual(landscape["label"], "Landscape Classic")
                self.assertEqual(portrait["width"], landscape["height"])
                self.assertEqual(portrait["height"], landscape["width"])

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
                (side, side, "1:1", "1:1", side, side * side / (1024 * 1024)),
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
        self.assertEqual(smart_image_size.SmartImageSize.RETURN_TYPES[4], "INT")
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

    def test_manufacturer_published_presets(self):
        data = smart_image_size.RESOLUTIONS
        hidream = {item["ratio"]: (item["width"], item["height"])
                   for item in data["HiDream-O1-Image / Dev"]["4 MP ~ 2K (2048 × 2048)"]}
        self.assertEqual(hidream["4:3"], (2304, 1728))
        self.assertEqual(hidream["21:9"], (3104, 1312))

        krea = data["Krea 2"]
        self.assertEqual(set(krea), {"1 MP ~ 1K (1024 × 1024)",
                                     "4 MP ~ 2K (2048 × 2048)"})
        self.assertTrue(all(item["width"] <= 2048 and item["height"] <= 2048
                            for item in krea["4 MP ~ 2K (2048 × 2048)"]))
        krea_1k = {item["ratio"]: (item["width"], item["height"])
                   for item in krea["1 MP ~ 1K (1024 × 1024)"]}
        self.assertEqual(krea_1k["13:19"], (832, 1216))
        self.assertEqual(krea_1k["19:13"], (1216, 832))
        self.assertEqual(krea_1k["2:3"], (832, 1248))
        self.assertEqual(krea_1k["3:2"], (1248, 832))

        z_image = data["Z-Image-Turbo"]
        self.assertNotIn("4 MP ~ 2K (2048 × 2048)", z_image)
        for tier, ratio, expected in (
            ("1 MP ~ 1K (1024 × 1024)", "21:9", (1344, 576)),
            ("1.56 MP ~ 1.25K (1280 × 1280)", "16:9", (1536, 864)),
            ("2.25 MP ~ 1.5K (1536 × 1536)", "4:3", (1728, 1296)),
        ):
            by_ratio = {item["ratio"]: (item["width"], item["height"])
                        for item in z_image[tier]}
            self.assertEqual(by_ratio[ratio], expected)
