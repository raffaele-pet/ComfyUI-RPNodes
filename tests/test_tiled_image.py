import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

try:
    import comfy_api.latest
except ModuleNotFoundError:
    raise unittest.SkipTest("Tile node integration tests require the ComfyUI Python environment")

import execution
import nodes
from comfy_extras.nodes_preview_any import PreviewAny

from image_sizing_and_resizing.tiled_image import (
    RPTileMerge, RPTileSplit, TILE_COUNTS, merge_tiles, split_tiles, tile_layout,
)


class TiledImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_every_preset_handles_portrait_landscape_and_extreme_sizes(self):
        for preset in TILE_COUNTS:
            for width, height in [(1239, 929), (929, 1239), (1, 1), (1, 3000), (3000, 1), (10007, 6007)]:
                with self.subTest(preset=preset, size=(width, height)):
                    plan = tile_layout(width, height, int(preset))
                    self.assertEqual(plan["rows"] * plan["columns"], int(preset))
                    self.assertEqual(len(plan["positions"]), int(preset))
                    self.assertGreaterEqual(plan["overlap"], 192)
                    self.assertLessEqual(plan["overlap"], 288)
                    cw, ch = plan["canvas_size"]
                    sw, sh = plan["content_size"]
                    left, top, right, bottom = plan["padding"]
                    self.assertEqual(left + sw + right, cw)
                    self.assertEqual(top + sh + bottom, ch)
                    self.assertTrue(sw == cw or sh == ch)
                    scale = min(cw / width, ch / height)
                    self.assertLessEqual(abs(sw - width * scale), 0.5)
                    self.assertLessEqual(abs(sh - height * scale), 1)
                    self.assertTrue(all(0 <= x <= cw - 1024 and 0 <= y <= ch - 1024 for x, y in plan["positions"]))

    def test_identity_reconstruction_removes_padding_without_seams(self):
        for width, height, preset in [(43, 29, 4), (29, 43, 6), (37, 37, 9), (47, 31, 12), (45, 17, 25)]:
            with self.subTest(preset=preset):
                source = torch.rand((1, height, width, 3))
                tiles, context, reference, description = split_tiles(source, preset, "bilinear")
                self.assertEqual(len(tiles), preset)
                self.assertTrue(all(tuple(tile.shape) == (1, 1024, 1024, 3) for tile in tiles))
                output = merge_tiles(tiles, context, "reinhard_lab", 0, 0)
                torch.testing.assert_close(output, reference, rtol=0, atol=3e-7)
                self.assertIn(f"{preset} tiles/image", description)

    def test_changed_padding_pixels_never_enter_reconstruction(self):
        source = torch.rand((1, 17, 41, 3))
        tiles, context, reference, _ = split_tiles(source, 4, "nearest-exact")
        left, top, _, _ = context["padding"]
        width, height = context["content_size"]
        for tile, (x, y) in zip(tiles, context["positions"]):
            xx = torch.arange(x, x + 1024)[None, :]
            yy = torch.arange(y, y + 1024)[:, None]
            outside = (xx < left) | (xx >= left + width) | (yy < top) | (yy >= top + height)
            tile[:, outside] = torch.tensor([1.0, 0.0, 1.0])
        torch.testing.assert_close(merge_tiles(tiles, context, "reinhard_lab", 0, 0), reference, rtol=0, atol=3e-7)

    def test_tile_order_batch_order_and_metadata_remain_aligned(self):
        source = torch.cat([torch.full((1, 11, 17, 3), 0.2), torch.full((1, 11, 17, 3), 0.8)])
        tiles, context, reference, _ = split_tiles(source, 6, "nearest-exact")
        self.assertEqual(len(tiles), 12)
        self.assertAlmostEqual(tiles[0].mean().item(), 0.2, places=5)
        self.assertAlmostEqual(tiles[6].mean().item(), 0.8, places=5)
        output = RPTileMerge.execute(tiles, [context], ["reinhard_lab"], [0], [0])[0]
        torch.testing.assert_close(output, reference, rtol=0, atol=3e-7)

    def test_merge_accepts_a_decoded_image_batch(self):
        tiles, context, reference, _ = split_tiles(torch.rand((1, 21, 35, 3)), 4, "bilinear")
        output = merge_tiles([torch.cat(tiles)], context, "reinhard_lab", 0, 0)
        torch.testing.assert_close(output, reference, rtol=0, atol=3e-7)

    def test_default_lanczos_accepts_one_pixel_and_downsamples_large_inputs(self):
        for source in [torch.full((1, 1, 1, 3), 0.5), torch.rand((1, 50, 5000, 3))]:
            with self.subTest(size=tuple(source.shape)):
                tiles, context, reference, _ = RPTileSplit.execute(source, "4").result
                self.assertTrue(all(tuple(tile.shape) == (1, 1024, 1024, 3) for tile in tiles))
                self.assertEqual(reference.shape[2], context["content_size"][0])
                if source.shape[2] == 5000:
                    self.assertLess(reference.shape[2], 5000)
                torch.testing.assert_close(merge_tiles(tiles, context, "reinhard_lab", 0, 0), reference, rtol=0, atol=3e-7)

    def test_color_matching_uses_only_real_pixels_and_runs_global_last(self):
        tiles, context, reference, _ = split_tiles(torch.rand((1, 19, 43, 3)), 4, "bilinear")
        calls = []

        def match(target, ref, method, stats, strength):
            calls.append((tuple(target.shape), tuple(ref.shape), method, strength))
            return (target,)

        with patch("image_sizing_and_resizing.tiled_image.ColorTransfer.execute", side_effect=match):
            output = merge_tiles(tiles, context, "reinhard_lab", 0.75, 0.25)
        torch.testing.assert_close(output, reference, rtol=0, atol=3e-7)
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(target == ref for target, ref, _, _ in calls))
        left, top, _, _ = context["padding"]
        width, height = context["content_size"]
        expected = [(1, min(y + 1024, top + height) - max(y, top),
                     min(x + 1024, left + width) - max(x, left), 3)
                    for x, y in context["positions"]]
        self.assertEqual([shape for shape, _, _, _ in calls[:-1]], expected)
        self.assertTrue(any(shape[1] < 1024 or shape[2] < 1024 for shape in expected))
        self.assertEqual(calls[-1], (tuple(reference.shape), tuple(reference.shape), "reinhard_lab", 0.25))

    def test_native_color_match_reduces_shift(self):
        source = torch.full((1, 13, 19, 3), 0.25)
        tiles, context, reference, _ = split_tiles(source, 4, "nearest-exact")
        shifted = [tile + 0.15 for tile in tiles]
        with patch("comfy.model_management.get_torch_device", return_value=torch.device("cpu")):
            result = merge_tiles(shifted, context, "reinhard_lab", 1.0, 0)
        self.assertLess((result - reference).abs().mean().item(), 0.005)

    def test_wrong_count_and_wrong_dimensions_fail_before_silent_misassembly(self):
        tiles, context, _, _ = split_tiles(torch.zeros((1, 13, 17, 3)), 4, "nearest-exact")
        with self.assertRaisesRegex(ValueError, "expected 4 tiles"):
            merge_tiles(tiles[:-1], context, "reinhard_lab", 0, 0)
        tiles[0] = tiles[0][:, :1000]
        with self.assertRaisesRegex(ValueError, "1024 x 1024 RGB"):
            merge_tiles(tiles, context, "reinhard_lab", 0, 0)

    def test_v3_list_contracts(self):
        self.assertEqual(list(RPTileSplit.OUTPUT_IS_LIST), [True, False, False, False])
        self.assertTrue(RPTileMerge.INPUT_IS_LIST)
        self.assertEqual(RPTileSplit.GET_SCHEMA().category, "RP/image_sizing_and_resizing")
        RPTileSplit.VALIDATE_CLASS()
        RPTileMerge.VALIDATE_CLASS()

    def test_real_comfy_executor_maps_tiles_and_collects_merge_once(self):
        async def run():
            source = torch.rand((1, 19, 31, 3))
            split_results = await execution._async_map_node_over_list(
                "rp-test", "split", RPTileSplit,
                {"image": [source], "tile_count": ["6"], "upscale_method": ["bilinear"]}, "execute", v3_data={})
            split_data, _, _ = execution.get_output_from_returns(split_results, RPTileSplit)
            self.assertEqual(len(split_data[0]), 6)
            self.assertEqual(len(split_data[1]), 1)
            edit_results = await execution._async_map_node_over_list(
                "rp-test", "edit", nodes.ImageInvert(), {"image": split_data[0]}, nodes.ImageInvert.FUNCTION)
            edit_data, _, _ = execution.get_output_from_returns(edit_results, nodes.ImageInvert)
            self.assertEqual(len(edit_data[0]), 6)
            merge_results = await execution._async_map_node_over_list(
                "rp-test", "merge", RPTileMerge,
                {"tiles": edit_data[0], "tile_context": split_data[1], "color_method": ["reinhard_lab"],
                 "tile_color_strength": [0], "global_color_strength": [0]}, "execute", v3_data={})
            merge_data, _, _ = execution.get_output_from_returns(merge_results, RPTileMerge)
            self.assertEqual(len(merge_results), 1)
            self.assertEqual(len(merge_data[0]), 1)
            torch.testing.assert_close(merge_data[0][0], 1 - split_data[2][0], rtol=0, atol=3e-7)

        asyncio.run(run())

    def test_layout_connection_passes_real_prompt_validation_at_its_new_slot(self):
        prompt = {
            "source": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0}},
            "2300": {"class_type": "RPTileSplit", "inputs": {"image": ["source", 0], "tile_count": "12", "upscale_method": "lanczos"}},
            "2303": {"class_type": "PreviewAny", "inputs": {"source": ["2300", 4]}},
        }
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, {"RPTileSplit": RPTileSplit, "PreviewAny": PreviewAny}):
            with patch("execution.logging.error"):
                stale = asyncio.run(execution.validate_prompt("rp-old-layout", prompt, None))
            self.assertFalse(stale[0])
            self.assertIn("list index out of range", str(stale))
            prompt["2303"]["inputs"]["source"][1] = 3
            repaired = asyncio.run(execution.validate_prompt("rp-new-layout", prompt, None))
            self.assertTrue(repaired[0], repaired)
            self.assertIn("2303", repaired[2])


if __name__ == "__main__":
    unittest.main()
