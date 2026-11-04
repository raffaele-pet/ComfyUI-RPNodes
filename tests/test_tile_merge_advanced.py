import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

try:
    import comfy_api.latest
except ModuleNotFoundError:
    raise unittest.SkipTest('Advanced merge integration tests require the ComfyUI Python environment')

import cv2
import execution
import nodes
from comfy_extras.nodes_preview_any import PreviewAny

sys.path.insert(0, str(Path(__file__).parents[1]))
from image_sizing_and_resizing.tile_merge_advanced import (
    RPTileMergeAdvanced, merge_advanced, minimum_seam, seam_composite,
)
from image_sizing_and_resizing.tiled_image import RPTileSplit, merge_tiles, split_tiles


def two_tiles_fixture():
    rng = np.random.default_rng(17)
    reference = cv2.GaussianBlur(rng.random((1024, 1792, 3), dtype=np.float32) * 0.12 + 0.14, (0, 0), 2)
    reference[509:515] = 0.95
    for x in range(50, 1792, 110):
        cv2.circle(reference, (x, 390 + (x % 170)), 9, (0.7, 0.3, 0.1), 2)
    context = dict(rows=1, columns=2, overlap=256, canvas_size=(1792, 1024), content_size=(1792, 1024),
                   padding=(0, 0, 0, 0), positions=[(0, 0), (768, 0)], original_size=(1792, 1024),
                   batch_size=1, reference=torch.from_numpy(reference)[None])
    yy, xx = np.mgrid[:1024, :1024].astype(np.float32)
    first = reference[:, :1024].copy()
    second = cv2.remap(reference[:, 768:].copy(), xx - 8, yy - 18,
                       cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return [torch.from_numpy(first)[None], torch.from_numpy(second)[None]], context


class AdvancedMergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(2)
        torch.set_num_threads(2)

    def test_minimum_seam_follows_known_low_cost_path(self):
        cost = np.full((20, 15), 10, np.float32)
        expected = np.minimum(12, 2 + np.arange(20) // 2)
        cost[np.arange(20), expected] = 0
        np.testing.assert_array_equal(minimum_seam(cost), expected)

    def test_hard_seam_never_creates_a_double_exposure(self):
        old = np.full((64, 96, 3), [0.9, 0.1, 0.1], np.float32)
        new = np.full((64, 96, 3), [0.1, 0.1, 0.9], np.float32)
        covered = np.zeros((64, 96), bool)
        covered[:, :48] = True
        result, alpha = seam_composite(old, new, old, covered, 48, 0, 0)
        self.assertTrue(np.all((alpha == 0) | (alpha == 1)))
        self.assertTrue(np.all(np.all(result == old, axis=-1) | np.all(result == new, axis=-1)))
        np.testing.assert_array_equal(result[:, 48:], new[:, 48:])

    def test_optional_feather_is_confined_to_the_selected_seam(self):
        old = np.zeros((64, 256, 3), np.float32)
        new = np.ones_like(old)
        covered = np.ones((64, 256), bool)
        _, alpha = seam_composite(old, new, old, covered, 128, 0, 3)
        self.assertLess(np.mean((alpha > 0) & (alpha < 1)), 0.04)

    def test_identity_preserves_geometry_padding_and_batch_order(self):
        source = torch.cat([torch.full((1, 31, 43, 3), 0.2), torch.full((1, 31, 43, 3), 0.8)])
        tiles, context, reference, _ = split_tiles(source, 4, 'nearest-exact')
        result, aligned, grid, correction, seams, report = merge_advanced(
            tiles, context, tile_color_strength=0, global_color_strength=0)
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        self.assertEqual(len(aligned), 8)
        self.assertEqual(grid.shape[0], 2)
        self.assertEqual(correction.max().item(), 0)
        self.assertEqual(tuple(seams.shape), (2, reference.shape[1], reference.shape[2]))
        self.assertIn('Image 2:', report)

    def test_grid_alignment_before_blending_reduces_injected_shift(self):
        tiles, context = two_tiles_fixture()
        ordinary = merge_tiles(tiles, context, 'reinhard_lab', 0, 0)
        advanced, _, _, correction, _, _ = merge_advanced(
            tiles, context, tile_color_strength=0, global_color_strength=0, max_shift=40, grid_spacing=16)
        ref = context['reference']
        roi = (slice(None), slice(450, 580), slice(800, 990), slice(None))
        before = (ordinary[roi] - ref[roi]).abs().mean().item()
        after = (advanced[roi] - ref[roi]).abs().mean().item()
        self.assertLess(after, before * 0.4)
        self.assertGreater(correction.max().item(), 0)

    def test_rgba_tiles_are_accepted_and_alpha_does_not_drive_rgb_alignment(self):
        tiles, context, reference, _ = split_tiles(torch.rand((1, 23, 35, 3)), 4, 'bilinear')
        rgba = [torch.cat([tile, torch.zeros_like(tile[..., :1])], dim=-1) for tile in tiles]
        result, aligned, _, _, _, _ = merge_advanced(rgba, context, tile_color_strength=0, global_color_strength=0)
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        self.assertTrue(all(tile.shape[-1] == 4 for tile in aligned))

    def test_color_match_keeps_padding_out_of_statistics(self):
        tiles, context, reference, _ = split_tiles(torch.rand((1, 17, 43, 3)), 4, 'bilinear')
        calls = []

        def match(target, ref, method, stats, strength):
            calls.append((tuple(target.shape), tuple(ref.shape), strength))
            return (target,)

        with patch('image_sizing_and_resizing.tile_merge_advanced.ColorTransfer.execute', side_effect=match):
            result, _, _, _, _, _ = merge_advanced(tiles, context)
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        self.assertTrue(all(a == b for a, b, _ in calls))
        self.assertEqual(calls[-1], (tuple(reference.shape), tuple(reference.shape), 0.25))
        self.assertTrue(any(a[1] < 1024 or a[2] < 1024 for a, _, _ in calls[:-1]))

    def test_wrong_count_and_size_are_rejected(self):
        tiles, context = two_tiles_fixture()
        with self.assertRaisesRegex(ValueError, 'expected 2 tiles'):
            merge_advanced(tiles[:-1], context)
        tiles[0] = tiles[0][:, :1000]
        with self.assertRaisesRegex(ValueError, '1024 x 1024'):
            merge_advanced(tiles, context)

    def test_real_executor_collects_tiles_once_and_report_slot_validates(self):
        tiles, context, _, _ = split_tiles(torch.full((1, 19, 31, 3), 0.4), 4, 'nearest-exact')
        inputs = dict(tiles=tiles, tile_context=[context], color_method=['reinhard_lab'],
                      tile_color_strength=[0], global_color_strength=[0], alignment_strength=[1],
                      grid_spacing=[16], max_shift=[96], anchor_tolerance=[1], seam_feather=[0],
                      rigidity=[0.5], minimum_match=[0.55])
        results = asyncio.run(execution._async_map_node_over_list(
            'rp-advanced', 'merge', RPTileMergeAdvanced, inputs, 'execute', v3_data={}))
        output, _, _ = execution.get_output_from_returns(results, RPTileMergeAdvanced)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(output[0]), 1)
        self.assertEqual(len(output[1]), 4)
        self.assertEqual(list(RPTileMergeAdvanced.OUTPUT_IS_LIST), [False, True, False, False, False, False])
        RPTileMergeAdvanced.VALIDATE_CLASS()
        settings = {key: value[0] for key, value in inputs.items() if key not in ('tiles', 'tile_context')}
        prompt = {
            'source': {'class_type': 'EmptyImage', 'inputs': {'width': 64, 'height': 64, 'batch_size': 1, 'color': 0}},
            'split': {'class_type': 'RPTileSplit', 'inputs': {'image': ['source', 0], 'tile_count': '4', 'upscale_method': 'lanczos'}},
            'merge': {'class_type': 'RPTileMergeAdvanced', 'inputs': {**settings, 'tiles': ['split', 0], 'tile_context': ['split', 1]}},
            'report': {'class_type': 'PreviewAny', 'inputs': {'source': ['merge', 5]}},
        }
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, {'RPTileSplit': RPTileSplit,
                                                   'RPTileMergeAdvanced': RPTileMergeAdvanced, 'PreviewAny': PreviewAny}):
            validation = asyncio.run(execution.validate_prompt('rp-advanced-validation', prompt, None))
        self.assertTrue(validation[0], validation)
