import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

try:
    import comfy_api.latest
except ModuleNotFoundError:
    raise unittest.SkipTest('Dense alignment node tests require the ComfyUI Python environment')

import execution
import nodes
from comfy_extras.nodes_preview_any import PreviewAny

sys.path.insert(0, str(Path(__file__).parents[1]))
from image_sizing_and_resizing.dense_grid_align_node import RPDenseGridAlign


class DenseGridAlignNodeTests(unittest.TestCase):
    def test_v3_schema_and_batched_rgba_outputs(self):
        RPDenseGridAlign.VALIDATE_CLASS()
        image = torch.rand((2, 24, 32, 4), dtype=torch.float32)
        result, preview, mask, report = RPDenseGridAlign.execute(image.clone(), image)
        torch.testing.assert_close(result, image, rtol=0, atol=0)
        self.assertEqual(preview.shape[0], 2)
        self.assertEqual(tuple(mask.shape), (2, 24, 32))
        self.assertEqual(mask.max().item(), 0)
        self.assertIn('Image 2:', report)

    def test_original_resolution_can_differ_and_strength_zero_preserves_edit(self):
        reference = torch.rand((1, 16, 24, 3))
        edited = torch.rand((1, 32, 48, 3))
        result, _, mask, _ = RPDenseGridAlign.execute(reference, edited, strength=0)
        torch.testing.assert_close(result, edited, rtol=0, atol=0)
        self.assertEqual(mask.max().item(), 0)

    def test_batch_mismatch_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, 'reference must contain'):
            RPDenseGridAlign.execute(torch.zeros((3, 16, 16, 3)), torch.zeros((2, 16, 16, 3)))

    def test_report_socket_passes_real_prompt_validation(self):
        prompt = {
            'source': {'class_type': 'EmptyImage', 'inputs': {'width': 64, 'height': 64, 'batch_size': 1, 'color': 0}},
            'align': {'class_type': 'RPDenseGridAlign', 'inputs': {
                'reference': ['source', 0], 'image': ['source', 0], 'strength': 1.0,
                'grid_spacing': 16, 'max_shift': 96.0, 'anchor_tolerance': 1.5,
                'rigidity': 0.5, 'analysis_max_side': 2048, 'minimum_match': 0.55,
            }},
            'preview': {'class_type': 'PreviewAny', 'inputs': {'source': ['align', 3]}},
        }
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, {'RPDenseGridAlign': RPDenseGridAlign, 'PreviewAny': PreviewAny}):
            result = asyncio.run(execution.validate_prompt('rp-dense-align', prompt, None))
        self.assertTrue(result[0], result)
        self.assertEqual(list(RPDenseGridAlign.RETURN_TYPES), ['IMAGE', 'IMAGE', 'MASK', 'STRING'])


if __name__ == '__main__':
    unittest.main()
