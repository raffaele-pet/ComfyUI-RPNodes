import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


spec = importlib.util.spec_from_file_location(
    "rp_face_detection_test", Path(__file__).parents[1] / "face_detection" / "face_detection.py"
)
face_detection = importlib.util.module_from_spec(spec)
registry = types.SimpleNamespace(NODE_CLASS_MAPPINGS={})
with patch.dict(sys.modules, {"nodes": registry}):
    spec.loader.exec_module(face_detection)


def rectangle(left=30, top=20, width=20, height=10, image_width=100, image_height=80):
    mask = torch.zeros((1, image_height, image_width))
    mask[:, top:top + height, left:left + width] = 1
    return mask


def bounds(mask):
    points = torch.nonzero(mask[0])
    top, left = points.amin(0).tolist()
    bottom, right = points.amax(0).tolist()
    return left, top, right - left + 1, bottom - top + 1


class FaceMaskTests(unittest.TestCase):
    def test_defaults_keep_the_exact_upstream_mask(self):
        mask = rectangle()
        self.assertIs(face_detection.resize_bbox_mask(mask), mask)

    def test_pixel_sizes_keep_center_and_scale_after_overrides(self):
        self.assertEqual(bounds(face_detection.resize_bbox_mask(rectangle(), 40, 30)), (20, 10, 40, 30))
        self.assertEqual(bounds(face_detection.resize_bbox_mask(rectangle(), 40, 0, 1.5)), (10, 18, 60, 15))

    def test_fractional_scale_and_square_use_original_long_side(self):
        self.assertEqual(bounds(face_detection.resize_bbox_mask(rectangle(), scale=1.1))[2:], (22, 11))
        self.assertEqual(bounds(face_detection.resize_bbox_mask(rectangle(), square=True, scale=1.5)), (25, 10, 30, 30))

    def test_square_size_override_keeps_equal_sides(self):
        self.assertEqual(bounds(face_detection.resize_bbox_mask(rectangle(), width=40, square=True))[2:], (40, 40))
        self.assertEqual(bounds(face_detection.resize_bbox_mask(rectangle(), width=8, square=True))[2:], (8, 8))

    def test_border_boxes_shift_to_preserve_size_and_square(self):
        mask = rectangle(left=0, top=0)
        self.assertEqual(bounds(face_detection.resize_bbox_mask(mask, scale=2, square=True)), (0, 0, 40, 40))
        self.assertEqual(bounds(face_detection.resize_bbox_mask(mask, scale=100, square=True)), (0, 0, 80, 80))

    def test_shrink_and_empty_mask(self):
        self.assertEqual(bounds(face_detection.resize_bbox_mask(rectangle(), scale=0.01))[2:], (1, 1))
        empty = torch.zeros((1, 80, 100))
        self.assertIs(face_detection.resize_bbox_mask(empty, scale=2), empty)


class FacePipelineTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        calls = self.calls

        class Provider:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"model_name": (["bbox/other.pt", face_detection.DEFAULT_MODEL],)}}

            def doit(self, model):
                calls.append(("model", model))
                return "detector", None

        class Detector:
            def doit(self, detector, image, threshold, dilation, crop_factor, drop_size, labels):
                calls.append(("detect", len(image), threshold, dilation, crop_factor, drop_size, labels))
                return (image,)

        class Masks:
            def doit(self, segs):
                return ([rectangle(), rectangle(left=60)],)

        class Crop:
            def crop(self, mask, frame, multiplier, alpha):
                calls.append(("crop", bounds(mask), multiplier, alpha))
                left, top, width, height = bounds(mask)
                return frame, frame[:, top:top + height, left:left + width], [(left, top, width, height)], width, height

        self.mapping = dict(zip(face_detection.DEPENDENCIES, (Provider, Detector, Masks, Crop)))
        self.patcher = patch.object(registry, "NODE_CLASS_MAPPINGS", self.mapping)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_default_workflow_settings_and_face_lists(self):
        image = torch.rand((2, 80, 100, 3))
        output = face_detection.RPFaceDetection().detect(image)
        self.assertEqual(tuple(map(len, output)), (4,) * 6)
        self.assertEqual(self.calls[1], ("detect", 1, 0.5, 10, 1.5, 10, "all"))
        self.assertEqual(self.calls[2], ("crop", (30, 20, 20, 10), 3.0, 0.5))
        torch.testing.assert_close(output[0][0], image[:1])
        torch.testing.assert_close(output[0][2], image[1:2])
        self.assertEqual(output[2][0], [(30, 20, 20, 10)])

    def test_resized_masks_are_used_for_both_crop_and_mask_output(self):
        output = face_detection.RPFaceDetection().detect(torch.rand((1, 80, 100, 3)), square=True, scale=1.5)
        self.assertEqual(bounds(output[5][0])[2:], (30, 30))
        self.assertEqual(output[2][0][0], bounds(output[5][0]))

    def test_tiny_masks_and_zero_smoothing_produce_valid_crops(self):
        image = torch.rand((1, 80, 100, 3))
        for settings in ({"width": 1, "height": 1}, {"scale": 0.01}, {"bbox_smooth_alpha": 0}):
            with self.subTest(settings=settings):
                result = face_detection.RPFaceDetection().detect(image, **settings)
                self.assertEqual(result[1][0].shape, (1, 1, 1, 3))
                self.assertEqual(result[2][0][0][2:], (1, 1))

    def test_hidden_advanced_values_are_still_used(self):
        face_detection.RPFaceDetection().detect(torch.rand((1, 80, 100, 3)), show_advanced=False, threshold=0.3, crop_size_mult=2)
        self.assertEqual(self.calls[1][2], 0.3)
        self.assertEqual(self.calls[2][2], 2)

    def test_default_model_is_preferred_and_missing_dependency_is_actionable(self):
        self.assertEqual(face_detection.RPFaceDetection.INPUT_TYPES()["required"]["model_name"][1]["default"], face_detection.DEFAULT_MODEL)
        del self.mapping["BatchCropFromMask"]
        with self.assertRaisesRegex(RuntimeError, "KJNodes"):
            face_detection.RPFaceDetection().detect(torch.rand((1, 80, 100, 3)))

    def test_no_detection_has_clear_error(self):
        self.mapping["ImpactSEGSToMaskList"] = lambda: types.SimpleNamespace(doit=lambda segs: ([torch.zeros((1, 80, 100))],))
        with self.assertRaisesRegex(ValueError, "no face detected in image 1"):
            face_detection.RPFaceDetection().detect(torch.rand((1, 80, 100, 3)))


if __name__ == "__main__":
    unittest.main()
