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

    def test_border_masks_are_clipped_without_shifting_original_center(self):
        mask = rectangle(left=0, top=0)
        self.assertEqual(bounds(face_detection.resize_bbox_mask(mask, scale=2, square=True)), (0, 0, 30, 25))
        self.assertEqual(bounds(face_detection.resize_bbox_mask(mask, scale=100, square=True)), (0, 0, 100, 80))
        self.assertEqual(face_detection.bbox_geometry(mask, scale=2, square=True), (9.5, 4.5, 40, 40))

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

        self.mapping = dict(zip(face_detection.DEPENDENCIES, (Provider, Detector, Masks)))
        self.patcher = patch.object(registry, "NODE_CLASS_MAPPINGS", self.mapping)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_default_workflow_settings_and_face_lists(self):
        image = torch.rand((2, 80, 100, 3))
        output = face_detection.RPFaceDetection().detect(image)
        self.assertEqual(tuple(map(len, output)), (4,) * 6)
        self.assertEqual(self.calls[1], ("detect", 1, 0.5, 10, 1.5, 10, "all"))
        torch.testing.assert_close(output[0][0], image[:1])
        torch.testing.assert_close(output[0][2], image[1:2])
        self.assertEqual(output[2][0], [(25, 18, 30, 12)])
        torch.testing.assert_close(output[1][0], image[:1, 18:30, 25:55], rtol=0, atol=0)

    def test_resized_masks_are_used_for_both_crop_and_mask_output(self):
        output = face_detection.RPFaceDetection().detect(torch.rand((1, 80, 100, 3)), square=True, scale=1.5)
        self.assertEqual(bounds(output[5][0])[2:], (30, 30))
        self.assertEqual(output[2][0][0], (18, 2, 45, 45))

    def test_tiny_masks_and_zero_smoothing_produce_valid_crops(self):
        image = torch.rand((1, 80, 100, 3))
        for settings in ({"width": 1, "height": 1}, {"scale": 0.01}, {"bbox_smooth_alpha": 0}):
            with self.subTest(settings=settings):
                result = face_detection.RPFaceDetection().detect(image, **settings)
                self.assertEqual(result[1][0].shape, (1, 2, 2, 3))
                self.assertEqual(result[2][0][0][2:], (2, 2))

    def test_hidden_advanced_values_are_still_used(self):
        result = face_detection.RPFaceDetection().detect(torch.rand((1, 80, 100, 3)), show_advanced=False, threshold=0.3, crop_size_mult=2)
        self.assertEqual(self.calls[1][2], 0.3)
        self.assertEqual(result[3][0], 20)
        self.assertEqual(result[4][0], 8)

    def test_default_model_is_preferred_and_missing_dependency_is_actionable(self):
        self.assertEqual(face_detection.RPFaceDetection.INPUT_TYPES()["required"]["model_name"][1]["default"], face_detection.DEFAULT_MODEL)
        del self.mapping["BboxDetectorSEGS"]
        with self.assertRaisesRegex(RuntimeError, "Impact Pack"):
            face_detection.RPFaceDetection().detect(torch.rand((1, 80, 100, 3)))

    def test_no_detection_has_clear_error(self):
        self.mapping["ImpactSEGSToMaskList"] = lambda: types.SimpleNamespace(doit=lambda segs: ([torch.zeros((1, 80, 100))],))
        with self.assertRaisesRegex(ValueError, "no face detected in image 1"):
            face_detection.RPFaceDetection().detect(torch.rand((1, 80, 100, 3)))


class FaceCropAlignmentTests(unittest.TestCase):
    def test_crop_uncrop_preserves_every_pixel_at_all_edges_and_scales(self):
        image = torch.rand((1, 80, 100, 3))
        for left, top in ((30, 20), (0, 0), (80, 0), (0, 70), (80, 70), (0, 30), (40, 0)):
            mask = rectangle(left=left, top=top)
            for scale in (0.01, 0.5, 1, 1.1, 1.5, 2, 3, 10, 100):
                for square in (False, True):
                    with self.subTest(left=left, top=top, scale=scale, square=square):
                        _, cropped, boxes, width, height = face_detection.crop_face(
                            mask, image, 3.0, 0.5, scale=scale, square=square
                        )
                        x, y, w, h = boxes[0]
                        self.assertEqual(cropped.shape, (1, height, width, 3))
                        self.assertEqual((w, h), (width, height))
                        restored = image.clone()
                        restored[:, y:y + h, x:x + w] = cropped
                        torch.testing.assert_close(restored, image, rtol=0, atol=0)

    def test_manual_sizes_and_odd_crop_sizes_preserve_pixels(self):
        image = torch.rand((1, 80, 100, 3))
        mask = rectangle(left=0, top=0)
        for width, height, multiplier, alpha in ((128, 192, 3, 0.5), (13, 17, 1.1, 0.7), (1, 50, 3, 0.5), (500, 5, 2, 0.9)):
            with self.subTest(width=width, height=height):
                _, cropped, boxes, _, _ = face_detection.crop_face(mask, image, multiplier, alpha, width, height)
                x, y, w, h = boxes[0]
                torch.testing.assert_close(cropped, image[:, y:y + h, x:x + w], rtol=0, atol=0)

    def test_clipped_mask_does_not_recenter_crop(self):
        image = torch.rand((1, 80, 100, 3))
        mask = rectangle(left=0, top=0)
        _, _, boxes, _, _ = face_detection.crop_face(mask, image, 1.0, 1.0, scale=2, square=True)
        self.assertEqual(boxes, [(0, 0, 29, 23)])

    def test_scale_is_applied_to_original_crop_size(self):
        image = torch.rand((1, 80, 100, 3))
        mask = rectangle(width=14, height=18)
        base = face_detection.crop_face(mask, image, 3, 0.5)
        scaled = face_detection.crop_face(mask, image, 3, 0.5, scale=2)
        self.assertEqual(scaled[3:5], tuple(size * 2 for size in base[3:5]))

    def test_tiny_border_crop_keeps_rgb_shape_for_batch_uncrop(self):
        mask = rectangle(left=0, top=0, width=1, height=1)
        image = torch.rand((1, 80, 100, 3))
        _, cropped, boxes, _, _ = face_detection.crop_face(mask, image, 3, 0.5, scale=0.01)
        self.assertEqual(boxes, [(0, 0, 2, 2)])
        torch.testing.assert_close(cropped, image[:, :2, :2], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
