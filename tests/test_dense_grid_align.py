import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from image_sizing_and_resizing.dense_grid_align import (
    _constrain_mesh, _interpolate_mesh, estimate_deformation, warp_image,
)


def image_fixture():
    rng = np.random.default_rng(21)
    image = cv2.GaussianBlur(rng.random((384, 512, 3), dtype=np.float32), (0, 0), 1.5)
    for _ in range(45):
        x, y = rng.integers([5, 5], [500, 375])
        cv2.circle(image, (int(x), int(y)), int(rng.integers(3, 17)), tuple(map(float, rng.random(3))), 2)
    letters = np.zeros(image.shape[:2], np.uint8)
    cv2.putText(letters, 'FIXED   LOCAL   FIXED', (15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    image[letters > 0] = (0.9, 0.1, 0.2)
    return image


def local_shift(reference):
    yy, xx = np.mgrid[:384, :512].astype(np.float32)
    bump = np.clip(1 - ((xx - 256) / 125) ** 2 - ((yy - 190) / 120) ** 2, 0, 1) ** 2
    edited = cv2.remap(reference, xx - 15 * bump, yy + 9 * bump, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return edited, bump


class DenseGridAlignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(2)

    def test_local_shift_is_corrected_and_unchanged_pixels_are_exact(self):
        reference = image_fixture()
        edited, bump = local_shift(reference)
        field, _, _, _ = estimate_deformation(reference, edited, max_shift=32, grid_spacing=8)
        result = warp_image(edited, field)
        moving = bump > 0.2
        before = np.abs(edited - reference)[moving].mean()
        after = np.abs(result - reference)[moving].mean()
        self.assertLess(after, before * 0.3)
        np.testing.assert_array_equal(result[bump == 0], edited[bump == 0])

    def test_identical_input_is_a_bit_exact_noop(self):
        reference = image_fixture()
        field, _, pinned, stats = estimate_deformation(reference, reference.copy())
        np.testing.assert_array_equal(field, np.zeros_like(field))
        np.testing.assert_array_equal(warp_image(reference, field), reference)
        self.assertTrue(pinned.all())
        self.assertEqual(stats['max_displacement'], 0)

    def test_local_stretch_is_corrected_without_a_global_transform(self):
        reference = image_fixture()
        yy, xx = np.mgrid[:384, :512].astype(np.float32)
        bump = np.clip(1 - ((xx - 256) / 150) ** 2 - ((yy - 190) / 140) ** 2, 0, 1) ** 2
        edited = cv2.remap(reference, xx - 0.35 * (xx - 256) * bump, yy + 0.2 * (yy - 190) * bump,
                           cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        field, _, _, _ = estimate_deformation(reference, edited, grid_spacing=8, max_shift=32)
        result = warp_image(edited, field)
        region = bump > 0.2
        self.assertLess(np.abs(result - reference)[region].mean(), np.abs(edited - reference)[region].mean() * 0.5)
        np.testing.assert_array_equal(result[bump == 0], edited[bump == 0])

    def test_color_change_does_not_trigger_geometric_correction(self):
        reference = image_fixture()
        edited = reference * 0.7 + 0.15
        field, _, _, _ = estimate_deformation(reference, edited)
        self.assertEqual(float(np.abs(field).max()), 0)

    def test_protected_pixels_remain_exact_even_inside_a_shifted_region(self):
        reference = image_fixture()
        edited, _ = local_shift(reference)
        mask = np.zeros(reference.shape[:2], np.float32)
        mask[160:210, 230:280] = 1
        field, _, _, _ = estimate_deformation(reference, edited, grid_spacing=8, protect_mask=mask)
        result = warp_image(edited, field)
        np.testing.assert_array_equal(result[mask == 1], edited[mask == 1])
        self.assertGreater(float(np.linalg.norm(field, axis=-1).max()), 2)

    def test_global_translation_moves_in_the_correct_sampling_direction(self):
        reference = image_fixture()
        yy, xx = np.mgrid[:384, :512].astype(np.float32)
        edited = cv2.remap(reference, xx - 12, yy + 6, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        field, _, _, _ = estimate_deformation(reference, edited, grid_spacing=16, max_shift=32)
        center = field[40:-40, 40:-40]
        self.assertLess(float(np.median(np.linalg.norm(center - [12, -6], axis=-1))), 1.0)

    def test_shift_limit_is_respected(self):
        reference = image_fixture()
        edited, _ = local_shift(reference)
        field, _, _, _ = estimate_deformation(reference, edited, max_shift=4)
        self.assertLessEqual(float(np.linalg.norm(field, axis=-1).max()), 4.0001)

    def test_constrained_mesh_cannot_fold_and_keeps_hard_anchors(self):
        rng = np.random.default_rng(19)
        mesh = rng.normal(0, 80, (25, 33, 2)).astype(np.float32)
        pinned = np.zeros((25, 33), bool)
        pinned[:, :5] = True
        mesh[pinned] = 0
        constrained, _ = _constrain_mesh(mesh, pinned, 8, 8)
        np.testing.assert_array_equal(constrained[pinned], 0)
        field = _interpolate_mesh(constrained, 257, 193)
        dx = np.diff(field, axis=1)[:-1]
        dy = np.diff(field, axis=0)[:, :-1]
        determinant = (1 + dx[..., 0]) * (1 + dy[..., 1]) - dx[..., 1] * dy[..., 0]
        self.assertGreater(float(determinant.min()), 0.1)

    def test_subpixel_rgba_warp_moves_alpha_with_rgb(self):
        image = image_fixture()
        alpha = np.linspace(0, 1, 512, dtype=np.float32)[None, :].repeat(384, axis=0)
        rgba = np.dstack([image, alpha])
        field = np.zeros((384, 512, 2), np.float32)
        field[..., 0] = 2.5
        result = warp_image(rgba, field)
        expected = cv2.remap(alpha, np.mgrid[:384, :512][1].astype(np.float32) + 2.5,
                             np.mgrid[:384, :512][0].astype(np.float32), cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
        np.testing.assert_allclose(result[..., 3], np.clip(expected, 0, 1), atol=1e-6)
        self.assertEqual(result.shape[-1], 4)

    def test_small_images_are_left_unchanged(self):
        image = np.full((1, 7, 3), 0.5, np.float32)
        field, _, _, _ = estimate_deformation(image, image * 0.8)
        np.testing.assert_array_equal(warp_image(image, field), image)


if __name__ == '__main__':
    unittest.main()
