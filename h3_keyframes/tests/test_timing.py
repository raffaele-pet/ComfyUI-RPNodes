import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timing import shot_frame_positions


class ShotFramePositionTests(unittest.TestCase):
    def test_example_timestamps_convert_at_24_fps(self):
        prompt = (
            "[Shot 1] Opening. "
            "[Shot 2] At 00:03.750, continuation. "
            "[Shot 3] At 00:07.500, surprise. "
            "[Shot 4] At 00:11.250, retrieves a sign."
        )
        self.assertEqual(
            shot_frame_positions(prompt, [1, 2, 3, 4], 300),
            [0, 90, 180, 270],
        )

    def test_ignores_shot_cross_references_before_detailed_description(self):
        prompt = """retention_analysis:
<Picture 1> from [Shot 1] remains consistent.

detailed_description:
[Shot 1] Opening image.
[Shot 2] At 00:01.500, second image.

overall_soundscape:
Room tone."""
        self.assertEqual(shot_frame_positions(prompt, [1, 2], 124), [0, 36])

    def test_missing_later_timestamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "At MM:SS.mmm"):
            shot_frame_positions("[Shot 1] A. [Shot 2] B.", [1, 2], 124)

    def test_position_outside_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the target range"):
            shot_frame_positions(
                "[Shot 1] A. [Shot 2] At 00:06.000, B.", [1, 2], 124
            )

    def test_subframe_times_round_half_up(self):
        self.assertEqual(
            shot_frame_positions(
                "[Shot 1] A. [Shot 2] At 00:00.063, B.", [1, 2], 124
            ),
            [0, 2],
        )


if __name__ == "__main__":
    unittest.main()
