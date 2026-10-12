import tempfile
import unittest
from pathlib import Path

from PIL import Image

from video_frame_processing import video_frames_workflow as workflow


class VideoFramesWorkflowTests(unittest.TestCase):
    def test_extract_every_variable_rate_frame_and_reassemble_video(self):
        try:
            ffmpeg = workflow._find_executable("ffmpeg")
        except RuntimeError as error:
            self.skipTest(str(error))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            for index, color in enumerate(colors):
                Image.new("RGB", (32, 32), color).save(root / f"source_{index}.png")
            concat = root / "source.ffconcat"
            concat.write_text(
                "ffconcat version 1.0\n"
                "file 'source_0.png'\nduration 0.04\n"
                "file 'source_1.png'\nduration 0.12\n"
                "file 'source_2.png'\n",
                encoding="utf-8",
            )
            source = root / "source.mkv"
            workflow._run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "concat",
                 "-safe", "0", "-i", str(concat), "-fps_mode", "passthrough",
                 "-c:v", "ffv1", str(source)],
                "creating the test video",
            )
            loader = workflow.RPVideoToFrames()
            frames = root / "frames"
            processed = root / "processed"
            result = loader.extract(str(source), str(frames), str(processed), False, False)
            self.assertEqual(result[4]["frame_count"], 3)
            for index, color in enumerate(colors):
                _, image, _, _, context = loader.extract(
                    str(source), str(frames), str(processed), False, False,
                    loop_index=index,
                )
                self.assertEqual(context["index"], index)
                self.assertEqual(tuple(image.shape), (1, 32, 32, 3))
                self.assertEqual(tuple(round(float(c) * 255) for c in image[0, 0, 0]), color)
                Image.fromarray((image[0].numpy() * 255).astype("uint8")).save(
                    processed / context["frame_filename"]
                )

            output = workflow.RPFramesToVideo().combine(
                "stub", image, context, "result.mp4", "libx264", 19, False, False
            )
            self.assertEqual(output[1], 3)
            self.assertEqual(output[2], result[4]["fps"])
            self.assertTrue(Path(output[0]).is_file())
            info = workflow._probe_video(Path(output[0]))
            self.assertEqual((info["width"], info["height"]), (32, 32))
            self.assertAlmostEqual(info["fps"], result[4]["fps"], places=2)


if __name__ == "__main__":
    unittest.main()
