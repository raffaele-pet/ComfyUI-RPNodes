import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from video_frame_processing import video_frames_workflow as workflow


class VideoFramesWorkflowTests(unittest.TestCase):
    def test_short_audio_does_not_drop_processed_video_frames(self):
        try:
            ffmpeg = workflow._find_executable("ffmpeg")
            ffprobe = workflow._find_executable("ffprobe")
        except RuntimeError as error:
            self.skipTest(str(error))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "short_audio.mp4"
            frame_count = 12
            rate = "24000/1001"
            workflow._run(
                [
                    ffmpeg, "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"color=c=red:s=32x32:r={rate}",
                    "-f", "lavfi", "-i", "sine=sample_rate=48000:duration=0.15",
                    "-frames:v", str(frame_count), "-c:v", "libx264", "-c:a", "aac",
                    str(source),
                ],
                "creating a video with a shorter audio track",
            )
            frames = root / "frames"
            processed = root / "processed"
            _, last_image, _, _, context = workflow.RPVideoToFrames().extract(
                str(source), str(frames), str(processed), False, False, loop_index=frame_count - 1,
            )
            for frame in frames.glob("frame_*.png"):
                shutil.copyfile(frame, processed / frame.name)
            output = workflow.RPFramesToVideo().combine(
                "stub", last_image, context, "result.mp4", "libx264", 19, True, False,
            )
            info = workflow._run(
                [
                    ffprobe, "-v", "error", "-show_entries",
                    "stream=codec_type,nb_frames,avg_frame_rate,duration", "-of", "json", output[0],
                ],
                "checking the assembled video and audio",
            )
            video, audio = json.loads(info.stdout)["streams"]
            self.assertEqual(int(video["nb_frames"]), frame_count)
            self.assertEqual(video["avg_frame_rate"], rate)
            self.assertEqual(audio["codec_type"], "audio")
            self.assertLess(float(audio["duration"]), float(video["duration"]))

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
            with_optional_audio = workflow.RPFramesToVideo().combine(
                "stub", image, context, "silent.mp4", "libx264", 19, True, False
            )
            self.assertEqual(with_optional_audio[1], 3)
            self.assertTrue(Path(with_optional_audio[0]).is_file())

            original_bytes = Path(output[0]).read_bytes()
            occupied = processed / "result_1.mp4"
            occupied.write_bytes(b"Keep this existing file")
            numbered = workflow.RPFramesToVideo().combine(
                "stub", image, context, "result.mp4", "libx264", 19, False, False
            )
            self.assertEqual(Path(numbered[0]).name, "result_2.mp4")
            self.assertTrue(Path(numbered[0]).is_file())
            self.assertEqual(Path(output[0]).read_bytes(), original_bytes)
            self.assertEqual(occupied.read_bytes(), b"Keep this existing file")

            next_numbered = workflow.RPFramesToVideo().combine(
                "stub", image, context, "result_2.mp4", "libx264", 19, False, False
            )
            self.assertEqual(Path(next_numbered[0]).name, "result_3.mp4")
            self.assertTrue(Path(next_numbered[0]).is_file())

            Path(output[0]).write_bytes(b"Replace when overwrite is enabled")
            overwritten = workflow.RPFramesToVideo().combine(
                "stub", image, context, "result.mp4", "libx264", 19, False, True
            )
            self.assertEqual(overwritten[0], output[0])
            self.assertEqual(workflow._probe_video(Path(overwritten[0]))["width"], 32)


if __name__ == "__main__":
    unittest.main()
