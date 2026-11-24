import json
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from video_frame_processing import video_stream


class StreamingVideoTests(unittest.TestCase):
    def test_selected_frames_and_short_source_audio_preserve_video_timing(self):
        try:
            ffmpeg = video_stream._find_executable("ffmpeg")
            ffprobe = video_stream._find_executable("ffprobe")
        except RuntimeError as error:
            self.skipTest(str(error))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / "source.mp4"
            subprocess.run([
                ffmpeg, "-v", "error", "-f", "lavfi",
                "-i", "testsrc2=size=64x48:rate=20:duration=2",
                "-f", "lavfi", "-i", "sine=sample_rate=48000:duration=0.45",
                "-c:v", "libx264", "-c:a", "aac", str(clip),
            ], check=True)

            def check_interrupt():
                pass

            with patch.object(video_stream, "_input_directory", return_value=root):
                source = video_stream.VideoSource(clip.name, 20, 65, 47, 12, 4, 2)
                self.assertTrue(source.has_audio)
                self.assertEqual(source.audio_start, Fraction(1, 5))
                path = root / "result.mp4"
                encoder = video_stream.VideoEncoder(path, source.fps, 19, source.path, source.audio_start)
                with source.frames() as frames:
                    first = next(frames).float().div_(255)[None]
                    with encoder.open(first, check_interrupt) as pipe:
                        encoder.write(pipe, first)
                        for frame in frames:
                            encoder.write(pipe, frame.float().div_(255)[None])

            result = subprocess.check_output([
                ffprobe, "-v", "error", "-count_frames", "-show_entries",
                "stream=codec_type,avg_frame_rate,nb_read_frames,duration,width,height", "-of", "json", str(path),
            ])
            video, audio = json.loads(result)["streams"]
            self.assertEqual(video["nb_read_frames"], "12")
            self.assertEqual(video["avg_frame_rate"], "10/1")
            self.assertEqual((video["width"], video["height"]), (66, 48))
            self.assertEqual(audio["codec_type"], "audio")
            self.assertAlmostEqual(float(audio["duration"]), 1.2, delta=0.05)
            self.assertFalse(path.with_suffix(".nut").exists())
