import math
import os
import subprocess
import tempfile
import wave
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

import torch

from .video_frames_workflow import VIDEO_EXTENSIONS, _find_executable, _input_directory, _probe_video


MAX_PIXELS = 8192 * 8192
MP4_OPTIONS = [
    # Keep muxing metadata bounded by the fragment size.
    "-movflags", "+frag_keyframe+delay_moov+default_base_moof+skip_trailer",
    "-flush_packets", "1", "-f", "mp4",
]


def input_video(value):
    root = _input_directory()
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Select or upload a video from ComfyUI/input: {value}")
    return path


@contextmanager
def ffmpeg(arguments, operation, check_interrupt=None, **pipes):
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            [_find_executable("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "error",
             "-filter_threads", "1", *arguments],
            stderr=errors,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            **pipes,
        )
        try:
            yield process
        except BrokenPipeError:
            process.wait()
            raise ffmpeg_error(errors, operation) from None
        else:
            while True:
                try:
                    returncode = process.wait(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    if check_interrupt is not None:
                        check_interrupt()
            if returncode:
                raise ffmpeg_error(errors, operation)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            for pipe in (process.stdin, process.stdout):
                if pipe is not None and not pipe.closed:
                    try:
                        pipe.close()
                    except BrokenPipeError:
                        pass


def ffmpeg_error(errors, operation):
    errors.seek(max(0, errors.seek(0, os.SEEK_END) - 4000))
    return RuntimeError(f"FFmpeg failed while {operation}.\n{errors.read().decode(errors='replace').strip()}")


class VideoSource:
    def __init__(self, video, force_rate, custom_width, custom_height, frame_load_cap,
                 skip_first_frames, select_every_nth):
        self.path = input_video(video)
        info = _probe_video(self.path, include_audio=True)
        rate = Fraction(force_rate or info["fps"]).limit_denominator(100_000)
        self.fps = rate / select_every_nth
        self.has_audio = info["has_audio"]
        self.audio_start = Fraction(skip_first_frames, 1) / rate
        self.estimated_frames = max(
            0, math.ceil((info["duration"] * rate - skip_first_frames) / select_every_nth)
        )
        if frame_load_cap:
            self.estimated_frames = min(frame_load_cap, self.estimated_frames)

        filters = [f"fps={rate}", f"trim=start_frame={skip_first_frames}"]
        if select_every_nth > 1:
            filters.append(f"select=not(mod(n\\,{select_every_nth}))")
        if custom_width or custom_height:
            filters.append(f"scale={custom_width or -2}:{custom_height or -2}")
        self.arguments = [
            "-xerror", "-threads:v", "2", "-max_pixels", str(MAX_PIXELS),
            "-i", str(self.path), "-map", "0:v:0",
            "-vf", ",".join(filters), "-fps_mode", "passthrough",
        ]
        if frame_load_cap:
            self.arguments.extend(["-frames:v", str(frame_load_cap)])
        # PPM supplies the actual dimensions after rotation and scaling on each frame.
        self.arguments.extend(["-c:v", "ppm", "-threads:v", "1", "-f", "image2pipe", "pipe:1"])

    @contextmanager
    def frames(self):
        with ffmpeg(self.arguments, "decoding video", stdout=subprocess.PIPE) as process:
            yield self._read_frames(process.stdout)

    @staticmethod
    def _read_frames(pipe):
        while header := pipe.readline(32):
            try:
                width, height = map(int, pipe.readline(64).split())
            except ValueError:
                raise RuntimeError("FFmpeg returned an invalid frame header.") from None
            if header != b"P6\n" or pipe.readline(32) != b"255\n":
                raise RuntimeError("FFmpeg returned an invalid RGB frame header.")
            if min(width, height) < 1 or width * height > MAX_PIXELS:
                raise ValueError(f"Unsupported video frame size: {width} x {height}.")
            pixels = bytearray(width * height * 3)
            if pipe.readinto(pixels) != len(pixels):
                raise RuntimeError("FFmpeg returned an incomplete RGB frame.")
            yield torch.frombuffer(pixels, dtype=torch.uint8).view(height, width, 3)


def write_audio(audio, path, check_interrupt):
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if waveform.ndim != 3 or waveform.shape[0] != 1 or not 1 <= waveform.shape[1] <= 8:
        raise ValueError("Connect a single AUDIO waveform with up to eight channels.")
    with wave.open(str(path), "wb") as output:
        output.setparams((waveform.shape[1], 2, sample_rate, 0, "NONE", "not compressed"))
        for offset in range(0, waveform.shape[-1], 65536):
            check_interrupt()
            block = waveform[0, :, offset:offset + 65536]
            samples = block.detach().to(device="cpu", dtype=torch.float32)
            samples = samples.mul(32767).round_().clamp_(-32768, 32767)
            output.writeframesraw(samples.T.contiguous().to(torch.int16).numpy().tobytes())


def audio_tempo(ratio):
    filters = []
    while ratio < 0.5:
        filters.append("atempo=0.5")
        ratio *= 2
    while ratio > 2:
        filters.append("atempo=2")
        ratio /= 2
    if abs(ratio - 1) > 1e-9:
        filters.append(f"atempo={ratio:.10g}")
    return filters


class VideoEncoder:
    def __init__(self, path, fps, crf, audio_path=None, audio_start=0, tempo=1):
        self.path = Path(path)
        self.fps = fps
        self.crf = crf
        self.audio_path = audio_path
        self.audio_start = audio_start
        self.tempo = tempo
        self.size = None
        self.count = 0

    @contextmanager
    def open(self, images, check_interrupt):
        if images.ndim != 4 or images.shape[-1] not in (1, 3, 4):
            raise ValueError("Connect a ComfyUI IMAGE with 1, 3 or 4 channels.")
        height, width = images.shape[1:3]
        if min(height, width) < 1 or height * width > MAX_PIXELS:
            raise ValueError(f"Unsupported output frame size: {width} x {height}.")
        self.size = (height, width)
        arguments = [
            "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}", "-framerate", str(self.fps), "-i", "pipe:0",
        ]
        arguments.extend([
            "-map", "0:v:0",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,setsar=1",
            "-c:v", "libx264", "-crf", str(self.crf), "-preset", "fast",
            "-tune", "zerolatency", "-threads:v", "2", "-pix_fmt", "yuv420p",
            "-g", str(max(1, round(self.fps * 2))),
        ])
        if self.audio_path is None:
            intermediate = self.path
            arguments.extend(MP4_OPTIONS)
        else:
            # Keep a compressed, index-free intermediate until the exact audio trim is known.
            intermediate = self.path.with_suffix(".nut")
            arguments.extend(["-strict", "experimental", "-write_index", "0", "-syncpoints", "none", "-f", "nut"])
        arguments.append(str(intermediate))
        with ffmpeg(arguments, "encoding video", check_interrupt, stdin=subprocess.PIPE) as process:
            yield process.stdin
            process.stdin.close()
        if self.audio_path is not None:
            check_interrupt()
            duration = f"{float(self.count / self.fps):.12f}"
            filters = ",".join([*audio_tempo(self.tempo), "apad", f"atrim=duration={duration}"])
            arguments = [
                "-y", "-f", "nut", "-i", "pipe:0",
                "-ss", f"{float(self.audio_start):.12f}", "-vn", "-sn", "-dn",
                "-i", str(self.audio_path), "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "copy", "-c:a", "aac", "-af", filters, "-t", duration,
                *MP4_OPTIONS, str(self.path),
            ]
            with intermediate.open("rb") as encoded:
                with ffmpeg(arguments, "adding audio", check_interrupt, stdin=encoded):
                    pass
            intermediate.unlink()

    def write(self, pipe, images):
        if images.ndim != 4 or tuple(images.shape[1:3]) != self.size or images.shape[-1] not in (1, 3, 4):
            raise ValueError("All video frames must have the same width and height and 1, 3 or 4 channels.")
        for frame in images:
            rgb = frame[..., :3].detach().to(device="cpu", dtype=torch.float32)
            if rgb.shape[-1] == 1:
                rgb = rgb.expand(-1, -1, 3)
            pixels = rgb.mul(255).round_().clamp_(0, 255).to(torch.uint8).contiguous()
            pipe.write(memoryview(pixels.numpy()).cast("B"))
            self.count += 1


@contextmanager
def video_destination(folder_paths, save_video, filename_prefix, size):
    if save_video:
        root = Path(folder_paths.get_output_directory()).resolve()
        directory, stem, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix.strip() or "RP_Video", str(root), size[1], size[0]
        )
        directory = Path(directory).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("The video must be saved within ComfyUI/output.")
    else:
        directory = Path(folder_paths.get_temp_directory())
        subfolder, stem, counter = "", f"rp_preview_{uuid4().hex}", 1
    directory.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".rp-video-", dir=directory) as staging:
        temporary = Path(staging) / "video.mp4"
        result = {}
        yield temporary, result
        while True:
            path = directory / f"{stem}_{counter:05}.mp4"
            try:
                with path.open("xb"):
                    pass
                break
            except FileExistsError:
                counter += 1
        try:
            temporary.replace(path)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        result.update(
            path=str(path), filename=path.name, subfolder=subfolder,
            type="output" if save_video else "temp", format="video/mp4",
        )
