import json
import math
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

try:
    import folder_paths
except ImportError:
    folder_paths = None


VIDEO_EXTENSIONS = {
    ".avi",
    ".m2ts",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mts",
    ".webm",
    ".wmv",
}


def _input_directory():
    if folder_paths is not None:
        return Path(folder_paths.get_input_directory()).resolve()
    return Path.cwd().resolve()


def _available_videos():
    input_directory = _input_directory()
    if not input_directory.exists():
        return [""]
    videos = sorted(
        path.relative_to(input_directory).as_posix()
        for path in input_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    return videos or [""]


def _find_executable(name):
    executable = shutil.which(name)
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if name == "ffmpeg" and ffmpeg.is_file():
            return str(ffmpeg)
        sibling = ffmpeg.with_name(f"{name}.exe" if ffmpeg.suffix.lower() == ".exe" else name)
        if sibling.is_file():
            return str(sibling)
    except (ImportError, RuntimeError):
        pass
    raise RuntimeError(
        f"{name} was not found. Install FFmpeg and make sure its bin folder is in PATH."
    )


def _run(command, operation):
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if process.returncode != 0:
        details = process.stderr.strip()[-4000:]
        raise RuntimeError(f"FFmpeg failed while {operation}.\n{details}")
    return process


def _probe_video(video):
    try:
        ffprobe = _find_executable("ffprobe")
    except RuntimeError:
        return _probe_video_with_ffmpeg(video)

    process = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,avg_frame_rate,r_frame_rate",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video),
        ],
        "reading video information",
    )
    data = json.loads(process.stdout)
    streams = data.get("streams") or []
    stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if stream is None:
        raise ValueError(f"No video stream found in: {video}")
    rate_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "24/1"
    try:
        fps = float(Fraction(rate_text))
    except (ValueError, ZeroDivisionError):
        fps = 24.0
    if not math.isfinite(fps) or fps <= 0:
        fps = 24.0
    return {
        "fps": fps,
        "duration": float((data.get("format") or {}).get("duration") or 0.0),
        "has_audio": any(stream.get("codec_type") == "audio" for stream in streams),
    }


def _probe_video_with_ffmpeg(video):
    """Read basic metadata from FFmpeg output when FFprobe is unavailable."""
    ffmpeg = _find_executable("ffmpeg")
    process = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(video)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    details = process.stderr
    video_line = next(
        (line for line in details.splitlines() if re.search(r"Stream #.*Video:", line)),
        "",
    )
    if not video_line:
        raise ValueError(f"No video stream found in: {video}")

    rate = re.search(r"(\d+(?:\.\d+)?)\s*fps\b", video_line)
    if rate is None:
        rate = re.search(r"(\d+(?:\.\d+)?)\s*tbr\b", video_line)
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", details)

    fps = float(rate.group(1)) if rate else 24.0
    if not math.isfinite(fps) or fps <= 0:
        fps = 24.0
    duration = 0.0
    if duration_match:
        hours, minutes, seconds = duration_match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    return {
        "fps": fps,
        "duration": duration,
        "has_audio": bool(re.search(r"Stream #.*Audio:", details)),
    }
