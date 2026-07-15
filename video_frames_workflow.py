import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from comfy_execution.graph_utils import GraphBuilder, is_link
except ImportError:
    GraphBuilder = None
    is_link = None


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
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MANIFEST_NAME = ".rp_video_frames.json"


def _input_directory():
    if folder_paths is not None:
        return Path(folder_paths.get_input_directory()).resolve()
    return Path.cwd().resolve()


def _output_directory():
    if folder_paths is not None:
        return Path(folder_paths.get_output_directory()).resolve()
    return (Path.cwd() / "output").resolve()


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


def _resolve_video(video):
    value = str(video or "").strip().strip('"')
    if not value:
        raise ValueError("Select or upload a video.")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = _input_directory() / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Video not found: {candidate}")
    if candidate.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {candidate.suffix}")
    return candidate


def _resolve_output_folder(value, label):
    text = str(value or "").strip().strip('"')
    if not text:
        raise ValueError(f"{label} cannot be empty.")
    folder = Path(text).expanduser()
    if not folder.is_absolute():
        parts = list(folder.parts)
        lowered = [part.casefold() for part in parts]
        if len(parts) >= 2 and lowered[:2] == ["comfyui", "output"]:
            parts = parts[2:]
        elif parts and lowered[0] == "output":
            parts = parts[1:]
        folder = _output_directory().joinpath(*parts)
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


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
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate,nb_frames,width,height",
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
    if not streams:
        raise ValueError(f"No video stream found in: {video}")
    stream = streams[0]
    rate_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "24/1"
    try:
        fps = float(Fraction(rate_text))
    except (ValueError, ZeroDivisionError):
        fps = 24.0
    if not math.isfinite(fps) or fps <= 0:
        fps = 24.0
    return {
        "fps": fps,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration": float((data.get("format") or {}).get("duration") or 0.0),
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

    dimensions = re.search(r"(?<!\d)(\d{2,6})x(\d{2,6})(?!\d)", video_line)
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
        "width": int(dimensions.group(1)) if dimensions else 0,
        "height": int(dimensions.group(2)) if dimensions else 0,
        "duration": duration,
    }


def _natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _next_available_output_path(directory, filename):
    """Return a collision-free video path without replacing an earlier render."""
    requested = directory / filename
    if not requested.exists():
        return requested

    stem = requested.stem
    suffix = requested.suffix
    counter = 1
    while True:
        candidate = directory / f"{stem}_{counter:04d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _image_files(folder):
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=_natural_key,
    )


def _remove_extracted_frames(folder):
    for path in folder.glob("frame_*.png"):
        if path.is_file():
            path.unlink()
    manifest = folder / MANIFEST_NAME
    if manifest.is_file():
        manifest.unlink()


def _clear_processed_images(folder):
    for path in _image_files(folder):
        _unlink_with_retry(path)


def _unlink_with_retry(path, attempts=20):
    for attempt in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise PermissionError(
                    f"Cannot remove '{path}' because another process is still using it. "
                    "Close image previews or file viewers and run the workflow again."
                )
            time.sleep(0.05 * (attempt + 1))


def _load_frame(path):
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        has_alpha = "A" in image.getbands()
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb).unsqueeze(0)
        if has_alpha:
            alpha = np.asarray(image.getchannel("A"), dtype=np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(alpha).unsqueeze(0)
        else:
            mask = torch.zeros((1, rgb.shape[0], rgb.shape[1]), dtype=torch.float32)
    return tensor, mask


def _save_processed_frame(image, path):
    if image is None or not hasattr(image, "shape") or len(image.shape) != 4:
        raise ValueError("processed_image must be a ComfyUI IMAGE tensor.")
    if int(image.shape[0]) != 1:
        raise ValueError(
            "The integrated video loop expects one image per iteration. "
            "Connect a processing chain that returns a single IMAGE."
        )

    array = image[0].detach().cpu().float().clamp(0.0, 1.0).numpy()
    array = np.rint(array * 255.0).astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    elif array.shape[-1] not in (3, 4):
        array = array[..., :3]

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    Image.fromarray(array).save(temporary, format="PNG")
    try:
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise PermissionError(
                        f"Cannot write '{path}' because another process is using it."
                    )
                time.sleep(0.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_matches(manifest, video):
    if not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        stat = video.stat()
        return (
            Path(data["source_video"]).resolve() == video
            and data["source_size"] == stat.st_size
            and data["source_mtime_ns"] == stat.st_mtime_ns
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


class RPVideoToFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (_available_videos(), {"video_upload": True}),
                "frames_folder": ("STRING", {"default": "video_frames/source"}),
                "processed_frames_folder": ("STRING", {"default": "video_frames/processed"}),
                "replace_existing_frames": ("BOOLEAN", {"default": False}),
                "clear_processed_frames": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "loop_index": ("INT",),
            },
        }

    RETURN_TYPES = ("FLOW_CONTROL", "IMAGE", "MASK", "STRING", "RP_VIDEO_CONTEXT")
    RETURN_NAMES = (
        "flow",
        "image",
        "mask",
        "frame_name",
        "video_context",
    )
    FUNCTION = "extract"
    CATEGORY = "video/RPNodes"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    def extract(
        self,
        video,
        frames_folder,
        processed_frames_folder,
        replace_existing_frames,
        clear_processed_frames,
        loop_index=0,
    ):
        loop_index = max(0, int(loop_index))
        source_video = _resolve_video(video)
        frames_directory = _resolve_output_folder(frames_folder, "Frames folder")
        processed_directory = _resolve_output_folder(
            processed_frames_folder, "Processed frames folder"
        )
        if frames_directory == processed_directory:
            raise ValueError("Source and processed frames folders must be different.")
        if clear_processed_frames and processed_directory == _output_directory():
            raise ValueError(
                "Do not use the ComfyUI/output root with clear_processed_frames enabled. "
                "Choose a dedicated subfolder such as 'video_frames/processed'."
            )

        if clear_processed_frames and loop_index == 0:
            _clear_processed_images(processed_directory)

        manifest = frames_directory / MANIFEST_NAME
        existing_frames = sorted(frames_directory.glob("frame_*.png"))
        reuse_existing = bool(existing_frames) and _manifest_matches(manifest, source_video)

        if not reuse_existing:
            if existing_frames and not replace_existing_frames:
                raise RuntimeError(
                    "The source frames folder already contains frames from another or changed "
                    "video. Enable replace_existing_frames or choose another folder."
                )
            _remove_extracted_frames(frames_directory)
            ffmpeg = _find_executable("ffmpeg")
            _run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source_video),
                    "-map",
                    "0:v:0",
                    "-vsync",
                    "0",
                    str(frames_directory / "frame_%08d.png"),
                ],
                "extracting video frames",
            )
            existing_frames = sorted(frames_directory.glob("frame_*.png"))
            if not existing_frames:
                raise RuntimeError("FFmpeg completed without creating any frames.")

        info = _probe_video(source_video)
        stat = source_video.stat()
        manifest.write_text(
            json.dumps(
                {
                    "source_video": str(source_video),
                    "source_size": stat.st_size,
                    "source_mtime_ns": stat.st_mtime_ns,
                    "fps": info["fps"],
                    "width": info["width"],
                    "height": info["height"],
                    "frame_count": len(existing_frames),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if loop_index >= len(existing_frames):
            raise IndexError(
                f"Video frame index {loop_index} is outside the extracted frame range "
                f"(0-{len(existing_frames) - 1})."
            )
        frame_path = existing_frames[loop_index]
        image, mask = _load_frame(frame_path)
        context = {
            "index": loop_index,
            "frame_count": len(existing_frames),
            "frame_filename": frame_path.name,
            "frames_directory": str(frames_directory),
            "processed_frames_directory": str(processed_directory),
            "source_video": str(source_video),
            "fps": float(info["fps"]),
        }
        return (
            "stub",
            image,
            mask,
            frame_path.stem,
            context,
        )


class RPFramesToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
                "processed_image": ("IMAGE",),
                "video_context": ("RP_VIDEO_CONTEXT",),
                "output_filename": ("STRING", {"default": "processed_video.mp4"}),
                "video_codec": (["libx264", "libx265"], {"default": "libx264"}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51}),
                "copy_original_audio": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT")
    RETURN_NAMES = ("video_path", "frame_count", "fps")
    FUNCTION = "combine"
    CATEGORY = "video/RPNodes"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    def _explore_dependencies(self, node_id, dynprompt, upstream):
        node_info = dynprompt.get_node(node_id)
        for value in node_info.get("inputs", {}).values():
            if not is_link(value):
                continue
            parent_id = value[0]
            if parent_id not in upstream:
                upstream[parent_id] = []
                self._explore_dependencies(parent_id, dynprompt, upstream)
            upstream[parent_id].append(node_id)

    def _collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self._collect_contained(child_id, upstream, contained)

    def _next_iteration(self, flow, next_index, dynprompt, unique_id):
        if GraphBuilder is None or is_link is None:
            raise RuntimeError(
                "This ComfyUI version does not provide dynamic graph support required by "
                "the integrated frame loop."
            )
        open_node = flow[0]
        upstream = {}
        self._explore_dependencies(unique_id, dynprompt, upstream)
        contained = {}
        self._collect_contained(open_node, upstream, contained)
        contained[open_node] = True
        contained[unique_id] = True

        graph = GraphBuilder()
        for node_id in contained:
            original = dynprompt.get_node(node_id)
            clone_id = "Recurse" if node_id == unique_id else node_id
            node = graph.node(original["class_type"], clone_id)
            node.set_override_display_id(node_id)

        for node_id in contained:
            original = dynprompt.get_node(node_id)
            clone_id = "Recurse" if node_id == unique_id else node_id
            node = graph.lookup_node(clone_id)
            for name, value in original.get("inputs", {}).items():
                if is_link(value) and value[0] in contained:
                    parent = graph.lookup_node(value[0])
                    node.set_input(name, parent.out(value[1]))
                else:
                    node.set_input(name, value)

        graph.lookup_node(open_node).set_input("loop_index", int(next_index))
        recursive_end = graph.lookup_node("Recurse")
        return {
            "result": tuple(recursive_end.out(i) for i in range(3)),
            "expand": graph.finalize(),
        }

    def combine(
        self,
        flow,
        processed_image,
        video_context,
        output_filename,
        video_codec,
        crf,
        copy_original_audio,
        dynprompt=None,
        unique_id=None,
        overwrite=None,
    ):
        context = dict(video_context)
        index = int(context["index"])
        frame_count = int(context["frame_count"])
        frames_directory = Path(context["processed_frames_directory"]).expanduser().resolve()
        if not frames_directory.is_dir():
            raise FileNotFoundError(f"Processed frames folder not found: {frames_directory}")
        frame_path = frames_directory / context["frame_filename"]
        _save_processed_frame(processed_image, frame_path)

        if index + 1 < frame_count:
            return self._next_iteration(flow, index + 1, dynprompt, unique_id)

        images = sorted(frames_directory.glob("frame_*.png"), key=_natural_key)[:frame_count]
        if len(images) != frame_count:
            raise RuntimeError(
                f"Expected {frame_count} processed frames, but found {len(images)} in "
                f"'{frames_directory}'."
            )

        source = _resolve_video(context["source_video"])
        rate = float(context["fps"])
        if not math.isfinite(rate) or rate <= 0:
            rate = _probe_video(source)["fps"]

        safe_name = Path(str(output_filename).strip()).name
        if not safe_name:
            safe_name = "processed_video.mp4"
        if Path(safe_name).suffix.lower() != ".mp4":
            safe_name = f"{Path(safe_name).stem}.mp4"
        output_path = _next_available_output_path(frames_directory, safe_name)

        duration = 1.0 / rate
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ffconcat", encoding="utf-8", delete=False
        ) as concat_file:
            concat_path = Path(concat_file.name)
            concat_file.write("ffconcat version 1.0\n")
            for image in images:
                escaped = image.resolve().as_posix().replace("'", "'\\''")
                concat_file.write(f"file '{escaped}'\n")
                concat_file.write(f"duration {duration:.12f}\n")
            escaped_last = images[-1].resolve().as_posix().replace("'", "'\\''")
            concat_file.write(f"file '{escaped_last}'\n")

        ffmpeg = _find_executable("ffmpeg")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
        ]
        if copy_original_audio:
            command.extend(["-i", str(source), "-map", "0:v:0", "-map", "1:a:0?"])
        else:
            command.extend(["-map", "0:v:0"])
        command.extend(
            [
                "-r",
                f"{rate:.12g}",
                "-c:v",
                video_codec,
                "-crf",
                str(int(crf)),
                "-pix_fmt",
                "yuv420p",
            ]
        )
        if copy_original_audio:
            command.extend(["-c:a", "aac", "-shortest"])
        command.extend(["-movflags", "+faststart", str(output_path)])

        try:
            _run(command, "building the processed video")
        finally:
            concat_path.unlink(missing_ok=True)

        preview = None
        try:
            relative = output_path.resolve().relative_to(_output_directory())
            preview = {
                "filename": relative.name,
                "subfolder": "" if relative.parent == Path(".") else relative.parent.as_posix(),
                "type": "output",
                "format": "video/mp4",
            }
        except ValueError:
            pass

        result = (str(output_path), len(images), rate)
        if preview is None:
            return result
        return {"ui": {"rp_video": [preview]}, "result": result}


NODE_CLASS_MAPPINGS = {
    "RPVideoToFrames": RPVideoToFrames,
    "RPFramesToVideo": RPFramesToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RPVideoToFrames": "RP Video to Frames",
    "RPFramesToVideo": "RP Frames to Video",
}
