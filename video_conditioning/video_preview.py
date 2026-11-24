from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from uuid import uuid4

import av
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, maximum_filter

import folder_paths

from ..video_frame_processing.video_frames_workflow import _find_executable, _run


PREVIEW_FILES = OrderedDict()
RENDER_LOCK = Lock()


def preview_pixels(image, max_size=1008, even_size=False, pixel_format="rgb24"):
    scale = min(1, max_size / max(image.width, image.height))
    alignment = 2 if even_size else 1
    width = alignment * max(1, round(image.width * scale / alignment))
    height = alignment * max(1, round(image.height * scale / alignment))
    pixels = image.reformat(width=width, height=height, format=pixel_format).to_ndarray()
    rotation = round(image.rotation / 90) % 4
    return np.ascontiguousarray(np.rot90(pixels, rotation))


def overlay_masks(pixels, index, tracks, styles, scale):
    height, width = pixels.shape[:2]
    for (first, frames), (color, expand, fill_holes, active) in zip(tracks, styles):
        offset = index - first
        if not 0 <= offset < len(frames):
            continue
        mask = np.unpackbits(frames[offset], axis=-1, bitorder="little")
        if not mask.any():
            continue
        mask = Image.fromarray(mask * 255).resize((width, height), Image.Resampling.BILINEAR)
        mask = np.asarray(mask) > 127
        if fill_holes:
            binary_fill_holes(mask, output=mask)
        radius = round(expand * scale)
        if radius:
            maximum_filter(mask, size=2 * radius + 1, mode="constant", output=mask)
        alpha = 0.55 if active else 0.38
        pixels[mask] = np.rint(pixels[mask] * (1 - alpha) + color * alpha).astype(np.uint8)


def render_video(source, output_path, tracks, styles, source_width, check):
    track_arrays = [(first, packed.numpy()) for first, packed in tracks]
    colors = [
        (np.array(tuple(bytes.fromhex(color[1:])), dtype=np.float32), expand, fill_holes, active)
        for color, expand, fill_holes, active in styles
    ]
    with av.open(str(source)) as original, av.open(str(output_path), "w", options={"movflags": "+faststart"}) as output:
        video = original.streams.video[0]
        video.thread_type = "AUTO"
        rate = video.average_rate or 24
        encoder = output.add_stream("libx264", rate=rate, options={"preset": "ultrafast", "crf": "20"})
        encoder.pix_fmt = "yuv420p"
        encoder.codec_context.time_base = video.time_base
        first_pts = None
        last_pts = -1
        index = 0
        for frame in original.decode(video):
            if frame.pts is None or frame.pts < 0:
                continue
            check()
            pixels = preview_pixels(frame, max_size=640, even_size=True)
            height, width = pixels.shape[:2]
            if first_pts is None:
                first_pts = frame.pts
                encoder.width = width
                encoder.height = height
            overlay_masks(pixels, index, track_arrays, colors, width / source_width)
            composed = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            composed.time_base = video.time_base
            # The source can repeat a timestamp. Encoders require strictly increasing PTS.
            last_pts = max(last_pts + 1, frame.pts - first_pts)
            composed.pts = last_pts
            if frame.duration:
                composed.duration = frame.duration
            for packet in encoder.encode(composed):
                output.mux(packet)
            index += 1
        if first_pts is None:
            raise ValueError("The video has no decodable frames.")
        for packet in encoder.encode():
            output.mux(packet)
        return bool(original.streams.audio)


def cached_preview(ticket, source, tracks, styles, source_width, check):
    while not RENDER_LOCK.acquire(timeout=0.25):
        check()
    try:
        check()
        key = ticket, styles
        existing = PREVIEW_FILES.get(key)
        if existing is not None and existing.is_file():
            PREVIEW_FILES.move_to_end(key)
            return existing.name

        temp = Path(folder_paths.get_temp_directory())
        temp.mkdir(parents=True, exist_ok=True)
        path = temp / f"rp-selective-{uuid4().hex}.mp4"
        with TemporaryDirectory(dir=temp, prefix="rp-selective-") as folder:
            video = Path(folder) / "video.mp4"
            has_audio = render_video(source, video, tracks, styles, source_width, check)
            if has_audio:
                check()
                muxed = Path(folder) / "preview.mp4"
                _run([
                    _find_executable("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-i", str(video), "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart", str(muxed),
                ], "adding source audio to the selective video preview")
                video = muxed
            check()
            video.replace(path)
        PREVIEW_FILES[key] = path
        while len(PREVIEW_FILES) > 8:
            _, expired = PREVIEW_FILES.popitem(last=False)
            try:
                expired.unlink(missing_ok=True)
            except OSError:
                # A browser may still have this file open on Windows. ComfyUI owns the temp folder.
                pass
        return path.name
    finally:
        RENDER_LOCK.release()
