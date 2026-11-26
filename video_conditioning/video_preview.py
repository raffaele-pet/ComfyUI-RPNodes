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

from ..video_frame_processing.video_files import _find_executable, _run


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


def cached_preview_file(key, suffix, render, check):
    while not RENDER_LOCK.acquire(timeout=0.25):
        check()
    try:
        check()
        existing = PREVIEW_FILES.get(key)
        if existing is not None and existing.is_file():
            PREVIEW_FILES.move_to_end(key)
            return existing.name

        temp = Path(folder_paths.get_temp_directory())
        temp.mkdir(parents=True, exist_ok=True)
        path = temp / f"rp-selective-{uuid4().hex}.{suffix}"
        with TemporaryDirectory(dir=temp, prefix="rp-selective-") as folder:
            video = render(Path(folder))
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


def cached_preview(ticket, source, tracks, styles, source_width, check):
    def render(folder):
        video = folder / "video.mp4"
        if not render_video(source, video, tracks, styles, source_width, check):
            return video
        check()
        muxed = folder / "preview.mp4"
        _run([
            _find_executable("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", str(video), "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart", str(muxed),
        ], "adding source audio to the selective video preview")
        return muxed

    return cached_preview_file((ticket, styles), "mp4", render, check)


def cached_prepared_preview(ticket, mode, count, rate, audio, draw, check):
    transparent = mode == "transparent"
    suffix = "webm" if transparent else "mp4"

    def render(folder):
        path = folder / f"video.{suffix}"
        options = {} if transparent else {"movflags": "+faststart"}
        with av.open(str(path), "w", options=options) as output:
            if transparent:
                encoder = output.add_stream("libvpx-vp9", rate=rate, options={
                    "deadline": "realtime", "cpu-used": "6", "crf": "23", "b": "0", "auto-alt-ref": "0",
                })
                encoder.pix_fmt = "yuva420p"
            else:
                encoder = output.add_stream("libx264", rate=rate, options={"preset": "ultrafast", "crf": "20"})
                encoder.pix_fmt = "yuv420p"
            if audio is not None and not transparent:
                samples = audio["waveform"][0, :2].float().cpu().numpy()
                soundtrack = output.add_stream("aac", rate=audio["sample_rate"])
                soundtrack.layout = "mono" if len(samples) == 1 else "stereo"
            for index in range(count):
                check()
                pixels = draw(index)
                if index == 0:
                    scale = min(1, 640 / max(pixels.size))
                    encoder.width, encoder.height = (2 * max(1, round(side * scale / 2)) for side in pixels.size)
                pixels = pixels.resize((encoder.width, encoder.height), Image.Resampling.BILINEAR)
                frame = av.VideoFrame.from_ndarray(np.asarray(pixels), format="rgba" if transparent else "rgb24")
                for packet in encoder.encode(frame):
                    output.mux(packet)
            for packet in encoder.encode():
                output.mux(packet)
            if audio is not None and not transparent:
                for start in range(0, samples.shape[-1], 4096):
                    check()
                    frame = av.AudioFrame.from_ndarray(
                        np.ascontiguousarray(samples[:, start:start + 4096]), format="fltp", layout=soundtrack.layout,
                    )
                    frame.sample_rate = soundtrack.rate
                    frame.pts = start
                    for packet in soundtrack.encode(frame):
                        output.mux(packet)
                for packet in soundtrack.encode():
                    output.mux(packet)
        return path

    return cached_preview_file((ticket, mode), suffix, render, check)
