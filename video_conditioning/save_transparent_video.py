import logging
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import av
import numpy as np
from PIL import Image

import comfy.model_management as model_management
import folder_paths
from comfy.cli_args import args
from comfy_api.latest import InputImpl, VideoComponents, io, ui

from .selective_video import VideoInfo
from .video_preview import preview_pixels


log = logging.getLogger(__name__)


def render_preview(source, destination):
    with av.open(str(source)) as original, av.open(
        str(destination), "w", options={"movflags": "+faststart"},
    ) as preview:
        stream = original.streams.video[0]
        stream.thread_type = "AUTO"
        encoder = preview.add_stream("libx264", rate=stream.average_rate,
                                     options={"preset": "ultrafast", "crf": "20"})
        encoder.pix_fmt = "yuv420p"
        checker = None
        visible = False
        for frame in original.decode(stream):
            model_management.throw_exception_if_processing_interrupted()
            pixels = preview_pixels(frame, max_size=640, even_size=True, pixel_format="rgba")
            visible = visible or bool(np.any(pixels[..., 3] > 2))
            if checker is None:
                encoder.height, encoder.width = pixels.shape[:2]
                rows, columns = np.ogrid[:encoder.height, :encoder.width]
                shades = (96 + (rows // 16 + columns // 16) % 2 * 48).astype(np.uint8)
                checker = Image.fromarray(shades).convert("RGBA")
            composed = Image.alpha_composite(checker, Image.fromarray(pixels)).convert("RGB")
            frame = av.VideoFrame.from_ndarray(np.asarray(composed), format="rgb24")
            for packet in encoder.encode(frame):
                preview.mux(packet)
        for packet in encoder.encode():
            preview.mux(packet)
        return visible


class RPSaveTransparentVideo(io.ComfyNode):
    title = "RP Save Transparent Video"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name=cls.title,
            category="RP/Video Conditioning",
            description="Save a ProRes 4444 MOV with alpha and play a checkerboard preview in ComfyUI.",
            inputs=[
                io.Video.Input("video", optional=True),
                io.String.Input("filename_prefix", default="video/RP_Guida_Trasparente"),
                io.Image.Input("frames", optional=True, tooltip="RGBA frames from frames_transparent_guide."),
                VideoInfo.Input("video_info", optional=True, tooltip="Use the frame rate from RP Selective Video Guides."),
                io.Float.Input("fps", default=24, min=0.001, max=240, optional=True, force_input=True,
                               tooltip="Frame rate for RGBA frames when video_info is absent."),
            ],
            outputs=[io.Video.Output("video")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, filename_prefix, video=None, frames=None, video_info=None, fps=24):
        if video is None:
            if frames is None or frames.ndim != 4 or not len(frames) or frames.shape[-1] != 4:
                raise ValueError("Connect a video or an RGBA image batch to export with transparency.")
            rate = Fraction(video_info["loaded_fps"] if video_info is not None else fps).limit_denominator(100_000)
            video = InputImpl.VideoFromComponents(VideoComponents(
                images=frames[..., :3], alpha=frames[..., 3], frame_rate=rate,
            ))
        width, height = video.get_dimensions()
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height,
        )
        master = Path(folder) / f"{name}_{counter:05}_.mov"
        metadata = None
        if not args.disable_metadata:
            metadata = dict(cls.hidden.extra_pnginfo or {})
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
        with TemporaryDirectory(dir=folder, prefix=".rp-export-") as working:
            encoded = Path(working) / "guide.mov"
            video.save_to(str(encoded), format="mov", codec="prores", chroma_subsampling="444", metadata=metadata)
            encoded.replace(master)

        result = {"rp_transparent_export": [ui.SavedResult(master.name, subfolder, io.FolderType.output)]}
        temp = Path(folder_paths.get_temp_directory())
        proxy = temp / f"rp-guide-{uuid4().hex}.mp4"
        try:
            with TemporaryDirectory(dir=temp, prefix="rp-guide-") as working:
                preview = Path(working) / "preview.mp4"
                visible = render_preview(master, preview)
                preview.replace(proxy)
            result.update(ui.PreviewVideo([ui.SavedResult(proxy.name, "", io.FolderType.temp)]).as_dict())
            result["rp_transparent_empty"] = [not visible]
        except (OSError, ValueError, av.error.FFmpegError):
            log.exception("Could not render the browser preview for %s", master.name)
        return io.NodeOutput(video, ui=result)
