import asyncio
import os
import tempfile
from contextlib import ExitStack
from contextvars import ContextVar
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import torch

from .video_frames_workflow import VIDEO_EXTENSIONS, _available_videos, _input_directory
from .video_stream import VideoEncoder, VideoSource, input_video, video_destination, write_audio


CATEGORY = "video/RPNodes"
STREAM_TYPE = "RP_VIDEO_STREAM"
_active_stream = ContextVar("rp_active_video_stream", default=None)


class FrameStream:
    def __init__(self, loader_id):
        self.loader_id = loader_id
        self.source = None
        self.reader = None
        self.frame = None
        self.index = 0

    def start(self, source):
        self.source = source
        self.reader = self.iterate(source)
        if not self.advance():
            raise ValueError("No frames match the video loading settings.")

    @staticmethod
    def iterate(source):
        with source.frames() as frames:
            yield from frames

    def advance(self):
        self.frame = next(self.reader, None)
        return self.frame is not None

    def close(self):
        if self.reader is not None:
            self.reader.close()
        self.frame = None


class RPLoadVideo:
    CATEGORY = CATEGORY
    DESCRIPTION = "Load an IMAGE batch or connect video_stream to RP Preview Video to process one frame at a time."
    FUNCTION = "load"
    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", STREAM_TYPE)
    RETURN_NAMES = ("images", "fps", "frame_count", "video_stream")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ([""] if _active_stream.get() else _available_videos(), {"rp_video_upload": True}),
                "force_rate": ("FLOAT", {"default": 0, "min": 0, "max": 240, "step": 0.001}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "max_batch_mib": ("INT", {"default": 512, "min": 1, "max": 32768,
                                         "tooltip": "Maximum IMAGE batch size when used alone. "
                                                    "The connected video_stream mode ignores this limit."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "_rp_frame": ("INT",)},
        }

    @classmethod
    def IS_CHANGED(cls, video, **_):
        if _active_stream.get() or video is None:
            return False
        stat = input_video(video).stat()
        return (stat.st_size, stat.st_mtime_ns)

    def load(self, video, force_rate, custom_width, custom_height, frame_load_cap,
             skip_first_frames, select_every_nth, max_batch_mib, unique_id=None, _rp_frame=None):
        from comfy.model_management import throw_exception_if_processing_interrupted as check_interrupt

        stream = _active_stream.get()
        paired = stream is not None and stream.loader_id == unique_id and _rp_frame is not None
        if not paired or stream.source is None:
            source = VideoSource(video, force_rate, custom_width, custom_height, frame_load_cap,
                                 skip_first_frames, select_every_nth)
            if paired:
                stream.start(source)
        if paired:
            image = stream.frame.to(torch.float32).div_(255).unsqueeze(0)
            return (image, float(stream.source.fps), 1, None)

        frames = []
        batch_bytes = 0
        with source.frames() as decoded_frames:
            for frame in decoded_frames:
                check_interrupt()
                batch_bytes += frame.numel() * 4
                if batch_bytes > max_batch_mib * 1024**2:
                    raise ValueError(
                        f"The IMAGE batch would exceed max_batch_mib ({max_batch_mib} MiB). "
                        "Connect video_stream to RP Preview Video for one-frame processing, "
                        "or reduce frame_load_cap or the image size."
                    )
                if frames and frame.shape != frames[0].shape:
                    raise ValueError("The video changes resolution. Set custom_width and custom_height.")
                frames.append(frame)
        if not frames:
            raise ValueError("No frames match the video loading settings.")
        images = torch.empty((len(frames), *frames[0].shape), dtype=torch.float32, device="cpu")
        for image, frame in zip(images, frames):
            image.copy_(frame)
        return (images.div_(255), float(source.fps), len(frames), None)


def processing_graph(dynprompt, image_link, loader_id):
    from comfy_execution.graph_utils import is_link

    graph = {}
    pending = [image_link[0]]
    while pending:
        node_id = pending.pop()
        if node_id in graph:
            continue
        original = dynprompt.get_node(node_id)
        if original["class_type"] == "RPPreviewVideo":
            raise ValueError("A streaming image branch cannot contain another RP Preview Video.")
        graph[node_id] = {key: value for key, value in original.items() if key != "is_changed"}
        pending.extend(value[0] for value in original["inputs"].values() if is_link(value))
    if loader_id not in graph:
        raise ValueError("Connect the selected RP Load Video images through your processing nodes to images.")
    return graph


class FrameGraph:
    # Run only the preview's IMAGE dependencies in an isolated ComfyUI cache.
    # A new loader signature invalidates every frame-dependent result before the next run.
    def __init__(self, graph, image_link, loader_id, prompt_id, extra_data):
        import execution
        from comfy_execution.caching import CacheKeySetInputSignature, HierarchicalCache

        self.execution = execution
        self.graph = graph
        self.image_link = image_link
        self.loader_id = loader_id
        self.prompt_id = prompt_id
        self.extra_data = extra_data
        self.outputs = HierarchicalCache(CacheKeySetInputSignature)
        self.asset_manager = execution.NoAssets(execution.args)

    # The enclosing prompt publishes the final result. Suppress intermediate node UI messages.
    client_id = None

    def send_sync(self, *_):
        pass

    async def process(self, index):
        from comfy_execution.caching import CacheKeySetID, HierarchicalCache
        from comfy_execution.graph import DynamicPrompt, ExecutionList

        execution = self.execution
        prompt = {node_id: dict(node) for node_id, node in self.graph.items()}
        loader = prompt[self.loader_id]
        loader["inputs"] = {**loader["inputs"], "_rp_frame": index}
        dynprompt = DynamicPrompt(prompt)
        outputs = self.outputs
        objects = HierarchicalCache(CacheKeySetID)
        caches = SimpleNamespace(outputs=outputs, objects=objects, all=(outputs, objects))
        changed = execution.IsChangedCache(self.prompt_id, dynprompt, outputs)
        for cache in caches.all:
            await cache.set_prompt(dynprompt, prompt.keys(), changed)
            cache.clean_unused()

        target_id, socket = self.image_link
        execution_list = ExecutionList(dynprompt, outputs)
        execution_list.add_node(self.loader_id)
        if target_id != self.loader_id:
            execution_list.add_node(target_id)
        executed, subgraphs, tasks, ui = set(), {}, {}, {}
        try:
            while not execution_list.is_empty():
                node_id, error, exception = await execution_list.stage_node_execution()
                if error:
                    raise exception
                result, error, exception = await execution.execute(
                    self, dynprompt, caches, node_id, self.extra_data, executed,
                    self.prompt_id, execution_list, subgraphs, tasks, ui, self.asset_manager,
                )
                if result == execution.ExecutionResult.FAILURE:
                    raise exception
                if result == execution.ExecutionResult.PENDING:
                    execution_list.unstage_node_execution()
                else:
                    execution_list.complete_node_execution()
        finally:
            running = [task for values in tasks.values() for task in values if isinstance(task, asyncio.Task)]
            for task in running:
                task.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)

        result = await outputs.get(target_id)
        images = result.outputs[socket] if result is not None else ()
        if (len(images) != 1 or not isinstance(images[0], torch.Tensor)
                or images[0].ndim != 4 or images[0].shape[0] != 1):
            raise ValueError("The processing chain must return exactly one IMAGE for each input frame.")
        return images[0]


class RPPreviewVideo:
    CATEGORY = CATEGORY
    DESCRIPTION = "Preview images as an MP4. Enable save_video to write it to ComfyUI/output."
    FUNCTION = "preview"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "INT", "FLOAT")
    RETURN_NAMES = ("video_path", "frame_count", "fps")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"lazy": True}),
                "fps": ("FLOAT", {"default": 0, "min": 0, "max": 240, "step": 0.001, "lazy": True,
                                  "tooltip": "0 uses the loader FPS when streaming, or 24 otherwise."}),
                "save_video": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "RP_Video"}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51}),
                "keep_source_audio": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "video_stream": (STREAM_TYPE, {"rawLink": True, "lazy": True}),
                "audio": ("AUDIO", {"lazy": True, "tooltip": "Optional audio for standalone use."}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO",
                "auth_token_comfy_org": "AUTH_TOKEN_COMFY_ORG", "api_key_comfy_org": "API_KEY_COMFY_ORG",
                "comfy_usage_source": "COMFY_USAGE_SOURCE",
            },
        }

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def check_lazy_status(self, images=None, fps=None, video_stream=None, audio=None,
                          dynprompt=None, unique_id=None, **_):
        inputs = dynprompt.get_node(unique_id)["inputs"]
        if video_stream is not None:
            # A direct FPS connection from the loader already resolves to the streaming rate.
            uses_stream_fps = inputs.get("fps") == [video_stream[0], 1]
            return ["fps"] if fps is None and not uses_stream_fps else []
        return [key for key, value in (("images", images), ("fps", fps), ("audio", audio))
                if key in inputs and value is None]

    async def preview(self, images, fps, save_video, filename_prefix, crf, keep_source_audio,
                      video_stream=None, audio=None, dynprompt=None, unique_id=None, extra_pnginfo=None,
                      auth_token_comfy_org=None, api_key_comfy_org=None, comfy_usage_source=None):
        import folder_paths
        from comfy.model_management import throw_exception_if_processing_interrupted as check_interrupt
        from comfy.utils import ProgressBar
        from comfy_execution.graph_utils import is_link
        from comfy_execution.utils import get_executing_context

        rate = Fraction(fps).limit_denominator(100_000) if fps else None
        with ExitStack() as stack:
            if video_stream is not None:
                if not is_link(video_stream):
                    raise ValueError("Connect video_stream directly from RP Load Video.")
                loader_id, socket = video_stream
                if socket != 3 or dynprompt.get_node(loader_id)["class_type"] != "RPLoadVideo":
                    raise ValueError("Connect video_stream directly from RP Load Video.")
                image_link = dynprompt.get_node(unique_id)["inputs"]["images"]
                graph = processing_graph(dynprompt, image_link, loader_id)
                stream = FrameStream(loader_id)
                stack.callback(stream.close)
                token = _active_stream.set(stream)
                stack.callback(_active_stream.reset, token)
                runner = FrameGraph(
                    graph, image_link, loader_id, get_executing_context().prompt_id,
                    dict(extra_pnginfo=extra_pnginfo, auth_token_comfy_org=auth_token_comfy_org,
                         api_key_comfy_org=api_key_comfy_org, comfy_usage_source=comfy_usage_source),
                )
                images = await runner.process(0)
                source = stream.source
                rate = rate or source.fps
                audio_path = source.path if keep_source_audio and source.has_audio else None
                audio_start = source.audio_start
                tempo = float(rate / source.fps)
                total = source.estimated_frames
            else:
                if not isinstance(images, torch.Tensor) or images.ndim != 4 or not images.shape[0]:
                    raise ValueError("Connect a non-empty IMAGE batch.")
                rate = rate or Fraction(24)
                audio_path = None
                if audio is not None:
                    audio_path = Path(stack.enter_context(tempfile.TemporaryDirectory())) / "audio.wav"
                    write_audio(audio, audio_path, check_interrupt)
                audio_start, tempo, total = 0, 1, images.shape[0]

            path, preview = stack.enter_context(
                video_destination(folder_paths, save_video, filename_prefix, images.shape[1:3])
            )
            encoder = VideoEncoder(path, rate, crf, audio_path, audio_start, tempo)
            progress = ProgressBar(max(total, 1), node_id=unique_id)
            with encoder.open(images, check_interrupt) as pipe:
                if video_stream is not None:
                    while True:
                        check_interrupt()
                        encoder.write(pipe, images)
                        del images
                        progress.update_absolute(encoder.count, max(total, encoder.count + 1))
                        await asyncio.sleep(0)
                        check_interrupt()
                        if not stream.advance():
                            break
                        images = await runner.process(encoder.count)
                else:
                    for frame in images:
                        check_interrupt()
                        encoder.write(pipe, frame.unsqueeze(0))
                        progress.update(1)
            check_interrupt()
            progress.update_absolute(encoder.count, encoder.count)

        path = preview.pop("path")
        return {"ui": {"rp_video": [preview]}, "result": (path, encoder.count, float(rate))}


def register_upload():
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return
    if not getattr(PromptServer, "instance", None):
        return

    @PromptServer.instance.routes.post("/rp/video/upload")
    async def upload_video(request):
        name = request.query.get("filename", "")
        if (not name or Path(name).name != name or "/" in name or "\\" in name
                or Path(name).suffix.lower() not in VIDEO_EXTENSIONS):
            raise web.HTTPBadRequest(text="Choose a supported video file.")
        root = _input_directory()
        directory = (root / "rp_videos").resolve()
        if not directory.is_relative_to(root):
            raise web.HTTPBadRequest(text="Invalid video upload directory.")
        directory.mkdir(parents=True, exist_ok=True)
        stamp = uuid4().hex[:12]
        path = directory / f"{stamp}_{name}"
        temporary = directory / f".{stamp}.part"
        try:
            with temporary.open("xb") as output:
                async for chunk in request.content.iter_chunked(1024 * 1024):
                    output.write(chunk)
            if temporary.stat().st_size == 0:
                raise web.HTTPBadRequest(text="The video file is empty.")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return web.json_response({"video": path.relative_to(root).as_posix()})


register_upload()
