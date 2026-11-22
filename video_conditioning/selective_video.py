import asyncio
import base64
import io as buffer_io
import json
import logging
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from threading import Lock

import av
import numpy as np
import torch
import torch.nn.functional as F
from aiohttp import web
from PIL import Image

import comfy.model_management as model_management
import comfy.utils
import folder_paths
from comfy_api.latest import InputImpl, io
from server import PromptServer

from ..video_frame_processing.video_frames_workflow import VIDEO_EXTENSIONS, _available_videos


CATEGORY = "RP/Video Conditioning"
CHECKPOINT = "sam3.1_multiplex_fp16.safetensors"
PREPROCESSORS = ("segmentation", "depth", "canny", "openpose")
MAX_INSTANCES = 12
MAX_TRACK_OBJECTS = 64
INFERENCE_LOCK = Lock()
log = logging.getLogger(__name__)


def input_video(filename):
    root = Path(folder_paths.get_input_directory()).resolve()
    path = (root / filename).resolve()
    if not filename or Path(filename).anchor or not path.is_relative_to(root):
        raise ValueError("Choose a video from the ComfyUI input directory.")
    if path.suffix.lower() not in VIDEO_EXTENSIONS or not path.is_file():
        raise ValueError(f"Input video not found or unsupported: {filename}")
    stat = path.stat()
    return path, stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=4)
def video_timeline(source):
    with av.open(str(source[0])) as container:
        if not container.streams.video:
            raise ValueError("The file has no video stream.")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frames = []
        seek_point = None
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < 0:
                continue
            if frame.key_frame:
                seek_point = frame.pts
            frames.append((frame.pts, seek_point))
        if not frames:
            raise ValueError("The video has no decodable frames.")
        time_base = float(stream.time_base)
        pts0 = frames[0][0]
        timestamps = [(pts - pts0) * time_base for pts, _ in frames]
        return frames, timestamps


def preview_frame(source, index):
    frames, _ = video_timeline(source)
    if type(index) is not int or not 0 <= index < len(frames):
        raise ValueError("Frame is outside the video.")
    pts, keyframe = frames[index]
    unique_pts = all(
        not 0 <= other < len(frames) or frames[other][0] != pts
        for other in (index - 1, index + 1)
    )
    with av.open(str(source[0])) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        image = None
        if unique_pts and keyframe is not None:
            container.seek(keyframe, stream=stream, backward=True)
            image = next((frame for frame in container.decode(stream) if frame.pts == pts), None)
        if image is None:
            container.seek(0, stream=stream, backward=True)
            decoded = (frame for frame in container.decode(stream) if frame.pts is not None and frame.pts >= 0)
            image = next((frame for i, frame in enumerate(decoded) if i == index), None)
        if image is None:
            raise ValueError("Could not decode the selected frame.")
        scale = min(1, 1008 / max(image.width, image.height))
        rgb = image.reformat(
            width=max(1, round(image.width * scale)),
            height=max(1, round(image.height * scale)),
            format="rgb24",
        ).to_ndarray()
        rotation = round(image.rotation / 90) % 4
        return np.ascontiguousarray(np.rot90(rgb, rotation))


def parse_layers(value, filename, frame_count):
    data = json.loads(value)
    if not isinstance(data, dict) or not isinstance(data.get("layers"), list):
        raise ValueError("Use the selection editor to configure the video layers.")
    layers = data["layers"]
    if layers and data.get("video") != filename:
        raise ValueError("These selections belong to a different video. Open the editor to select the new video.")
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("Invalid video layer.")
        validate_prompt(layer.get("text"), layer.get("threshold"))
        color = layer.get("color", "#1678ff")
        if not isinstance(color, str) or len(color) != 7 or not color.startswith("#"):
            raise ValueError("Selection color must be a hex RGB color.")
        int(color[1:], 16)
        if layer.get("preprocessor", "segmentation") not in PREPROCESSORS:
            raise ValueError("Unknown selection preprocessor.")
        anchor, start, end = layer.get("frame"), layer.get("start", 0), layer.get("end")
        end = frame_count - 1 if end is None else end
        if not all(type(x) is int for x in (anchor, start, end)) or not 0 <= start <= anchor <= end < frame_count:
            raise ValueError("The selection range must contain its anchor frame and stay inside the video.")
        instances = layer.get("instances")
        if instances is not None and (
            not isinstance(instances, list) or not instances
            or any(type(index) is not int or not 0 <= index < MAX_INSTANCES for index in instances)
            or len(set(instances)) != len(instances)
        ):
            raise ValueError("Choose at least one detected instance in the editor.")
    return layers


def validate_prompt(prompt, threshold):
    if not isinstance(prompt, str) or not 0 < len(prompt.strip()) <= 120 or "," in prompt or ":" in prompt:
        raise ValueError("Enter one short description per layer, without commas or colons.")
    if type(threshold) not in (int, float) or not 0.01 <= threshold <= 0.99:
        raise ValueError("Detection confidence must be between 0.01 and 0.99.")


@lru_cache(maxsize=1)
def load_sam(path, modified, size):
    import comfy.sd

    model, clip, *_ = comfy.sd.load_checkpoint_guess_config(path, output_vae=False)
    return model, clip


def detect_objects(frame, prompt, threshold):
    from comfy_extras.nodes_sam3 import _extract_text_prompts, _refine_mask

    path = folder_paths.get_full_path("checkpoints", CHECKPOINT)
    if not path:
        raise ValueError(f"Place {CHECKPOINT} in ComfyUI/models/checkpoints to enable SAM 3.1.")
    stat = Path(path).stat()
    model, clip = load_sam(path, stat.st_mtime_ns, stat.st_size)
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(prompt.strip()))
    model_management.load_model_gpu(model)
    device = model_management.get_torch_device()
    dtype = model.model.get_dtype()
    sam = model.model.diffusion_model
    embeddings, text_mask, _ = _extract_text_prompts(conditioning, device, dtype)[0]
    image = torch.from_numpy(frame).float().div_(255)
    height, width = image.shape[:2]
    prepared = comfy.utils.common_upscale(image.movedim(-1, 0).unsqueeze(0), 1008, 1008, "bilinear", "disabled")
    result = sam(prepared.to(device=device, dtype=dtype), text_embeddings=embeddings,
                 text_mask=text_mask, threshold=threshold, orig_size=(height, width))

    scores = result["scores"][0].sigmoid()
    order = scores.argsort(descending=True)[:MAX_INSTANCES]
    order = order[scores[order] > threshold]
    masks = [
        _refine_mask(sam, image, result["masks"][0, i], result["boxes"][0, i].cpu(),
                     height, width, device, dtype, 2).to(device="cpu", dtype=torch.bool)
        for i in order.tolist()
    ]
    mask_batch = torch.cat(masks) if masks else torch.empty((0, height, width), dtype=torch.bool)
    return model, mask_batch, scores[order].tolist()


def resize_frames(images, height, width):
    if images.shape[1:3] == (height, width):
        return images
    return F.interpolate(images.movedim(-1, 1), (height, width), mode="bilinear", align_corners=False).movedim(1, -1)


def track_layers(source, images, layers):
    from comfy_extras.nodes_sam3 import SAM3_VideoTrack

    grouped = defaultdict(list)
    frame_count, height, width, _ = images.shape
    for index, layer in enumerate(layers):
        end = layer.get("end")
        grouped[layer["frame"], layer.get("start", 0), frame_count - 1 if end is None else end].append(index)

    scale = min(1, 1008 / max(width, height))
    tracking_size = max(1, round(height * scale)), max(1, round(width * scale))
    tracking_frames = images
    if tracking_size != (height, width):
        tracking_frames = torch.empty((frame_count, *tracking_size, 3), dtype=images.dtype)
        for offset in range(0, frame_count, 8):
            tracking_frames[offset:offset + 8] = resize_frames(images[offset:offset + 8], *tracking_size)

    tracks = [None] * len(layers)
    for (anchor, start, end), indexes in grouped.items():
        frame = preview_frame(source, anchor)
        seeds = []
        for index in indexes:
            layer = layers[index]
            model_management.throw_exception_if_processing_interrupted()
            model, masks, _ = detect_objects(frame, layer["text"], layer["threshold"])
            chosen = layer.get("instances")
            if not len(masks) or (chosen is not None and max(chosen) >= len(masks)):
                raise ValueError(f'Could not find the selected instances for "{layer["text"]}" on frame {anchor + 1}. Recheck the layer in the editor.')
            seeds.append(masks if chosen is None else masks[chosen])

        # SAM tracks independent objects. Keep each layer's instance slices intact when batching.
        offset = 0
        while offset < len(seeds):
            limit = offset
            count = 0
            while limit < len(seeds) and count + len(seeds[limit]) <= MAX_TRACK_OBJECTS:
                count += len(seeds[limit])
                limit += 1
            batch = torch.cat(seeds[offset:limit]).float()
            forward = SAM3_VideoTrack.execute(
                tracking_frames[anchor:end + 1], model, initial_mask=batch, max_objects=count,
            ).result[0]["packed_masks"].cpu()
            packed = forward
            if start < anchor:
                backward = SAM3_VideoTrack.execute(
                    tracking_frames[start:anchor + 1].flip(0), model, initial_mask=batch, max_objects=count,
                ).result[0]["packed_masks"].cpu()
                packed = torch.cat((backward[1:].flip(0), forward))
            column = 0
            for index, masks in zip(indexes[offset:limit], seeds[offset:limit]):
                union = packed[:, column].clone()
                for instance in range(column + 1, column + len(masks)):
                    union |= packed[:, instance]
                tracks[index] = (start, union)
                column += len(masks)
            offset = limit
    return tracks


def prepare_preprocessors(kinds):
    from comfy_extras.nodes_canny import Canny

    device = model_management.get_torch_device()
    processors = {}
    if "canny" in kinds:
        def canny(images):
            height, width = images.shape[1:3]
            scale = min(1, 1024 / max(height, width))
            images = resize_frames(images, max(3, round(height * scale)), max(3, round(width * scale)))
            return Canny.execute(images, 0.1, 0.3).result[0]
        processors["canny"] = canny

    if kinds & {"depth", "openpose"}:
        try:
            from custom_controlnet_aux.depth_anything_v2 import DepthAnythingV2Detector
            from custom_controlnet_aux.open_pose import OpenposeDetector
        except ImportError as exc:
            raise RuntimeError("Install comfyui_controlnet_aux to use Depth or OpenPose.") from exc
        if "depth" in kinds:
            depth = DepthAnythingV2Detector.from_pretrained(filename="depth_anything_v2_vitl.pth").to(device)
            processors["depth"] = lambda frame: depth(frame, detect_resolution=1024, max_depth=1, output_type="np")
        if "openpose" in kinds:
            pose = OpenposeDetector.from_pretrained().to(device)
            processors["openpose"] = lambda frame: pose(
                frame, detect_resolution=512, include_hand=True, include_face=True, output_type="np",
            )
    return processors


def apply_preprocessors(images, layers, tracks):
    from comfy.ldm.sam3.tracker import unpack_masks

    kinds = {layer.get("preprocessor", "segmentation") for layer in layers} - {"segmentation"}
    processors = prepare_preprocessors(kinds) if kinds else {}
    progress = comfy.utils.ProgressBar(len(images))
    height, width = images.shape[1:3]
    for start in range(0, len(images), 4):
        model_management.throw_exception_if_processing_interrupted()
        chunk = images[start:start + 4]
        active = []
        for layer, (first, packed) in zip(layers, tracks):
            left = max(start, first)
            right = min(start + len(chunk), first + len(packed))
            if left < right:
                active.append((layer, slice(left - start, right - start), packed[left - first:right - first]))

        needed = {layer.get("preprocessor", "segmentation") for layer, _, _ in active} & processors.keys()
        maps = {}
        # Evaluate each map from the original RGB before any layer writes.
        if needed & {"depth", "openpose"}:
            rgb = (chunk.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        for name in needed:
            processor = processors[name]
            if name == "canny":
                mapped = processor(chunk).to(chunk)
            else:
                mapped = torch.from_numpy(np.stack([processor(frame) for frame in rgb])).to(chunk).div_(255)
            maps[name] = resize_frames(mapped, height, width)

        for layer, section, packed in active:
            mask = unpack_masks(packed).unsqueeze(1).float()
            mask = F.interpolate(mask, (height, width), mode="bilinear", align_corners=False).squeeze(1) > 0.5
            kind = layer.get("preprocessor", "segmentation")
            if kind == "segmentation":
                value = layer.get("color", "#1678ff")
                color = chunk.new_tensor([int(value[i:i + 2], 16) / 255 for i in (1, 3, 5)])
                chunk[section][mask] = color
            else:
                chunk[section][mask] = maps[kind][section][mask]
        progress.update(len(chunk))


class RPSelectiveVideoInput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RPSelectiveVideoInput",
            display_name="RP Selective Video Input",
            category=CATEGORY,
            description="Select objects by text before queueing and apply a different map to each tracked silhouette.",
            search_aliases=["SAM3 video", "selective depth", "video segmentation", "video restoration"],
            inputs=[
                io.Combo.Input("video", options=_available_videos(), upload=io.UploadType.video),
                io.String.Input("selections", default='{"layers":[]}'),
            ],
            outputs=[io.Video.Output(display_name="video"), io.Image.Output(display_name="frames")],
        )

    @classmethod
    def fingerprint_inputs(cls, video, selections):
        return input_video(video)[1:]

    @classmethod
    def validate_inputs(cls, video):
        try:
            input_video(video)
        except (OSError, ValueError) as exc:
            return str(exc)
        return True

    @classmethod
    def execute(cls, video, selections):
        source = input_video(video)
        original = InputImpl.VideoFromFile(str(source[0]))
        components = original.get_components()
        layers = parse_layers(selections, video, len(components.images))
        if not layers:
            return io.NodeOutput(original, components.images)
        with INFERENCE_LOCK:
            tracks = track_layers(source, components.images, layers)
            apply_preprocessors(components.images, layers, tracks)
        output = InputImpl.VideoFromComponents(
            components, original.get_bit_depth(), original.get_color_space(), original.get_chroma_subsampling(),
        )
        return io.NodeOutput(output, components.images)


def encoded_image(pixels, format):
    stream = buffer_io.BytesIO()
    Image.fromarray(pixels).save(stream, format=format)
    return stream.getvalue()


def mask_overlay(mask):
    rgba = np.full((*mask.shape, 4), 255, dtype=np.uint8)
    rgba[..., 3] = mask.numpy() * 255
    return rgba


def detect_preview(source, frame, prompt, threshold):
    with INFERENCE_LOCK, torch.inference_mode():
        _, masks, scores = detect_objects(preview_frame(source, frame), prompt, threshold)
    return {
        "candidates": [
            {
                "score": score,
                "mask": base64.b64encode(encoded_image(mask_overlay(mask), "PNG")).decode("ascii"),
            }
            for mask, score in zip(masks, scores)
        ],
    }


if PromptServer.instance is not None:
    routes = PromptServer.instance.routes

    @routes.get("/rp/selective-video/info")
    async def selective_video_info(request):
        try:
            source = input_video(request.query.get("video", ""))
            _, timestamps = await asyncio.to_thread(video_timeline, source)
            return web.json_response({"timestamps": timestamps})
        except (OSError, ValueError, av.error.FFmpegError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.get("/rp/selective-video/frame")
    async def selective_video_frame(request):
        try:
            source = input_video(request.query.get("video", ""))
            index = int(request.query.get("frame", "-1"))
            pixels = await asyncio.to_thread(preview_frame, source, index)
            return web.Response(body=encoded_image(pixels, "JPEG"), content_type="image/jpeg")
        except (OSError, ValueError, av.error.FFmpegError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.post("/rp/selective-video/detect")
    async def selective_video_detect(request):
        try:
            body = await request.json()
            if not isinstance(body, dict) or not isinstance(body.get("video"), str):
                raise ValueError("Choose an input video first.")
            source = input_video(body["video"])
            frame, prompt, threshold = body.get("frame"), body.get("text"), body.get("threshold")
            validate_prompt(prompt, threshold)
            if PromptServer.instance.prompt_queue.get_tasks_remaining() or INFERENCE_LOCK.locked():
                return web.json_response({"error": "ComfyUI is busy. Try detection when the queue is idle."}, status=409)
            result = await asyncio.to_thread(detect_preview, source, frame, prompt, threshold)
            return web.json_response(result)
        except (OSError, ValueError, av.error.FFmpegError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            log.exception("Selective video detection failed")
            return web.json_response({"error": str(exc)}, status=500)
