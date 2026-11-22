import asyncio
import base64
import io as buffer_io
import json
import logging
from collections import OrderedDict, defaultdict
from functools import lru_cache
from pathlib import Path
from threading import Event, Lock
from time import monotonic
from uuid import uuid4

import av
import numpy as np
import torch
import torch.nn.functional as F
from aiohttp import web
from PIL import Image
from scipy.ndimage import maximum_filter

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
DEFAULT_EXPANSION = 10
MAX_EXPANSION = 256
TRACK_CACHE = OrderedDict()
TRACK_CACHE_LOCK = Lock()
PREVIEW_JOBS = OrderedDict()
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
        size = None
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < 0:
                continue
            if size is None:
                size = (frame.height, frame.width) if round(frame.rotation / 90) % 2 else (frame.width, frame.height)
            if frame.key_frame:
                seek_point = frame.pts
            frames.append((frame.pts, seek_point))
        if not frames:
            raise ValueError("The video has no decodable frames.")
        time_base = float(stream.time_base)
        pts0 = frames[0][0]
        timestamps = [(pts - pts0) * time_base for pts, _ in frames]
        return frames, timestamps, size


def preview_frame(source, index):
    frames, _, _ = video_timeline(source)
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
        return preview_pixels(image)


def preview_pixels(image):
    scale = min(1, 1008 / max(image.width, image.height))
    rgb = image.reformat(
        width=max(1, round(image.width * scale)),
        height=max(1, round(image.height * scale)),
        format="rgb24",
    ).to_ndarray()
    rotation = round(image.rotation / 90) % 4
    return np.ascontiguousarray(np.rot90(rgb, rotation))


def parse_groups(data, filename, frame_count):
    if not isinstance(data, dict):
        raise ValueError("Use the editor to configure selection groups.")
    groups = data.get("groups")
    if groups is None and isinstance(data.get("layers"), list):
        groups = []
        for layer in data["layers"]:
            if not isinstance(layer, dict):
                raise ValueError("Invalid video selection.")
            groups.append({**layer, "name": layer.get("text"), "parts": [layer]})
    if not isinstance(groups, list):
        raise ValueError("Use the editor to configure selection groups.")
    if groups and data.get("video") != filename:
        raise ValueError("These selections belong to a different video. Open the editor to select the new video.")

    normalized = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("parts"), list) or not group["parts"]:
            raise ValueError("Every group must contain at least one selection.")
        name = group.get("name")
        if not isinstance(name, str) or not 0 < len(name.strip()) <= 120:
            raise ValueError("Enter a group name of up to 120 characters.")
        kind = group.get("preprocessor", "segmentation")
        if kind not in PREPROCESSORS:
            raise ValueError("Unknown group preprocessor.")
        color = group.get("color", "#1678ff")
        if not isinstance(color, str) or len(color) != 7 or not color.startswith("#"):
            raise ValueError("Group color must be a hex RGB color.")
        int(color[1:], 16)
        expand = group.get("expand", DEFAULT_EXPANSION)
        if type(expand) is not int or not 0 <= expand <= MAX_EXPANSION:
            raise ValueError(f"Mask expansion must be between 0 and {MAX_EXPANSION} pixels.")
        parts = [parse_part(part, frame_count) for part in group["parts"]]
        normalized.append({"name": name, "preprocessor": kind, "color": color, "expand": expand, "parts": parts})
    return normalized


def parse_part(part, frame_count):
    if not isinstance(part, dict):
        raise ValueError("Invalid selection.")
    validate_prompt(part.get("text"), part.get("threshold"))
    anchor, start, end = part.get("frame"), part.get("start", 0), part.get("end")
    end = frame_count - 1 if end is None else end
    if not all(type(x) is int for x in (anchor, start, end)) or not 0 <= start <= anchor <= end < frame_count:
        raise ValueError("The tracking range must contain its anchor frame and stay inside the video.")
    instances = part.get("instances")
    if instances is not None and (
        not isinstance(instances, list) or not instances
        or any(type(index) is not int or not 0 <= index < MAX_INSTANCES for index in instances)
        or len(set(instances)) != len(instances)
    ):
        raise ValueError("Choose at least one matching mask in the editor.")
    return {
        "text": part["text"].strip(), "threshold": part["threshold"], "frame": anchor,
        "start": start, "end": end, "instances": instances,
    }


def validate_prompt(prompt, threshold):
    if not isinstance(prompt, str) or not 0 < len(prompt.strip()) <= 120 or "," in prompt or ":" in prompt:
        raise ValueError("Enter one short description per selection, without commas or colons.")
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


def read_tracking_frames(source, start, end, check):
    """Decode only the needed span at the tracker preview size."""
    with av.open(str(source[0])) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        index = -1
        frames = None
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < 0:
                continue
            index += 1
            if index < start:
                continue
            if check:
                check()
            pixels = preview_pixels(frame)
            if frames is None:
                frames = np.empty((end - start + 1, *pixels.shape), dtype=np.float16)
            np.multiply(pixels, np.float32(1 / 255), out=frames[index - start], dtype=np.float32)
            if index == end:
                return torch.from_numpy(frames)
    raise ValueError("The video changed before all tracking frames could be decoded.")


class PreviewProgress:
    def __init__(self, check):
        self.check = check

    def update(self, _):
        self.check()


def run_tracker(images, model, seeds, check):
    model_management.load_model_gpu(model)
    device = model_management.get_torch_device()
    dtype = model.model.get_dtype()
    progress = PreviewProgress(check) if check else comfy.utils.ProgressBar(len(images))
    result = model.model.diffusion_model.forward_video(
        images=images.movedim(-1, 1),
        initial_masks=seeds.unsqueeze(1).to(device=device, dtype=dtype),
        pbar=progress,
        max_objects=len(seeds),
        target_device=device,
        target_dtype=dtype,
    )
    return result["packed_masks"].cpu()


def track_parts(source, parts, check=None):
    grouped = defaultdict(list)
    first_frame = min(part["start"] for part in parts)
    last_frame = max(part["end"] for part in parts)
    tracking_frames = read_tracking_frames(source, first_frame, last_frame, check)
    for index, part in enumerate(parts):
        grouped[part["frame"], part["start"], part["end"]].append(index)

    tracks = [None] * len(parts)
    for (anchor, start, end), indexes in grouped.items():
        frame = preview_frame(source, anchor)
        seeds = []
        for index in indexes:
            part = parts[index]
            (check or model_management.throw_exception_if_processing_interrupted)()
            model, masks, _ = detect_objects(frame, part["text"], part["threshold"])
            chosen = part["instances"]
            if not len(masks) or (chosen is not None and max(chosen) >= len(masks)):
                raise ValueError(f'Could not find the selected masks for "{part["text"]}" on frame {anchor + 1}. Recheck the selection in the editor.')
            seeds.append(masks if chosen is None else masks[chosen])

        offset = 0
        while offset < len(seeds):
            limit = offset
            count = 0
            while limit < len(seeds) and count + len(seeds[limit]) <= MAX_TRACK_OBJECTS:
                count += len(seeds[limit])
                limit += 1
            batch = torch.cat(seeds[offset:limit])
            local_anchor, local_start = anchor - first_frame, start - first_frame
            forward = run_tracker(tracking_frames[local_anchor:end - first_frame + 1], model, batch, check)
            packed = forward
            if start < anchor:
                backward = run_tracker(tracking_frames[local_start:local_anchor + 1].flip(0), model, batch, check)
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


def union_tracks(tracks):
    if len(tracks) == 1:
        return tracks[0]
    start = min(first for first, _ in tracks)
    end = max(first + len(masks) for first, masks in tracks)
    union = tracks[0][1].new_zeros((end - start, *tracks[0][1].shape[1:]))
    for first, masks in tracks:
        offset = first - start
        union[offset:offset + len(masks)] |= masks
    return start, union


def track_groups(source, groups, check=None):
    key = source, json.dumps([group["parts"] for group in groups], sort_keys=True)
    with TRACK_CACHE_LOCK:
        cached = next((entry for entry in TRACK_CACHE.values() if entry[0] == key), None)
        if cached is not None:
            TRACK_CACHE.move_to_end(cached[1])
            return cached[1], cached[2]

    parts = [part for group in groups for part in group["parts"]]
    tracks = track_parts(source, parts, check)
    grouped = []
    offset = 0
    for group in groups:
        end = offset + len(group["parts"])
        grouped.append(union_tracks(tracks[offset:end]))
        offset = end

    ticket = uuid4().hex
    with TRACK_CACHE_LOCK:
        TRACK_CACHE[ticket] = key, ticket, grouped
        while len(TRACK_CACHE) > 2:
            TRACK_CACHE.popitem(last=False)
    return ticket, grouped


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


def expanded_masks(packed, height, width, radius):
    from comfy.ldm.sam3.tracker import unpack_masks

    mask = unpack_masks(packed).unsqueeze(1).float()
    mask = F.interpolate(mask, (height, width), mode="bilinear", align_corners=False).squeeze(1) > 0.5
    if radius:
        pixels = mask.numpy()
        diameter = 2 * radius + 1
        maximum_filter(pixels, size=(1, diameter, diameter), mode="constant", output=pixels)
    return mask


def apply_preprocessors(images, groups, tracks):
    kinds = {group["preprocessor"] for group in groups} - {"segmentation"}
    processors = prepare_preprocessors(kinds) if kinds else {}
    progress = comfy.utils.ProgressBar(len(images))
    height, width = images.shape[1:3]
    for start in range(0, len(images), 4):
        model_management.throw_exception_if_processing_interrupted()
        chunk = images[start:start + 4]
        active = []
        for group, (first, packed) in zip(groups, tracks):
            left = max(start, first)
            right = min(start + len(chunk), first + len(packed))
            if left < right:
                active.append((group, slice(left - start, right - start), packed[left - first:right - first]))

        needed = {group["preprocessor"] for group, _, _ in active} & processors.keys()
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

        for group, section, packed in active:
            mask = expanded_masks(packed, height, width, group["expand"])
            kind = group["preprocessor"]
            if kind == "segmentation":
                value = group["color"]
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
            description="Select objects by text before queueing and apply a different map to each tracked group.",
            search_aliases=["SAM3 video", "selective depth", "video segmentation", "video restoration"],
            inputs=[
                io.Combo.Input("video", options=_available_videos(), upload=io.UploadType.video),
                io.String.Input("selections", default='{"groups":[]}'),
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
        groups = parse_groups(json.loads(selections), video, len(components.images))
        if not groups:
            return io.NodeOutput(original, components.images)
        with INFERENCE_LOCK:
            _, tracks = track_groups(source, groups)
            apply_preprocessors(components.images, groups, tracks)
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
    return {"candidates": [{"score": score, "mask": encode_mask(mask)} for mask, score in zip(masks, scores)]}


def check_saved_prompts(source, frame, prompts):
    counts = []
    union = None
    with INFERENCE_LOCK, torch.inference_mode():
        pixels = preview_frame(source, frame)
        for prompt, threshold in prompts:
            _, masks, _ = detect_objects(pixels, prompt, threshold)
            counts.append({"text": prompt, "matches": len(masks)})
            if len(masks):
                combined = masks.any(dim=0)
                union = combined if union is None else union | combined
    return {"mask": encode_mask(union) if union is not None else None, "results": counts}


def preview_inference_busy():
    return (PromptServer.instance.prompt_queue.get_tasks_remaining() or INFERENCE_LOCK.locked()
            or any(not job["done"] for job in PREVIEW_JOBS.values()))


def encode_mask(mask):
    return base64.b64encode(encoded_image(mask_overlay(mask), "PNG")).decode("ascii")


def preview_group_frame(tracks, frame):
    from comfy.ldm.sam3.tracker import unpack_masks

    return [
        encode_mask(unpack_masks(packed[frame - first]))
        if first <= frame < first + len(packed) and packed[frame - first].any()
        else None
        for first, packed in tracks
    ]


def preview_tracking(source, groups, job):
    def check():
        if job["cancel"].is_set() or monotonic() - job["last_seen"] > 30:
            raise InterruptedError("Tracking preview was cancelled.")
        if PromptServer.instance.prompt_queue.get_tasks_remaining():
            raise InterruptedError("Tracking stopped when a workflow was queued.")

    with INFERENCE_LOCK, torch.inference_mode():
        check()
        ticket, _ = track_groups(source, groups, check)
    return ticket


async def run_preview_job(source, groups, job):
    try:
        job["ticket"] = await asyncio.to_thread(preview_tracking, source, groups, job)
    except InterruptedError as exc:
        job["error"] = str(exc)
    except Exception as exc:
        log.exception("Selective video tracking preview failed")
        job["error"] = str(exc)
    finally:
        job["done"] = True


if getattr(PromptServer, "instance", None) is not None:
    routes = PromptServer.instance.routes

    @routes.get("/rp/selective-video/info")
    async def selective_video_info(request):
        try:
            source = input_video(request.query.get("video", ""))
            _, timestamps, (width, height) = await asyncio.to_thread(video_timeline, source)
            return web.json_response({"timestamps": timestamps, "width": width, "height": height})
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
            if preview_inference_busy():
                return web.json_response({"error": "ComfyUI is busy. Try detection when the queue is idle."}, status=409)
            result = await asyncio.to_thread(detect_preview, source, frame, prompt, threshold)
            return web.json_response(result)
        except (OSError, ValueError, av.error.FFmpegError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            log.exception("Selective video detection failed")
            return web.json_response({"error": str(exc)}, status=500)

    @routes.post("/rp/selective-video/check")
    async def selective_video_check(request):
        try:
            body = await request.json()
            if not isinstance(body, dict) or not isinstance(body.get("video"), str):
                raise ValueError("Choose an input video first.")
            parts = body.get("prompts")
            if not isinstance(parts, list) or not parts:
                raise ValueError("Choose a saved selection or group first.")
            prompts = {}
            for part in parts:
                if not isinstance(part, dict):
                    raise ValueError("Invalid saved selection.")
                prompt, threshold = part.get("text"), part.get("threshold")
                validate_prompt(prompt, threshold)
                prompts[prompt.strip(), threshold] = None
            source = input_video(body["video"])
            if preview_inference_busy():
                return web.json_response({"error": "ComfyUI is busy. Try detection when the queue is idle."}, status=409)
            result = await asyncio.to_thread(check_saved_prompts, source, body.get("frame"), prompts)
            return web.json_response(result)
        except (OSError, ValueError, av.error.FFmpegError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            log.exception("Selective video check failed")
            return web.json_response({"error": str(exc)}, status=500)


    @routes.post("/rp/selective-video/track")
    async def selective_video_track(request):
        try:
            data = await request.json()
            if not isinstance(data, dict) or not isinstance(data.get("video"), str):
                raise ValueError("Choose an input video first.")
            source = input_video(data["video"])
            frames, _, _ = await asyncio.to_thread(video_timeline, source)
            groups = parse_groups(data, data["video"], len(frames))
            if not groups:
                raise ValueError("Add at least one selection to preview tracking.")
            if preview_inference_busy():
                return web.json_response({"error": "ComfyUI is busy. Try tracking when the queue is idle."}, status=409)
            job_id = uuid4().hex
            job = {"done": False, "ticket": None, "error": None, "cancel": Event(), "last_seen": monotonic()}
            PREVIEW_JOBS[job_id] = job
            while len(PREVIEW_JOBS) > 4:
                PREVIEW_JOBS.popitem(last=False)
            job["task"] = asyncio.create_task(run_preview_job(source, groups, job))
            return web.json_response({"job": job_id}, status=202)
        except (OSError, ValueError, av.error.FFmpegError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.get("/rp/selective-video/track-status")
    async def selective_video_track_status(request):
        job = PREVIEW_JOBS.get(request.query.get("job", ""))
        if job is None:
            return web.json_response({"error": "Tracking preview expired. Run Track & preview again."}, status=404)
        job["last_seen"] = monotonic()
        return web.json_response({key: job[key] for key in ("done", "ticket", "error")})

    @routes.delete("/rp/selective-video/track")
    async def selective_video_track_cancel(request):
        job = PREVIEW_JOBS.get(request.query.get("job", ""))
        if job is not None:
            job["cancel"].set()
        return web.Response(status=204)

    @routes.get("/rp/selective-video/track-frame")
    async def selective_video_track_frame(request):
        ticket = request.query.get("ticket", "")
        with TRACK_CACHE_LOCK:
            cached = TRACK_CACHE.get(ticket)
            if cached is not None:
                TRACK_CACHE.move_to_end(ticket)
        if cached is None:
            return web.json_response({"error": "Tracking preview expired. Run Track & preview again."}, status=404)
        source = cached[0][0]
        try:
            frame = int(request.query.get("frame", "-1"))
            frames, _, _ = await asyncio.to_thread(video_timeline, source)
            if not 0 <= frame < len(frames):
                raise ValueError("Frame is outside the video.")
            masks = await asyncio.to_thread(preview_group_frame, cached[2], frame)
            return web.json_response({"masks": masks})
        except (OSError, ValueError, av.error.FFmpegError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
