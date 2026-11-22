import asyncio
import base64
import io as buffer_io
import json
import logging
from bisect import bisect_left
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from fractions import Fraction
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
from scipy.ndimage import binary_fill_holes, maximum_filter

import comfy.model_management as model_management
import comfy.utils
import folder_paths
from comfy_api.latest import InputImpl, VideoComponents, io
from nodes import MAX_RESOLUTION
from server import PromptServer

from ..video_frame_processing.video_frames_workflow import VIDEO_EXTENSIONS, _available_videos
from .video_preview import cached_preview, preview_pixels


CATEGORY = "RP/Video Conditioning"
CHECKPOINT = "sam3.1_multiplex_fp16.safetensors"
PREPROCESSORS = ("segmentation", "depth", "canny", "openpose")
MAX_INSTANCES = 12
MAX_TRACK_OBJECTS = 64
DEFAULT_EXPANSION = 10
MAX_EXPANSION = 256
# Spatial alignment and frame-count grid for the VHS loading presets.
LOAD_FORMATS = {
    "None": (1, 1, 0),
    "AnimateDiff": (8, 1, 0),
    "Mochi": (16, 6, 1),
    "LTXV": (32, 8, 1),
    "Hunyuan": (16, 4, 1),
    "Cosmos": (16, 8, 1),
    "Wan": (8, 4, 1),
    "H3": (32, 17, 5),
}
VideoInfo = io.Custom("VHS_VIDEOINFO")
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


@dataclass(frozen=True)
class VideoTimeline:
    frames: list[tuple[int, int | None]]
    timestamps: list[float]
    size: tuple[int, int]
    time_base: Fraction
    rate: Fraction
    duration: float


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
        last_duration = None
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < 0:
                continue
            if size is None:
                size = (frame.height, frame.width) if round(frame.rotation / 90) % 2 else (frame.width, frame.height)
            if frame.key_frame:
                seek_point = frame.pts
            frames.append((frame.pts, seek_point))
            last_duration = frame.duration
        if not frames:
            raise ValueError("The video has no decodable frames.")
        time_base = Fraction(stream.time_base)
        rate = Fraction(stream.average_rate or 1)
        pts0 = frames[0][0]
        seconds_per_tick = float(time_base)
        timestamps = [(pts - pts0) * seconds_per_tick for pts, _ in frames]
        last_span = last_duration * time_base if last_duration else 1 / rate
        duration = float((frames[-1][0] - pts0) * time_base + last_span)
        return VideoTimeline(frames, timestamps, size, time_base, rate, duration)


def preview_frame(source, index):
    frames = video_timeline(source).frames
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
        if not isinstance(group, dict) or not isinstance(group.get("parts"), list):
            raise ValueError("Invalid selection group.")
        if not group["parts"]:
            continue
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
        fill_holes = group.get("fill_holes", True)
        if type(fill_holes) is not bool:
            raise ValueError("Fill holes must be enabled or disabled.")
        parts = [parse_part(part, frame_count) for part in group["parts"]]
        normalized.append({
            "name": name, "preprocessor": kind, "color": color, "expand": expand,
            "fill_holes": fill_holes, "parts": parts,
        })
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


def expanded_masks(packed, height, width, radius, fill_holes=True):
    from comfy.ldm.sam3.tracker import unpack_masks

    mask = unpack_masks(packed).unsqueeze(1).float()
    mask = F.interpolate(mask, (height, width), mode="bilinear", align_corners=False).squeeze(1) > 0.5
    pixels = mask.numpy()
    if fill_holes:
        for frame in pixels:
            binary_fill_holes(frame, output=frame)
    if radius:
        diameter = 2 * radius + 1
        maximum_filter(pixels, size=(1, diameter, diameter), mode="constant", output=pixels)
    return mask


def select_frames(timeline, force_rate, skip, stride, cap, format):
    rate = Fraction(str(force_rate)).limit_denominator(100_000) if force_rate else timeline.rate
    _, period, remainder = LOAD_FORMATS[format]
    available = len(timeline.frames)
    if force_rate:
        available = max(1, int(np.ceil(timeline.duration * float(rate) - 1e-8)))
    slots = range(skip, available, stride)
    if cap:
        slots = slots[:cap]
    count = len(slots)
    if not count:
        raise ValueError("No frames remain after applying the video loading settings.")
    count -= (count - remainder) % period
    if count < 1:
        raise ValueError(f"{format} needs at least {remainder} loaded frames.")
    slots = slots[:count]
    if not force_rate:
        return slots, rate / stride, timeline.timestamps[skip]
    times = np.arange(slots.start, slots.stop, slots.step, dtype=np.float64) / float(rate)
    indices = np.searchsorted(timeline.timestamps, times + 1e-9, side="right") - 1
    return indices, rate / stride, skip / float(rate)


def load_selected_components(original, timeline, indices, rate, start):
    if isinstance(indices, range) and indices.step == 1 and indices == range(len(timeline.frames)):
        return original.get_components()

    time_base = float(timeline.time_base)
    pts = timeline.frames
    audio_start = pts[0][0] * time_base + start
    audio_end = min(pts[0][0] * time_base + timeline.duration, audio_start + len(indices) / float(rate))
    first = int(indices[0])
    last = int(indices[-1])
    # Decode a bounded window straddling the selected PTS and the audio span.
    # Use whole source ticks for frame boundaries, including duplicate timestamps.
    first_tick = max(0, min(pts[first][0], int(audio_start / time_base)) - 1)
    stop_tick = max(pts[last][0] + 1, int(np.ceil(audio_end / time_base)))
    decode_start = (first_tick + 0.25) * time_base if first_tick else 0
    decode_end = (stop_tick + 0.25) * time_base
    first_index = bisect_left(pts, first_tick, key=lambda item: item[0])
    stop_index = bisect_left(pts, stop_tick, key=lambda item: item[0])
    components = original.as_trimmed(decode_start, decode_end - decode_start, strict_duration=False).get_components()
    if len(components.images) != stop_index - first_index:
        raise ValueError("Decoded frames do not match the source timeline.")
    if not (isinstance(indices, range) and indices == range(first_index, stop_index)):
        local_indices = torch.as_tensor(indices, dtype=torch.long) - first_index
        components.images = components.images.index_select(0, local_indices)
        if components.alpha is not None:
            components.alpha = components.alpha.index_select(0, local_indices)
    components.frame_rate = rate
    if components.audio is not None:
        sample_rate = components.audio["sample_rate"]
        offset = max(0, round((audio_start - decode_start) * sample_rate))
        sample_count = round((audio_end - audio_start) * sample_rate)
        samples = components.audio["waveform"][..., offset:offset + sample_count]
        components.audio = {"sample_rate": sample_rate, "waveform": samples} if samples.shape[-1] else None
    return components


def target_size(width, height, custom_width, custom_height, format):
    alignment = LOAD_FORMATS[format][0]
    target_width = custom_width or (width * custom_height / height if custom_height else width)
    target_height = custom_height or (height * custom_width / width if custom_width else height)
    return tuple(max(alignment, int(value / alignment + 0.5) * alignment)
                 for value in (target_width, target_height))


def resize_selection_batch(batch, width, height, method):
    source_height, source_width = batch.shape[1:3]
    x = min((source_width - 1) // 2, max(0, round((source_width - source_height * width / height) / 2)))
    y = min((source_height - 1) // 2, max(0, round((source_height - source_width * height / width) / 2)))
    cropped = batch[:, y:source_height - y, x:source_width - x]
    return comfy.utils.common_upscale(cropped.movedim(-1, 1), width, height, method, "disabled").movedim(1, -1)


def resize_components(components, width, height):
    images = components.images
    resized = images.new_empty((len(images), height, width, images.shape[-1]))
    for start in range(0, len(images), 4):
        model_management.throw_exception_if_processing_interrupted()
        resized[start:start + 4] = resize_selection_batch(images[start:start + 4], width, height, "bilinear")
    components.images = resized
    if components.alpha is not None:
        alpha = components.alpha.reshape(len(images), *images.shape[1:3], 1)
        components.alpha = resize_selection_batch(alpha, width, height, "bilinear")


def apply_preprocessors(images, groups, tracks, indices, source_size):
    kinds = {group["preprocessor"] for group in groups} - {"segmentation"}
    processors = prepare_preprocessors(kinds) if kinds else {}
    progress = comfy.utils.ProgressBar(len(images))
    height, width = images.shape[1:3]
    source_width, source_height = source_size
    mask_frames = torch.zeros((len(images), height, width), dtype=torch.float32)
    for start in range(0, len(images), 4):
        model_management.throw_exception_if_processing_interrupted()
        chunk = images[start:start + 4]
        frame_indices = indices[start:start + len(chunk)]
        active = []
        for group, (first, packed) in zip(groups, tracks):
            left = bisect_left(frame_indices, first)
            right = bisect_left(frame_indices, first + len(packed))
            if left < right:
                offsets = torch.as_tensor(frame_indices[left:right], dtype=torch.long) - first
                active.append((group, slice(left, right), packed.index_select(0, offsets)))

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

        union = mask_frames[start:start + len(chunk)]
        for group, section, packed in active:
            mask = expanded_masks(packed, source_height, source_width, group["expand"], group["fill_holes"])
            if (width, height) != source_size:
                mask = resize_selection_batch(mask[..., None].float(), width, height, "nearest-exact")[..., 0].bool()
            union[section][mask] = 1
            kind = group["preprocessor"]
            if kind == "segmentation":
                value = group["color"]
                color = chunk.new_tensor([int(value[i:i + 2], 16) / 255 for i in (1, 3, 5)])
                chunk[section][mask] = color
            else:
                chunk[section][mask] = maps[kind][section][mask]
        progress.update(len(chunk))
    return mask_frames


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
                io.Float.Input("force_rate", default=0, min=0, max=60, step=1, optional=True,
                               tooltip="Resample to this FPS before skipping or selecting frames. 0 keeps the source rate."),
                io.Int.Input("custom_width", default=0, min=0, max=MAX_RESOLUTION, optional=True,
                             tooltip="Output width. 0 uses the source aspect ratio."),
                io.Int.Input("custom_height", default=0, min=0, max=MAX_RESOLUTION, optional=True,
                             tooltip="Output height. 0 uses the source aspect ratio."),
                io.Int.Input("frame_load_cap", default=0, min=0, optional=True,
                             tooltip="Maximum output frames after skipping and sampling. 0 means no limit."),
                io.Int.Input("skip_first_frames", default=0, min=0, optional=True,
                             tooltip="Skip this many frames after applying force_rate."),
                io.Int.Input("select_every_nth", default=1, min=1, optional=True),
                io.Combo.Input("format", options=list(LOAD_FORMATS), default="None", optional=True,
                               tooltip="Align dimensions and frame count to a model's grid."),
            ],
            outputs=[
                io.Video.Output(display_name="video"),
                io.Image.Output(display_name="frames"),
                io.Int.Output(display_name="frame_count"),
                io.Audio.Output(display_name="audio"),
                VideoInfo.Output(display_name="video_info"),
                io.Video.Output(display_name="video_mask", tooltip="White selected regions on black, without audio."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, video, **kwargs):
        return input_video(video)[1:]

    @classmethod
    def validate_inputs(cls, video):
        try:
            input_video(video)
        except (OSError, ValueError) as exc:
            return str(exc)
        return True

    @classmethod
    def execute(cls, video, selections, force_rate=0, custom_width=0, custom_height=0,
                frame_load_cap=0, skip_first_frames=0, select_every_nth=1, format="None"):
        source = input_video(video)
        original = InputImpl.VideoFromFile(str(source[0]))
        timeline = video_timeline(source)
        indices, rate, start = select_frames(
            timeline, force_rate, skip_first_frames, select_every_nth, frame_load_cap, format,
        )
        components = load_selected_components(original, timeline, indices, rate, start)
        count = len(components.images)
        if count != len(indices):
            raise ValueError("Decoded frames do not match the source timeline.")
        width, height = target_size(*timeline.size, custom_width, custom_height, format)
        resized = (width, height) != timeline.size
        if resized:
            resize_components(components, width, height)
        groups = parse_groups(json.loads(selections), video, len(timeline.frames))
        if groups:
            with INFERENCE_LOCK:
                _, tracks = track_groups(source, groups)
                masks = apply_preprocessors(components.images, groups, tracks, indices, timeline.size)
        else:
            masks = torch.zeros(1, 1, 1).expand(count, height, width)
        output = original
        if groups or resized or force_rate or count != len(timeline.frames) or rate != timeline.rate:
            output = InputImpl.VideoFromComponents(
                components, original.get_bit_depth(), original.get_color_space(), original.get_chroma_subsampling(),
            )
        source_width, source_height = timeline.size
        info = {
            "source_fps": float(timeline.rate),
            "source_frame_count": len(timeline.frames),
            "source_duration": timeline.duration,
            "source_width": source_width,
            "source_height": source_height,
            "loaded_fps": float(components.frame_rate),
            "loaded_frame_count": count,
            "loaded_duration": count / float(components.frame_rate),
            "loaded_width": width,
            "loaded_height": height,
        }
        mask_video = InputImpl.VideoFromComponents(
            VideoComponents(images=masks[..., None].expand(-1, -1, -1, 3), frame_rate=components.frame_rate),
        )
        return io.NodeOutput(output, components.images, count, components.audio, info, mask_video)


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
            or any(job["inference"] and not job["done"] for job in PREVIEW_JOBS.values()))


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


def tracked_group_at(source, group, ticket, group_index, frame):
    if not isinstance(ticket, str) or type(group_index) is not int:
        return None
    with TRACK_CACHE_LOCK:
        cached = TRACK_CACHE.get(ticket)
    if cached is None or cached[0][0] != source or not 0 <= group_index < len(cached[2]):
        return None
    if json.loads(cached[0][1])[group_index] != group["parts"]:
        return None
    if any(not part["start"] <= frame <= part["end"] for part in group["parts"]):
        return None
    start, packed = cached[2][group_index]
    return packed[frame - start]


def check_guided_group(source, group, frame, ticket, group_index, check):
    packed = tracked_group_at(source, group, ticket, group_index, frame)
    if packed is None:
        paths = [
            {**part, "start": min(part["frame"], frame), "end": max(part["frame"], frame)}
            for part in group["parts"]
        ]
        first, masks = union_tracks(track_parts(source, paths, check))
        packed = masks[frame - first]

    # A current-frame match has to cover pixels belonging to an approved track.
    # Intersect accepted masks with that track so a fused detection cannot add a nearby subject.
    pixels = preview_frame(source, frame)
    guide = expanded_masks(packed.unsqueeze(0), *pixels.shape[:2], radius=0, fill_holes=False)[0]
    union = torch.zeros_like(guide)
    results = []
    prompts = dict.fromkeys((part["text"], part["threshold"]) for part in group["parts"])
    for prompt, threshold in prompts:
        check()
        _, masks, _ = detect_objects(pixels, prompt, threshold)
        overlaps = masks & guide
        area = masks.sum(dim=(-2, -1)).clamp_min(1)
        matched = overlaps.sum(dim=(-2, -1)) >= area * 0.2
        if matched.any():
            union |= overlaps[matched].any(dim=0)
        results.append({"text": prompt, "matches": int(matched.sum())})
    return {"detection": {"mask": encode_mask(union) if union.any() else None, "results": results}}


def native_video_preview(ticket, source, tracks, groups, active_group, frame, check):
    timeline = video_timeline(source)
    styles = tuple(
        (group["color"], group["expand"], group["fill_holes"], index == active_group)
        for index, group in enumerate(groups)
    )
    filename = cached_preview(ticket, source[0], tracks, styles, timeline.size[0], check)
    return {"video": {"filename": filename, "type": "temp", "time": timeline.timestamps[frame]}}


def tracking_preview(source, groups, check):
    ticket, _ = track_groups(source, groups, check)
    return {"ticket": ticket}


def preview_work(job, operation, args):
    def check():
        if job["cancel"].is_set() or monotonic() - job["last_seen"] > 30:
            raise InterruptedError("The preview was cancelled.")
        if job["inference"] and PromptServer.instance.prompt_queue.get_tasks_remaining():
            raise InterruptedError("The preview stopped when a workflow was queued.")

    if not job["inference"]:
        check()
        return operation(*args, check=check)
    with INFERENCE_LOCK, torch.inference_mode():
        check()
        return operation(*args, check=check)


async def run_preview_job(job, operation, args):
    try:
        job.update(await asyncio.to_thread(preview_work, job, operation, args))
    except InterruptedError as exc:
        job["error"] = str(exc)
    except Exception as exc:
        log.exception("Selective video preview failed")
        job["error"] = str(exc)
    finally:
        job["done"] = True


def start_preview_job(operation, *args, inference=True):
    job_id = uuid4().hex
    job = {"done": False, "error": None, "cancel": Event(), "last_seen": monotonic(), "inference": inference}
    PREVIEW_JOBS[job_id] = job
    completed = [key for key, entry in PREVIEW_JOBS.items() if entry["done"]]
    for key in completed[:-4]:
        del PREVIEW_JOBS[key]
    job["task"] = asyncio.create_task(run_preview_job(job, operation, args))
    return web.json_response({"job": job_id}, status=202)


if getattr(PromptServer, "instance", None) is not None:
    routes = PromptServer.instance.routes

    @routes.get("/rp/selective-video/info")
    async def selective_video_info(request):
        try:
            source = input_video(request.query.get("video", ""))
            timeline = await asyncio.to_thread(video_timeline, source)
            width, height = timeline.size
            return web.json_response({"timestamps": timeline.timestamps, "width": width, "height": height})
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


    @routes.post("/rp/selective-video/check-group")
    async def selective_video_check_group(request):
        try:
            data = await request.json()
            if not isinstance(data, dict) or not isinstance(data.get("video"), str):
                raise ValueError("Choose an input video first.")
            source = input_video(data["video"])
            frames = (await asyncio.to_thread(video_timeline, source)).frames
            groups = parse_groups({"video": data["video"], "groups": [data.get("group")]},
                                  data["video"], len(frames))
            if not groups:
                raise ValueError("Add a selection to this group before checking it.")
            frame = data.get("frame")
            if type(frame) is not int or not 0 <= frame < len(frames):
                raise ValueError("Frame is outside the video.")
            if preview_inference_busy():
                return web.json_response({"error": "ComfyUI is busy. Try checking when the queue is idle."}, status=409)
            return start_preview_job(check_guided_group, source, groups[0], frame, data.get("ticket"), data.get("group_index"))
        except (OSError, ValueError, av.error.FFmpegError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.post("/rp/selective-video/track")
    async def selective_video_track(request):
        try:
            data = await request.json()
            if not isinstance(data, dict) or not isinstance(data.get("video"), str):
                raise ValueError("Choose an input video first.")
            source = input_video(data["video"])
            frames = (await asyncio.to_thread(video_timeline, source)).frames
            groups = parse_groups(data, data["video"], len(frames))
            if not groups:
                raise ValueError("Add at least one selection to preview tracking.")
            if preview_inference_busy():
                return web.json_response({"error": "ComfyUI is busy. Try tracking when the queue is idle."}, status=409)
            return start_preview_job(tracking_preview, source, groups)
        except (OSError, ValueError, av.error.FFmpegError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.get("/rp/selective-video/track-status")
    async def selective_video_track_status(request):
        job = PREVIEW_JOBS.get(request.query.get("job", ""))
        if job is None:
            return web.json_response({"error": "Tracking preview expired. Run Track & preview again."}, status=404)
        job["last_seen"] = monotonic()
        return web.json_response({key: job[key] for key in ("done", "ticket", "detection", "video", "error") if key in job})

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
            frames = (await asyncio.to_thread(video_timeline, source)).frames
            if not 0 <= frame < len(frames):
                raise ValueError("Frame is outside the video.")
            masks = await asyncio.to_thread(preview_group_frame, cached[2], frame)
            return web.json_response({"masks": masks})
        except (OSError, ValueError, av.error.FFmpegError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.post("/rp/selective-video/preview-video")
    async def selective_video_preview_video(request):
        try:
            data = await request.json()
            if not isinstance(data, dict) or not isinstance(data.get("video"), str):
                raise ValueError("Choose an input video first.")
            ticket = data.get("ticket")
            if not isinstance(ticket, str):
                raise ValueError("Run Track & preview before preparing a preview video.")
            source = input_video(data["video"])
            with TRACK_CACHE_LOCK:
                cached = TRACK_CACHE.get(ticket)
                if cached is not None and cached[0][0] == source:
                    TRACK_CACHE.move_to_end(ticket)
            if cached is None or cached[0][0] != source:
                return web.json_response({"error": "Tracking preview expired. Run Track & preview again."}, status=404)

            frames = (await asyncio.to_thread(video_timeline, source)).frames
            groups = parse_groups(data, data["video"], len(frames))
            if [group["parts"] for group in groups] != json.loads(cached[0][1]):
                raise ValueError("The selections changed. Run Track & preview again.")
            frame = data.get("frame", 0)
            active_group = data.get("activeGroup", -1)
            if type(frame) is not int or not 0 <= frame < len(frames):
                raise ValueError("Frame is outside the video.")
            if type(active_group) is not int or not -1 <= active_group < len(groups):
                raise ValueError("The active group is not in the tracking preview.")
            return start_preview_job(
                native_video_preview, ticket, source, cached[2], groups, active_group, frame, inference=False,
            )
        except (OSError, ValueError, av.error.FFmpegError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
