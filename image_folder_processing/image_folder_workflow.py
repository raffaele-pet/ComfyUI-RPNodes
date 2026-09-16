import os
import re
import time
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


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGE_FORMATS = {
    ".bmp": "BMP",
    ".jpeg": "JPEG",
    ".jpg": "JPEG",
    ".png": "PNG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
}


def _input_directory():
    if folder_paths is not None:
        return Path(folder_paths.get_input_directory()).resolve()
    return Path.cwd().resolve()


def _output_directory():
    if folder_paths is not None:
        return Path(folder_paths.get_output_directory()).resolve()
    return (Path.cwd() / "output").resolve()


def _relative_parts(path, directory_name):
    parts = list(path.parts)
    lowered = [part.casefold() for part in parts]
    if len(parts) >= 2 and lowered[:2] == ["comfyui", directory_name]:
        return parts[2:]
    if parts and lowered[0] == directory_name:
        return parts[1:]
    return parts


def _resolve_input_folder(value):
    text = str(value or "").strip().strip('"')
    if not text:
        raise ValueError("Images folder cannot be empty.")
    folder = Path(text).expanduser()
    if not folder.is_absolute():
        folder = _input_directory().joinpath(*_relative_parts(folder, "input"))
    folder = folder.resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Images folder not found: {folder}")
    return folder


def _resolve_output_folder(value):
    text = str(value or "").strip().strip('"')
    if not text:
        raise ValueError("Output folder cannot be empty.")
    folder = Path(text).expanduser()
    if not folder.is_absolute():
        folder = _output_directory().joinpath(*_relative_parts(folder, "output"))
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _natural_key(path):
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


def _image_files(folder):
    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=_natural_key,
    )


def _load_image(path):
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


def _clear_images(folder):
    for path in _image_files(folder):
        _unlink_with_retry(path)


def _image_from_tensor(image):
    if image is None or not hasattr(image, "shape") or len(image.shape) != 4:
        raise ValueError("processed_image must be a ComfyUI IMAGE tensor.")
    if int(image.shape[0]) != 1:
        raise ValueError(
            "The integrated image loop expects one image per iteration. "
            "Connect a processing chain that returns a single IMAGE."
        )

    array = image[0].detach().cpu().float().clamp(0.0, 1.0).numpy()
    array = np.rint(array * 255.0).astype(np.uint8)
    channels = int(array.shape[-1])
    if channels == 1:
        return Image.fromarray(array[..., 0], mode="L")
    if channels == 4:
        return Image.fromarray(array, mode="RGBA")
    if channels < 3:
        raise ValueError("processed_image must have 1, 3, or 4 channels.")
    return Image.fromarray(array[..., :3], mode="RGB")


def _save_image(image, path, overwrite):
    image_format = IMAGE_FORMATS.get(path.suffix.lower())
    if image_format is None:
        raise ValueError(f"Unsupported output image extension: {path.suffix}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output image already exists: {path}")

    output_image = _image_from_tensor(image)
    if image_format in {"BMP", "JPEG"} and output_image.mode not in {"L", "RGB"}:
        output_image = output_image.convert("RGB")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.{os.getpid()}.tmp{path.suffix.lower()}"
    )
    output_image.save(temporary, format=image_format)
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


class RPLoadImagesFromFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_folder": ("STRING", {"default": "images"}),
            },
            "hidden": {
                "loop_index": ("INT",),
            },
        }

    RETURN_TYPES = (
        "FLOW_CONTROL",
        "IMAGE",
        "MASK",
        "STRING",
        "RP_IMAGE_FOLDER_CONTEXT",
    )
    RETURN_NAMES = (
        "flow",
        "image",
        "mask",
        "image_name",
        "image_context",
    )
    FUNCTION = "load"
    CATEGORY = "image/RPNodes"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    def load(self, images_folder, loop_index=0):
        loop_index = max(0, int(loop_index))
        source_directory = _resolve_input_folder(images_folder)
        images = _image_files(source_directory)
        if not images:
            raise ValueError(
                f"No supported images found in '{source_directory}'. Supported extensions: "
                f"{', '.join(sorted(IMAGE_EXTENSIONS))}."
            )
        if loop_index >= len(images):
            raise IndexError(
                f"Image index {loop_index} is outside the folder range "
                f"(0-{len(images) - 1})."
            )

        image_path = images[loop_index]
        image, mask = _load_image(image_path)
        context = {
            "index": loop_index,
            "image_count": len(images),
            "image_filename": image_path.name,
            "source_directory": str(source_directory),
            "source_image": str(image_path),
        }
        return (
            "stub",
            image,
            mask,
            image_path.stem,
            context,
        )


class RPSaveImagesToFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
                "processed_image": ("IMAGE",),
                "image_context": ("RP_IMAGE_FOLDER_CONTEXT",),
                "output_folder": ("STRING", {"default": "processed_images"}),
                "clear_output_folder": ("BOOLEAN", {"default": False}),
                "overwrite": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("folder_path", "image_count", "last_image_path")
    FUNCTION = "save"
    CATEGORY = "image/RPNodes"
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
                "the integrated image loop."
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

    def save(
        self,
        flow,
        processed_image,
        image_context,
        output_folder,
        clear_output_folder,
        overwrite,
        dynprompt=None,
        unique_id=None,
    ):
        context = dict(image_context)
        index = int(context["index"])
        image_count = int(context["image_count"])
        if image_count < 1 or index < 0 or index >= image_count:
            raise ValueError("image_context contains an invalid index or image count.")

        source_directory = Path(context["source_directory"]).expanduser().resolve()
        output_directory = _resolve_output_folder(output_folder)
        if source_directory == output_directory:
            raise ValueError("Source and output image folders must be different.")
        if clear_output_folder and output_directory == _output_directory():
            raise ValueError(
                "Do not use the ComfyUI/output root with clear_output_folder enabled. "
                "Choose a dedicated subfolder such as 'processed_images'."
            )
        if clear_output_folder and index == 0:
            _clear_images(output_directory)

        image_filename = Path(str(context["image_filename"])).name
        if not image_filename:
            raise ValueError("image_context does not contain a valid image filename.")
        output_path = output_directory / image_filename
        _save_image(processed_image, output_path, bool(overwrite))

        if index + 1 < image_count:
            return self._next_iteration(flow, index + 1, dynprompt, unique_id)

        return (str(output_directory), image_count, str(output_path))


NODE_CLASS_MAPPINGS = {
    "RPLoadImagesFromFolder": RPLoadImagesFromFolder,
    "RPSaveImagesToFolder": RPSaveImagesToFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RPLoadImagesFromFolder": "RP Load Images from Folder",
    "RPSaveImagesToFolder": "RP Save Images to Folder",
}
