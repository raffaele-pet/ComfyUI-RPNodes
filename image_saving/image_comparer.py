"""Image comparison preview with a configurable download filename."""

from datetime import datetime
import os
import re
from uuid import uuid4

import folder_paths
from nodes import PreviewImage


DEFAULT_DISPLAY_NAME = "#1"
DEFAULT_SAVE_NAME = "%display_name-date:yyyy-MM-dd_HHmmss%"


def _format_date(pattern, value):
    """Format the date tokens used by ComfyUI-style filename templates."""
    replacements = {
        "yyyy": f"{value.year:04d}",
        "MM": f"{value.month:02d}",
        "dd": f"{value.day:02d}",
        "HH": f"{value.hour:02d}",
        "mm": f"{value.minute:02d}",
        "ss": f"{value.second:02d}",
    }
    return re.sub(
        "|".join(replacements),
        lambda match: replacements[match.group(0)],
        pattern,
    )


def format_save_name(display_name, save_name, now=None):
    """Expand RP Image Comparer filename tokens and return a safe basename."""
    value = now or datetime.now()
    display_name = str(display_name or DEFAULT_DISPLAY_NAME).strip()
    template = str(save_name or DEFAULT_SAVE_NAME).strip()

    def display_and_date(match):
        return f"{display_name}-{_format_date(match.group(1), value)}"

    template = re.sub(
        r"%display_name-date:([^%]+)%",
        display_and_date,
        template,
    )
    template = template.replace("%display_name%", display_name)
    template = re.sub(
        r"%date:([^%]+)%",
        lambda match: _format_date(match.group(1), value),
        template,
    )

    # Browser downloads cannot preserve directories. Keep the chosen basename
    # portable across Windows, macOS, and Linux.
    template = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", template).strip(" .")
    return template or DEFAULT_DISPLAY_NAME


def png_filename(save_name, batch_index=0):
    """Return a PNG filename while preserving dots inside model names."""
    basename = re.sub(r"\.(?:png|jpe?g|webp)$", "", save_name, flags=re.IGNORECASE)
    suffix = "" if batch_index == 0 else f"_{batch_index + 1}"
    return f"{basename}{suffix}.png"


class RPImageComparer(PreviewImage):
    """Compare two images and download either preview with a chosen name."""

    FUNCTION = "compare_images"
    CATEGORY = "image/saving"
    DESCRIPTION = (
        "Compares two images with a hover slider, click mode, or side-by-side "
        "view and downloads the selected image using the configured save name."
    )

    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name
        return {
            "required": {
                "display_name": (
                    "STRING",
                    {
                        "default": DEFAULT_DISPLAY_NAME,
                        "label": "Display name",
                        "tooltip": "Name used by the save-name template.",
                    },
                ),
                "save_name": (
                    "STRING",
                    {
                        "default": DEFAULT_SAVE_NAME,
                        "label": "Save name",
                        "tooltip": (
                            "Download filename template. Supports %display_name%, "
                            "%date:...%, and %display_name-date:...%."
                        ),
                    },
                ),
            },
            "optional": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def compare_images(
        self,
        display_name=DEFAULT_DISPLAY_NAME,
        save_name=DEFAULT_SAVE_NAME,
        image_a=None,
        image_b=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        download_name = format_save_name(display_name, save_name)
        run_id = uuid4().hex
        result = {
            "ui": {
                "a_images": [],
                "b_images": [],
                "download_name": [download_name],
            }
        }
        if image_a is not None and len(image_a) > 0:
            result["ui"]["a_images"] = self._save_named_previews(
                image_a, download_name, run_id, "a", prompt, extra_pnginfo
            )

        if image_b is not None and len(image_b) > 0:
            result["ui"]["b_images"] = self._save_named_previews(
                image_b, download_name, run_id, "b", prompt, extra_pnginfo
            )

        return result

    def _save_named_previews(
        self,
        images,
        download_name,
        run_id,
        side,
        prompt,
        extra_pnginfo,
    ):
        """Save previews in isolated temp folders with download-ready names."""
        saved = self.save_images(
            images,
            "rp_image_comparer.preview",
            prompt,
            extra_pnginfo,
        )["ui"]["images"]
        target_subfolder = os.path.join("rp_image_comparer", run_id, side)
        target_folder = os.path.join(folder_paths.get_temp_directory(), target_subfolder)
        os.makedirs(target_folder, exist_ok=True)

        for batch_index, image_data in enumerate(saved):
            source_path = os.path.join(
                folder_paths.get_temp_directory(),
                image_data.get("subfolder", ""),
                image_data["filename"],
            )
            filename = png_filename(download_name, batch_index)
            os.replace(source_path, os.path.join(target_folder, filename))
            image_data["filename"] = filename
            image_data["subfolder"] = target_subfolder

        return saved


NODE_CLASS_MAPPINGS = {"RPImageComparer": RPImageComparer}
NODE_DISPLAY_NAME_MAPPINGS = {"RPImageComparer": "RP Image Comparer"}
