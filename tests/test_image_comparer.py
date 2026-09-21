from datetime import datetime
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest


class PreviewImageStub:
    def save_images(self, images, filename_prefix, prompt, extra_pnginfo):
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filenames = [f"{filename_prefix}_{index}.png" for index, _ in enumerate(images)]
        for filename in filenames:
            (output_dir / filename).write_bytes(b"preview")
        return {
            "ui": {
                "images": [
                    {
                        "filename": filename,
                        "subfolder": "",
                        "type": "temp",
                    }
                    for filename in filenames
                ]
            }
        }


MODULE_PATH = Path(__file__).parents[1] / "image_saving" / "image_comparer.py"
SPEC = importlib.util.spec_from_file_location("rp_image_comparer", MODULE_PATH)
image_comparer = importlib.util.module_from_spec(SPEC)
previous_nodes_module = sys.modules.get("nodes")
previous_folder_paths_module = sys.modules.get("folder_paths")
folder_paths_stub = types.SimpleNamespace(get_temp_directory=lambda: "")
sys.modules["nodes"] = types.SimpleNamespace(PreviewImage=PreviewImageStub)
sys.modules["folder_paths"] = folder_paths_stub
try:
    SPEC.loader.exec_module(image_comparer)
finally:
    if previous_nodes_module is None:
        sys.modules.pop("nodes", None)
    else:
        sys.modules["nodes"] = previous_nodes_module
    if previous_folder_paths_module is None:
        sys.modules.pop("folder_paths", None)
    else:
        sys.modules["folder_paths"] = previous_folder_paths_module


class ImageComparerTests(unittest.TestCase):
    def test_default_filename_template(self):
        now = datetime(2026, 9, 21, 15, 52, 4)
        self.assertEqual(
            image_comparer.format_save_name("#1", image_comparer.DEFAULT_SAVE_NAME, now),
            "#1-2026-09-21_155204",
        )
        self.assertEqual(
            image_comparer.format_save_name(
                "qwen_image_2.1", image_comparer.DEFAULT_SAVE_NAME, now
            ),
            "qwen_image_2.1-2026-09-21_155204",
        )

    def test_individual_tokens_and_filename_sanitizing(self):
        now = datetime(2026, 1, 2, 3, 4, 5)
        self.assertEqual(
            image_comparer.format_save_name(
                "klein/model", "%display_name%_%date:yyyyMMdd-HHmmss%", now
            ),
            "klein_model_20260102-030405",
        )

    def test_png_filename_preserves_model_version_dots(self):
        self.assertEqual(
            image_comparer.png_filename("qwen_image_2.1-2026-09-21_155204"),
            "qwen_image_2.1-2026-09-21_155204.png",
        )
        self.assertEqual(
            image_comparer.png_filename("qwen_image_2.1-2026-09-21_155204", 1),
            "qwen_image_2.1-2026-09-21_155204_2.png",
        )

    def test_inputs_have_requested_defaults(self):
        required = image_comparer.RPImageComparer.INPUT_TYPES()["required"]
        self.assertEqual(required["display_name"][1]["default"], "#1")
        self.assertEqual(
            required["save_name"][1]["default"],
            "%display_name-date:yyyy-MM-dd_HHmmss%",
        )

    def test_compare_images_keeps_rgthree_ui_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_comparer.folder_paths.get_temp_directory = lambda: temp_dir
            node = image_comparer.RPImageComparer()
            node.output_dir = temp_dir
            result = node.compare_images(
                display_name="krea-2",
                save_name="fixed-name",
                image_a=[object()],
                image_b=[object(), object()],
            )

            self.assertEqual(len(result["ui"]["a_images"]), 1)
            self.assertEqual(len(result["ui"]["b_images"]), 2)
            self.assertEqual(result["ui"]["download_name"], ["fixed-name"])
            self.assertEqual(result["ui"]["a_images"][0]["filename"], "fixed-name.png")
            self.assertEqual(result["ui"]["b_images"][0]["filename"], "fixed-name.png")
            self.assertEqual(result["ui"]["b_images"][1]["filename"], "fixed-name_2.png")
            for side in ("a_images", "b_images"):
                for image in result["ui"][side]:
                    self.assertTrue(
                        (Path(temp_dir) / image["subfolder"] / image["filename"]).is_file()
                    )


if __name__ == "__main__":
    unittest.main()
