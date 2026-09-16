import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from image_folder_processing import image_folder_workflow as workflow


class _FolderPaths:
    def __init__(self, input_directory, output_directory):
        self.input_directory = input_directory
        self.output_directory = output_directory

    def get_input_directory(self):
        return str(self.input_directory)

    def get_output_directory(self):
        return str(self.output_directory)


class _GraphNode:
    def __init__(self, class_type, node_id):
        self.class_type = class_type
        self.node_id = node_id
        self.inputs = {}
        self.display_id = None

    def set_override_display_id(self, node_id):
        self.display_id = node_id

    def set_input(self, name, value):
        self.inputs[name] = value

    def out(self, index):
        return [self.node_id, index]


class _GraphBuilder:
    last_instance = None

    def __init__(self):
        self.nodes = {}
        _GraphBuilder.last_instance = self

    def node(self, class_type, node_id):
        node = _GraphNode(class_type, node_id)
        self.nodes[node_id] = node
        return node

    def lookup_node(self, node_id):
        return self.nodes[node_id]

    def finalize(self):
        return {"nodes": self.nodes}


class _DynamicPrompt:
    def __init__(self, nodes):
        self.nodes = nodes

    def get_node(self, node_id):
        return self.nodes[node_id]

    def all_node_ids(self):
        return set(self.nodes)


class ImageFolderWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_directory = self.root / "input"
        self.output_directory = self.root / "output"
        self.input_directory.mkdir()
        self.output_directory.mkdir()
        self.original_folder_paths = workflow.folder_paths
        workflow.folder_paths = _FolderPaths(
            self.input_directory, self.output_directory
        )

    def tearDown(self):
        workflow.folder_paths = self.original_folder_paths
        self.temporary_directory.cleanup()

    def _write_image(self, path, color, mode="RGB"):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(mode, (4, 3), color).save(path)

    def test_loader_uses_natural_order_and_returns_mask_and_context(self):
        source = self.input_directory / "source"
        self._write_image(source / "image10.png", (10, 20, 30))
        self._write_image(source / "image2.png", (40, 50, 60, 128), mode="RGBA")
        (source / "ignored.txt").write_text("not an image", encoding="utf-8")

        result = workflow.RPLoadImagesFromFolder().load("source", loop_index=0)

        self.assertEqual(result[0], "stub")
        self.assertEqual(tuple(result[1].shape), (1, 3, 4, 3))
        self.assertEqual(tuple(result[2].shape), (1, 3, 4))
        self.assertAlmostEqual(float(result[2][0, 0, 0]), 1.0 - (128 / 255.0))
        self.assertEqual(result[3], "image2")
        self.assertEqual(result[4]["image_filename"], "image2.png")
        self.assertEqual(result[4]["image_count"], 2)

        second = workflow.RPLoadImagesFromFolder().load("source", loop_index=1)
        self.assertEqual(second[4]["image_filename"], "image10.png")
        self.assertTrue(torch.count_nonzero(second[2]) == 0)

    def test_loader_rejects_empty_folder(self):
        (self.input_directory / "empty").mkdir()

        with self.assertRaisesRegex(ValueError, "No supported images"):
            workflow.RPLoadImagesFromFolder().load("empty")

    def test_prompt_loader_selects_block_using_image_context_index(self):
        prompt_file = self.root / "prompts.txt"
        prompt_file.write_text(
            "First prompt\nwith two lines\n\n*\n\nSecond prompt\n\n*\n\nThird prompt\n",
            encoding="utf-8",
        )
        loader = workflow.RPLoadPromptFromFile()
        context = {"index": 1, "image_count": 3}

        prompt, number = loader.load_prompt(context, str(prompt_file), "*", True)

        self.assertEqual(prompt, "Second prompt")
        self.assertEqual(number, 2)

    def test_prompt_loader_rejects_prompt_image_count_mismatch(self):
        prompt_file = self.root / "prompts.txt"
        prompt_file.write_text("First\n*\nSecond\n", encoding="utf-8")
        context = {"index": 0, "image_count": 3}

        with self.assertRaisesRegex(ValueError, "Prompt/image count mismatch"):
            workflow.RPLoadPromptFromFile().load_prompt(
                context, str(prompt_file), "*", True
            )

    def test_saver_preserves_filename_and_format_and_controls_overwrite(self):
        source = self.input_directory / "source"
        self._write_image(source / "photo.jpg", (20, 40, 60))
        loaded = workflow.RPLoadImagesFromFolder().load("source")
        saver = workflow.RPSaveImagesToFolder()

        result = saver.save(
            ["loader", 0],
            loaded[1],
            loaded[4],
            "results",
            False,
            False,
        )

        saved_path = Path(result[2])
        self.assertEqual(result[:2], (str(self.output_directory / "results"), 1))
        self.assertEqual(saved_path.name, "photo.jpg")
        with Image.open(saved_path) as saved:
            self.assertEqual(saved.format, "JPEG")
            self.assertEqual(saved.size, (4, 3))

        with self.assertRaisesRegex(FileExistsError, "already exists"):
            saver.save(
                ["loader", 0],
                loaded[1],
                loaded[4],
                "results",
                False,
                False,
            )

        saver.save(
            ["loader", 0],
            loaded[1],
            loaded[4],
            "results",
            False,
            True,
        )

    def test_inline_saver_saves_and_passes_image_through_without_looping(self):
        source = self.input_directory / "source"
        self._write_image(source / "photo.png", (20, 40, 60))
        loaded = workflow.RPLoadImagesFromFolder().load("source")

        returned_image, saved_path = workflow.RPSaveImageToFolder().save(
            loaded[1], loaded[4], "intermediate", False, False
        )

        self.assertIs(returned_image, loaded[1])
        self.assertEqual(Path(saved_path).name, "photo.png")
        self.assertTrue(Path(saved_path).is_file())

    def test_loop_saver_rejects_two_controllers_for_the_same_loader(self):
        source = self.input_directory / "source"
        self._write_image(source / "photo.png", (20, 40, 60))
        loaded = workflow.RPLoadImagesFromFolder().load("source")
        nodes = {
            "loader": {
                "class_type": "RPLoadImagesFromFolder",
                "inputs": {"images_folder": "source"},
            },
            "saver-a": {
                "class_type": "RPSaveImagesToFolder",
                "inputs": {"flow": ["loader", 0]},
            },
            "saver-b": {
                "class_type": "RPSaveImagesToFolder",
                "inputs": {"flow": ["loader", 0]},
            },
        }

        with self.assertRaisesRegex(ValueError, "Multiple RP Save Images"):
            workflow.RPSaveImagesToFolder().save(
                ["loader", 0],
                loaded[1],
                loaded[4],
                "results",
                False,
                False,
                dynprompt=_DynamicPrompt(nodes),
                unique_id="saver-a",
            )

    def test_saver_clears_only_destination_images_at_first_iteration(self):
        source = self.input_directory / "source"
        self._write_image(source / "new.png", (1, 2, 3))
        loaded = workflow.RPLoadImagesFromFolder().load("source")
        destination = self.output_directory / "results"
        self._write_image(destination / "old.png", (4, 5, 6))
        marker = destination / "keep.txt"
        marker.write_text("keep", encoding="utf-8")

        workflow.RPSaveImagesToFolder().save(
            ["loader", 0],
            loaded[1],
            loaded[4],
            "results",
            True,
            False,
        )

        self.assertFalse((destination / "old.png").exists())
        self.assertTrue((destination / "new.png").is_file())
        self.assertTrue(marker.is_file())

    def test_saver_rejects_source_as_destination(self):
        source = self.input_directory / "source"
        self._write_image(source / "image.png", (1, 2, 3))
        loaded = workflow.RPLoadImagesFromFolder().load("source")

        with self.assertRaisesRegex(ValueError, "must be different"):
            workflow.RPSaveImagesToFolder().save(
                ["loader", 0],
                loaded[1],
                loaded[4],
                str(source),
                False,
                True,
            )

    def test_next_iteration_clones_the_contained_graph_and_advances_loader(self):
        nodes = {
            "loader": {
                "class_type": "RPLoadImagesFromFolder",
                "inputs": {"images_folder": "source", "loop_index": 0},
            },
            "processor": {
                "class_type": "ExampleProcessor",
                "inputs": {"image": ["loader", 1]},
            },
            "saver": {
                "class_type": "RPSaveImagesToFolder",
                "inputs": {
                    "flow": ["loader", 0],
                    "processed_image": ["processor", 0],
                    "image_context": ["loader", 4],
                    "output_folder": "results",
                },
            },
        }
        original_builder = workflow.GraphBuilder
        original_is_link = workflow.is_link
        workflow.GraphBuilder = _GraphBuilder
        workflow.is_link = lambda value: (
            isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
        )
        try:
            expansion = workflow.RPSaveImagesToFolder()._next_iteration(
                ["loader", 0], 3, _DynamicPrompt(nodes), "saver"
            )
        finally:
            workflow.GraphBuilder = original_builder
            workflow.is_link = original_is_link

        graph = _GraphBuilder.last_instance
        self.assertEqual(graph.nodes["loader"].inputs["loop_index"], 3)
        self.assertEqual(graph.nodes["processor"].inputs["image"], ["loader", 1])
        self.assertIn("Recurse", graph.nodes)
        self.assertEqual(
            expansion["result"], (["Recurse", 0], ["Recurse", 1], ["Recurse", 2])
        )


if __name__ == "__main__":
    unittest.main()
