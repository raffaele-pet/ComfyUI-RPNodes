import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / "example_workflows" / "RP_All_Nodes_Examples.json"


def registered_nodes():
    names = set()
    for path in ROOT.rglob("*.py"):
        if "tests" in path.relative_to(ROOT).parts:
            continue
        for item in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if (isinstance(item, ast.Assign) and isinstance(item.value, ast.Dict)
                    and any(isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS" for target in item.targets)):
                names.update(key.value for key in item.value.keys if isinstance(key, ast.Constant) and isinstance(key.value, str))
    return names


class ExampleWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))

    def test_all_registered_rp_nodes_are_demonstrated(self):
        present = {node["type"] for node in self.workflow["nodes"]}
        self.assertTrue(registered_nodes())
        self.assertEqual(registered_nodes() - present, set())

    def test_links_reference_existing_nodes_and_matching_slots(self):
        nodes = {node["id"]: node for node in self.workflow["nodes"]}
        self.assertEqual(len(nodes), len(self.workflow["nodes"]))
        links = {link[0]: link for link in self.workflow["links"]}
        self.assertEqual(len(links), len(self.workflow["links"]))
        for link_id, source, source_slot, target, target_slot, _ in self.workflow["links"]:
            with self.subTest(link=link_id):
                self.assertIn(source, nodes)
                self.assertIn(target, nodes)
                self.assertIn(link_id, nodes[source]["outputs"][source_slot]["links"])
                self.assertEqual(nodes[target]["inputs"][target_slot]["link"], link_id)
        for node in nodes.values():
            for port in node.get("inputs", []):
                if port.get("link") is not None:
                    self.assertIn(port["link"], links)
            for port in node.get("outputs", []):
                for link in port.get("links") or []:
                    self.assertIn(link, links)

    def test_face_group_has_bilingual_note_inside_its_bounds(self):
        group = next(group for group in self.workflow["groups"] if group["title"] == "15 RP Face Detection")
        note = next(node for node in self.workflow["nodes"] if node["type"] == "Note" and node.get("title") == "RP Face Detection")
        self.assertIn("ENGLISH", note["widgets_values"][0])
        self.assertIn("ITALIANO", note["widgets_values"][0])
        self.assertEqual(note["widgets_values_named"]["text"], note["widgets_values"][0])
        x, y, width, height = group["bounding"]
        self.assertGreaterEqual(note["pos"][0], x)
        self.assertGreaterEqual(note["pos"][1], y)
        self.assertLessEqual(note["pos"][0] + note["size"][0], x + width)
        self.assertLessEqual(note["pos"][1] + note["size"][1], y + height)
        self.assertFalse((ROOT / "example_workflows" / "RP_Face_Detection.json").exists())


if __name__ == "__main__":
    unittest.main()
