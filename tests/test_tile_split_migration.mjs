import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/js/image_sizing_and_resizing/tile_split.js", import.meta.url), "utf8")
    .replace('import { app } from "/scripts/app.js";', "")
    .replace("export function", "function");
let extension;
const context = vm.createContext({ app: { registerExtension(value) { extension = value; } } });
vm.runInContext(source, context);

function fixture(version = 0.4, duplicate = false) {
    const links = [[1, 2300, 0, 2284, 1, "IMAGE"], [2, 2300, 4, 2303, 0, "STRING"],
        [3, 2300, 3, 2302, 0, "IMAGE"], [4, 2300, 1, 2301, 1, "RP_TILE_CONTEXT"]];
    return {
        version,
        nodes: [{ id: 2300, type: "RPTileSplit", pos: [400, 600], widgets_values: ["12", "lanczos"], outputs: [
            { name: "tiles", type: "IMAGE", links: [1], shape: 6 },
            { name: "tile_context", type: "RP_TILE_CONTEXT", links: [4] },
            { name: "scaled_image", type: "IMAGE", links: [] },
            { name: duplicate ? "layout" : "tile_preview", type: duplicate ? "STRING" : "IMAGE", links: [3] },
            { name: "layout", type: "STRING", links: [2], slot_index: 4 },
        ] }],
        links: version === 1 ? links.map(([id, origin_id, origin_slot, target_id, target_slot, type]) =>
            ({ id, origin_id, origin_slot, target_id, target_slot, type })) : links,
    };
}

for (const version of [0.4, 1]) {
    for (const duplicate of [false, true]) {
        test(`migrate old five outputs and screenshot's duplicated layout, v${version}, duplicate=${duplicate}`, () => {
            const workflow = fixture(version, duplicate);
            const original = structuredClone(workflow.nodes[0]);
            extension.beforeConfigureGraph(workflow);
            const splitter = workflow.nodes[0];
            assert.equal(splitter.outputs.length, 4);
            assert.deepEqual(Array.from(splitter.outputs, (output) => output.name), ["tiles", "tile_context", "scaled_image", "layout"]);
            assert.deepEqual(splitter.pos, original.pos);
            assert.deepEqual(splitter.widgets_values, original.widgets_values);
            assert.deepEqual(Array.from(splitter.outputs[0].links), [1, 3]);
            assert.deepEqual(Array.from(splitter.outputs[3].links), [2]);
            const slots = workflow.links.map((link) => Array.isArray(link) ? link[2] : link.origin_slot);
            assert.deepEqual(slots, [0, 3, 0, 1]);
            const migrated = JSON.stringify(workflow);
            assert.equal(context.migrateTileSplitOutputs(workflow), 0);
            assert.equal(JSON.stringify(workflow), migrated);
        });
    }
}

test("merge both duplicated layout links while preserving their target IDs", () => {
    const workflow = fixture(0.4, true);
    workflow.links[2][5] = "STRING";
    context.migrateTileSplitOutputs(workflow);
    assert.deepEqual(Array.from(workflow.nodes[0].outputs[3].links), [2, 3]);
    assert.deepEqual(workflow.links[2], [3, 2300, 3, 2302, 0, "STRING"]);
});

test("repair a stale fifth-port link even when the serialized node already has four outputs", () => {
    const workflow = fixture();
    workflow.nodes[0].outputs.splice(3, 1);
    workflow.links.splice(2, 1);
    context.migrateTileSplitOutputs(workflow);
    assert.equal(workflow.links[1][2], 3);
    assert.deepEqual(Array.from(workflow.nodes[0].outputs[3].links), [2]);
});

test("nested subgraphs and floating preview links are migrated", () => {
    const inner = fixture(1);
    inner.floatingLinks = [{ id: 9, origin_id: 2300, origin_slot: 4, target_id: -1, target_slot: -1, type: "STRING" }];
    const outer = { nodes: [], links: [], definitions: { subgraphs: [inner] } };
    context.migrateTileSplitOutputs(outer);
    assert.equal(inner.floatingLinks[0].origin_slot, 3);
    assert.deepEqual(Array.from(inner.nodes[0].outputs[3].links), [2, 9]);
});

test("current workflow is an exact no-op and unrelated nodes are untouched", () => {
    const workflow = JSON.parse(readFileSync(new URL("../example_workflows/FLUX.3 Klein Distilled - RP Tiled Upscale.json", import.meta.url), "utf8"));
    const original = JSON.stringify(workflow);
    assert.equal(context.migrateTileSplitOutputs(workflow), 0);
    assert.equal(JSON.stringify(workflow), original);
    const unrelated = fixture();
    unrelated.nodes[0].type = "SomeOtherNode";
    assert.equal(context.migrateTileSplitOutputs(unrelated), 0);
    assert.equal(unrelated.nodes[0].outputs.length, 5);
});
