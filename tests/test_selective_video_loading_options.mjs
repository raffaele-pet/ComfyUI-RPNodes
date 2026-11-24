import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

let extension;
const source = readFileSync(new URL("../web/js/video_conditioning/selective_video.js", import.meta.url), "utf8")
    .replace(/^import[\s\S]*?;\r?\n/gm, "");
vm.runInNewContext(source, {
    app: { registerExtension(value) { extension = value; } },
    attachNativePreview() { return { update() {} }; },
});

function fixture(options) {
    const widgets = { video: { name: "video" }, selections: { name: "selections" } };
    for (const [name, value] of Object.entries(options)) widgets[name] = { name, value };
    const node = {
        comfyClass: "RPSelectiveVideoInput",
        widgets: Object.values(widgets),
        addWidget() { return {}; },
        setDirtyCanvas() {},
        onConfigure(values) {
            for (const [name, value] of Object.entries(values)) widgets[name].value = value;
        },
    };
    return { node, widgets };
}

test("saved blank loader options recover defaults while explicit options are preserved", () => {
    const saved = {
        force_rate: "image",
        custom_width: null,
        custom_height: 576,
        frame_load_cap: NaN,
        skip_first_frames: 11,
        select_every_nth: null,
        format: null,
    };
    const { node, widgets } = fixture(saved);
    extension.nodeCreated(node);
    node.onConfigure(saved);
    const values = Object.fromEntries(Object.keys(saved).map((name) => [name, widgets[name].value]));
    assert.deepEqual(values, {
        force_rate: 0, custom_width: 0, custom_height: 576, frame_load_cap: 0,
        skip_first_frames: 11, select_every_nth: 1, format: "None",
    });
});

test("prompt serialization handles controls cleared after loading the graph", async () => {
    const { node, widgets } = fixture({ custom_width: 640, select_every_nth: 2, force_rate: 23.976 });
    widgets.force_rate.serializeValue = function (scale) { return this.value * scale; };
    extension.nodeCreated(node);
    widgets.custom_width.value = NaN;
    widgets.select_every_nth.value = "";
    assert.equal(await widgets.custom_width.serializeValue(), 0);
    assert.equal(await widgets.select_every_nth.serializeValue(), 1);
    assert.equal(await widgets.force_rate.serializeValue(1), 23.976);
});

test("old video-mask links migrate without moving the original video output", () => {
    const node = {
        id: 587,
        type: "RPSelectiveVideoInput",
        outputs: [
            ...["video", "frames", "frame_count", "audio", "video_info"].map((name) => ({ name })),
            { name: "video_mask", localized_name: "video_mask", type: "VIDEO", links: [619], slot_index: 5 },
        ],
    };
    const nested = {
        nodes: [structuredClone(node)],
        links: [{ id: 619, origin_id: 587, origin_slot: 5, target_id: -2, target_slot: 0, type: "VIDEO" }],
        floatingLinks: [{ id: 620, origin_id: 587, origin_slot: 5, type: "VIDEO" }],
    };
    const workflow = {
        nodes: [node],
        links: [[618, 587, 0, 588, 0, "VIDEO"], [619, 587, 5, 589, 0, "VIDEO"]],
        definitions: { subgraphs: [nested] },
    };
    extension.beforeConfigureGraph(workflow);
    assert.equal(node.outputs[2].name, "original_frames");
    assert.equal(node.outputs[6].name, "MASK_IMAGE");
    assert.equal(node.outputs[6].type, "IMAGE");
    assert.equal(node.outputs[7].name, "mask_video");
    assert.equal(node.outputs[7].localized_name, "mask_video");
    assert.equal(node.outputs[7].slot_index, 7);
    assert.deepEqual(node.outputs[7].links, [619]);
    assert.deepEqual(workflow.links, [[618, 587, 0, 588, 0, "VIDEO"], [619, 587, 7, 589, 0, "VIDEO"]]);
    assert.equal(nested.links[0].origin_slot, 7);
    assert.equal(nested.floatingLinks[0].origin_slot, 7);
    const migrated = JSON.stringify(workflow);
    extension.beforeConfigureGraph(workflow);
    assert.equal(JSON.stringify(workflow), migrated);
});

test("reordering original frames preserves all output links on both RP variants", () => {
    const names = ["video", "frames", "frame_count", "audio", "video_info", "MASK_IMAGE", "mask_video", "MASK", "original_frames"];
    for (const type of ["RPSelectiveVideoInput", "RPSelectiveVideoEdgeFill"]) {
        const node = {
            id: 80, type,
            outputs: names.map((name, index) => ({ name, links: [100 + index], slot_index: index })),
        };
        const workflow = {
            nodes: [node],
            links: names.map((_, index) => [100 + index, node.id, index, 81 + index, 0]),
        };
        extension.beforeConfigureGraph(workflow);
        const orderedNames = ["video", "frames", "original_frames", ...names.slice(2, -1)];
        assert.deepEqual(Array.from(node.outputs, ({ name }) => name), orderedNames);
        for (const [link, _, slot] of workflow.links) {
            assert.equal(node.outputs[slot].name, names[link - 100]);
            assert.equal(node.outputs[slot].slot_index, slot);
            assert.deepEqual(node.outputs[slot].links, [link]);
        }
    }
});

test("old Guides workflows rewire the transparent saver and preserve the other output links", () => {
    const names = ["video", "frames", "original_frames", "frame_count", "audio", "video_info",
        "MASK_IMAGE", "mask_video", "MASK", "guide_video"];
    const guides = {
        id: 1,
        type: "RPSelectiveVideoGuides",
        outputs: names.map((name, slot) => ({ name, type: "", links: [4, 8, 9].includes(slot) ? [slot + 1] : null })),
    };
    const saver = { id: 2, type: "RPSaveTransparentVideo", inputs: [{ name: "video", type: "VIDEO", link: 10 }] };
    const workflow = {
        last_link_id: 10,
        nodes: [guides, saver],
        links: [[5, 1, 4, 3, 0, "AUDIO"], [9, 1, 8, 4, 0, "MASK"], [10, 1, 9, 2, 0, "VIDEO"]],
    };
    extension.beforeConfigureGraph(workflow);
    assert.deepEqual(Array.from(guides.outputs, ({ name }) => name), [
        "video", "frames_map_guide", "frames_transparent_guide", ...names.slice(2, 9),
    ]);
    assert.equal(guides.outputs[2].type, "IMAGE");
    assert.deepEqual(JSON.parse(JSON.stringify(saver.inputs)), [
        { name: "video", type: "VIDEO", link: null },
        { name: "frames", type: "IMAGE", link: 10 },
        { name: "video_info", type: "VHS_VIDEOINFO", link: 11 },
    ]);
    assert.deepEqual(JSON.parse(JSON.stringify(workflow.links)), [
        [5, 1, 5, 3, 0, "AUDIO"], [9, 1, 9, 4, 0, "MASK"],
        [10, 1, 2, 2, 1, "IMAGE"], [11, 1, 6, 2, 2, "VHS_VIDEOINFO"],
    ]);
    const migrated = JSON.stringify(workflow);
    extension.beforeConfigureGraph(workflow);
    assert.equal(JSON.stringify(workflow), migrated);
});
