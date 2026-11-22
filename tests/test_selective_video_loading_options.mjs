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
