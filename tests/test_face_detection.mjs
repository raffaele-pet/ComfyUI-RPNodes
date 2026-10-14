import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/js/face_detection/face_detection.js", import.meta.url), "utf8")
    .replace('import { app } from "/scripts/app.js";', "")
    .replace("export function", "function");
let extension;
const context = vm.createContext({ app: { registerExtension(value) { extension = value; } } });
vm.runInContext(source, context);

function fixture() {
    const widgets = ["show_advanced", "threshold", "dilation", "crop_factor", "drop_size", "crop_size_mult", "bbox_smooth_alpha", "scale"]
        .map((name) => ({ name, type: name === "show_advanced" ? "toggle" : "number", value: name === "show_advanced" ? false : 1.5 }));
    return { widgets, size: [320, 300], computeSize: () => [300, 200], setSize(value) { this.size = value; }, setDirtyCanvas() {} };
}

test("advanced options start hidden, toggle without losing values, and serialize", () => {
    const node = fixture();
    const original = () => [250, 20];
    node.widgets[1].computeSize = original;
    context.configureAdvancedWidgets(node);
    for (const widget of node.widgets.slice(1, 7)) {
        assert.equal(widget.hidden, true);
        assert.equal(widget.computeSize()[1], -4);
        assert.equal(widget.value, 1.5);
    }
    assert.equal(node.widgets[7].type, "number");
    node.widgets[0].value = true;
    node.widgets[0].callback(true);
    assert.equal(node.widgets[1].computeSize, original);
    assert.equal(node.widgets[2].computeSize, undefined);
    assert.ok(node.widgets.slice(1, 7).every((widget) => widget.type === "number" && !widget.hidden));
    node.widgets[1].value = 0.7;
    node.widgets[0].value = false;
    node.widgets[0].callback(false);
    assert.equal(node.widgets[1].value, 0.7);
    assert.deepEqual(JSON.parse(JSON.stringify(node.widgets.map((widget) => widget.value))), [false, 0.7, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]);
});

test("configure restores saved visibility and existing hooks are preserved", () => {
    let created = 0;
    let configured = 0;
    function Node() { Object.assign(this, fixture()); }
    Node.prototype.onNodeCreated = function () { created++; return 42; };
    Node.prototype.onConfigure = function (data) { configured++; this.widgets[0].value = data.advanced; };
    extension.beforeRegisterNodeDef(Node, { name: "RPFaceDetection" });
    const node = new Node();
    assert.equal(node.onNodeCreated(), 42);
    node.onConfigure({ advanced: true });
    assert.equal(node.widgets[1].hidden, false);
    node.onConfigure({ advanced: false });
    assert.equal(node.widgets[1].hidden, true);
    assert.equal(created, 1);
    assert.equal(configured, 2);
});

test("visibility follows ComfyUI widget type restoration semantics", () => {
    const node = fixture();
    const widget = node.widgets[1];
    let type = widget.type;
    let hidden = false;
    let registeredHidden = false;
    Object.defineProperties(widget, {
        type: {
            get: () => type,
            set(value) {
                const wasConverted = type === "converted-widget";
                type = value;
                if (type === "converted-widget") hidden = true;
                else if (wasConverted) hidden = registeredHidden;
            },
        },
        hidden: {
            get: () => hidden,
            set(value) {
                hidden = value;
                if (type !== "converted-widget") registeredHidden = value;
            },
        },
    });
    context.configureAdvancedWidgets(node);
    assert.equal(widget.hidden, true);
    node.widgets[0].value = true;
    node.widgets[0].callback(true);
    assert.equal(widget.hidden, false);
});
