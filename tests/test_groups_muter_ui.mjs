import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import * as logic from "../web/js/workflow_utilities/groups_muter_logic.js";

let extension;
let tick;
const source = readFileSync(new URL("../web/js/workflow_utilities/groups_muter.js", import.meta.url), "utf8")
    .replace(/^import .*;\r?\n/gm, "");
vm.runInNewContext(source, {
    ...logic,
    app: { registerExtension(value) { extension = value; } },
    LGraphCanvas: { node_colors: {} },
    LiteGraph: { NODE_WIDGET_HEIGHT: 20 },
    setInterval(callback) { tick = callback; return 1; },
    clearInterval() { tick = undefined; },
});

class Node {
    constructor(graph) { this.graph = graph; this.widgets = []; this.properties = {}; this.size = [300, 100]; }
    addProperty(name, value) { this.properties[name] = value; }
    setProperty(name, value) { this.properties[name] = value; this.onPropertyChanged(); }
    addWidget(type, name, value, callback, options) {
        const widget = { type, name, value, callback, options };
        this.widgets.push(widget);
        return widget;
    }
    addCustomWidget(widget) { this.widgets.push(widget); return widget; }
    removeWidget(widget) { this.widgets = this.widgets.filter((entry) => entry !== widget); }
    computeSize() { return [300, 50 + 24 * this.widgets.length]; }
    setSize(size) { this.size = size; }
    setDirtyCanvas() {}
}
await extension.beforeRegisterNodeDef(Node, { name: "RPGroupsMuter" });

function fixture(t) {
    const graph = {
        nodes: [{ mode: 0, pos: [20, 20], size: [10, 10] }, { mode: 0, pos: [160, 20], size: [10, 10] }],
        groups: [], changes: [], beforeChange() { this.changes.push("before"); },
        afterChange() { this.changes.push("after"); }, setDirtyCanvas() {},
    };
    graph.groups = [{ title: "First", pos: [0, 0], size: [100, 100], graph },
        { title: "Second", pos: [140, 0], size: [100, 100], graph }];
    function muter() {
        const node = new Node(graph);
        node.onNodeCreated();
        node.onAdded();
        t.after(() => node.onRemoved());
        return node;
    }
    return { graph, muter };
}

test("bottom button mutes/unmutes all and synchronizes other muters and manual changes", (t) => {
    const { graph, muter } = fixture(t);
    const first = muter();
    const second = muter();
    assert.equal(first.widgets.at(-1), first.muteAllWidget);
    assert.equal(first.muteAllWidget.name, "Mute all");
    first.muteAllWidget.callback();
    assert.deepEqual(graph.nodes.map((node) => node.mode), [2, 2]);
    assert.equal(first.muteAllWidget.label, "Unmute all");
    assert.equal(second.muteAllWidget.label, "Unmute all");
    assert.deepEqual(graph.changes, ["before", "after"]);
    second.muteAllWidget.callback();
    assert.deepEqual(graph.nodes.map((node) => node.mode), [0, 0]);
    assert.equal(first.muteAllWidget.name, "Mute all");
    graph.nodes.forEach((node) => { node.mode = 2; });
    tick();
    assert.equal(first.muteAllWidget.name, "Unmute all");
    graph.nodes[0].mode = 0;
    tick();
    assert.equal(first.muteAllWidget.name, "Mute all");
    assert.equal(first.widgets.at(-1), first.muteAllWidget);
    assert.equal(first.muteAllWidget.serialize, false);
});

test("bulk button honors nested visibility and title filter", (t) => {
    const { graph, muter } = fixture(t);
    graph.groups.push({ title: "Nested", pos: [10, 10], size: [40, 40], graph });
    const node = muter();
    node.setProperty("matchTitle", "^Nested$");
    node.muteAllWidget.callback();
    assert.deepEqual(graph.nodes.map((entry) => entry.mode), [0, 0]);
    assert.deepEqual(graph.changes, []);
    node.hideNestedGroupsWidget.callback(false);
    assert.equal(node.widgets[1].group.title, "Nested");
    node.muteAllWidget.callback();
    assert.deepEqual(graph.nodes.map((entry) => entry.mode), [2, 0]);
    assert.equal(node.muteAllWidget.name, "Unmute all");
    node.muteAllWidget.callback();
    assert.deepEqual(graph.nodes.map((entry) => entry.mode), [0, 0]);
});
