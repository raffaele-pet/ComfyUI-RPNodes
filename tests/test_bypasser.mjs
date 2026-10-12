import assert from "node:assert/strict";
import { test } from "node:test";
import { getConnectedTargets, setTargetsEnabled } from "../web/js/workflow_utilities/bypasser_logic.js";

function fixture() {
    const graph = { nodes: new Map(), links: new Map(), changes: [],
        getNodeById(id) { return this.nodes.get(id); },
        beforeChange() { this.changes.push("before"); },
        afterChange() { this.changes.push("after"); }, setDirtyCanvas() {},
    };
    const targets = [0, 2, 4].map((mode, index) => ({ id: index + 1, mode, title: "Same name", type: "ImageScaleBy", graph }));
    for (const node of targets) graph.nodes.set(node.id, node);
    const controller = { id: 10, graph, inputs: [{ link: 0 }, { link: 2 }, { link: 3 }, { link: null }] };
    graph.links.set(0, { origin_id: 1 });graph.links.set(2, { origin_id: 2 });graph.links.set(3, { origin_id: 3 });
    graph.nodes.set(10, controller);
    return { graph, targets, controller };
}

test("reading connections never activates muted or bypassed nodes, including link ID zero", () => {
    const { controller, targets } = fixture();
    assert.deepEqual(getConnectedTargets(controller).map(row => row.node), targets);
    assert.deepEqual(targets.map(node => node.mode), [0, 2, 4]);
});

test("single-node toggles and Bypass all change only connected nodes in one history operation", () => {
    const { controller, graph, targets } = fixture();
    setTargetsEnabled(controller, [targets[2]], true);
    assert.deepEqual(targets.map(node => node.mode), [0, 2, 0]);
    setTargetsEnabled(controller, targets, false);
    assert.deepEqual(targets.map(node => node.mode), [4, 4, 4]);
    assert.deepEqual(graph.changes, ["before", "after", "before", "after"]);
    setTargetsEnabled(controller, targets, false);
    assert.equal(graph.changes.length, 4);
});

test("duplicate source connections remain separate rows but update the node only once", () => {
    const { controller, graph, targets } = fixture();
    graph.links.set(2, { origin_id: 1 });
    const rows = getConnectedTargets(controller);
    assert.equal(rows[0].node, rows[1].node);
    setTargetsEnabled(controller, rows.map(row => row.node), false);
    assert.deepEqual(targets.map(node => node.mode), [4, 2, 4]);
    assert.deepEqual(graph.changes, ["before", "after"]);
});

test("reroute chains resolve the actual source; cycles, missing sources and self-links are ignored", () => {
    const { controller, graph, targets } = fixture();
    const reroute = { id: 20, type: "Reroute", inputs: [{ link: 30 }] };
    graph.nodes.set(20, reroute);graph.links.set(0, { origin_id: 20 });graph.links.set(30, { origin_id: 1 });
    assert.equal(getConnectedTargets(controller)[0].node, targets[0]);
    graph.links.set(30, { origin_id: 20 });graph.links.set(2, { origin_id: 999 });graph.links.set(3, { origin_id: 10 });
    assert.deepEqual(getConnectedTargets(controller), []);
});
