import assert from "node:assert/strict";
import { test } from "node:test";
import { fixWorkflow } from "../web/js/workflow_utilities/merge_workflow.js";

function fixture(version = 0.4) {
    const link = { id: 8, origin_id: 1525, origin_slot: 0, target_id: 2, target_slot: 0, type: "INT" };
    return {
        id: "workflow-id", revision: 7, version, last_node_id: 1525, last_link_id: 2967,
        nodes: [
            { id: 1525, type: "Integer Selector", title: "Control Map Selector", pos: [5600, 540], size: [200, 100],
                outputs: [{ name: "Integer", type: "INT", links: [8] }], widgets_values: [2], properties: { custom: true } },
            { id: 2, type: "Sink", pos: [5900, 540], size: [200, 100], inputs: [{ name: "value", type: "INT", link: 8 }],
                outputs: [{ name: "unused", links: [] }], widgets_values: ["text"] },
        ],
        links: version === 1 ? [link] : [[8, 1525, 0, 2, 0, "INT"]],
        groups: [{ id: 1, title: "Klein", bounding: [5500, 500, 800, 300] }],
        config: { custom: "keep" }, extra: { ds: { scale: 0.5, offset: [50, 60] }, node_versions: { custom: "1" } },
        models: [{ name: "model", url: "local" }],
    };
}

for (const version of [0.4, 1]) {
    test(`repair Essential Klein's orphaned link without moving or renumbering nodes, v${version}`, () => {
        const workflow = fixture(version);
        const goodLink = structuredClone(workflow.links[0]);
        workflow.links.push(version === 1
            ? { id: 2967, origin_id: 1525, origin_slot: 0, target_id: 1547, target_slot: 0, type: "INT" }
            : [2967, 1525, 0, 1547, 0, "INT"]);
        workflow.nodes[0].outputs[0].links.push(2967);
        const original = structuredClone(workflow);
        const { workflow: fixed, ...repairs } = fixWorkflow(workflow);
        assert.deepEqual(workflow, original);
        assert.deepEqual(repairs, { links: 1, references: 1, reroutes: 0 });
        assert.deepEqual(fixed.links, [goodLink]);
        assert.deepEqual(fixed.nodes[0].outputs[0].links, [8]);
        for (let i = 0; i < workflow.nodes.length; i++) {
            for (const field of ["id", "type", "title", "pos", "size", "widgets_values", "properties"]) {
                assert.deepEqual(fixed.nodes[i][field], workflow.nodes[i][field]);
            }
        }
        for (const field of ["id", "revision", "version", "last_node_id", "last_link_id", "groups", "config", "extra", "models"]) {
            assert.deepEqual(fixed[field], workflow[field]);
        }
        assert.deepEqual(fixWorkflow(fixed), { workflow: fixed, links: 0, references: 0, reroutes: 0 });
    });

    test(`healthy workflow is an exact no-op, including empty output arrays, v${version}`, () => {
        const workflow = fixture(version);
        assert.deepEqual(fixWorkflow(workflow), { workflow, links: 0, references: 0, reroutes: 0 });
    });
}

test("clear stale inputs, reroute parents, and references while retaining a valid floating link", () => {
    const workflow = fixture(1);
    workflow.nodes[1].inputs.push({ name: "stale", link: 999 });
    workflow.floatingLinks = [{ id: 9, origin_id: 1525, origin_slot: 0, target_id: -1, target_slot: -1, type: "INT", parentId: 1 }];
    workflow.reroutes = [{ id: 1, pos: [5800, 560], linkIds: [8, 9, 999], parentId: 999 }];
    workflow.links[0].parentId = 999;
    const { workflow: fixed, ...repairs } = fixWorkflow(workflow);
    assert.deepEqual(repairs, { links: 0, references: 4, reroutes: 0 });
    assert.equal(fixed.nodes[1].inputs[1].link, null);
    assert.deepEqual(fixed.reroutes[0].linkIds, [8, 9]);
    assert.equal(fixed.reroutes[0].parentId, undefined);
    assert.equal(fixed.links[0].parentId, undefined);
    assert.deepEqual(fixed.floatingLinks, workflow.floatingLinks);
    assert.deepEqual(fixed.reroutes[0].pos, workflow.reroutes[0].pos);
});

test("repair nested subgraphs while preserving their boundary connections and IDs", () => {
    const workflow = fixture();
    const subgraph = fixture(1);
    subgraph.id = "subgraph-id";
    subgraph.inputNode = { id: -10 };
    subgraph.outputNode = { id: -20 };
    subgraph.links.push(
        { id: 9, origin_id: -10, origin_slot: 0, target_id: 1525, target_slot: 0, type: "INT" },
        { id: 10, origin_id: 2, origin_slot: 0, target_id: -20, target_slot: 0, type: "INT" },
        { id: 11, origin_id: 999, origin_slot: 0, target_id: 2, target_slot: 1, type: "INT" },
    );
    subgraph.inputs = [{ linkIds: [9, 999] }];
    subgraph.outputs = [{ linkIds: [10] }];
    subgraph.nodes[1].inputs.push({ link: 11 });
    const outer = fixture(1);
    outer.id = "outer-id";
    outer.definitions = { subgraphs: [subgraph] };
    workflow.definitions = { subgraphs: [outer] };
    const { workflow: fixed, ...repairs } = fixWorkflow(workflow);
    assert.deepEqual(repairs, { links: 1, references: 2, reroutes: 0 });
    assert.deepEqual(fixed.nodes, workflow.nodes);
    const inner = fixed.definitions.subgraphs[0].definitions.subgraphs[0];
    assert.equal(inner.id, subgraph.id);
    assert.deepEqual(inner.links.map(link => link.id), [8, 9, 10]);
    assert.deepEqual(inner.inputs[0].linkIds, [9]);
    assert.deepEqual(inner.outputs[0].linkIds, [10]);
    assert.equal(inner.nodes[1].inputs[1].link, null);
});
