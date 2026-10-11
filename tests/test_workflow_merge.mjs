import assert from "node:assert/strict";
import { test } from "node:test";
import { appendWorkflows, validateWorkflow } from "../web/js/workflow/merge_workflow.js";

function fixture(version = 0.4) {
    const link = { id: 8, origin_id: 1, origin_slot: 0, target_id: 2, target_slot: 0, type: "IMAGE", parentId: 2 };
    const reroutes = [
        { id: 1, pos: [250, 60], linkIds: [8] },
        { id: 2, pos: [270, 80], parentId: 1, linkIds: [8] },
    ];
    return {
        id: "current-workflow", revision: 5, version,
        last_node_id: 50, last_link_id: 60,
        state: { lastNodeId: 50, lastLinkId: 60, lastGroupId: 20, lastRerouteId: 30 },
        nodes: [
            { id: 1, type: "Source", pos: [-20, 30], size: { 0: 200, 1: 100 }, outputs: [{ links: [8] }], widgets_values: ["prompt", 42] },
            { id: 2, type: "Sink", pos: [300, 30], size: [200, 100], inputs: [{ link: 8 }], widgets_values: ["output"] },
        ],
        links: version === 1 ? [link] : [[8, 1, 0, 2, 0, "IMAGE"]],
        groups: [{ id: 3, title: "Source group", bounding: [-60, -10, 600, 200], color: "#fff" }],
        extra: { ds: { scale: 0.7, offset: [120, 30] }, customSetting: { enabled: true },
            ...(version === 0.4 ? { reroutes, linkExtensions: [{ id: 8, parentId: 2 }] } : {}) },
        ...(version === 1 ? { reroutes } : {}),
    };
}

for (const currentVersion of [0.4, 1]) {
    for (const importVersion of [0.4, 1]) {
        test(`preserve current canvas and connected reroutes: ${importVersion} into ${currentVersion}`, () => {
            const current = fixture(currentVersion);
            const imported = fixture(importVersion);
            const originals = structuredClone([current, imported]);
            const { workflow: result, importedNodeIds } = appendWorkflows(current, [{ name: "import.json", data: imported }]);
            assert.deepEqual([current, imported], originals);
            assert.equal(result.id, current.id);
            assert.equal(result.revision, current.revision);
            assert.deepEqual(result.nodes.slice(0, 2), current.nodes);
            assert.deepEqual(result.links[0], current.links[0]);
            assert.deepEqual(result.groups[0], current.groups[0]);
            assert.deepEqual(result.extra.ds, current.extra.ds);
            assert.deepEqual(result.extra.customSetting, current.extra.customSetting);
            assert.ok(importedNodeIds.every((id) => id > 50));
            const link = currentVersion === 1 ? result.links[1] : {
                id: result.links[1][0], origin_id: result.links[1][1], target_id: result.links[1][3],
                parentId: result.extra.linkExtensions[1].parentId,
            };
            assert.equal(link.origin_id, importedNodeIds[0]);
            assert.equal(link.target_id, importedNodeIds[1]);
            assert.equal(result.nodes[2].outputs[0].links[0], link.id);
            assert.equal(result.nodes[3].inputs[0].link, link.id);
            const reroutes = currentVersion === 1 ? result.reroutes : result.extra.reroutes;
            assert.equal(link.parentId, reroutes[3].id);
            assert.equal(reroutes[3].parentId, reroutes[2].id);
            assert.deepEqual(reroutes[2].linkIds, [link.id]);
            assert.deepEqual(result.nodes[2].widgets_values, imported.nodes[0].widgets_values);
            const wrapper = result.groups.at(-1);
            assert.equal(wrapper.title, "import");
            assert.ok(wrapper.bounding[0] >= current.groups[0].bounding[0] + current.groups[0].bounding[2] + 160);
            validateWorkflow(result);
        });
    }
}

test("multiple imports have independent IDs and groups, placed below without moving existing nodes", () => {
    const current = fixture();
    const imported = fixture();
    const { workflow: result } = appendWorkflows(current, [
        { name: "first.json", data: imported }, { name: "second.json", data: imported },
    ], { placement: "below", gap: 200 });
    assert.equal(new Set(result.nodes.map((node) => node.id)).size, 6);
    assert.equal(new Set(result.links.map((link) => link[0])).size, 3);
    assert.equal(new Set(result.groups.map((group) => group.id)).size, 5);
    assert.ok(result.groups[2].bounding[1] >= 390);
    assert.ok(result.groups[4].bounding[1] >= result.groups[2].bounding[1] + result.groups[2].bounding[3] + 200);
    assert.deepEqual(result.nodes.slice(0, 2), current.nodes);
    validateWorkflow(result);
});

test("colliding subgraph IDs are independent and widget strings remain untouched", () => {
    const current = fixture();
    const source = fixture();
    const inner = fixture(1);
    inner.id = "nested-id";
    const outer = fixture(1);
    outer.id = "outer-id";
    outer.nodes[0].type = inner.id;
    outer.definitions = { subgraphs: [inner] };
    source.nodes[0].type = outer.id;
    source.nodes[0].widgets_values = [outer.id, inner.id];
    source.definitions = { subgraphs: [outer] };
    current.definitions = structuredClone(source.definitions);
    const { workflow: result } = appendWorkflows(current, [{ name: "nested.json", data: source }]);
    const existing = result.definitions.subgraphs[0];
    const imported = result.definitions.subgraphs[1];
    assert.deepEqual(existing, current.definitions.subgraphs[0]);
    assert.notEqual(imported.id, existing.id);
    assert.equal(result.nodes[2].type, imported.id);
    assert.equal(imported.nodes[0].type, imported.definitions.subgraphs[0].id);
    assert.notEqual(imported.definitions.subgraphs[0].id, inner.id);
    assert.deepEqual(result.nodes[2].widgets_values, [outer.id, inner.id]);
});

test("floating link endpoints and reroutes follow newly allocated IDs", () => {
    const source = fixture(1);
    source.floatingLinks = [{ id: 9, origin_id: 1, origin_slot: 0, target_id: -1, target_slot: -1, type: "IMAGE", parentId: 2 }];
    source.reroutes[1].linkIds.push(9);
    const { workflow: result, importedNodeIds } = appendWorkflows(fixture(), [{ name: "floating.json", data: source }]);
    assert.equal(result.floatingLinks[0].origin_id, importedNodeIds[0]);
    assert.equal(result.floatingLinks[0].target_id, -1);
    assert.equal(result.floatingLinks[0].parentId, result.extra.reroutes.at(-1).id);
    assert.ok(result.extra.reroutes.at(-1).linkIds.includes(result.floatingLinks[0].id));
});

test("merge model and node version metadata without changing current values", () => {
    const current = fixture();
    current.models = [{ name: "existing", url: "local" }];
    current.extra.node_versions = { existing: "1" };
    const source = fixture();
    source.models = [...current.models, { name: "new", url: "local2" }];
    source.extra.node_versions = { existing: "2", added: "3" };
    const { workflow: result } = appendWorkflows(current, [{ name: "models.json", data: source }], { groupImport: false });
    assert.equal(result.models.length, 2);
    assert.deepEqual(result.extra.node_versions, { existing: "1", added: "3" });
    assert.equal(result.groups.length, 2);
});

test("reject malformed graphs and API prompts before mutating the current graph", () => {
    const current = fixture();
    const original = structuredClone(current);
    for (const invalid of [null, { "1": { class_type: "KSampler", inputs: {} } },
        { nodes: [], links: [] },
        { ...fixture(), nodes: [fixture().nodes[0], fixture().nodes[0]] },
        { ...fixture(), links: [[8, 999, 0, 2, 0, "IMAGE"]] },
        { ...fixture(), links: [] },
        { ...fixture(), links: [...fixture().links, ...fixture().links] },
        { ...fixture(), links: [[8, -1, 0, 2, 0, "IMAGE"]] },
        { ...fixture(), extra: { reroutes: [{ id: 1, linkIds: [999] }] } },
        { ...fixture(), extra: { reroutes: [{ id: 1, parentId: 999 }] } },
        { ...fixture(), extra: { reroutes: [{ id: 1 }, { id: 1 }] } },
    ]) {
        assert.throws(() => appendWorkflows(current, [{ name: "bad.json", data: invalid }]));
        assert.deepEqual(current, original);
    }
    assert.throws(() => appendWorkflows(current, [
        { name: "good.json", data: fixture() }, { name: "bad.json", data: null },
    ]));
    assert.deepEqual(current, original);
});
