import assert from "node:assert/strict";
import { test } from "node:test";
import { getGroupNodes, getMatchingGroups, isGroupEnabled, setGroupEnabled } from "../web/js/workflow_utilities/groups_muter_logic.js";

function fixture() {
    const graph = { nodes: [], groups: [], changes: [],
        beforeChange() { this.changes.push("before"); },
        afterChange() { this.changes.push("after"); },
        setDirtyCanvas() {},
    };
    const node = (type, mode, x = 20) => ({ type, mode, pos: [x, 20], size: [40, 40] });
    graph.nodes = [node("LoadImage", 2), node("PreviewImage", 2),
        node("Note", 0), node("MarkdownNote", 0), node("Label (rgthree)", 0),
        node("Reroute", 0), node("RPGroupsMuter", 0),
        { ...node("Other editor", 0), isVirtualNode: true }, node("Outside", 0, 150)];
    const group = { title: "Group 1", graph, pos: [0, 0], size: [100, 100], color: "#aabbcc", rgthree_hasAnyActiveNode: true };
    graph.groups = [group];
    return { graph, group };
}

test("muted group stays off despite active notes, editor controls, and stale rgthree state", () => {
    const { graph, group } = fixture();
    assert.equal(isGroupEnabled(group), false);
    assert.deepEqual(getGroupNodes(group), graph.nodes.slice(0, 2));
    graph.nodes[0].mode = 4;
    assert.equal(isGroupEnabled(group), false);
    graph.nodes[1].mode = 0;
    assert.equal(isGroupEnabled(group), true);
    graph.nodes[1].mode = 2;
    assert.equal(isGroupEnabled(group), false);
});

test("toggles update only current executable members and use a single graph change", () => {
    const { graph, group } = fixture();
    setGroupEnabled(group, true);
    assert.deepEqual(graph.nodes.map(n => n.mode), [0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert.deepEqual(graph.changes, ["before", "after"]);
    setGroupEnabled(group, false);
    assert.deepEqual(graph.nodes.map(n => n.mode), [2, 2, 0, 0, 0, 0, 0, 0, 0]);
    setGroupEnabled(group, false);
    assert.equal(graph.changes.length, 4);
});

test("membership follows moves and resizing without relying on render bounds or cached group nodes", () => {
    const { graph, group } = fixture();
    graph.nodes[0].mode = 0;
    assert.equal(isGroupEnabled(group), true);
    graph.nodes[0].pos[0] = 100;
    assert.equal(isGroupEnabled(group), false);
    group.size[0] = 200;
    assert.equal(isGroupEnabled(group), true);
    group.nodes = [];
    assert.equal(getGroupNodes(group).length, 3);
});

test("duplicate names, nested groups, and empty groups keep independent actual states", () => {
    const { graph, group } = fixture();
    const second = { ...group, pos: [140, 0], size: [100, 100] };
    assert.equal(isGroupEnabled(group), false);
    assert.equal(isGroupEnabled(second), true);
    setGroupEnabled(second, false);
    assert.equal(graph.nodes.at(-1).mode, 2);
    const outer = { ...group, size: [250, 150] };
    setGroupEnabled(outer, true);
    assert.equal(isGroupEnabled(group), true);
    assert.equal(isGroupEnabled(second), true);
    assert.equal(isGroupEnabled({ ...group, pos: [500, 0] }), false);
});

test("filters support title patterns, hex shorthand, palette colors and natural ordering", () => {
    const { graph, group } = fixture();
    graph.groups = [{ ...group, title: "Group 10", pos: [0, 0] },
        { ...group, title: "Group 2", pos: [100, 0] }, { ...group, title: "Other", color: "#ff0000", pos: [0, 100] }];
    assert.deepEqual(getMatchingGroups(graph, { matchTitle: "^group", matchColors: "#abc", sort: "alphanumeric" })
        .map(g => g.title), ["Group 2", "Group 10"]);
    assert.equal(getMatchingGroups(graph, { matchColors: "red" }, { red: { groupcolor: "#f00" } })[0].title, "Other");
    assert.equal(getMatchingGroups(graph, { sort: "position" })[0].title, "Group 10");
    assert.equal(graph.groups[0].title, "Group 10");
    group.title = "Group [";graph.groups = [group];
    assert.equal(getMatchingGroups(graph, { matchTitle: "[" }).length, 1);
});
