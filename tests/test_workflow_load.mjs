import assert from "node:assert/strict";
import { test } from "node:test";
import { beginWorkflowLoad, endWorkflowLoad, loadMergedWorkflow, prepareNodeForLoad } from "../web/js/workflow_utilities/load_workflow.js";

function fixture(load) {
    const before = { nodes: [{ id: "1" }], links: [] };
    const initialState = { nodes: [], links: [] };
    const tracker = {
        initialState, activeState: before, changeCount: 0, undoQueue: [],
        captureCanvasState() {
            if (this.changeCount) return;
            if (this.activeState !== app.graphData) {
                this.undoQueue.push(this.activeState);
                this.activeState = app.graphData;
            }
        },
        beforeChange() { this.changeCount++; },
        afterChange() { this.changeCount--; this.captureCanvasState(); },
    };
    const activeWorkflow = { changeTracker: tracker };
    const app = {
        graphData: before,
        canvas: { ds: { scale: 0.8, offset: [100, 200] } },
        extensionManager: { workflow: { activeWorkflow } },
        async loadGraphData(data, clean, restoreView, workflow, options) {
            assert.equal(clean, false);
            assert.equal(restoreView, false);
            assert.equal(workflow, activeWorkflow);
            assert.equal(options.skipAssetScans, true);
            assert.equal(tracker.changeCount, 1);
            app.graphData = data;
            tracker.initialState = data;
            tracker.activeState = data;
            app.canvas.ds.scale = 1;
            app.canvas.ds.offset = [0, 0];
            beginWorkflowLoad();
            try {
                return load?.(data) ?? true;
            } finally {
                endWorkflowLoad();
            }
        },
    };
    return { app, tracker, before, initialState };
}

test("normal loading hooks run without replacing the saved baseline or undo history", async () => {
    let configured = false;
    const { app, tracker, before, initialState } = fixture(() => { configured = true; });
    const merged = { nodes: [...before.nodes, { id: "2" }], links: [] };
    await loadMergedWorkflow(app, merged, before);
    assert.equal(configured, true);
    assert.equal(app.graphData, merged);
    assert.equal(tracker.initialState, initialState);
    assert.equal(tracker.activeState, merged);
    assert.deepEqual(tracker.undoQueue, [before]);
    assert.deepEqual(app.canvas.ds, { scale: 0.8, offset: [100, 200] });
    assert.equal(tracker.changeCount, 0);
});

for (const outcome of [false, new Error("custom node failed")]) {
    test(`failed load rolls back the canvas and preserves history: ${outcome}`, async () => {
        const { app, tracker, before, initialState } = fixture(data => {
            if (data === before) return true;
            if (outcome instanceof Error) throw outcome;
            return outcome;
        });
        await assert.rejects(loadMergedWorkflow(app, { nodes: [{ id: "2" }], links: [] }, before));
        assert.equal(app.graphData, before);
        assert.equal(tracker.initialState, initialState);
        assert.equal(tracker.activeState, before);
        assert.deepEqual(tracker.undoQueue, []);
        assert.equal(tracker.changeCount, 0);
        assert.deepEqual(app.canvas.ds, { scale: 0.8, offset: [100, 200] });
    });
}

test("legacy custom-node callbacks resolve string IDs during import and restore connect afterwards", async () => {
    const target = { findInputSlot: () => 3 };
    const node = {
        graph: { getNodeById: id => id === "target" ? target : null },
        connect(slot, targetNode, name) {
            return { slot, targetNode, input: targetNode.findInputSlot(name) };
        },
    };
    const original = node.connect;
    const { app, before } = fixture(() => {
        prepareNodeForLoad(node);
        // Impact Pack calls connect(slot, node.id, inputName) while configuring.
        assert.deepEqual(node.connect(0, "target", "input1"), { slot: 0, targetNode: target, input: 3 });
        assert.deepEqual(node.connect(0, target, "input1"), { slot: 0, targetNode: target, input: 3 });
    });
    prepareNodeForLoad(node);
    assert.equal(node.connect, original);
    await loadMergedWorkflow(app, { nodes: [{ id: "2" }], links: [] }, before);
    assert.equal(node.connect, original);
});

test("connection compatibility also applies to undo, redo, and reopening a saved graph", () => {
    for (let load = 0; load < 3; load++) {
        const node = {
            graph: { getNodeById: () => node },
            connect(slot, target) {
                if (target === this) return null;
                return target.findInputSlot("input1");
            },
        };
        const original = node.connect;
        beginWorkflowLoad();
        prepareNodeForLoad(node);
        assert.equal(node.connect(0, "123"), null);
        endWorkflowLoad();
        assert.equal(node.connect, original);
    }
});
