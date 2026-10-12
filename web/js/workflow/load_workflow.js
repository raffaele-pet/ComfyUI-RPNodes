let connectionRestorers = null;

export function beginWorkflowLoad() {
    endWorkflowLoad();
    connectionRestorers = [];
}

export function endWorkflowLoad() {
    for (const restore of connectionRestorers ?? []) restore();
    connectionRestorers = null;
}

export function prepareNodeForLoad(node) {
    if (!connectionRestorers || typeof node.connect !== "function") return;
    const connect = node.connect;
    const resolveTarget = function (slot, target, ...args) {
        // Legacy extensions pass node.id here, but recent frontends use string
        // IDs while connect() accepts a node object or a numeric ID.
        if (typeof target === "string") target = this.graph.getNodeById(target);
        return connect.call(this, slot, target, ...args);
    };
    node.connect = resolveTarget;
    connectionRestorers.push(() => {
        if (node.connect === resolveTarget) node.connect = connect;
    });
}

export async function loadMergedWorkflow(app, workflow, before) {
    const activeWorkflow = app.extensionManager.workflow.activeWorkflow;
    const tracker = activeWorkflow.changeTracker;
    tracker.captureCanvasState();
    const initialState = tracker.initialState;
    const previousState = tracker.activeState;
    const scale = app.canvas.ds.scale;
    const offset = [...app.canvas.ds.offset];
    const options = {
        checkForRerouteMigration: false,
        skipAssetScans: true,
        deferWarnings: true,
    };
    tracker.beforeChange();
    try {
        // Use the normal loading lifecycle so custom nodes recognize restoration
        // and run their before/after configure hooks instead of reconnecting slots.
        const loaded = await app.loadGraphData(workflow, false, false, activeWorkflow, options);
        if (loaded === false) throw new Error("ComfyUI could not load the updated workflow.");
    } catch (error) {
        await app.loadGraphData(before, false, false, activeWorkflow, options);
        throw error;
    } finally {
        // The loader resets the saved baseline. This is an edit to the same
        // workflow, so keep its baseline and let afterChange record one undo step.
        tracker.initialState = initialState;
        tracker.activeState = previousState;
        app.canvas.ds.scale = scale;
        app.canvas.ds.offset = offset;
        tracker.afterChange();
    }
}
