import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { appendWorkflows } from "./merge_workflow.js";

const EMPTY = "(select a saved workflow)";
let importing = false;

function setStatus(node, message) {
    const current = node.graph ? node : app.rootGraph?.getNodeById(node.id);
    if (current?.rpImportStatus) current.rpImportStatus.textContent = message;
    current?.graph?.setDirtyCanvas(true, true);
}

function notify(node, message, severity = "error") {
    setStatus(node, message);
    app.extensionManager.toast.add({ severity, summary: "RP Import Workflow", detail: message, life: 7000 });
}

async function refreshWorkflows(node, showStatus = true) {
    const response = await api.fetchApi("/userdata?dir=workflows&recurse=true&split=false");
    if (!response.ok && response.status !== 404) {
        throw new Error(`Cannot list saved workflows: HTTP ${response.status}.`);
    }
    const files = response.status === 404 ? [] : await response.json();
    const values = files.filter((file) => file.toLowerCase().endsWith(".json"))
        .sort((a, b) => a.localeCompare(b));
    const widget = node.widgets.find((widget) => widget.name === "workflow");
    widget.options.values = [EMPTY, ...values];
    if (!values.includes(widget.value)) widget.value = EMPTY;
    if (showStatus) {
        setStatus(node, values.length ? `${values.length} saved workflows available.` : "No saved workflows. Choose a JSON file.");
    }
}

function options(node) {
    const value = (name) => node.widgets.find((widget) => widget.name === name)?.value;
    return { placement: value("placement"), gap: value("gap"), groupImport: value("group_import") };
}

async function importIntoCanvas(node, readItems) {
    if (importing) return;
    const graph = app.rootGraph;
    if (node.graph !== graph || app.canvas.graph !== graph) {
        notify(node, "Return to the main workflow canvas to import.");
        return;
    }
    importing = true;
    const graphId = graph.id;
    const importOptions = options(node);
    setStatus(node, "Reading workflow JSON...");
    try {
        const items = await readItems();
        if (graph.id !== graphId || node.graph !== graph || app.canvas.graph !== graph) {
            throw new Error("The active workflow changed. Start the import again.");
        }
        const before = JSON.parse(JSON.stringify(graph.serialize()));
        const result = appendWorkflows(before, items, importOptions);
        const scale = app.canvas.ds.scale;
        const offset = [...app.canvas.ds.offset];
        const tracker = app.extensionManager.workflow.activeWorkflow.changeTracker;
        tracker.captureCanvasState();
        tracker.beforeChange();
        try {
            graph.configure(result.workflow);
        } catch (error) {
            graph.configure(before);
            throw error;
        } finally {
            app.canvas.ds.scale = scale;
            app.canvas.ds.offset = offset;
            tracker.afterChange();
        }
        app.canvas.selectNodes(result.importedNodeIds.map((id) => graph.getNodeById(id)));
        notify(node, `Imported ${items.length} workflow(s), ${result.importedNodeIds.length} nodes. Ctrl+Z to undo.`, "success");
    } catch (error) {
        notify(node, error.message ?? String(error));
    } finally {
        importing = false;
    }
}

app.registerExtension({
    name: "RPNodes.ImportWorkflow",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "RPImportWorkflow") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            this.isVirtualNode = true;
            this.addWidget("button", "Refresh saved workflows", null, () => {
                refreshWorkflows(this).catch((error) => notify(this, error.message));
            }, { serialize: false }).serialize = false;
            this.addWidget("button", "Import saved workflow", null, () => {
                const filename = this.widgets.find((widget) => widget.name === "workflow").value;
                if (!filename || filename === EMPTY) {
                    notify(this, "Select a saved workflow first.");
                    return;
                }
                void importIntoCanvas(this, async () => {
                    const response = await api.getUserData(`workflows/${filename}`);
                    if (!response.ok) throw new Error(`Cannot read ${filename}: HTTP ${response.status}.`);
                    return [{ name: filename, data: await response.json() }];
                });
            }, { serialize: false }).serialize = false;
            this.addWidget("button", "Choose file to import", null, () => {
                const input = document.createElement("input");
                input.type = "file";
                input.accept = ".json,application/json";
                input.multiple = true;
                input.onchange = () => {
                    const files = [...input.files];
                    if (!files.length) return;
                    void importIntoCanvas(this, () => Promise.all(files.map(async (file) => ({
                        name: file.name, data: JSON.parse(await file.text()),
                    }))));
                };
                input.click();
            }, { serialize: false }).serialize = false;
            const statusElement = document.createElement("div");
            statusElement.textContent = "Use the buttons; no queue needed.";
            Object.assign(statusElement.style, {
                padding: "6px 10px", fontSize: "12px", lineHeight: "16px",
                whiteSpace: "normal", overflow: "auto", color: "var(--input-text)",
            });
            this.rpImportStatus = statusElement;
            const status = this.addDOMWidget("import_status", "text", statusElement, { serialize: false });
            status.serialize = false;
            status.computeSize = (width) => [width, 48];
            status.computeLayoutSize = () => ({ minHeight: 48, minWidth: 0 });
            this.setSize([Math.max(this.size[0], 340), this.computeSize()[1]]);
            return result;
        };
        const onAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function () {
            const result = onAdded?.apply(this, arguments);
            // Wait until configure() has restored the saved combo value.
            queueMicrotask(() => refreshWorkflows(this, false).catch((error) => notify(this, error.message)));
            return result;
        };
    },
});
