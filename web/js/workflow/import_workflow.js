import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { appendWorkflows } from "./merge_workflow.js";
import { beginWorkflowLoad, endWorkflowLoad, loadMergedWorkflow, prepareNodeForLoad } from "./load_workflow.js";

const EMPTY = "(select a saved workflow)";
let importing = false;

function notify(message, severity = "error") {
    app.extensionManager.toast.add({ severity, summary: "RP Import Workflow", detail: message, life: 7000 });
}

async function refreshWorkflows(node) {
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
    node.graph?.setDirtyCanvas(true, true);
}

function options(node) {
    const value = (name) => node.widgets.find((widget) => widget.name === name)?.value;
    return { placement: value("placement"), gap: value("gap"), groupImport: value("group_import") };
}

async function importIntoCanvas(node, readItems) {
    if (importing) return;
    const graph = app.rootGraph;
    if (node.graph !== graph || app.canvas.graph !== graph) {
        notify("Return to the main workflow canvas to import.");
        return;
    }
    importing = true;
    const graphId = graph.id;
    const importOptions = options(node);
    try {
        const items = await readItems();
        if (graph.id !== graphId || node.graph !== graph || app.canvas.graph !== graph) {
            throw new Error("The active workflow changed. Start the import again.");
        }
        const before = JSON.parse(JSON.stringify(graph.serialize()));
        const result = appendWorkflows(before, items, importOptions);
        await loadMergedWorkflow(app, result.workflow, before);
        app.canvas.selectNodes(result.importedNodeIds.map((id) => graph.getNodeById(id)));
        const count = result.importedNodeIds.length;
        notify(`Imported ${count} ${count === 1 ? "node" : "nodes"}.`, "success");
    } catch (error) {
        notify(error.message ?? String(error));
    } finally {
        importing = false;
    }
}

app.registerExtension({
    name: "RPNodes.ImportWorkflow",
    beforeLoadGraph() {
        beginWorkflowLoad();
    },
    afterConfigureGraph() {
        endWorkflowLoad();
    },
    onGraphLoadError() {
        endWorkflowLoad();
    },
    nodeCreated(node) {
        prepareNodeForLoad(node);
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "RPImportWorkflow") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            this.isVirtualNode = true;
            this.addWidget("button", "Refresh saved workflows", null, () => {
                refreshWorkflows(this).catch((error) => notify(error.message));
            }, { serialize: false }).serialize = false;
            this.addWidget("button", "Import saved workflow", null, () => {
                const filename = this.widgets.find((widget) => widget.name === "workflow").value;
                if (!filename || filename === EMPTY) {
                    notify("Select a saved workflow first.");
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
            this.setSize([Math.max(this.size[0], 340), this.computeSize()[1]]);
            return result;
        };
        const onAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function () {
            const result = onAdded?.apply(this, arguments);
            // Wait until configure() has restored the saved size and combo value.
            queueMicrotask(() => {
                this.setSize([this.size[0], this.computeSize()[1]]);
                refreshWorkflows(this).catch((error) => notify(error.message));
            });
            return result;
        };
    },
});
