import { app } from "/scripts/app.js";
import { fixWorkflow } from "./merge_workflow.js";
import { loadMergedWorkflow } from "./load_workflow.js";

let fixing = false;

function notify(message, severity = "success") {
    app.extensionManager.toast.add({ severity, summary: "RP Fix Workflow", detail: message, life: 7000 });
}

async function fixCurrentWorkflow() {
    if (fixing) return;
    fixing = true;
    try {
        const before = JSON.parse(JSON.stringify(app.rootGraph.serialize()));
        const result = fixWorkflow(before);
        const changes = [
            [result.links, "link"],
            [result.references, "reference"],
            [result.reroutes, "reroute"],
        ].filter(([count]) => count > 0);
        if (!changes.length) {
            notify("Workflow is already clean.");
            return;
        }
        await loadMergedWorkflow(app, result.workflow, before);
        notify(`Fixed ${changes.map(([count, name]) => `${count} ${name}${count === 1 ? "" : "s"}`).join(", ")}.`);
    } catch (error) {
        notify(error.message ?? String(error), "error");
    } finally {
        fixing = false;
    }
}

app.registerExtension({
    name: "RPNodes.FixWorkflow",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "RPFixWorkflow") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            this.isVirtualNode = true;
            this.addWidget("button", "Fix workflow", null, () => {
                void fixCurrentWorkflow();
            }, { serialize: false }).serialize = false;
            return result;
        };
    },
});
