import { app } from "/scripts/app.js";

const NODES = new Set(["RPGroupsMuter", "RPBypasser", "RPImportWorkflow", "RPFixWorkflow"]);

app.registerExtension({
    name: "RPNodes.WorkflowUtilities.Layout",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODES.has(nodeData.name)) return;
        if (nodeData.name === "RPBypasser") {
            nodeType.slot_start_y = LiteGraph.NODE_SLOT_HEIGHT;
            return;
        }
        // Native first-slot center (0.7 rows), half-slot bounds, then widget inset.
        const widgetStart = LiteGraph.NODE_SLOT_HEIGHT * (0.7 + 0.5) + 2;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            this.widgets_start_y = widgetStart;
            return onNodeCreated?.apply(this, arguments);
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            this.widgets_start_y = widgetStart;
            return result;
        };
    },
});
