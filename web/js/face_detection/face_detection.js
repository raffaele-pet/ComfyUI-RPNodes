import { app } from "/scripts/app.js";

const ADVANCED = new Set([
    "threshold", "dilation", "crop_factor", "drop_size",
    "crop_size_mult", "bbox_smooth_alpha",
]);

export function configureAdvancedWidgets(node) {
    const toggle = node.widgets?.find((widget) => widget.name === "show_advanced");
    if (!toggle) return;
    const originals = new Map();
    for (const widget of node.widgets) {
        if (ADVANCED.has(widget.name)) {
            originals.set(widget, { type: widget.type, computeSize: widget.computeSize });
        }
    }
    function update() {
        for (const [widget, original] of originals) {
            widget.type = toggle.value ? original.type : "converted-widget";
            widget.hidden = !toggle.value;
            if (toggle.value) {
                if (original.computeSize) widget.computeSize = original.computeSize;
                else delete widget.computeSize;
            } else {
                widget.computeSize = () => [0, -4];
            }
        }
        const size = node.computeSize();
        node.setSize([Math.max(node.size[0], size[0]), size[1]]);
        node.setDirtyCanvas(true, true);
    }
    const callback = toggle.callback;
    toggle.callback = function (...args) {
        callback?.apply(this, args);
        update();
    };
    node.rpFaceAdvancedUpdate = update;
    update();
}

app.registerExtension({
    name: "RPNodes.FaceDetection.AdvancedOptions",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "RPFaceDetection") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = onNodeCreated?.apply(this, args);
            configureAdvancedWidgets(this);
            return result;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = onConfigure?.apply(this, args);
            this.rpFaceAdvancedUpdate?.();
            return result;
        };
    },
});
