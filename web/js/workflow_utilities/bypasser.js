import { app } from "/scripts/app.js";
import { getConnectedTargets, setTargetsEnabled } from "./bypasser_logic.js";

const bypassers = new Set();
let timer;
let loading = false;

function refreshAll() {
    if (!loading) for (const node of bypassers) refresh(node);
}

function refresh(node) {
    if (!node.graph || loading || node.rpRefreshing) return;
    node.rpRefreshing = true;
    try {
        let changed = false;
        if (!node.inputs.length || node.inputs.at(-1).link != null) {
            node.addInput("", "*");
            changed = true;
        }
        for (let i = node.inputs.length - 2; i >= 0; i--) {
            if (node.inputs[i].link == null) {
                node.removeInput(i);
                changed = true;
            }
        }
        const rows = getConnectedTargets(node);
        const button = node.widgets[0];
        const buttonName = rows.length && rows.every((row) => row.node.mode === 4) ? "Enable all" : "Bypass all";
        const buttonChanged = button.name !== buttonName;
        button.name = buttonName;
        const previous = node.rpBypassRows ?? [];
        const state = rows.map(({ index, node: target }) => ({ index, id: target.id, enabled: target.mode === 0 }));
        const stateChanged = JSON.stringify(state) !== JSON.stringify(previous);
        node.rpBypassRows = state;
        for (const [index, input] of node.inputs.entries()) {
            const target = rows.find((row) => row.index === index)?.node;
            const label = target?.title ?? "";
            if (input.name !== label) { input.name = label;changed = true; }
        }
        if (changed) {
            const size = node.computeSize();
            node.setSize([Math.max(node.size[0], 360, size[0] + 70), size[1]]);
        }
        if (changed || stateChanged || buttonChanged) node.setDirtyCanvas(true, true);
    } finally {
        node.rpRefreshing = false;
    }
}

function rowY(node, index) {
    return node.getConnectionPos(true, index)[1] - node.pos[1];
}

app.registerExtension({
    name: "RPNodes.Bypasser",
    beforeLoadGraph() { loading = true; },
    afterConfigureGraph() { loading = false;refreshAll(); },
    onGraphLoadError() { loading = false;refreshAll(); },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "RPBypasser") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            this.isVirtualNode = true;
            this.serialize_widgets = false;
            this.addInput("", "*");
            this.addWidget("button", "Bypass all", null, () => {
                const targets = getConnectedTargets(this).map((row) => row.node);
                setTargetsEnabled(this, targets, targets.length > 0 && targets.every((node) => node.mode === 4));
                refreshAll();
            }, { serialize: false }).serialize = false;
            this.color = "#223";this.bgcolor = "#335";
            this.size[0] = Math.max(this.size[0], 360);
            return result;
        };
        const onAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function () {
            const result = onAdded?.apply(this, arguments);
            bypassers.add(this);
            timer ??= setInterval(refreshAll, 250);
            queueMicrotask(refreshAll);
            return result;
        };
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            bypassers.delete(this);
            if (!bypassers.size) { clearInterval(timer);timer = undefined; }
            return onRemoved?.apply(this, arguments);
        };
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = onConnectionsChange?.apply(this, arguments);
            queueMicrotask(refreshAll);
            return result;
        };
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const result = onDrawForeground?.apply(this, arguments);
            if (this.flags.collapsed) return result;
            ctx.save();
            ctx.font = `${LiteGraph.NODE_TEXT_SIZE}px Arial`;
            ctx.textBaseline = "middle";
            for (const { index, node } of getConnectedTargets(this)) {
                const y = rowY(this, index), x = this.size[0] - 85, enabled = node.mode === 0;
                ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
                ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
                ctx.beginPath();ctx.roundRect(x, y - 10, 70, 20, 10);ctx.fill();ctx.stroke();
                ctx.fillStyle = enabled ? LiteGraph.WIDGET_TEXT_COLOR : LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
                ctx.textAlign = "right";ctx.fillText(enabled ? "yes" : "no", x + 43, y);
                ctx.fillStyle = enabled ? "#89a" : "#333";
                ctx.beginPath();ctx.arc(x + 56, y, 7, 0, Math.PI * 2);ctx.fill();
            }
            ctx.restore();
            return result;
        };
        const onMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (event, pos) {
            if (event.button === 0 && pos[0] >= this.size[0] - 85 && pos[0] <= this.size[0] - 15) {
                for (const { index, node } of getConnectedTargets(this)) {
                    if (Math.abs(pos[1] - rowY(this, index)) <= 10) {
                        setTargetsEnabled(this, [node], node.mode !== 0);
                        refreshAll();
                        return true;
                    }
                }
            }
            return onMouseDown?.apply(this, arguments);
        };
        const onConnectInput = nodeType.prototype.onConnectInput;
        nodeType.prototype.onConnectInput = function (index, type, slot, node) {
            if (node === this) return false;
            return onConnectInput?.apply(this, arguments) ?? true;
        };
    },
});
