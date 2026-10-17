import { app } from "/scripts/app.js";
import { getMatchingGroups, isGroupEnabled, setGroupEnabled } from "./groups_muter_logic.js";

const muters = new Set();
let timer;

function refreshAll() {
    for (const node of muters) refreshGroups(node);
}

function navigateToGroup(group) {
    const canvas = app.canvas;
    if (canvas.graph !== group.graph) return;
    canvas.centerOnNode(group);
    const rect = canvas.canvas.getBoundingClientRect();
    canvas.setZoom(Math.min(canvas.ds.scale, rect.width / (group.size[0] + 80),
        rect.height / (group.size[1] + 80)), [rect.width / 2, rect.height / 2]);
    canvas.setDirty(true, true);
}

function createGroupWidget(node, group) {
    return node.addCustomWidget({
        type: "rp_group_toggle",
        name: `Enable ${group.title}`,
        group,
        value: isGroupEnabled(group),
        serialize: false,
        options: { serialize: false },
        computeSize() { return [300, LiteGraph.NODE_WIDGET_HEIGHT]; },
        draw(ctx, node, width, y, height) {
            this.value = isGroupEnabled(this.group);
            this.last_y = y;
            const right = width - 15;
            const navWidth = node.properties.showNav ? 30 : 0;
            const circleX = right - navWidth - 16;
            ctx.save();
            ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
            ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
            ctx.beginPath();
            ctx.roundRect(15, y, width - 30, height, height / 2);
            ctx.fill();ctx.stroke();
            ctx.fillStyle = this.value ? "#89a" : "#333";
            ctx.beginPath();ctx.arc(circleX, y + height / 2, height * 0.36, 0, Math.PI * 2);ctx.fill();
            ctx.fillStyle = this.value ? LiteGraph.WIDGET_TEXT_COLOR : LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
            ctx.font = `${LiteGraph.NODE_TEXT_SIZE}px Arial`;
            ctx.textAlign = "right";ctx.textBaseline = "middle";
            ctx.fillText(this.value ? "yes" : "no", circleX - 14, y + height / 2);
            ctx.textAlign = "left";
            ctx.save();ctx.beginPath();ctx.rect(24, y, Math.max(0, circleX - 80), height);ctx.clip();
            ctx.fillText(this.name, 24, y + height / 2);ctx.restore();
            if (navWidth) {
                ctx.beginPath();ctx.moveTo(right - navWidth, y);ctx.lineTo(right - navWidth, y + height);ctx.stroke();
                ctx.fillStyle = "#89a";
                const cx = right - 15, cy = y + height / 2;
                ctx.beginPath();ctx.moveTo(cx - 7, cy - 3);ctx.lineTo(cx, cy - 3);
                ctx.lineTo(cx, cy - 6);ctx.lineTo(cx + 7, cy);ctx.lineTo(cx, cy + 6);
                ctx.lineTo(cx, cy + 3);ctx.lineTo(cx - 7, cy + 3);ctx.closePath();ctx.fill();
            }
            ctx.restore();
        },
        mouse(event, pos, node) {
            if (event.type !== "pointerdown") return true;
            if (node.properties.showNav && pos[0] >= node.size[0] - 45) {
                navigateToGroup(this.group);
            } else {
                setGroupEnabled(this.group, !isGroupEnabled(this.group));
                refreshAll();
            }
            return true;
        },
    });
}

function refreshGroups(node) {
    if (!node.graph) return;
    const groups = getMatchingGroups(node.graph, node.properties, LGraphCanvas.node_colors);
    const previous = [...(node.widgets ?? [])];
    const groupWidgets = groups.map((group) => previous.find((widget) => widget.group === group)
        ?? createGroupWidget(node, group));
    const toggle = node.hideNestedGroupsWidget;
    const hideNestedGroups = node.properties.hideNestedGroups !== false;
    let changed = toggle.value !== hideNestedGroups;
    toggle.value = hideNestedGroups;
    const widgets = [toggle, ...groupWidgets];
    for (const widget of groupWidgets) {
        const name = `Enable ${widget.group.title}`;
        const enabled = isGroupEnabled(widget.group);
        changed ||= widget.name !== name || widget.value !== enabled;
        widget.name = name;widget.value = enabled;
    }
    const structureChanged = previous.length !== widgets.length || widgets.some((widget, index) => previous[index] !== widget);
    changed ||= structureChanged;
    for (const widget of previous) if (!widgets.includes(widget)) node.removeWidget(widget);
    if (structureChanged) {
        node.widgets = widgets;
        const size = node.computeSize();
        node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    }
    if (changed) node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "RPNodes.GroupsMuter",
    afterConfigureGraph() { refreshAll(); },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "RPGroupsMuter") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            this.isVirtualNode = true;
            this.serialize_widgets = false;
            this.addProperty("hideNestedGroups", true, "boolean");
            this.hideNestedGroupsWidget = this.addWidget("toggle", "Hide nested groups", true,
                (value) => this.setProperty("hideNestedGroups", value),
                { on: "yes", off: "no", serialize: false });
            this.hideNestedGroupsWidget.serialize = false;
            this.addProperty("matchTitle", "", "string");
            this.addProperty("matchColors", "", "string");
            this.addProperty("showNav", true, "boolean");
            this.addProperty("sort", "position", "enum", { values: ["position", "alphanumeric"] });
            return result;
        };
        const onAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function () {
            const result = onAdded?.apply(this, arguments);
            muters.add(this);
            timer ??= setInterval(refreshAll, 250);
            refreshGroups(this);
            return result;
        };
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            muters.delete(this);
            if (!muters.size) { clearInterval(timer);timer = undefined; }
            return onRemoved?.apply(this, arguments);
        };
        const onPropertyChanged = nodeType.prototype.onPropertyChanged;
        nodeType.prototype.onPropertyChanged = function () {
            const result = onPropertyChanged?.apply(this, arguments);
            refreshGroups(this);
            this.setDirtyCanvas(true, true);
            return result;
        };
    },
});
