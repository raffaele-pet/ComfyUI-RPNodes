import { app } from "/scripts/app.js";
import {
  populateLegacyFramePrompts,
  restoreWidgetValues,
} from "./rp_h3_widget_migration.mjs";

const TARGET = "RPH3I2VPromptWriter";
const FRAME_INPUT_PATTERN = /^(?:frames\.)?frame_([1-9])$/;
const FRAME_PROMPT_PATTERN = /^prompt_([1-9])$/;

function frameSlot(input) {
  const name = input?.label || input?.name || "";
  return FRAME_INPUT_PATTERN.exec(name)?.[1] || "";
}

function rememberWidgetLayout(widget) {
  if (widget._rpH3FramePromptLayout) return;
  widget._rpH3FramePromptLayout = {
    type: widget.type,
    computeSize: widget.computeSize,
    hidden: widget.hidden,
  };
}

function setWidgetVisible(widget, visible) {
  if (!widget) return;
  rememberWidgetLayout(widget);
  const original = widget._rpH3FramePromptLayout;
  if (visible) {
    widget.type = original.type;
    widget.computeSize = original.computeSize;
    widget.hidden = original.hidden;
  } else {
    widget.type = "converted-widget:rp-h3-frame-prompt";
    widget.computeSize = () => [0, -4];
    widget.hidden = true;
  }
}

function refreshFramePrompts(node) {
  const connected = new Set(
    (node.inputs || [])
      .filter((input) => input.link != null)
      .map(frameSlot)
      .filter(Boolean),
  );

  for (const widget of node.widgets || []) {
    const match = FRAME_PROMPT_PATTERN.exec(widget?.name || "");
    if (match) setWidgetVisible(widget, connected.has(match[1]));
  }

  const size = node.computeSize?.();
  if (Array.isArray(size)) {
    node.setSize?.([Math.max(node.size?.[0] || 0, size[0]), size[1]]);
  }
  node.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "RP.H3PromptWriter.FramePrompts",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TARGET) return;

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated?.apply(this, arguments);
      queueMicrotask(() => refreshFramePrompts(this));
      return result;
    };

    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (config) {
      const result = originalConfigure?.apply(this, arguments);
      const legacyPrompt = restoreWidgetValues(this, config);
      queueMicrotask(() => {
        populateLegacyFramePrompts(this, legacyPrompt);
        refreshFramePrompts(this);
      });
      return result;
    };

    const originalConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const result = originalConnectionsChange?.apply(this, arguments);
      queueMicrotask(() => refreshFramePrompts(this));
      return result;
    };
  },
});
