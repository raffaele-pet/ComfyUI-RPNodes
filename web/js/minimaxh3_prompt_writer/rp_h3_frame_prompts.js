import { app } from "/scripts/app.js";

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

function hideLegacyPrompt(widget) {
  if (!widget) return;
  rememberWidgetLayout(widget);
  const original = widget._rpH3FramePromptLayout;
  // Keep the original widget type so ComfyUI continues serializing its value at
  // the historical index while the control remains completely invisible.
  widget.type = original.type;
  widget.computeSize = () => [0, -4];
  widget.hidden = true;
}

function splitNumberedPrompts(text) {
  const source = String(text || "");
  const markers = [...source.matchAll(/^\s*([1-9])\s*[.)]\s*/gm)];
  if (!markers.length) return new Map();

  const prompts = new Map();
  for (let index = 0; index < markers.length; index += 1) {
    const marker = markers[index];
    const next = markers[index + 1];
    const slot = Number(marker[1]);
    const value = source
      .slice(marker.index + marker[0].length, next?.index ?? source.length)
      .trim();
    if (value) prompts.set(slot, value);
  }

  const slots = [...prompts.keys()];
  const complete = slots.every((slot, index) => slot === index + 1);
  return complete ? prompts : new Map();
}

function populateFramePromptsFromLegacy(node) {
  const legacy = (node.widgets || []).find((widget) => widget?.name === "prompt");
  const promptWidgets = (node.widgets || []).filter((widget) =>
    FRAME_PROMPT_PATTERN.test(widget?.name || ""),
  );
  if (!legacy || !promptWidgets.length) return;
  if (promptWidgets.some((widget) => String(widget.value || "").trim())) return;

  const prompts = splitNumberedPrompts(legacy.value);
  for (const widget of promptWidgets) {
    const slot = Number(FRAME_PROMPT_PATTERN.exec(widget.name)?.[1]);
    if (prompts.has(slot)) widget.value = prompts.get(slot);
  }
}

function refreshFramePrompts(node) {
  populateFramePromptsFromLegacy(node);
  const connected = new Set(
    (node.inputs || [])
      .filter((input) => input.link != null)
      .map(frameSlot)
      .filter(Boolean),
  );

  for (const widget of node.widgets || []) {
    if (widget?.name === "prompt") {
      hideLegacyPrompt(widget);
      continue;
    }
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
    nodeType.prototype.onConfigure = function () {
      const result = originalConfigure?.apply(this, arguments);
      queueMicrotask(() => refreshFramePrompts(this));
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
