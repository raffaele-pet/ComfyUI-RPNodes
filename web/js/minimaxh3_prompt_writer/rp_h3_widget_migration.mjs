const PARAMETER_NAMES = [
  "max_token_length",
  "media_analysis_tokens",
  "sampling",
  "temperature",
  "top_k",
  "top_p",
  "min_p",
  "repetition_penalty",
  "seed",
  "control_after_generate",
  "strict_validation",
];

const PROMPT_NAMES = Array.from({ length: 9 }, (_, index) =>
  `prompt_${index + 1}`,
);

// RP H3-I2V before per-frame prompts: one global prompt before parameters.
export const LEGACY_WIDGET_NAMES = [
  "skill",
  "duration_seconds",
  "prompt",
  ...PARAMETER_NAMES,
  ...PROMPT_NAMES,
];

// Short-lived broken layout: parameters first, per-frame prompts appended.
export const PARAMETERS_FIRST_WIDGET_NAMES = [
  "skill",
  "duration_seconds",
  ...PARAMETER_NAMES,
  ...PROMPT_NAMES,
];

// Previous stable layout: prompt_1...prompt_9 replaced the old prompt.
export const FRAME_PROMPTS_FIRST_WIDGET_NAMES = [
  "skill",
  "duration_seconds",
  ...PROMPT_NAMES,
  ...PARAMETER_NAMES,
];

// Current layout: one sequence-wide prompt plus the nine frame-local prompts.
export const GLOBAL_FRAME_PROMPTS_FIRST_WIDGET_NAMES = [
  "skill",
  "duration_seconds",
  "global_prompt",
  ...PROMPT_NAMES,
  ...PARAMETER_NAMES,
];

export function splitNumberedPrompts(text) {
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
  const contiguous = slots.every((slot, index) => slot === index + 1);
  return contiguous ? prompts : new Map();
}

export function detectWidgetLayout(values) {
  if (!Array.isArray(values)) return null;
  if (
    typeof values[2] === "string" &&
    typeof values[3] === "string" &&
    typeof values[12] === "number" &&
    typeof values[14] === "boolean"
  ) {
    return GLOBAL_FRAME_PROMPTS_FIRST_WIDGET_NAMES;
  }
  if (
    typeof values[2] === "string" &&
    typeof values[3] === "string" &&
    typeof values[11] === "number" &&
    typeof values[13] === "boolean"
  ) {
    return FRAME_PROMPTS_FIRST_WIDGET_NAMES;
  }
  if (
    typeof values[2] === "string" &&
    typeof values[3] === "number" &&
    typeof values[5] === "boolean"
  ) {
    return LEGACY_WIDGET_NAMES;
  }
  if (
    typeof values[2] === "number" &&
    typeof values[3] === "number" &&
    typeof values[4] === "boolean"
  ) {
    return PARAMETERS_FIRST_WIDGET_NAMES;
  }
  return null;
}

export function restoreWidgetValues(node, config) {
  const values = config?.widgets_values;
  const names = detectWidgetLayout(values);
  if (!names) return "";

  const widgets = new Map(
    (node.widgets || []).map((widget) => [widget?.name, widget]),
  );
  for (const name of ["global_prompt", ...PROMPT_NAMES]) {
    const widget = widgets.get(name);
    if (widget) widget.value = "";
  }
  names.forEach((name, index) => {
    const widget = widgets.get(name);
    if (widget && index < values.length) widget.value = values[index];
  });

  return names === LEGACY_WIDGET_NAMES ? String(values[2] || "").trim() : "";
}

export function populateLegacyFramePrompts(node, legacyPrompt) {
  if (!legacyPrompt) return;
  const promptWidgets = (node.widgets || []).filter((widget) =>
    /^prompt_[1-9]$/.test(widget?.name || ""),
  );
  if (promptWidgets.some((widget) => String(widget.value || "").trim())) return;

  const prompts = splitNumberedPrompts(legacyPrompt);
  if (!prompts.size) prompts.set(1, legacyPrompt);
  for (const widget of promptWidgets) {
    const slot = Number(widget.name.slice("prompt_".length));
    if (prompts.has(slot)) widget.value = prompts.get(slot);
  }
}
