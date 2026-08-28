import assert from "node:assert/strict";
import test from "node:test";

import {
  FRAME_PROMPTS_FIRST_WIDGET_NAMES,
  PARAMETERS_FIRST_WIDGET_NAMES,
  populateLegacyFramePrompts,
  restoreWidgetValues,
} from "../web/js/minimaxh3_prompt_writer/rp_h3_widget_migration.mjs";

const parameters = [2048, 256, false, 0.7, 64, 0.95, 0.05, 1.05, 42, "fixed", true];

function makeNode() {
  return {
    widgets: FRAME_PROMPTS_FIRST_WIDGET_NAMES.map((name) => ({
      name,
      value: `wrong:${name}`,
    })),
  };
}

function value(node, name) {
  return node.widgets.find((widget) => widget.name === name)?.value;
}

function assertParameters(node) {
  const names = PARAMETERS_FIRST_WIDGET_NAMES.slice(2, 13);
  assert.deepEqual(names.map((name) => value(node, name)), parameters);
}

test("restores original global-prompt workflows by widget name", () => {
  const node = makeNode();
  const legacy = "1. Opening pose.\n2. The subject waves.\n3. The camera moves closer.";
  const legacyPrompt = restoreWidgetValues(node, {
    widgets_values: ["Auto", 15, legacy, ...parameters],
  });
  populateLegacyFramePrompts(node, legacyPrompt);

  assertParameters(node);
  assert.equal(value(node, "prompt_1"), "Opening pose.");
  assert.equal(value(node, "prompt_2"), "The subject waves.");
  assert.equal(value(node, "prompt_3"), "The camera moves closer.");
  assert.equal(value(node, "prompt_4"), "");
});

test("restores the short-lived parameters-first layout", () => {
  const node = makeNode();
  const prompts = ["A", "B", "C", "D", "E", "F", "G", "H", "I"];
  const legacyPrompt = restoreWidgetValues(node, {
    widgets_values: ["Auto", 15, ...parameters, ...prompts],
  });

  assert.equal(legacyPrompt, "");
  assertParameters(node);
  assert.deepEqual(
    prompts.map((_, index) => value(node, `prompt_${index + 1}`)),
    prompts,
  );
});

test("restores parameters when the broken layout saved no frame prompts", () => {
  const node = makeNode();
  restoreWidgetValues(node, {
    widgets_values: ["Auto", 15, ...parameters],
  });

  assertParameters(node);
  assert.deepEqual(
    Array.from({ length: 9 }, (_, index) => value(node, `prompt_${index + 1}`)),
    Array(9).fill(""),
  );
});

test("round-trips the stable prompts-first layout", () => {
  const node = makeNode();
  const prompts = ["A", "B", "C", "D", "E", "F", "G", "H", "I"];
  restoreWidgetValues(node, {
    widgets_values: ["Auto", 15, ...prompts, ...parameters],
  });

  assertParameters(node);
  assert.deepEqual(
    prompts.map((_, index) => value(node, `prompt_${index + 1}`)),
    prompts,
  );
});
