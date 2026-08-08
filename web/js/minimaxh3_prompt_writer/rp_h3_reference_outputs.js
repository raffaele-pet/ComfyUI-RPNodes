import { app } from "/scripts/app.js";

const TARGETS = new Set(["RPH3T2VPromptWriter", "RPH3REF2VPromptWriter"]);
const REF2V_TARGET = "RPH3REF2VPromptWriter";
const BASE_OUTPUT_COUNT = 3;
const REFERENCE_PATTERN = /^ref_(?:image_[0-8]|video_[0-2]|video_audio_[0-2]|audio_[0-2])$/;
const GROUP_ORDER = new Map([
  ["image", 0],
  ["video", 1],
  ["video_audio", 2],
  ["audio", 3],
]);

function socketParts(name) {
  const match = /^ref_(image|video|video_audio|audio)_(\d+)$/.exec(name);
  return match ? [GROUP_ORDER.get(match[1]), Number(match[2])] : [99, 99];
}

function socketType(name) {
  return name.startsWith("ref_audio_") || name.startsWith("ref_video_audio_")
    ? "AUDIO"
    : "IMAGE";
}

function socketName(input) {
  const name = input?.label || input?.name?.split(".").at(-1) || "";
  return REFERENCE_PATTERN.test(name) ? name : "";
}

function autogrowGroup(name) {
  if (name.startsWith("ref_image_")) return "ref_images";
  if (name.startsWith("ref_video_audio_")) return "ref_video_audios";
  if (name.startsWith("ref_video_")) return "ref_videos";
  return "ref_audios";
}

function upgradeLegacyInputs(config) {
  for (const input of config?.inputs || []) {
    if (!REFERENCE_PATTERN.test(input.name || "")) continue;
    const legacyName = input.name;
    const qualifiedName = `${autogrowGroup(legacyName)}.${legacyName}`;
    input.name = qualifiedName;
    input.localized_name = qualifiedName;
    input.label = legacyName;
  }
}

function connectedReferenceNames(node) {
  return (node.inputs || [])
    .map((input) => ({ name: socketName(input), link: input.link }))
    .filter((input) => input.name && input.link != null)
    .map((input) => input.name)
    .sort((left, right) => {
      const [leftGroup, leftIndex] = socketParts(left);
      const [rightGroup, rightIndex] = socketParts(right);
      return leftGroup - rightGroup || leftIndex - rightIndex;
    });
}

function graphLink(graph, linkId) {
  return graph?.links?.[linkId] ?? graph?.links?.get?.(linkId);
}

function syncReferenceOutputs(node) {
  const desiredNames = connectedReferenceNames(node);
  const desiredSet = new Set(desiredNames);
  const baseOutputs = (node.outputs || []).slice(0, BASE_OUTPUT_COUNT);
  const existing = new Map(
    (node.outputs || [])
      .slice(BASE_OUTPUT_COUNT)
      .filter((output) => REFERENCE_PATTERN.test(output.name))
      .map((output) => [output.name, output]),
  );

  for (let index = (node.outputs || []).length - 1; index >= BASE_OUTPUT_COUNT; index--) {
    const output = node.outputs[index];
    if (output?.links?.length && !desiredSet.has(output.name)) {
      node.disconnectOutput(index);
    }
  }

  const referenceOutputs = desiredNames.map((name) =>
    existing.get(name) || { name, type: socketType(name), links: null },
  );
  node.outputs = [...baseOutputs, ...referenceOutputs];

  referenceOutputs.forEach((output, offset) => {
    output.name = desiredNames[offset];
    output.type = socketType(output.name);
    for (const linkId of output.links || []) {
      const link = graphLink(node.graph, linkId);
      if (link) link.origin_slot = BASE_OUTPUT_COUNT + offset;
    }
  });
  node.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "RP.H3PromptWriter.ReferenceOutputs",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!TARGETS.has(nodeData.name)) return;

    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (config) {
      upgradeLegacyInputs(config);
      const result = originalConfigure?.apply(this, arguments);
      if (nodeData.name === REF2V_TARGET) {
        queueMicrotask(() => syncReferenceOutputs(this));
      }
      return result;
    };

    if (nodeData.name !== REF2V_TARGET) return;

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated?.apply(this, arguments);
      syncReferenceOutputs(this);
      return result;
    };

    const originalConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const result = originalConnectionsChange?.apply(this, arguments);
      queueMicrotask(() => syncReferenceOutputs(this));
      return result;
    };
  },
});
