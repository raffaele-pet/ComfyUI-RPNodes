import { app } from "/scripts/app.js";

const TARGETS = new Set(["RPH3T2VPromptWriter", "RPH3REF2VPromptWriter"]);
const BASE_OUTPUT_COUNT = 3;
const REFERENCE_PATTERN = /^ref_(?:image_[0-8]|video_[0-2]|video_audio_[0-2]|audio_[0-2])$/;
const GROUP_ORDER = new Map([
  ["image", 0],
  ["video", 1],
  ["video_audio", 2],
  ["audio", 3],
]);

function socketName(input) {
  const name = input?.label || input?.name?.split(".").at(-1) || "";
  return REFERENCE_PATTERN.test(name) ? name : "";
}

function socketRank(name) {
  const match = /^ref_(image|video|video_audio|audio)_(\d+)$/.exec(name);
  return match ? GROUP_ORDER.get(match[1]) * 10 + Number(match[2]) : -1;
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

function canonicalInputs(inputs) {
  return (inputs || [])
    .map((input, index) => ({ input, index, name: socketName(input) }))
    .sort((left, right) => {
      if (!left.name && !right.name) return left.index - right.index;
      if (!left.name) return -1;
      if (!right.name) return 1;
      return socketRank(left.name) - socketRank(right.name) || left.index - right.index;
    })
    .map(({ input }) => input);
}

function canonicalizeConfigInputs(config) {
  upgradeLegacyInputs(config);
  if (Array.isArray(config?.inputs)) {
    config.inputs = canonicalInputs(config.inputs);
  }
}

function graphLink(graph, linkId) {
  return graph?.links?.[linkId] ?? graph?.links?.get?.(linkId);
}

function legacyReferenceLinkIds(config) {
  return (config?.outputs || [])
    .slice(BASE_OUTPUT_COUNT)
    .filter((output) => REFERENCE_PATTERN.test(output?.name || ""))
    .flatMap((output) => output.links || []);
}

function stripLegacyReferenceOutputs(config) {
  if (Array.isArray(config?.outputs)) {
    config.outputs = config.outputs.slice(0, BASE_OUTPUT_COUNT);
  }
}

function removeGraphLink(graph, linkId) {
  const link = graphLink(graph, linkId);
  if (!link) return;
  if (typeof graph.removeLink === "function") {
    graph.removeLink(linkId);
    return;
  }
  const target = graph.getNodeById?.(link.target_id);
  const input = target?.inputs?.[link.target_slot];
  if (input?.link === linkId) input.link = null;
  if (typeof graph.links?.delete === "function") graph.links.delete(linkId);
  else if (graph.links) delete graph.links[linkId];
}

function stripNodeReferenceOutputs(node, legacyLinkIds) {
  const linkIds = new Set(legacyLinkIds);
  for (const output of (node.outputs || []).slice(BASE_OUTPUT_COUNT)) {
    for (const linkId of output?.links || []) linkIds.add(linkId);
  }
  for (const linkId of linkIds) removeGraphLink(node.graph, linkId);
  node.outputs = (node.outputs || []).slice(0, BASE_OUTPUT_COUNT);
}

function canonicalizeNodeInputs(node) {
  node.inputs = canonicalInputs(node.inputs);
  node.inputs.forEach((input, index) => {
    if (input.link == null) return;
    const link = graphLink(node.graph, input.link);
    if (link) link.target_slot = index;
  });
  node.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "RP.H3PromptWriter.ReferenceInputs",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!TARGETS.has(nodeData.name)) return;

    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (config) {
      const legacyLinkIds = legacyReferenceLinkIds(config);
      canonicalizeConfigInputs(config);
      stripLegacyReferenceOutputs(config);
      const result = originalConfigure?.apply(this, arguments);
      queueMicrotask(() => {
        stripNodeReferenceOutputs(this, legacyLinkIds);
        canonicalizeNodeInputs(this);
      });
      return result;
    };

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated?.apply(this, arguments);
      queueMicrotask(() => canonicalizeNodeInputs(this));
      return result;
    };

    const originalConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const result = originalConnectionsChange?.apply(this, arguments);
      queueMicrotask(() => canonicalizeNodeInputs(this));
      return result;
    };
  },
});
