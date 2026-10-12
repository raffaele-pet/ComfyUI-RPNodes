export function getConnectedTargets(controller) {
    const graph = controller.graph;
    if (!graph) return [];
    const rows = [];
    for (const [index, input] of controller.inputs.entries()) {
        let link = graph.links.get(input.link);
        let node = link && graph.getNodeById(link.origin_id);
        const visited = new Set();
        while (node?.type === "Reroute" && !visited.has(node)) {
            visited.add(node);
            link = graph.links.get(node.inputs[0]?.link);
            node = link && graph.getNodeById(link.origin_id);
        }
        if (node && node !== controller && node.type !== "Reroute") rows.push({ index, node });
    }
    return rows;
}

export function setTargetsEnabled(controller, targets, enabled) {
    const mode = enabled ? 0 : 4;
    const nodes = [...new Set(targets)].filter((node) => node.graph === controller.graph && node.mode !== mode);
    if (!nodes.length) return;
    controller.graph.beforeChange();
    try {
        for (const node of nodes) node.mode = mode;
    } finally {
        controller.graph.afterChange();
        controller.graph.setDirtyCanvas(true, true);
    }
}
