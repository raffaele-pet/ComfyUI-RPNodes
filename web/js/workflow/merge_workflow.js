// Adapted from raffaele-pet/comfyui-workflow-merger for an existing canvas.
const clone = (value) => JSON.parse(JSON.stringify(value));
const key = (id) => String(id);
const linkId = (link) => Array.isArray(link) ? link[0] : link.id;
const reroutes = (workflow) => workflow.reroutes ?? workflow.extra?.reroutes ?? [];
const isV1 = (workflow) => Number(workflow.version) >= 1;

function objectLink(link) {
    if (!Array.isArray(link)) return { ...link };
    return {
        id: link[0], origin_id: link[1], origin_slot: link[2],
        target_id: link[3], target_slot: link[4], type: link[5],
    };
}

export function validateWorkflow(workflow, name = "Workflow", subgraph = false) {
    const fail = (message) => { throw new Error(`${name}: ${message}`); };
    if (!workflow || !Array.isArray(workflow.nodes) || !Array.isArray(workflow.links)) {
        fail("choose a workflow saved as JSON (nodes/links required), not an API prompt.");
    }
    const nodeIds = new Set();
    for (const node of workflow.nodes) {
        if (node?.id == null || typeof node.type !== "string") fail("invalid node.");
        if (nodeIds.has(key(node.id))) fail(`duplicate node ID ${node.id}.`);
        nodeIds.add(key(node.id));
    }
    if (subgraph) {
        nodeIds.add(key(workflow.inputNode?.id ?? -10));
        nodeIds.add(key(workflow.outputNode?.id ?? -20));
    }
    const linkIds = new Set();
    const floatingIds = new Set((workflow.floatingLinks ?? []).map((link) => key(linkId(link))));
    for (const source of [...workflow.links, ...(workflow.floatingLinks ?? [])]) {
        if (!source || (Array.isArray(source) && source.length < 6)) fail("invalid link.");
        const link = objectLink(source);
        if (link.id == null || linkIds.has(key(link.id))) fail("missing or duplicate link ID.");
        linkIds.add(key(link.id));
        for (const endpoint of [link.origin_id, link.target_id]) {
            if ((endpoint == null || endpoint === -1) && !floatingIds.has(key(link.id))) {
                fail(`link ${link.id} has a missing endpoint.`);
            }
            if (endpoint != null && endpoint !== -1 && !nodeIds.has(key(endpoint))) {
                fail(`link ${link.id} refers to missing node ${endpoint}.`);
            }
        }
    }
    const rerouteIds = new Set();
    for (const reroute of reroutes(workflow)) {
        if (reroute.id == null || rerouteIds.has(key(reroute.id))) fail("missing or duplicate reroute ID.");
        rerouteIds.add(key(reroute.id));
        for (const id of reroute.linkIds ?? []) {
            if (!linkIds.has(key(id))) fail(`reroute ${reroute.id} refers to missing link ${id}.`);
        }
    }
    for (const reroute of reroutes(workflow)) {
        if (reroute.parentId != null && !rerouteIds.has(key(reroute.parentId))) {
            fail(`reroute ${reroute.id} refers to missing parent ${reroute.parentId}.`);
        }
    }
    for (const link of [...workflow.links, ...(workflow.floatingLinks ?? []), ...(workflow.extra?.linkExtensions ?? [])]) {
        if (link.parentId != null && !rerouteIds.has(key(link.parentId))) {
            fail(`link ${linkId(link)} refers to missing reroute ${link.parentId}.`);
        }
    }
    for (const node of workflow.nodes) {
        for (const input of node.inputs ?? []) {
            if (input.link != null && !linkIds.has(key(input.link))) {
                fail(`node ${node.id} refers to missing link ${input.link}.`);
            }
        }
        for (const output of node.outputs ?? []) {
            for (const id of output.links ?? []) {
                if (!linkIds.has(key(id))) fail(`node ${node.id} refers to missing link ${id}.`);
            }
        }
    }
    for (const definition of workflow.definitions?.subgraphs ?? []) {
        validateWorkflow(definition, `${name} / ${definition.name ?? definition.id}`, true);
    }
}

function point(value, fallback = [0, 0]) {
    return [Number(value?.[0]) || fallback[0], Number(value?.[1]) || fallback[1]];
}

function bounds(workflow) {
    const rectangles = workflow.nodes.map((node) => [
        ...point(node.pos), ...point(node.size, [220, 120]),
    ]);
    rectangles.push(...(workflow.groups ?? []).map((group) => group.bounding));
    rectangles.push(...reroutes(workflow).map((reroute) => [...point(reroute.pos), 0, 0]));
    if (!rectangles.length) return { x: 0, y: 0, width: 0, height: 0 };
    const x = Math.min(...rectangles.map((rect) => rect[0]));
    const y = Math.min(...rectangles.map((rect) => rect[1]));
    return {
        x, y,
        width: Math.max(...rectangles.map((rect) => rect[0] + rect[2])) - x,
        height: Math.max(...rectangles.map((rect) => rect[1] + rect[3])) - y,
    };
}

function idCounters(workflow) {
    const counters = { node: 0, link: 0, group: 0, reroute: 0 };
    const reserve = (kind, id) => {
        const number = Number(id);
        if (Number.isSafeInteger(number)) counters[kind] = Math.max(counters[kind], number);
    };
    const visit = (graph) => {
        for (const node of graph.nodes ?? []) reserve("node", node.id);
        for (const link of [...(graph.links ?? []), ...(graph.floatingLinks ?? [])]) reserve("link", linkId(link));
        for (const group of graph.groups ?? []) reserve("group", group.id);
        for (const reroute of reroutes(graph)) reserve("reroute", reroute.id);
        reserve("node", graph.last_node_id ?? graph.state?.lastNodeId);
        reserve("link", graph.last_link_id ?? graph.state?.lastLinkId);
        reserve("group", graph.state?.lastGroupId);
        reserve("reroute", graph.state?.lastRerouteId);
        for (const definition of graph.definitions?.subgraphs ?? []) visit(definition);
    };
    visit(workflow);
    return counters;
}

// Change only definition IDs and node types. Widget strings may contain the same UUID.
function remapSubgraphTypes(graph, ids) {
    for (const node of graph.nodes ?? []) {
        if (ids.has(node.type)) node.type = ids.get(node.type);
    }
    for (const definition of graph.definitions?.subgraphs ?? []) {
        definition.id = ids.get(definition.id) ?? definition.id;
        remapSubgraphTypes(definition, ids);
    }
}

function importDefinitions(out, source) {
    const definitions = source.definitions?.subgraphs ?? [];
    if (!definitions.length) return;
    const ids = new Map();
    const collect = (items) => {
        for (const definition of items) {
            if (!ids.has(definition.id)) ids.set(definition.id, crypto.randomUUID());
            collect(definition.definitions?.subgraphs ?? []);
        }
    };
    collect(definitions);
    remapSubgraphTypes(source, ids);
    out.definitions ??= {};
    out.definitions.subgraphs ??= [];
    out.definitions.subgraphs.push(...definitions);
}

export function appendWorkflows(current, items, { placement = "right", gap = 160, groupImport = true } = {}) {
    const out = clone(current);
    out.extra ??= {};
    out.groups ??= [];
    const counters = idCounters(out);
    const importedNodeIds = [];
    const margin = groupImport ? 40 : 0;
    gap = Math.max(0, Number(gap) || 0);

    for (const { name, data } of items) {
        validateWorkflow(data, name);
        if (!data.nodes.length) throw new Error(`${name}: workflow has no nodes to import.`);
        const source = clone(data);
        const existingBounds = bounds(out);
        const sourceBounds = bounds(source);
        const dx = existingBounds.x - sourceBounds.x + margin
            + (placement === "right" ? existingBounds.width + gap : 0);
        const dy = existingBounds.y - sourceBounds.y + margin
            + (placement === "below" ? existingBounds.height + gap : 0);
        importDefinitions(out, source);

        // Subgraph internals are normalized by ComfyUI when configuring the graph.
        // Reserve their IDs here so newly allocated root IDs cannot collide with them.
        const reserved = idCounters(out);
        for (const kind of Object.keys(counters)) counters[kind] = Math.max(counters[kind], reserved[kind]);
        const nodeIds = new Map(source.nodes.map((node) => [key(node.id), ++counters.node]));
        const linkIds = new Map([...source.links, ...(source.floatingLinks ?? [])]
            .map((link) => [key(linkId(link)), ++counters.link]));
        const rerouteIds = new Map(reroutes(source).map((reroute) => [key(reroute.id), ++counters.reroute]));

        for (const node of source.nodes) {
            node.id = nodeIds.get(key(node.id));
            importedNodeIds.push(node.id);
            const [x, y] = point(node.pos);
            node.pos = [x + dx, y + dy];
            for (const input of node.inputs ?? []) {
                if (input.link != null) input.link = linkIds.get(key(input.link));
            }
            for (const output of node.outputs ?? []) {
                if (Array.isArray(output.links)) output.links = output.links.map((id) => linkIds.get(key(id)));
            }
            out.nodes.push(node);
        }

        const extensions = new Map((source.extra?.linkExtensions ?? []).map((extension) => [key(extension.id), extension]));
        const mapLink = (sourceLink) => {
            const link = objectLink(sourceLink);
            const extension = extensions.get(key(link.id));
            if (extension) Object.assign(link, extension);
            link.id = linkIds.get(key(link.id));
            link.origin_id = nodeIds.get(key(link.origin_id)) ?? link.origin_id;
            link.target_id = nodeIds.get(key(link.target_id)) ?? link.target_id;
            if (link.parentId != null) link.parentId = rerouteIds.get(key(link.parentId));
            return link;
        };
        for (const sourceLink of source.links) {
            const link = mapLink(sourceLink);
            if (isV1(out)) out.links.push(link);
            else {
                out.links.push([link.id, link.origin_id, link.origin_slot, link.target_id, link.target_slot, link.type]);
                if (link.parentId != null) {
                    out.extra.linkExtensions ??= [];
                    out.extra.linkExtensions.push({ id: link.id, parentId: link.parentId });
                }
            }
        }
        if (source.floatingLinks?.length) {
            out.floatingLinks ??= [];
            out.floatingLinks.push(...source.floatingLinks.map(mapLink));
        }
        for (const reroute of reroutes(source)) {
            reroute.id = rerouteIds.get(key(reroute.id));
            if (reroute.parentId != null) reroute.parentId = rerouteIds.get(key(reroute.parentId));
            if (reroute.linkIds) reroute.linkIds = reroute.linkIds.map((id) => linkIds.get(key(id)));
            const [x, y] = point(reroute.pos);
            reroute.pos = [x + dx, y + dy];
            const owner = isV1(out) ? out : out.extra;
            owner.reroutes ??= [];
            owner.reroutes.push(reroute);
        }
        for (const group of source.groups ?? []) {
            group.id = ++counters.group;
            group.bounding[0] += dx;
            group.bounding[1] += dy;
            out.groups.push(group);
        }
        if (groupImport) {
            out.groups.push({
                id: ++counters.group, title: name.replace(/\.json$/i, ""),
                bounding: [sourceBounds.x + dx - margin, sourceBounds.y + dy - margin,
                    sourceBounds.width + margin * 2, sourceBounds.height + margin * 2],
                color: "#3f789e", flags: {},
            });
        }
        if (source.models?.length) {
            out.models ??= [];
            const seen = new Set(out.models.map((model) => JSON.stringify(model)));
            for (const model of source.models) {
                if (!seen.has(JSON.stringify(model))) {
                    out.models.push(model);
                    seen.add(JSON.stringify(model));
                }
            }
        }
        if (source.extra?.node_versions) {
            out.extra.node_versions = { ...source.extra.node_versions, ...out.extra.node_versions };
        }
    }
    if (isV1(out)) {
        out.state = { ...out.state, lastNodeId: counters.node, lastLinkId: counters.link,
            lastGroupId: counters.group, lastRerouteId: counters.reroute };
    } else {
        out.last_node_id = counters.node;
        out.last_link_id = counters.link;
    }
    return { workflow: out, importedNodeIds };
}
