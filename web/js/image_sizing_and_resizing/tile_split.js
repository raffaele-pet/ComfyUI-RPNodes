import { app } from "/scripts/app.js";

export function migrateTileSplitOutputs(workflow) {
    let changed = 0;
    const links = [...(workflow.links ?? []), ...(workflow.floatingLinks ?? [])];
    for (const node of workflow.nodes ?? []) {
        if (node.type !== "RPTileSplit" || !node.outputs || node.outputs.length < 4) continue;
        let needsUpdate = node.outputs.length !== 4 || node.outputs[3].name !== "layout";
        const moved = [];
        for (const link of links) {
            const source = Array.isArray(link) ? link[1] : link.origin_id;
            if (source !== node.id) continue;
            const slot = Array.isArray(link) ? link[2] : link.origin_slot;
            const type = Array.isArray(link) ? link[5] : link.type;
            // Old layout was slot 4. Old image previews can use the existing tiles list.
            const newSlot = slot === 4 ? 3 : slot === 3 && type === "IMAGE" ? 0 : slot;
            if (newSlot !== slot) {
                if (Array.isArray(link)) link[2] = newSlot;
                else link.origin_slot = newSlot;
                needsUpdate = true;
                moved.push({ id: Array.isArray(link) ? link[0] : link.id, slot: newSlot });
            }
        }
        if (!needsUpdate) continue;
        const layout = node.outputs.find((output, index) => index >= 3 && output.name === "layout") ?? {};
        const outputs = node.outputs.slice(0, 3).map((output) => ({ ...output }));
        outputs.push({ ...layout, name: "layout", type: "STRING", links: [] });
        if ("slot_index" in outputs[3]) outputs[3].slot_index = 3;
        for (const link of links) {
            const source = Array.isArray(link) ? link[1] : link.origin_id;
            const slot = Array.isArray(link) ? link[2] : link.origin_slot;
            if (source === node.id && slot === 3) outputs[3].links.push(Array.isArray(link) ? link[0] : link.id);
        }
        for (const { id, slot } of moved) {
            if (slot !== 0) continue;
            const ids = outputs[0].links ?? [];
            outputs[0].links = ids.includes(id) ? ids : [...ids, id];
        }
        node.outputs = outputs;
        changed++;
    }
    for (const graph of workflow.definitions?.subgraphs ?? []) changed += migrateTileSplitOutputs(graph);
    return changed;
}

app.registerExtension({
    name: "RPNodes.TileSplit.OutputMigration",
    beforeConfigureGraph(workflow) {
        migrateTileSplitOutputs(workflow);
    },
});
