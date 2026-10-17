export function getGroupNodes(group) {
    const [x, y] = group.pos;
    const [width, height] = group.size;
    return (group.graph?.nodes ?? []).filter((node) => {
        const centerX = node.pos[0] + node.size[0] / 2;
        const centerY = node.pos[1] + node.size[1] / 2;
        return centerX >= x && centerX < x + width && centerY >= y && centerY < y + height;
    });
}

export function isGroupEnabled(group) {
    return getGroupNodes(group).some((node) => node.mode === 0);
}

export function setGroupEnabled(group, enabled) {
    const mode = enabled ? 0 : 2;
    const nodes = getGroupNodes(group).filter((node) => node.mode !== mode);
    if (!nodes.length) return;
    group.graph.beforeChange();
    try {
        for (const node of nodes) node.mode = mode;
    } finally {
        group.graph.afterChange();
        group.graph.setDirtyCanvas(true, true);
    }
}

function normalizeColor(color, palette) {
    color = String(color ?? "").trim().toLowerCase();
    color = palette[color]?.groupcolor ?? color;
    if (/^#?[\da-f]{3}$/i.test(color)) {
        color = color.replace("#", "").replace(/./g, (digit) => digit + digit);
    }
    return color.replace(/^#/, "").toLowerCase();
}

export function getMatchingGroups(graph, properties, palette = {}) {
    let titlePattern;
    const matchTitle = String(properties.matchTitle ?? "").trim();
    if (matchTitle) {
        try { titlePattern = new RegExp(matchTitle, "i"); } catch { /* Use literal text below. */ }
    }
    const colors = String(properties.matchColors ?? "").split(",").filter((color) => color.trim())
        .map((color) => normalizeColor(color, palette));
    const allGroups = graph?.groups ?? [];
    const groups = allGroups.filter((group) => {
        if (properties.hideNestedGroups !== false && allGroups.some((parent) =>
            parent !== group
            && parent.pos[0] <= group.pos[0] && parent.pos[1] <= group.pos[1]
            && parent.pos[0] + parent.size[0] >= group.pos[0] + group.size[0]
            && parent.pos[1] + parent.size[1] >= group.pos[1] + group.size[1]
            && (parent.size[0] > group.size[0] || parent.size[1] > group.size[1]))) return false;
        if (matchTitle && !(titlePattern ? titlePattern.test(group.title)
            : group.title.toLowerCase().includes(matchTitle.toLowerCase()))) return false;
        return !colors.length || colors.includes(normalizeColor(group.color, palette));
    });
    return groups.sort(properties.sort === "alphanumeric"
        ? (a, b) => a.title.localeCompare(b.title, undefined, { numeric: true })
        : (a, b) => Math.floor(a.pos[1] / 30) - Math.floor(b.pos[1] / 30) || a.pos[0] - b.pos[0]);
}
