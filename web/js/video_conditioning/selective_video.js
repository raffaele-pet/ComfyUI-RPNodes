import { app } from "/scripts/app.js";
import { createGuideEditor, guideModes, guideStyle } from "./selective_video_guides.js";
import {
    attachNativePreview, colors, createPainter, decodeMask, defaultExpansion, loadGroupOverlays,
    checkerboard, loadPhoto, loadTrackedMasks, post, previewJob, releaseOverlays, request, sourceVideoURL, videoURL,
} from "./selective_video_preview.js";
const guideMaps = { segmentation: "Segmentation", depth: "Depth", canny: "Canny" };
const loadOptionDefaults = {
    force_rate: 0,
    custom_width: 0,
    custom_height: 0,
    frame_load_cap: 0,
    skip_first_frames: 0,
    select_every_nth: 1,
    format: "None",
};

const selectiveVideoOutputs = Object.entries({
    video: "VIDEO", frames: "IMAGE", original_frames: "IMAGE", frame_count: "INT", audio: "AUDIO",
    video_info: "VHS_VIDEOINFO", MASK_IMAGE: "IMAGE", mask_video: "VIDEO", MASK: "MASK",
});
const selectiveGuidesOutputs = [
    selectiveVideoOutputs[0], ["frames_map_guide", "IMAGE"], ["frames_transparent_guide", "IMAGE"],
    ...selectiveVideoOutputs.slice(2),
];

function migrateSelectiveVideoOutputs(workflow) {
    const remaps = new Map();
    const guideVideoSlots = new Map();
    const nodes = new Map((workflow.nodes ?? []).map((node) => [node.id, node]));
    for (const node of nodes.values()) {
        if (node.type === "RPSelectiveVideoInput") {
            node.type = "RPSelectiveVideoGuides";
            if (node.properties?.["Node name for S&R"] === "RPSelectiveVideoInput") {
                node.properties["Node name for S&R"] = node.type;
            }
            if (node.title === "RP Selective Video Input") node.title = "RP Selective Video Guides";
        }
        const guides = node.type === "RPSelectiveVideoGuides";
        if (!guides && node.type !== "RPSelectiveVideoEdgeFill") continue;
        const schema = guides ? selectiveGuidesOutputs : selectiveVideoOutputs;
        const canonical = schema.map(([name]) => name);
        const outputs = node.outputs;
        if (!outputs?.length ||
            (outputs.length === canonical.length && outputs.every(({ name }, index) => name === canonical[index]))) continue;

        const aliases = guides ? {
            video_mask: "mask_video", frames: "frames_map_guide", guide_video: "frames_transparent_guide",
        } : { video_mask: "mask_video" };
        const slots = outputs.map(({ name }) => canonical.indexOf(aliases[name] ?? name));
        if (slots.includes(-1) || new Set(slots).size !== slots.length) continue;

        const removedVideo = outputs.findIndex(({ name }) => name === "guide_video");
        if (guides && removedVideo >= 0) guideVideoSlots.set(node.id, removedVideo);
        const saved = new Map(slots.map((slot, index) => [slot, outputs[index]]));
        node.outputs = schema.map(([name, type], slot) => {
            const old = saved.get(slot);
            if (!old) return { name, type, links: null };
            if ("slot_index" in old) old.slot_index = slot;
            if (old.name !== name && "localized_name" in old) old.localized_name = name;
            old.name = name;
            old.type = type;
            return old;
        });
        remaps.set(node.id, slots);
    }
    if (remaps.size) {
        const links = [...(workflow.links ?? []), ...(workflow.floatingLinks ?? [])];
        const removed = new Set();
        let lastId = links.reduce((max, link) => Math.max(max, (Array.isArray(link) ? link[0] : link.id) ?? 0),
            Math.max(workflow.last_link_id ?? 0, workflow.state?.lastLinkId ?? 0));
        for (const link of links) {
            const array = Array.isArray(link);
            const [id, sourceId, oldSlot, targetId, targetSlot] = array ? link
                : [link.id, link.origin_id, link.origin_slot, link.target_id, link.target_slot];
            const slot = remaps.get(sourceId)?.[oldSlot] ?? oldSlot;
            if (guideVideoSlots.has(sourceId) && guideVideoSlots.get(sourceId) === oldSlot) {
                const source = nodes.get(sourceId);
                const target = nodes.get(targetId);
                const input = target?.inputs?.[targetSlot];
                const framesInput = target?.inputs?.find(({ name }) => name === "frames");
                if (input?.link === id) input.link = null;
                if (target?.type !== "RPSaveTransparentVideo" || input?.name !== "video" || framesInput?.link != null) {
                    source.outputs[slot].links = (source.outputs[slot].links ?? []).filter((value) => value !== id);
                    removed.add(link);
                    continue;
                }
                const [frames, framesSlot] = ensureInput(target, "frames", "IMAGE");
                frames.link = id;
                if (array) {
                    link[4] = framesSlot;
                    link[5] = "IMAGE";
                } else {
                    link.target_slot = framesSlot;
                    link.type = "IMAGE";
                }
                const [info, infoSlot] = ensureInput(target, "video_info", "VHS_VIDEOINFO");
                if (info.link == null) {
                    const rateSlot = selectiveGuidesOutputs.findIndex(([name]) => name === "video_info");
                    info.link = ++lastId;
                    (source.outputs[rateSlot].links ??= []).push(lastId);
                    (workflow.links ??= []).push(array
                        ? [lastId, sourceId, rateSlot, targetId, infoSlot, "VHS_VIDEOINFO"]
                        : { id: lastId, origin_id: sourceId, origin_slot: rateSlot, target_id: targetId,
                            target_slot: infoSlot, type: "VHS_VIDEOINFO" });
                    if (workflow.state) workflow.state.lastLinkId = lastId;
                    if (array || "last_link_id" in workflow) workflow.last_link_id = lastId;
                }
            }
            if (array) link[2] = slot;
            else link.origin_slot = slot;
        }
        if (removed.size) {
            for (const key of ["links", "floatingLinks"]) {
                if (workflow[key]) workflow[key] = workflow[key].filter((link) => !removed.has(link));
            }
        }
    }
    for (const graph of workflow.definitions?.subgraphs ?? []) migrateSelectiveVideoOutputs(graph);
}

function ensureInput(node, name, type) {
    const inputs = (node.inputs ??= []);
    const existing = inputs.findIndex((input) => input.name === name);
    if (existing >= 0) return [inputs[existing], existing];
    const port = { name, type, link: null };
    inputs.push(port);
    return [port, inputs.length - 1];
}

function loadOptionValue(value, fallback) {
    if (typeof value === "string") {
        value = value.trim();
        if (value && typeof fallback === "number") value = Number(value);
    }
    if (typeof fallback === "number") return Number.isFinite(value) ? value : fallback;
    return typeof value === "string" && value ? value : fallback;
}

function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text) node.textContent = text;
    if (className) node.className = className;
    return node;
}

function button(title, action, className) {
    const control = element("button", title, className);
    control.type = "button";
    control.addEventListener("click", action);
    return control;
}

function input(type, value, label) {
    const control = element("input");
    control.type = type;
    control.value = value;
    control.setAttribute("aria-label", label);
    return control;
}

function row(...contents) {
    const node = element("div", "", "rp-row");
    node.append(...contents);
    return node;
}

function label(text, control) {
    const node = element("label", text);
    node.append(control);
    return node;
}

function swatch(hex) {
    const dot = element("span", "", "rp-dot");
    dot.style.backgroundColor = hex;
    return dot;
}

const selectionColors = {
    Red: "#ef4444", Orange: "#f97316", Yellow: "#eab308", Green: "#22c55e",
    Cyan: "#06b6d4", Blue: "#1678ff", Purple: "#a855f7", Pink: "#ec4899",
    White: "#ffffff", Black: "#111111",
};

function colorPicker(onColor, palette) {
    const sample = swatch("");
    const hex = input("text", "", "Hex color");
    hex.maxLength = 7;
    const channels = ["Red", "Green", "Blue"].map((channel) => {
        const slider = input("range", 0, channel);
        slider.min = 0;
        slider.max = 255;
        return slider;
    });
    const panel = element("div", "", "rp-color-panel");
    panel.hidden = true;
    const toggle = button("", () => {
        if (!panel.hidden) {
            close();
            return;
        }
        panel.hidden = false;
        toggle.setAttribute("aria-expanded", "true");
        if (palette) paletteRow.firstElementChild.focus();
        else {
            hex.focus();
            hex.select();
        }
    });
    toggle.append(sample);
    toggle.setAttribute("aria-label", palette ? "Selection color" : "Group color");
    toggle.setAttribute("aria-expanded", "false");

    const closeButton = button("×", () => close(true));
    closeButton.setAttribute("aria-label", "Close color picker");
    const paletteRow = element("div", "", "rp-color-swatches");
    for (const [name, color] of Object.entries(palette ?? {})) {
        const chip = button("", () => {
            setColor(color);
            onColor(color);
            close(true);
        });
        chip.setAttribute("aria-label", name);
        chip.title = name;
        chip.style.backgroundColor = color;
        paletteRow.append(chip);
    }
    if (palette) panel.append(paletteRow);
    panel.append(row(hex, closeButton), ...channels.map((slider) => label(slider.ariaLabel, slider)));
    panel.addEventListener("keydown", (event) => {
        if (event.key !== "Escape") return;
        event.preventDefault();
        event.stopPropagation();
        close(true);
    });
    const wrapper = element("div", "", "rp-color-picker");
    wrapper.append(toggle, panel);

    function close(restoreFocus = false) {
        panel.hidden = true;
        toggle.setAttribute("aria-expanded", "false");
        if (restoreFocus) toggle.focus();
    }

    function setColor(color) {
        sample.style.backgroundColor = color;
        hex.value = color;
        channels.forEach((slider, index) => { slider.value = parseInt(color.slice(1 + 2 * index, 3 + 2 * index), 16); });
    }

    function channelColor() {
        return `#${channels.map((slider) => Number(slider.value).toString(16).padStart(2, "0")).join("")}`;
    }

    hex.addEventListener("input", () => {
        const value = hex.value.trim();
        if (!/^#?[\da-f]{6}$/i.test(value)) return;
        const color = `#${value.replace(/^#/, "").toLowerCase()}`;
        setColor(color);
        onColor(color);
    });
    hex.addEventListener("blur", () => { hex.value = channelColor(); });
    for (const slider of channels) {
        slider.addEventListener("input", () => {
            const color = channelColor();
            setColor(color);
            onColor(color);
        });
    }
    return { wrapper, setColor, close };
}

function readConfig(widget, video) {
    try {
        const data = JSON.parse(widget.value);
        if (data.video !== video) return { groups: [] };
        if (Array.isArray(data.groups)) return data;
        const groups = (data.selections ?? data.layers ?? []).map((layer) => {
            if (Array.isArray(layer.parts)) return layer;
            const {
                text, preprocessor, color, expand = defaultExpansion, fill_holes = true, guide_scale, guide_offset,
                guide_rotation, segmentation_opacity, ...selection
            } = layer;
            return {
                name: text, preprocessor, color, expand, fill_holes, guide_scale, guide_offset, guide_rotation,
                segmentation_opacity, parts: [{ text, ...selection }],
            };
        });
        return { ...data, groups };
    } catch {
        return { groups: [] };
    }
}

function guideSelections(groups) {
    return groups.filter(({ parts }) => parts.length);
}

function preparationSignature(groups, settings) {
    const selections = guideSelections(withoutPreviews(groups)).map((entry) => Object.fromEntries(
        Object.entries(entry).sort(([a], [b]) => a.localeCompare(b)),
    ));
    return JSON.stringify([settings, selections]);
}

function canShowPrepared(groups, settings, saved, activeGroup) {
    const current = preparationSignature(groups, settings);
    if (saved === current) return true;
    if (!Object.hasOwn(guideModes, activeGroup?.preprocessor)) return false;
    const index = groups.filter(({ parts }) => parts.length).indexOf(activeGroup);
    if (index < 0) return false;
    // The editor draws the active pose with the current transform.
    const withoutPlacement = (key) => {
        const data = JSON.parse(key);
        for (const field of ["guide_scale", "guide_offset", "guide_rotation"]) {
            delete data[1][index][field];
        }
        return JSON.stringify(data);
    };
    try {
        return withoutPlacement(saved) === withoutPlacement(current);
    } catch {
        return false;
    }
}

function withoutPreviews(groups) {
    return groups.filter((group) => group.parts.length).map((group) => ({
        ...group, parts: group.parts.map(({ preview, ...part }) => part),
    }));
}

function signature(groups) {
    return JSON.stringify(withoutPreviews(groups).map((group) => group.parts));
}

function mergedPreview(items, selected) {
    const masks = items.filter((_, index) => selected == null || selected.includes(index)).map(({ mask }) => mask);
    if (!masks.length) return null;
    const canvas = element("canvas");
    const { width, height } = masks[0];
    const scale = Math.min(1, 700 / Math.max(width, height));
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const context = canvas.getContext("2d");
    for (const mask of masks) context.drawImage(mask, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/png").split(",")[1];
}

const editorStyle = `
    .rp-selective-editor[open] { box-sizing:border-box; width:98vw; max-width:none;
        height:96dvh; max-height:none; display:flex; flex-direction:column; gap:12px;
        overflow:hidden; color:var(--fg-color,#ebedf1); background:var(--comfy-menu-bg,#202125);
        border:1px solid var(--border-color,#545761); border-radius:12px; padding:16px;
        font:14px/1.4 system-ui,sans-serif; }
    .rp-selective-editor::backdrop { background:#000b; }
    .rp-selective-editor * { box-sizing:border-box; }
    .rp-selective-editor h2,.rp-selective-editor h3,.rp-selective-editor p { margin:0; }
    .rp-selective-editor h2 { font-size:19px; }
    .rp-selective-editor h3 { font-size:15px; }
    .rp-selective-editor header,.rp-selective-editor footer,.rp-selective-editor .rp-row {
        display:flex; align-items:center; gap:8px; min-width:0; }
    .rp-selective-editor header,.rp-selective-editor footer,.rp-selective-editor .rp-between { justify-content:space-between; }
    .rp-selective-editor .rp-layout { display:grid; grid-template-columns:minmax(0,1fr) minmax(300px,380px);
        gap:14px; flex:1; min-height:0; }
    .rp-selective-editor .rp-view { display:flex; flex-direction:column; gap:8px; min-width:0; min-height:0; }
    .rp-selective-editor .rp-stage { position:relative; flex:1; min-height:120px; overflow:hidden;
        background:#101115; border-radius:8px; }
    .rp-selective-editor .rp-stage canvas { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; display:block; }
    .rp-selective-editor .rp-stage video { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; }
    .rp-selective-editor .rp-stage video[hidden] { display:none; }
    .rp-selective-editor .rp-stage.rp-playing .rp-guide-layer,
    .rp-selective-editor .rp-stage.rp-playing .rp-guide-controls { visibility:hidden; }
    .rp-selective-editor .rp-timeline .rp-play { min-width:34px; display:grid; place-items:center; }
    .rp-selective-editor .rp-play svg { width:14px; height:14px; fill:currentColor; }
    .rp-selective-editor .rp-prepare-spinner:not([hidden]) { position:absolute; inset:0; z-index:2;
        display:grid; place-items:center; background:#0006; }
    .rp-selective-editor .rp-prepare-spinner::after { content:""; width:44px; height:44px; border:4px solid #fff5;
        border-top-color:#fff; border-radius:50%; animation:rp-prepare-spin .75s linear infinite; }
    @keyframes rp-prepare-spin { to { transform:rotate(360deg); } }
    .rp-selective-editor .rp-preview-toolbar { display:grid; gap:6px; padding:6px 10px;
        border:1px solid var(--border-color,#545761); border-radius:8px; }
    .rp-selective-editor .rp-preview-toolbar .rp-preview-top { display:grid; grid-template-columns:max-content minmax(0,1fr);
        height:34px; }
    .rp-selective-editor .rp-preview-toolbar .rp-preview-options { column-gap:12px; }
    .rp-selective-editor .rp-preview-options input { margin:0; accent-color:#6792eb; }
    .rp-selective-editor .rp-preview-toolbar .rp-anchors { height:34px; min-height:0; min-width:0;
        gap:6px; padding:4px 6px 0 9px; overflow-x:auto; overflow-y:hidden; scrollbar-width:none;
        border-inline-start:1px solid var(--border-color,#545761); }
    .rp-selective-editor .rp-preview-toolbar .rp-anchors:empty { visibility:hidden; }
    .rp-selective-editor .rp-preview-toolbar .rp-preview-actions { padding-top:6px; justify-content:space-between;
        border-top:1px solid var(--border-color,#545761); }
    .rp-selective-editor .rp-preview-actions .rp-status { min-width:0; overflow:hidden;
        white-space:nowrap; text-overflow:ellipsis; text-align:right; }
    .rp-selective-editor .rp-timeline input[type=range] { flex:1; min-width:48px; }
    .rp-selective-editor .rp-frame { width:70px; }
    .rp-selective-editor .rp-time { min-width:64px; text-align:right; font-variant-numeric:tabular-nums; }
    .rp-selective-editor aside { display:flex; flex-direction:column; min-width:0; min-height:0; overflow:auto; gap:10px; padding-right:4px; }
    .rp-selective-editor .rp-flex { flex:1; min-width:0; }
    .rp-selective-editor button,.rp-selective-editor select,.rp-selective-editor input:not([type=range]):not([type=checkbox]) {
        color:inherit; border:1px solid var(--border-color,#545761); background:var(--comfy-input-bg,#191b20);
        border-radius:6px; padding:6px 9px; font:inherit; }
    .rp-selective-editor button { cursor:pointer; white-space:nowrap; }
    .rp-selective-editor button:disabled { opacity:.45; cursor:default; }
    .rp-selective-editor button.rp-primary { color:#fff; background:#315fbc; border-color:#638ae0; }
    .rp-selective-editor button[aria-pressed=true] { border-color:#83a8ff; }
    .rp-selective-editor button:focus-visible,.rp-selective-editor input:focus-visible,.rp-selective-editor select:focus-visible {
        outline:2px solid #83a8ff; outline-offset:1px; }
    .rp-selective-editor .rp-list { display:grid; gap:4px; }
    .rp-selective-editor .rp-group-list { gap:9px; }
    .rp-selective-editor .rp-group-card { display:grid; gap:10px; min-width:0; padding:9px;
        border:1px solid var(--rp-group-color); border-radius:9px; }
    .rp-selective-editor .rp-group-content { display:grid; gap:8px; min-width:0; }
    .rp-selective-editor .rp-check-result { line-height:1.35; font-size:13px; overflow-wrap:anywhere; }
    .rp-selective-editor .rp-check-result:empty,.rp-selective-editor .rp-status:empty { display:none; }
    .rp-selective-editor .rp-group-title { font-weight:650; }
    .rp-selective-editor .rp-part-list:empty,.rp-selective-editor .rp-candidates:empty { display:none; }
    .rp-selective-editor .rp-part-label { border-color:transparent; background:transparent; }
    .rp-selective-editor .rp-part-label:hover { background:var(--comfy-input-bg,#191b20); }
    .rp-selective-editor .rp-part-actions { gap:5px; }
    .rp-selective-editor .rp-part-actions button { padding:3px 7px; }
    .rp-selective-editor .rp-selection-divider { width:100%; margin:2px 0 0; border:0;
        border-top:1px solid var(--border-color,#545761); }
    .rp-selective-editor .rp-truncate { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; text-align:left; min-width:0; }
    .rp-selective-editor .rp-status { min-height:20px; opacity:.86; }
    .rp-selective-editor .rp-dot { display:inline-block; height:10px; width:10px; flex:0 0 10px; border-radius:50%; }
    .rp-selective-editor .rp-anchors { display:flex; align-items:center; gap:7px; min-height:37px; overflow:auto; padding:4px 3px; }
    .rp-selective-editor .rp-anchor { display:inline-flex; position:relative; flex:0 0 auto; }
    .rp-selective-editor .rp-anchor .rp-anchor-remove { position:absolute; right:-4px; top:-5px; width:16px;
        height:16px; padding:0; border-radius:50%; display:grid; place-items:center; }
    .rp-selective-editor .rp-anchor-remove::before,.rp-selective-editor .rp-anchor-remove::after {
        content:""; grid-area:1/1; width:7px; height:1px; background:currentColor; transform:rotate(45deg); }
    .rp-selective-editor .rp-anchor-remove::after { transform:rotate(-45deg); }
    .rp-selective-editor .rp-anchors button { padding:3px 8px; }
    .rp-selective-guides .rp-anchor { margin-inline-end:3px; }
    .rp-selective-guides .rp-anchor button:first-child { min-width:26px; height:22px; padding:0 5px;
        font-size:12px; line-height:1; }
    .rp-selective-guides .rp-anchor .rp-anchor-remove { width:12px; height:12px; top:-4px; right:-4px; }
    .rp-selective-guides .rp-anchor-remove::before,.rp-selective-guides .rp-anchor-remove::after { width:5px; }
    .rp-selective-guides .rp-anchor button:first-child { border-color:var(--rp-anchor-color); }
    .rp-selective-guides .rp-anchor button[aria-pressed="true"]:first-child {
        background:color-mix(in srgb,var(--rp-anchor-color) 24%,var(--comfy-input-bg,#202125)); }
    .rp-selective-editor .rp-color-picker { position:relative; }
    .rp-selective-editor .rp-color-picker > button { display:grid; place-items:center; width:41px; height:34px; }
    .rp-selective-guides .rp-selection-remove,.rp-selective-guides .rp-color-picker > button {
        display:grid; place-items:center; width:40px; height:34px; padding:0; flex:none; }
    .rp-selective-guides select.rp-guide-placeholder { color:var(--descrip-text,#a4a9b4); }
    .rp-selective-guides select option,.rp-selective-guides select optgroup { color:var(--fg-color,#ebedf1); }
    .rp-selective-editor .rp-color-picker .rp-dot { width:19px; height:15px; border-radius:3px; }
    .rp-selective-editor .rp-color-panel:not([hidden]) { display:grid; gap:7px; position:absolute;
        top:calc(100% + 5px); right:0; z-index:1; width:214px; padding:9px; border-radius:8px;
        border:1px solid var(--border-color,#545761); background:var(--comfy-menu-bg,#202125);
        box-shadow:0 6px 18px #0008; }
    .rp-selective-editor .rp-color-panel .rp-row > input,.rp-selective-editor .rp-color-panel input[type=range] {
        flex:1; min-width:0; }
    .rp-selective-editor .rp-color-panel label { display:grid; grid-template-columns:40px 1fr; }
    .rp-selective-editor .rp-color-panel input[type=range] { width:100%; }
    .rp-selective-editor .rp-color-swatches { display:grid; grid-template-columns:repeat(5,1fr); gap:5px; }
    .rp-selective-editor .rp-color-swatches button { min-height:25px; padding:0; }
    .rp-selective-editor .rp-add-selection { align-self:flex-start; }
    .rp-selective-guides footer { justify-content:flex-end; }
    .rp-selective-editor input[aria-label="Group name"] { text-transform:uppercase; }
    .rp-selective-editor .rp-small-number { width:70px; }
    .rp-selective-editor details .rp-row { margin:5px 0; flex-wrap:wrap; }
    .rp-selective-editor label { display:flex; align-items:center; gap:6px; }
    @media(max-width:760px) {
        .rp-selective-editor[open] { padding:9px; }
        .rp-selective-editor .rp-layout { grid-template-columns:minmax(0,1fr); grid-template-rows:minmax(245px,1fr) minmax(0,1fr); }
        .rp-selective-editor footer > span { display:none; }
        .rp-selective-editor footer { justify-content:flex-end; }
    }
`;

function openEditor(video, groups, session, edgeFill, withGuides, loadingSettings, onSave) {
    const modal = element("dialog", "", "rp-selective-editor");
    if (withGuides) modal.classList.add("rp-selective-guides");
    modal.setAttribute("aria-label", "Selective Video · Selection editor");
    const life = new AbortController();
    const canvas = element("canvas");
    const player = element("video");
    player.hidden = true;
    player.playsInline = true;
    player.loop = true;
    const paint = createPainter(canvas);
    const slider = input("range", 0, "Video frame timeline");
    slider.min = slider.max = 0;
    const frameInput = input("number", 1, "Frame number");
    frameInput.className = "rp-frame";
    frameInput.min = 1;
    const time = element("span", "—", "rp-time");
    const viewStatus = element("p", "Reading video…", "rp-status");
    viewStatus.setAttribute("role", "status");
    const status = element("p", "", "rp-status");
    status.setAttribute("role", "status");

    const name = input("text", "", "Group name");
    name.maxLength = 120;
    name.placeholder = "Group name";
    name.className = "rp-flex";
    name.hidden = true;
    const mode = edgeFill ? input("checkbox", "", "Fill inward from edges") : element("select");
    if (edgeFill) {
        mode.checked = true;
        mode.title = "Copy the nearest background pixels inward. Turn off to leave the selection black.";
        mode.addEventListener("change", () => { activeGroup.fill_from_edges = mode.checked; });
    } else {
        mode.setAttribute("aria-label", "Selection preprocessor");
        mode.append(new Option("Select the guide", "none"));
        for (const [label, options] of [
            ["Map guide", guideMaps],
            ["Transparent guide", guideModes],
        ]) {
            const section = element("optgroup");
            section.label = label;
            for (const [value, title] of Object.entries(options)) section.append(new Option(title, value));
            mode.append(section);
        }
        mode.addEventListener("change", () => {
            if (!activeGroup) return;
            activeGroup.preprocessor = mode.value;
            mode.classList.toggle("rp-guide-placeholder", mode.value === "none");
            showTransparentGuide(mode.value);
            guideEditor.update(activeGroup, currentFrame);
            draw();
        });
    }
    const color = colorPicker((value) => {
        activeGroup.color = value;
        syncGroupCard(activeGroup);
        if (withGuides) renderAnchors();
        draw();
    }, withGuides ? selectionColors : undefined);
    const expansion = input("number", defaultExpansion, "Mask expansion in source pixels");
    expansion.className = "rp-small-number";
    expansion.min = 0;
    expansion.max = 256;
    const fillHoles = input("checkbox", "", "Fill holes");
    fillHoles.checked = true;

    const prompt = input("text", "", "Object description");
    prompt.className = "rp-flex";
    prompt.placeholder = "e.g. person";
    prompt.maxLength = 120;
    const threshold = input("number", 0.5, "Detection confidence");
    threshold.min = 0.01;
    threshold.max = 0.99;
    threshold.step = 0.05;
    threshold.className = "rp-small-number";
    const from = input("number", "", "First frame to track");
    const to = input("number", "", "Last frame to track");
    for (const field of [from, to]) {
        field.min = 1;
        field.className = "rp-small-number";
    }
    from.placeholder = "Start";
    to.placeholder = "End";
    const groupList = element("div", "", "rp-list rp-group-list");
    const partList = element("div", "", "rp-list rp-part-list");
    const candidateList = element("div", "", "rp-list rp-candidates");
    const anchors = element("div", "", "rp-anchors");
    const showSelection = input("checkbox", "", "Selection");
    const showGuide = input("checkbox", "", "Guide");
    showSelection.checked = true;
    const spinner = element("div", "", "rp-prepare-spinner");
    spinner.hidden = true;
    spinner.setAttribute("role", "status");
    spinner.setAttribute("aria-label", "Preparing selections");
    const checkSummary = element("p", "", "rp-check-result");
    checkSummary.setAttribute("role", "status");
    const groupHeaders = new Map();
    const currentParts = new WeakMap();
    const anchorBitmaps = new Map();
    const frameMasks = new Map();
    const restoreFailed = new WeakSet();

    function newSelection() {
        const palette = Object.values(selectionColors).slice(0, -2);
        const unused = palette.filter((color) => !groups.some((selection) => selection.color === color));
        const available = unused.length ? unused : palette.filter((color) => color !== groups.at(-1)?.color);
        const color = available[Math.floor(Math.random() * available.length)];
        return { name: "", color, expand: defaultExpansion, fill_holes: true, preprocessor: "none",
            segmentation_opacity: 0.5, parts: [] };
    }
    if (withGuides && !groups.length) groups.push(newSelection());
    let activeGroup = groups[0] ?? null;
    let currentPart = null;
    let timestamps = [];
    let sourceWidth = 1;
    let currentFrame = 0;
    let shownFrame = -1;
    let photo;
    let candidates = [];
    let checked = [];
    let detecting = false;
    let checkRun;
    let checkResult;
    let restoring = false;
    let tracking = false;
    let frameFetch;
    let preparedFetch;
    let trackFetch;
    let draftVersion = 0;
    let drawVersion = 0;
    let scheduledSeek;
    let playback;
    let ticket = session.signature === signature(groups) ? session.ticket : null;
    let prepared = session.prepared;
    let preparedSignature = session.preparedSignature;
    if (withGuides && session.previewMode === "guide") {
        showSelection.checked = false;
        showGuide.checked = true;
    }

    function getName(group) {
        return (group.name.trim() || group.parts[0]?.text || `Group ${groups.indexOf(group) + 1}`).toUpperCase();
    }

    async function getCandidates(part) {
        const data = await post("/detect", {
            video, frame: part.frame, text: part.text, threshold: part.threshold,
        }, life.signal);
        return Promise.all(data.candidates.map(async ({ score, mask }) => ({ score, mask: await decodeMask(mask) })));
    }

    function storedBitmap(encoded) {
        if (!anchorBitmaps.has(encoded)) {
            anchorBitmaps.set(encoded, decodeMask(encoded).catch(() => null));
        }
        return anchorBitmaps.get(encoded);
    }

    function showTransparentGuide(kind) {
        if (Object.hasOwn(guideModes, kind)) {
            showSelection.checked = false;
            showGuide.checked = true;
        }
    }

    async function draw() {
        if (!photo || shownFrame !== currentFrame || life.signal.aborted) return;
        const frame = shownFrame;
        const version = ++drawVersion;
        preparedFetch?.abort();
        if (withGuides && prepared && canShowPrepared(groups, loadingSettings, preparedSignature, activeGroup)) {
            const controller = new AbortController();
            preparedFetch = controller;
            const query = new URLSearchParams({
                ticket: prepared, frame, active: groups.filter(({ parts }) => parts.length).indexOf(activeGroup),
                segmentation: showSelection.checked ? 1 : 0, guide: showGuide.checked ? 1 : 0,
            });
            try {
                const result = await (await request(`/prepared-frame?${query}`, { signal: controller.signal })).json();
                if (version !== drawVersion || controller.signal.aborted) return;
                if (!result.skipped) {
                    const bitmap = await decodeMask(result.photo);
                    try {
                        if (version !== drawVersion || controller.signal.aborted) return;
                        paint(bitmap, []);
                        guideEditor.showPrepared(activeGroup, [bitmap.width, bitmap.height], result.artwork);
                        return;
                    } finally {
                        bitmap.close();
                    }
                }
            } catch (error) {
                if (controller.signal.aborted || version !== drawVersion) return;
                if (error.status === 404) prepared = session.prepared = null;
                else status.textContent = error.message;
            }
        }
        const scale = photo.width / sourceWidth;
        if (checkResult?.frame === frame) {
            const { group, bitmap } = checkResult;
            const overlay = bitmap ? {
                bitmaps: [bitmap], color: group.color, opacity: 0.57,
                radius: Math.round((group.expand ?? defaultExpansion) * scale), fillHoles: group.fill_holes ?? true,
            } : null;
            paintFrame(overlay ? [overlay] : [], group === activeGroup ? overlay : null);
            return;
        }

        const visibleGroups = groups.filter(({ parts }) => parts.length);
        const tracked = frameMasks.get(frame) ?? [];
        const overlays = await loadGroupOverlays(visibleGroups, frame, tracked,
            visibleGroups.indexOf(activeGroup), scale, storedBitmap);
        try {
            if (life.signal.aborted || version !== drawVersion || shownFrame !== frame || currentFrame !== frame) return;
            const draft = candidates.flatMap(({ mask }, index) => checked.includes(index) ? [] : [{
                bitmaps: [mask], color: colors[index % colors.length], opacity: 0.16,
            }]);
            paintFrame([...overlays, ...draft], overlays[visibleGroups.indexOf(activeGroup)]);
        } finally {
            releaseOverlays(overlays);
        }
    }

    function paintFrame(overlays, activeOverlay) {
        if (withGuides && !showSelection.checked) {
            for (const overlay of overlays) {
                if (!showGuide.checked || overlay.preprocessor !== "segmentation") overlay.opacity = 0;
            }
        }
        if (!guideEditor || !showGuide.checked || !Object.hasOwn(guideModes, activeGroup?.preprocessor)) {
            guideEditor?.update(activeGroup, shownFrame);
            paint(photo, overlays);
            return;
        }
        if (!activeOverlay?.bitmaps.length) {
            guideEditor.update(activeGroup, shownFrame);
            paint(photo, overlays);
            return;
        }
        if (!withGuides) activeOverlay.opacity = 0;
        paint(photo, overlays, (overlay, mask) => {
            if (overlay === activeOverlay) guideEditor.update(activeGroup, shownFrame, mask);
        });
    }

    function setFrameStatus() {
        if (tracking) viewStatus.textContent = withGuides ? "Preparing selections and guides…" : "Tracking groups with SAM 3.1…";
        else if (shownFrame >= 0) {
            const state = checkResult ? "Detection check"
                : ticket ? "Tracked masks" : "Saved masks on anchor frames";
            viewStatus.textContent = `Frame ${shownFrame + 1} of ${timestamps.length} · ${state}`;
        }
    }

    function updateButtons() {
        const ready = shownFrame === currentFrame && shownFrame >= 0;
        const busy = detecting || restoring || tracking || Boolean(checkRun);
        detect.disabled = busy || !!playback || !ready || !activeGroup || !prompt.value.trim();
        add.disabled = busy || (!withGuides && !currentPart);
        track.disabled = !tracking && (!groups.some(({ parts }) => parts.length) || !ready || busy);
        for (const [group, { check }] of groupHeaders) {
            if (check) check.disabled = busy || !ready || !group.parts.length;
        }
        clearCheckButton.hidden = !checkResult && !checkRun;
        track.textContent = tracking ? (withGuides ? "Cancel preparation" : "Cancel tracking") : "Track & preview";
        const paused = withGuides && tracking;
        spinner.hidden = !paused;
        side.inert = paused || !!playback;
        anchors.inert = paused;
        apply.disabled = withGuides && (busy || !ready);
        slider.disabled = frameInput.disabled = paused || !timestamps.length;
        previous.disabled = paused || !timestamps.length || !currentFrame;
        next.disabled = paused || !timestamps.length || currentFrame >= timestamps.length - 1;
        play.disabled = paused || !timestamps.length || detecting || restoring;
    }

    function clearDetection() {
        draftVersion += 1;
        candidates.forEach(({ mask }) => mask.close());
        candidates = [];
        checked = [];
        candidateList.replaceChildren();
        updateButtons();
        draw();
    }

    function clearCheck() {
        checkRun?.abort();
        checkResult?.bitmap?.close();
        checkResult = null;
        checkSummary.textContent = "";
        setFrameStatus();
        updateButtons();
        draw();
    }

    async function checkGroup(group) {
        const frame = shownFrame;
        clearCheck();
        const controller = new AbortController();
        checkRun = controller;
        updateButtons();
        checkSummary.textContent = `Checking frame ${frame + 1} using approved masks…`;
        try {
            const { detection } = await previewJob("/check-group", {
                video, frame, ticket,
                group: { ...withoutPreviews([group])[0], name: getName(group) },
                group_index: groups.filter(({ parts }) => parts.length).indexOf(group),
            }, controller.signal);
            const { mask, results } = detection;
            const bitmap = mask ? await decodeMask(mask) : null;
            if (controller.signal.aborted || frame !== shownFrame) {
                bitmap?.close();
                return;
            }
            checkResult = { bitmap, group, frame };
            const summary = results.map(({ text, matches }) => `${text} (${matches} ${matches === 1 ? "match" : "matches"})`);
            checkSummary.textContent = `Frame ${frame + 1} · ${summary.join(" · ")}`;
            setFrameStatus();
            draw();
        } catch (error) {
            if (!controller.signal.aborted) checkSummary.textContent = error.message;
        } finally {
            if (checkRun === controller) checkRun = null;
            updateButtons();
            restoreAnchors(shownFrame);
        }
    }

    const clearCheckButton = button("Clear check", clearCheck);

    function openSelection(part) {
        clearCheck();
        currentPart = part;
        if (activeGroup) currentParts.set(activeGroup, part);
        prompt.value = part?.text ?? "";
        threshold.value = part?.threshold ?? 0.5;
        from.value = part?.start ? part.start + 1 : "";
        to.value = part?.end == null ? "" : part.end + 1;
        settings.open = false;
        status.textContent = "";
        clearDetection();
    }

    function openGroup(group) {
        if (withGuides && group !== activeGroup) showTransparentGuide(group?.preprocessor);
        activeGroup = group;
        guideEditor?.update(group, currentFrame);
        const part = currentParts.has(group) ? currentParts.get(group) : group?.parts.at(-1);
        openSelection(group?.parts.includes(part) ? part : null);
        name.hidden = true;
        color.close();
        if (group) {
            if (edgeFill) mode.checked = group.fill_from_edges ?? true;
            else {
                mode.value = group.preprocessor ?? "none";
                mode.classList.toggle("rp-guide-placeholder", mode.value === "none");
            }
            color.setColor(group.color ?? colors[0]);
            expansion.value = group.expand ?? defaultExpansion;
            fillHoles.checked = group.fill_holes ?? true;
        }
        renderGroups();
        draw();
    }

    async function restoreAnchors(index) {
        if (restoring || detecting || tracking || checkRun || index !== shownFrame || index < 0) return;
        const missing = groups.flatMap(({ parts }) => parts).filter((part) => (
            part.frame === index && !part.preview && !restoreFailed.has(part)
        ));
        if (!missing.length) return;
        restoring = true;
        updateButtons();
        try {
            for (const part of missing) {
                if (life.signal.aborted || index !== currentFrame) return;
                try {
                    const items = await getCandidates(part);
                    part.preview = mergedPreview(items, part.instances);
                    items.forEach(({ mask }) => mask.close());
                    if (!part.preview) restoreFailed.add(part);
                    draw();
                } catch (error) {
                    if (life.signal.aborted) return;
                    if (error.status !== 409) restoreFailed.add(part);
                    status.textContent = error.message;
                }
            }
        } finally {
            restoring = false;
            updateButtons();
            if (index !== shownFrame) restoreAnchors(shownFrame);
        }
    }

    async function loadFrameMasks(index, signal) {
        if (!ticket || frameMasks.has(index)) return;
        const selectedTicket = ticket;
        try {
            const masks = await loadTrackedMasks(ticket, index, signal);
            if (ticket !== selectedTicket || signal.aborted) return;
            frameMasks.set(index, masks);
            if (frameMasks.size > 48) frameMasks.delete(frameMasks.keys().next().value);
            draw();
        } catch (error) {
            if (signal.aborted || ticket !== selectedTicket) return;
            if (error.status === 404) {
                ticket = null;
                session.ticket = null;
                frameMasks.clear();
            }
            status.textContent = error.message;
            setFrameStatus();
        }
    }

    function updateTimeline(frame) {
        frameInput.value = frame + 1;
        slider.value = frame;
        time.textContent = `${timestamps[frame].toFixed(2)} s`;
    }

    function previewMode() {
        if (!showGuide.checked) return "selection";
        if (activeGroup?.preprocessor === "none" || !activeGroup?.parts.length) {
            const selected = groups.find(({ preprocessor, parts }) => parts.length && preprocessor !== "none");
            return Object.hasOwn(guideModes, selected?.preprocessor) ? "transparent" : "map";
        }
        return Object.hasOwn(guideModes, activeGroup.preprocessor) ? "transparent" : "map";
    }

    function updatePlayButton(playing) {
        playIcon.setAttribute("d", playing ? "M4 3h4v14H4z M12 3h4v14h-4z" : "M5 2l12 8-12 8z");
        const label = playing ? "Pause video" : "Play video";
        play.setAttribute("aria-label", label);
        play.title = label;
    }

    function stopPlayback(sync = true) {
        if (!playback) return;
        const state = playback;
        playback = null;
        state.controller.abort();
        cancelAnimationFrame(state.tick);
        player.pause();
        player.hidden = true;
        player.removeAttribute("src");
        player.load();
        stage.classList.remove("rp-playing");
        updatePlayButton(false);
        updateButtons();
        if (sync) seek(state.frame);
    }

    async function startPlayback() {
        if (playback) {
            stopPlayback();
            return;
        }
        clearTimeout(scheduledSeek);
        const controller = new AbortController();
        const { signal } = controller;
        const state = { controller, frame: slider.valueAsNumber };
        playback = state;
        updatePlayButton(true);
        updateButtons();
        try {
            let url = sourceVideoURL(video);
            let start = timestamps[state.frame];
            let transparent = false;
            if (prepared && preparedSignature === preparationSignature(groups, loadingSettings)) {
                try {
                    const result = await previewJob("/prepared-video", {
                        ticket: prepared, mode: previewMode(), frame: state.frame,
                    }, signal);
                    ({ indices: state.indices, fps: state.fps } = result);
                    url = videoURL(result.video);
                    start = result.video.time;
                    transparent = result.video.transparent;
                } catch (error) {
                    if (error.status !== 404) throw error;
                    prepared = session.prepared = null;
                }
            }
            signal.throwIfAborted();
            player.style.background = transparent ? checkerboard : "";
            player.muted = false;
            player.addEventListener("loadedmetadata", () => { player.currentTime = start; }, { once: true, signal });
            player.src = url;
            try {
                await player.play();
            } catch (error) {
                if (error.name !== "NotAllowedError") throw error;
                player.muted = true;
                await player.play();
            }
            signal.throwIfAborted();
            player.hidden = false;
            stage.classList.add("rp-playing");
            let sourceIndex = state.frame;
            function tick() {
                const time = player.currentTime;
                if (state.indices) {
                    const index = Math.min(state.indices.length - 1, Math.floor(time * state.fps + 0.001));
                    state.frame = state.indices[index];
                } else {
                    while (sourceIndex > 0 && timestamps[sourceIndex] > time + 0.001) sourceIndex--;
                    while (sourceIndex + 1 < timestamps.length && timestamps[sourceIndex + 1] <= time + 0.001) sourceIndex++;
                    state.frame = sourceIndex;
                }
                updateTimeline(state.frame);
                state.tick = requestAnimationFrame(tick);
            }
            tick();
        } catch (error) {
            if (!signal.aborted) {
                stopPlayback();
                viewStatus.textContent = error.message;
            }
        }
    }

    async function seek(index) {
        if (!timestamps.length || !Number.isInteger(index)) return;
        stopPlayback(false);
        const target = Math.min(timestamps.length - 1, Math.max(0, index));
        updateTimeline(target);
        if (target === shownFrame && target === currentFrame) {
            restoreAnchors(target);
            return;
        }
        if (target !== currentFrame) {
            clearCheck();
            clearDetection();
            drawVersion += 1;
            status.textContent = "";
        }
        currentFrame = target;
        guideEditor?.update(activeGroup, target);
        session.frame = target;
        frameFetch?.abort();
        const controller = new AbortController();
        frameFetch = controller;
        shownFrame = -1;
        renderAnchors();
        updateButtons();
        if (!tracking) viewStatus.textContent = "Loading frame…";
        try {
            const image = await loadPhoto(video, target, controller.signal);
            if (controller.signal.aborted) {
                image.close();
                return;
            }
            photo?.close();
            photo = image;
            shownFrame = target;
            draw();
            setFrameStatus();
            loadFrameMasks(target, controller.signal);
            restoreAnchors(target);
        } catch (error) {
            if (!controller.signal.aborted) viewStatus.textContent = error.message;
        } finally {
            updateButtons();
        }
    }

    function invalidateTracking() {
        ticket = null;
        session.ticket = null;
        prepared = session.prepared = null;
        frameMasks.clear();
        trackFetch?.abort();
        setFrameStatus();
        draw();
    }

    async function runPreparation(applyAfter = false) {
        if (tracking) {
            trackFetch.abort();
            return;
        }
        stopPlayback();
        clearCheck();
        const requestSignature = signature(groups);
        const readyKey = withGuides ? preparationSignature(groups, loadingSettings) : null;
        const controller = new AbortController();
        const { signal } = controller;
        trackFetch = controller;
        tracking = true;
        status.textContent = "";
        updateButtons();
        setFrameStatus();
        try {
            const trackedGroups = withoutPreviews(groups.map((group) => ({ ...group, name: getName(group) })));
            const result = withGuides
                ? await previewJob("/prepare", { video, selections: guideSelections(trackedGroups), settings: loadingSettings }, signal)
                : await previewJob("/track", { video, groups: trackedGroups }, signal);
            if (signal.aborted || requestSignature !== signature(groups)) return;
            ticket = result.ticket;
            Object.assign(session, { signature: requestSignature, ticket });
            if (withGuides) {
                prepared = result.prepared;
                preparedSignature = readyKey;
                Object.assign(session, { prepared, preparedSignature });
                if (applyAfter) {
                    const mode = previewMode();
                    const { video: asset } = await previewJob("/prepared-video", {
                        ticket: prepared, mode, frame: currentFrame,
                    }, signal);
                    if (!signal.aborted) saveAndClose(asset);
                    return;
                }
                draw();
            }
            frameMasks.clear();
            if (shownFrame >= 0) await loadFrameMasks(shownFrame, signal);
        } catch (error) {
            if (!signal.aborted) status.textContent = error.message;
        } finally {
            tracking = false;
            updateButtons();
            setFrameStatus();
            restoreAnchors(shownFrame);
        }
    }

    const track = button("Track & preview", () => runPreparation(), "rp-primary");

    const detect = button("Detect", async () => {
        const group = activeGroup;
        const text = prompt.value.trim();
        if (!group || !text) return;
        const frame = currentFrame;
        const start = from.value === "" ? 0 : from.valueAsNumber - 1;
        const end = to.value === "" ? null : to.valueAsNumber - 1;
        if (!Number.isInteger(start) || start < 0 || start > frame ||
            (end !== null && (!Number.isInteger(end) || end < frame || end >= timestamps.length))) {
            status.textContent = "The tracking range must include the current frame.";
            return;
        }
        const radius = expansion.valueAsNumber;
        if (!Number.isInteger(radius) || radius < 0 || radius > 256) {
            status.textContent = "Enter a mask expansion between 0 and 256 pixels.";
            return;
        }
        const confidence = threshold.valueAsNumber;
        if (!Number.isFinite(confidence) || confidence < 0.01 || confidence > 0.99) {
            status.textContent = "Enter a detection confidence between 0.01 and 0.99.";
            return;
        }
        const part = { text, frame, threshold: confidence };
        if (start) part.start = start;
        if (end !== null) part.end = end;
        const groupSettings = { expand: radius, fill_holes: fillHoles.checked };
        if (!edgeFill) groupSettings.preprocessor = mode.value;
        const previous = withGuides ? group.parts.find((saved) => (
            saved.frame === frame && saved.text.toLowerCase() === text.toLowerCase()
        )) : currentPart;
        clearCheck();
        clearDetection();
        const version = draftVersion;
        detecting = true;
        updateButtons();
        status.textContent = "Detecting with SAM 3.1…";
        try {
            const items = await getCandidates(part);
            if (life.signal.aborted || version !== draftVersion) {
                items.forEach(({ mask }) => mask.close());
                return;
            }
            if (!items.length) {
                status.textContent = "No match. Try another description, frame or confidence.";
                return;
            }
            candidates = items;
            if (previous?.text === text && previous.frame === frame && previous.threshold === confidence) {
                checked = previous.instances?.filter((index) => index < items.length) ?? [];
            }
            if (!checked.length) checked = items.map((_, index) => index);
            Object.assign(group, groupSettings);
            saveDetection(group, part);
            candidates.forEach(({ score }, index) => {
                const checkbox = input("checkbox", index, `Mask ${index + 1}`);
                checkbox.checked = checked.includes(index);
                checkbox.addEventListener("change", () => {
                    if (!checkbox.checked && checked.length === 1) {
                        checkbox.checked = true;
                        status.textContent = "Keep at least one mask for this selection.";
                        return;
                    }
                    checked = checkbox.checked ? [...checked, index] : checked.filter((i) => i !== index);
                    clearCheck();
                    saveDetection(group, part);
                });
                const entry = label("", checkbox);
                entry.append(swatch(colors[index % colors.length]), `Mask ${index + 1} · ${Math.round(score * 100)}%`);
                candidateList.append(entry);
            });
        } catch (error) {
            if (!life.signal.aborted && version === draftVersion) status.textContent = error.message;
        } finally {
            detecting = false;
            updateButtons();
            restoreAnchors(shownFrame);
        }
    });

    function saveDetection(group, detection) {
        const previousSignature = signature(groups);
        const part = {
            ...detection,
            instances: [...checked].sort((a, b) => a - b),
            preview: mergedPreview(candidates, checked),
        };
        if (!group.name.trim()) group.name = part.text.toUpperCase();
        const index = withGuides ? group.parts.findIndex((saved) => (
            saved.frame === part.frame && saved.text.toLowerCase() === part.text.toLowerCase()
        )) : group.parts.indexOf(currentPart);
        if (index < 0) group.parts.push(part);
        else group.parts[index] = part;
        currentPart = part;
        currentParts.set(group, part);
        if (signature(groups) !== previousSignature) invalidateTracking();
        else draw();
        syncGroupCard(group);
        renderAnchors();
        status.textContent = `Saved on frame ${part.frame + 1}.`;
    }

    const add = button("Add selection", () => {
        if (withGuides) {
            const selection = newSelection();
            groups.push(selection);
            openGroup(selection);
            prompt.focus();
            return;
        }
        openSelection(null);
        renderParts();
        prompt.focus();
    });

    function renderParts() {
        partList.replaceChildren();
        if (withGuides || !activeGroup) return;
        const group = activeGroup;
        for (const part of group.parts) {
            if (part === currentPart) continue;
            const open = button(`${part.text} · Frame ${part.frame + 1}`, () => {
                openSelection(part);
                renderParts();
                seek(part.frame);
                prompt.focus();
            }, "rp-flex rp-truncate rp-part-label");
            open.disabled = !timestamps.length;
            const remove = button("×", () => {
                group.parts.splice(group.parts.indexOf(part), 1);
                clearCheck();
                invalidateTracking();
                syncGroupCard(group);
                renderParts();
                renderAnchors();
                updateButtons();
            });
            remove.setAttribute("aria-label", `Remove selection ${part.text} on frame ${part.frame + 1}`);
            const actions = row(open, remove);
            actions.classList.add("rp-part-actions");
            partList.append(actions);
        }
    }

    function renderAnchors() {
        anchors.replaceChildren();
        const entries = withGuides ? groups.flatMap((group) => (
            [...new Set(group.parts.map(({ frame }) => frame))].map((frame) => ({ frame, group }))
        )) : [...new Set(groups.flatMap(({ parts }) => parts.map(({ frame }) => frame)))].map((frame) => ({ frame }));
        if (!entries.length) return;
        anchors.append(element("span", "Anchors"));
        entries.sort((a, b) => a.frame - b.frame);
        for (const { frame, group } of entries) {
            const name = group?.parts.filter((part) => part.frame === frame).map(({ text }) => text).join(", ");
            const jump = button(`#${frame + 1}`, () => {
                if (group) {
                    if (activeGroup !== group) openGroup(group);
                    openSelection(group.parts.filter((part) => part.frame === frame).at(-1));
                }
                seek(frame);
            });
            jump.setAttribute("aria-label", `Go to anchor frame ${frame + 1}`);
            if (name) jump.title = name;
            jump.setAttribute("aria-pressed", String(frame === currentFrame));
            jump.disabled = !timestamps.length;
            const remove = button("", () => {
                for (const target of group ? [group] : groups) {
                    target.parts = target.parts.filter((part) => part.frame !== frame);
                }
                if ((!group || group === activeGroup) && currentPart?.frame === frame) {
                    openSelection(withGuides ? group.parts.at(-1) : null);
                }
                else clearCheck();
                invalidateTracking();
                for (const group of groups) syncGroupCard(group);
                renderParts();
                renderAnchors();
                updateButtons();
            }, "rp-anchor-remove");
            remove.setAttribute("aria-label", `Remove anchor frame ${frame + 1}`);
            remove.title = group ? `Remove this card’s detections on frame ${frame + 1}`
                : "Remove all selections anchored to this frame";
            remove.disabled = !timestamps.length;
            const chip = element("span", "", "rp-anchor");
            if (group) chip.style.setProperty("--rp-anchor-color", group.color);
            chip.append(jump, remove);
            anchors.append(chip);
        }
    }

    function syncGroupCard(group) {
        const header = groupHeaders.get(group);
        if (!header) return;
        header.card.style.setProperty("--rp-group-color", group.color ?? colors[0]);
        if (withGuides) {
            const title = group.parts[0]?.text ?? "New selection";
            if (header.open) header.open.textContent = title;
            header.remove.setAttribute("aria-label", `Remove selection ${title}`);
            return;
        }
        const title = getName(group);
        header.open.textContent = title;
        header.open.title = title;
        header.rename.disabled = !group.parts.length;
        header.remove.setAttribute("aria-label", `Remove group ${title}`);
    }

    function renderGroups() {
        groupList.replaceChildren();
        groupHeaders.clear();
        if (withGuides) {
            for (const selection of groups) {
                const card = element("section", "", "rp-group-card");
                const remove = button("×", () => {
                    const index = groups.indexOf(selection);
                    groups.splice(index, 1);
                    if (selection.parts.length) invalidateTracking();
                    if (!groups.length) groups.push(newSelection());
                    openGroup(selection === activeGroup ? groups[Math.min(index, groups.length - 1)] : activeGroup);
                }, "rp-selection-remove");
                const open = selection === activeGroup ? null : button("", () => openGroup(selection), "rp-flex rp-truncate");
                if (open) card.append(row(open, remove));
                else {
                    selectionRow.replaceChildren(prompt, detect, remove);
                    card.append(draft);
                }
                groupHeaders.set(selection, { card, open, remove });
                syncGroupCard(selection);
                groupList.append(card);
            }
            renderAnchors();
            updateButtons();
            return;
        }
        for (const group of groups) {
            const active = group === activeGroup;
            const card = element("section", "", "rp-group-card");
            const open = button("", () => openGroup(active ? null : group), "rp-flex rp-truncate rp-group-title");
            open.setAttribute("aria-expanded", String(active));
            const rename = button("✎", () => {
                if (!active) openGroup(group);
                name.hidden = false;
                name.value = getName(group);
                name.focus();
                name.select();
            });
            rename.title = "Rename group";
            rename.setAttribute("aria-label", "Rename group");
            const check = button("Check group", () => {
                if (!active) openGroup(group);
                checkGroup(group);
            });
            check.title = "Check this group on the current frame using its approved masks";
            const remove = button("×", () => {
                const index = groups.indexOf(group);
                groups.splice(index, 1);
                if (group.parts.length) invalidateTracking();
                if (active) {
                    openGroup(groups[Math.min(index, groups.length - 1)] ?? null);
                    return;
                }
                clearCheck();
                name.hidden = true;
                renderGroups();
            });
            card.append(row(open, rename, check, remove));
            groupHeaders.set(group, { card, open, rename, check, remove });
            syncGroupCard(group);
            if (active) card.append(name, partList, checkSummary, draft);
            groupList.append(card);
        }
        renderParts();
        renderAnchors();
        updateButtons();
    }

    const play = button("", startPlayback, "rp-play");
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    const playIcon = document.createElementNS(icon.namespaceURI, "path");
    icon.setAttribute("viewBox", "0 0 20 20");
    icon.setAttribute("aria-hidden", "true");
    icon.append(playIcon);
    play.append(icon);
    updatePlayButton(false);
    const previous = button("‹", () => seek(slider.valueAsNumber - 1));
    const next = button("›", () => seek(slider.valueAsNumber + 1));
    previous.setAttribute("aria-label", "Previous frame");
    next.setAttribute("aria-label", "Next frame");
    slider.addEventListener("input", () => {
        clearTimeout(scheduledSeek);
        const frame = slider.valueAsNumber;
        stopPlayback(false);
        updateTimeline(frame);
        scheduledSeek = setTimeout(() => seek(frame), 75);
    });
    slider.addEventListener("change", () => {
        clearTimeout(scheduledSeek);
        seek(slider.valueAsNumber);
    });
    frameInput.addEventListener("change", () => seek(frameInput.valueAsNumber - 1));
    for (const toggle of [showSelection, showGuide]) toggle.addEventListener("change", () => {
        stopPlayback();
        draw();
    });
    if (withGuides) {
        for (const field of [expansion, fillHoles]) field.addEventListener("change", () => {
            if (!activeGroup || !expansion.checkValidity() || !Number.isInteger(expansion.valueAsNumber)) return;
            activeGroup.expand = expansion.valueAsNumber;
            activeGroup.fill_holes = fillHoles.checked;
            draw();
        });
    }
    for (const field of [prompt, threshold]) {
        field.addEventListener("input", () => {
            status.textContent = "";
            clearDetection();
        });
    }
    prompt.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            if (!detect.disabled) detect.click();
        }
    });
    name.addEventListener("change", () => {
        if (!activeGroup || !name.value.trim()) return;
        activeGroup.name = name.value.trim().toUpperCase();
        syncGroupCard(activeGroup);
    });
    name.addEventListener("blur", () => { name.hidden = true; });
    name.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            name.value = getName(activeGroup);
            name.blur();
        } else if (event.key === "Enter") {
            event.preventDefault();
            name.blur();
        }
    });

    const stage = element("div", "", "rp-stage");
    stage.append(canvas, player, spinner);
    const guideEditor = withGuides ? createGuideEditor(stage, video, () => showGuide.checked, draw) : null;
    const timeline = row(...(withGuides ? [play] : []), previous, slider, next, frameInput, time);
    timeline.classList.add("rp-timeline");
    const view = element("section", "", "rp-view");
    view.append(stage, timeline);
    if (withGuides) {
        const toggles = row();
        for (const [toggle, name] of [[showSelection, "Selection"], [showGuide, "Guide"]]) {
            const control = element("label");
            control.append(toggle, name);
            toggles.append(control);
        }
        toggles.classList.add("rp-preview-options");
        toggles.setAttribute("role", "group");
        toggles.setAttribute("aria-label", "Preview layers");
        const top = row(toggles, anchors);
        top.classList.add("rp-preview-top");
        const actions = row(track, viewStatus);
        actions.classList.add("rp-preview-actions");
        const toolbar = element("div", "", "rp-preview-toolbar");
        toolbar.append(top, actions);
        view.append(toolbar);
    } else view.append(anchors, row(track, clearCheckButton, viewStatus));

    const settings = element("details");
    settings.append(element("summary", "Detection confidence and tracking range"),
        row(label("Confidence", threshold)), row(label("From", from), label("To", to)));
    const draft = element("section", "", "rp-group-content");
    const effect = edgeFill ? label("Fill inward from edges", mode) : mode;
    effect.classList.add("rp-flex");
    const selectionRow = row(prompt, detect);
    draft.append(selectionRow, row(effect, color.wrapper),
        row(label("Expand (px)", expansion), label("Fill holes", fillHoles)));
    draft.append(candidateList, status, settings);
    if (!withGuides) draft.append(element("hr", "", "rp-selection-divider"), row(add));
    const newButton = button("New group", () => {
        const group = {
            name: "", color: colors[groups.length % colors.length],
            expand: defaultExpansion, fill_holes: true, parts: [],
            ...(edgeFill ? { fill_from_edges: true } : { preprocessor: "segmentation" }),
        };
        groups.push(group);
        openGroup(group);
        prompt.focus();
    });
    const groupHeader = row(element("h3", "Selection groups"), newButton);
    groupHeader.classList.add("rp-between");
    const side = element("aside");
    if (withGuides) {
        add.classList.add("rp-add-selection");
        side.append(groupList, add);
    } else side.append(groupHeader, groupList);
    const layout = element("div", "", "rp-layout");
    layout.append(view, side);

    const header = element("header");
    const close = button("×", () => modal.close());
    close.setAttribute("aria-label", "Close editor");
    header.append(element("h2", "Video selections"), close);
    const footer = element("footer");
    function saveAndClose(asset) {
        const saved = groups.map((group) => ({
            ...group, name: group.parts.length ? getName(group) : group.name,
        }));
        const preview = ticket || prepared || asset ? {
            ticket, frame: currentFrame, prepared, preparedSignature,
            activeGroup: groups.filter(({ parts }) => parts.length).indexOf(activeGroup),
            previewMode: showGuide.checked ? "guide" : "selection", asset,
        } : null;
        onSave(saved, preview);
        modal.close();
    }
    const apply = button("Apply selections", () => withGuides ? runPreparation(true) : saveAndClose(), "rp-primary");
    if (!withGuides) footer.append(element("span", video, "rp-truncate"));
    footer.append(row(button("Cancel", () => modal.close()), apply));
    modal.append(element("style", withGuides ? editorStyle + guideStyle : editorStyle), header, layout, footer);
    modal.addEventListener("close", () => {
        life.abort();
        stopPlayback(false);
        guideEditor?.disconnect();
        frameFetch?.abort();
        preparedFetch?.abort();
        trackFetch?.abort();
        checkRun?.abort();
        checkResult?.bitmap?.close();
        clearTimeout(scheduledSeek);
        photo?.close();
        candidates.forEach(({ mask }) => mask.close());
        for (const promise of anchorBitmaps.values()) promise.then((bitmap) => bitmap?.close());
        modal.remove();
    }, { once: true });

    document.body.append(modal);
    modal.showModal();
    openGroup(activeGroup);
    request(`/info?${new URLSearchParams({ video })}`, { signal: life.signal })
        .then((response) => response.json())
        .then((info) => {
            if (life.signal.aborted) return;
            timestamps = info.timestamps;
            sourceWidth = info.width;
            slider.max = timestamps.length - 1;
            frameInput.max = from.max = to.max = timestamps.length;
            renderParts();
            seek(session.frame ?? activeGroup?.parts[0]?.frame ?? 0);
        })
        .catch((error) => { if (!life.signal.aborted) viewStatus.textContent = error.message; });
    return modal;
}

app.registerExtension({
    name: "RPNodes.SelectiveVideo",
    beforeConfigureGraph(workflow) {
        migrateSelectiveVideoOutputs(workflow);
    },
    nodeCreated(node) {
        const edgeFill = node.comfyClass === "RPSelectiveVideoEdgeFill";
        const withGuides = node.comfyClass === "RPSelectiveVideoGuides";
        if (!edgeFill && !withGuides) return;
        const video = node.widgets.find((widget) => widget.name === "video");
        const selections = node.widgets.find((widget) => widget.name === "selections");
        if (!video || !selections) return;
        const loadOptions = node.widgets.filter((widget) => Object.hasOwn(loadOptionDefaults, widget.name));
        for (const widget of loadOptions) {
            const serializeValue = widget.serializeValue;
            const fallback = loadOptionDefaults[widget.name];
            widget.serializeValue = async function (...args) {
                const value = serializeValue ? await serializeValue.apply(this, args) : this.value;
                return loadOptionValue(value, fallback);
            };
        }
        selections.type = "converted-widget";
        selections.hidden = true;
        selections.computeSize = () => [0, -4];
        selections.element?.style.setProperty("display", "none");

        let modal;
        let session = {};
        let previewStatus = "";
        const editor = node.addWidget("button", "Edit video selections", null, () => {
            if (modal?.open || !video.value) return;
            const filename = video.value;
            const config = readConfig(selections, filename);
            const { trackingPreview } = config;
            const groups = withGuides ? config.groups.map((group) => ({
                ...group,
                preprocessor: ({ openpose: "pose_guide", rectangle_guide: "none" })[group.preprocessor]
                    ?? group.preprocessor ?? (config.version >= 3 ? "none" : "segmentation"),
                segmentation_opacity: group.segmentation_opacity ?? 0.5,
            })) : config.groups;
            const loadingSettings = Object.fromEntries(Object.entries(loadOptionDefaults).map(([name, fallback]) => [
                name, loadOptionValue(loadOptions.find((widget) => widget.name === name)?.value, fallback),
            ]));
            if (session.video !== filename) {
                session = {
                    video: filename, ticket: trackingPreview?.ticket, frame: trackingPreview?.frame,
                    signature: signature(groups),
                    prepared: trackingPreview?.prepared, preparedSignature: trackingPreview?.preparedSignature,
                    previewMode: trackingPreview?.previewMode,
                };
            }
            modal = openEditor(filename, groups, session, edgeFill, withGuides, loadingSettings, (savedGroups, preview) => {
                app.graph?.beforeChange?.();
                const data = withGuides
                    ? { version: 4, video: filename, selections: guideSelections(savedGroups) }
                    : { version: 2, video: filename, groups: savedGroups };
                if (preview) data.trackingPreview = preview;
                selections.value = JSON.stringify(data);
                session.previewMode = preview?.previewMode;
                selections.callback?.(selections.value);
                refresh();
                app.graph?.afterChange?.();
            });
        }, { serialize: false });
        const updateLabel = () => {
            const { groups } = readConfig(selections, video.value);
            const title = groups.length ? `Edit video selections (${groups.length})` : "Edit video selections";
            editor.label = previewStatus === "Preparing preview…" ? previewStatus : title;
            editor.tooltip = previewStatus || undefined;
            node.setDirtyCanvas(true, true);
        };
        const player = attachNativePreview(node, (status) => {
            previewStatus = status;
            updateLabel();
        });
        const refresh = () => {
            const { groups, trackingPreview } = readConfig(selections, video.value);
            updateLabel();
            player.update(video.value, withoutPreviews(groups), trackingPreview);
        };
        const onConfigure = node.onConfigure;
        node.onConfigure = function (...args) {
            const result = onConfigure?.apply(this, args);
            for (const widget of loadOptions) {
                widget.value = loadOptionValue(widget.value, loadOptionDefaults[widget.name]);
            }
            refresh();
            return result;
        };
        const onRemoved = node.onRemoved;
        node.onRemoved = function (...args) {
            modal?.close();
            player.disconnect();
            return onRemoved?.apply(this, args);
        };
        const callback = video.callback;
        video.callback = function (...args) {
            const result = callback?.apply(this, args);
            refresh();
            return result;
        };
        refresh();
    },
});
