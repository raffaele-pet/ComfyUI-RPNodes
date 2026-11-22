import { app } from "/scripts/app.js";
import {
    attachNativePreview, colors, createPainter, decodeMask, defaultExpansion, loadGroupOverlays,
    loadPhoto, loadTrackedMasks, post, previewJob, releaseOverlays, request,
} from "./selective_video_preview.js";
const maps = { segmentation: "Segmentation", depth: "Depth", canny: "Canny", openpose: "OpenPose" };

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

function readConfig(widget, video) {
    try {
        const data = JSON.parse(widget.value);
        if (data.video !== video) return { groups: [] };
        if (Array.isArray(data.groups)) return data;
        const groups = (data.layers ?? []).map(({ text, preprocessor, color, ...selection }) => ({
            name: text, preprocessor, color, expand: defaultExpansion, fill_holes: true, parts: [{ text, ...selection }],
        }));
        return { groups };
    } catch {
        return { groups: [] };
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
    .rp-selective-editor .rp-part-actions { gap:5px; }
    .rp-selective-editor .rp-part-actions button { padding-inline:7px; }
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
    .rp-selective-editor input[type=color] { width:42px; height:34px; }
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

function openEditor(video, groups, session, onSave) {
    const modal = element("dialog", "", "rp-selective-editor");
    modal.setAttribute("aria-label", "Selective Video · Selection editor");
    const life = new AbortController();
    const canvas = element("canvas");
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
    const mode = element("select", "", "rp-flex");
    mode.setAttribute("aria-label", "Group preprocessor");
    for (const [value, title] of Object.entries(maps)) mode.append(new Option(title, value));
    const color = input("color", colors[0], "Group color");
    const expansion = input("number", defaultExpansion, "Mask expansion in source pixels");
    expansion.className = "rp-small-number";
    expansion.min = 0;
    expansion.max = 256;
    const fillHoles = input("checkbox", "", "Fill holes");
    fillHoles.checked = true;

    const prompt = input("text", "", "Object description");
    prompt.className = "rp-flex";
    prompt.placeholder = "e.g. katana";
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
    const partList = element("div", "", "rp-list");
    const candidateList = element("div", "", "rp-list");
    const anchors = element("div", "", "rp-anchors");
    const groupSettings = element("section", "", "rp-group-content");
    const checkSummary = element("p", "", "rp-check-result");
    checkSummary.setAttribute("role", "status");
    const groupHeaders = new Map();
    const firstSearches = new WeakMap();
    const anchorBitmaps = new Map();
    const frameMasks = new Map();
    const restoreFailed = new WeakSet();

    let activeGroup = groups[0] ?? null;
    let editingPart = null;
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
    let trackFetch;
    let draftVersion = 0;
    let drawVersion = 0;
    let scheduledSeek;
    let ticket = session.signature === signature(groups) ? session.ticket : null;

    function getName(group) {
        return group.name.trim() || group.parts[0]?.text || firstSearches.get(group) || "New selection group";
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

    async function draw() {
        if (!photo || shownFrame !== currentFrame || life.signal.aborted) return;
        const frame = shownFrame;
        const version = ++drawVersion;
        const scale = photo.width / sourceWidth;
        if (checkResult?.frame === frame) {
            const { group, bitmap } = checkResult;
            paint(photo, bitmap ? [{
                bitmaps: [bitmap], color: group.color, opacity: 0.57,
                radius: Math.round((group.expand ?? defaultExpansion) * scale), fillHoles: group.fill_holes ?? true,
            }] : []);
            return;
        }

        const visibleGroups = groups.filter(({ parts }) => parts.length);
        const tracked = frameMasks.get(frame) ?? [];
        const overlays = await loadGroupOverlays(visibleGroups, frame, tracked,
            visibleGroups.indexOf(activeGroup), scale, storedBitmap);
        try {
            if (life.signal.aborted || version !== drawVersion || shownFrame !== frame || currentFrame !== frame) return;
            const draft = candidates.map(({ mask }, index) => ({
                bitmaps: [mask], color: colors[index % colors.length],
                opacity: checked.includes(index) ? 0.58 : 0.16,
            }));
            paint(photo, [...overlays, ...draft]);
        } finally {
            releaseOverlays(overlays);
        }
    }

    function setFrameStatus() {
        if (tracking) viewStatus.textContent = "Tracking groups with SAM 3.1…";
        else if (shownFrame >= 0) {
            const state = checkResult ? "Detection check"
                : ticket ? "Tracked masks" : "Saved masks on anchor frames";
            viewStatus.textContent = `Frame ${shownFrame + 1} of ${timestamps.length} · ${state}`;
        }
    }

    function updateButtons() {
        const ready = shownFrame === currentFrame && shownFrame >= 0;
        const busy = detecting || restoring || tracking || Boolean(checkRun);
        detect.disabled = busy || !ready || !activeGroup || !prompt.value.trim();
        add.disabled = busy || !ready || !activeGroup || !checked.length;
        track.disabled = !tracking && (!groups.some(({ parts }) => parts.length) || !ready || busy);
        for (const control of groupList.querySelectorAll("[data-rp-check]")) {
            control.disabled = busy || !ready;
        }
        clearCheckButton.hidden = !checkResult && !checkRun;
        track.textContent = tracking ? "Cancel tracking" : "Track & preview";
        slider.disabled = frameInput.disabled = !timestamps.length;
        previous.disabled = !timestamps.length || !currentFrame;
        next.disabled = !timestamps.length || currentFrame >= timestamps.length - 1;
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

    async function checkSaved(group, parts, guided = false) {
        const frame = shownFrame;
        clearCheck();
        const controller = new AbortController();
        checkRun = controller;
        updateButtons();
        checkSummary.textContent = guided
            ? `Checking frame ${frame + 1} using approved masks…` : `Checking frame ${frame + 1}…`;
        try {
            const { signal } = controller;
            const detection = guided
                ? (await previewJob("/check-group", {
                    video, frame, ticket,
                    group: { ...withoutPreviews([group])[0], name: getName(group) },
                    group_index: groups.filter(({ parts }) => parts.length).indexOf(group),
                }, signal)).detection
                : await post("/check", {
                    video, frame, prompts: parts.map(({ text, threshold }) => ({ text, threshold })),
                }, signal);
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

    function resetDraft() {
        clearCheck();
        editingPart = null;
        add.textContent = "Add selection";
        reset.hidden = true;
        prompt.value = "";
        threshold.value = 0.5;
        from.value = to.value = "";
        status.textContent = "";
        clearDetection();
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

    async function seek(index) {
        if (!timestamps.length || !Number.isInteger(index)) return;
        const target = Math.min(timestamps.length - 1, Math.max(0, index));
        frameInput.value = target + 1;
        slider.value = target;
        time.textContent = `${timestamps[target].toFixed(2)} s`;
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
        frameMasks.clear();
        trackFetch?.abort();
        setFrameStatus();
        draw();
    }

    const track = button("Track & preview", async () => {
        if (tracking) {
            trackFetch.abort();
            return;
        }
        clearCheck();
        const requestSignature = signature(groups);
        const controller = new AbortController();
        const { signal } = controller;
        trackFetch = controller;
        tracking = true;
        status.textContent = "";
        updateButtons();
        setFrameStatus();
        try {
            const trackedGroups = withoutPreviews(groups.map((group) => ({ ...group, name: getName(group) })));
            const result = await previewJob("/track", { video, groups: trackedGroups }, signal);
            if (signal.aborted || requestSignature !== signature(groups)) return;
            ticket = result.ticket;
            Object.assign(session, { signature: requestSignature, ticket });
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
    }, "rp-primary");

    const detect = button("Detect", async () => {
        const group = activeGroup;
        const text = prompt.value.trim();
        if (!group || !text) return;
        if (!group.name.trim() && !group.parts.length) {
            firstSearches.set(group, text);
            syncGroupCard(group);
        }
        clearCheck();
        const previousChoices = checked.length ? [...checked] : null;
        clearDetection();
        const version = draftVersion;
        detecting = true;
        updateButtons();
        status.textContent = "Detecting with SAM 3.1…";
        try {
            const items = await getCandidates({ frame: currentFrame, text, threshold: threshold.valueAsNumber });
            if (life.signal.aborted || version !== draftVersion) {
                items.forEach(({ mask }) => mask.close());
                return;
            }
            candidates = items;
            checked = previousChoices?.filter((i) => i < candidates.length) ?? candidates.map((_, i) => i);
            candidates.forEach(({ score }, index) => {
                const checkbox = input("checkbox", index, `Mask ${index + 1}`);
                checkbox.checked = checked.includes(index);
                checkbox.addEventListener("change", () => {
                    checked = checkbox.checked ? [...checked, index] : checked.filter((i) => i !== index);
                    updateButtons();
                    draw();
                });
                const entry = label("", checkbox);
                entry.append(swatch(colors[index % colors.length]), `Mask ${index + 1} · ${Math.round(score * 100)}%`);
                candidateList.append(entry);
            });
            draw();
            status.textContent = candidates.length ? "Choose the masks to add." : "No match. Try another description, frame or confidence.";
        } catch (error) {
            if (!life.signal.aborted && version === draftVersion) status.textContent = error.message;
        } finally {
            detecting = false;
            updateButtons();
            restoreAnchors(shownFrame);
        }
    });

    const add = button("Add selection", () => {
        if (!activeGroup) return;
        const start = from.value === "" ? 0 : from.valueAsNumber - 1;
        const end = to.value === "" ? null : to.valueAsNumber - 1;
        if (!Number.isInteger(start) || start < 0 || start > currentFrame ||
            (end !== null && (!Number.isInteger(end) || end < currentFrame || end >= timestamps.length))) {
            status.textContent = "The tracking range must include the current frame.";
            return;
        }
        const part = {
            text: prompt.value.trim(), frame: currentFrame, threshold: threshold.valueAsNumber,
            instances: [...checked].sort((a, b) => a - b),
            preview: candidates.length ? mergedPreview(candidates, checked) : editingPart?.preview,
        };
        if (start) part.start = start;
        if (end !== null) part.end = end;
        if (!activeGroup.name.trim()) activeGroup.name = part.text;
        const index = activeGroup.parts.indexOf(editingPart);
        if (index < 0) activeGroup.parts.push(part);
        else activeGroup.parts[index] = part;
        resetDraft();
        invalidateTracking();
        renderSettings();
        renderGroups();
        status.textContent = `Added to ${getName(activeGroup)} on frame ${part.frame + 1}.`;
    });

    const reset = button("Cancel edit", resetDraft);
    reset.hidden = true;

    async function editPart(part) {
        resetDraft();
        editingPart = part;
        prompt.value = part.text;
        threshold.value = part.threshold;
        from.value = part.start ? part.start + 1 : "";
        to.value = part.end == null ? "" : part.end + 1;
        add.textContent = "Update selection";
        reset.hidden = false;
        const version = draftVersion + Number(part.frame !== currentFrame);
        await seek(part.frame);
        if (editingPart === part && draftVersion === version && shownFrame === part.frame && part.instances?.length) {
            checked = [...part.instances];
            updateButtons();
        }
    }

    function renderSettings() {
        groupSettings.hidden = !activeGroup;
        partList.replaceChildren();
        if (!activeGroup) return;
        name.value = activeGroup.name;
        mode.value = activeGroup.preprocessor ?? "segmentation";
        color.value = activeGroup.color ?? colors[0];
        expansion.value = activeGroup.expand ?? defaultExpansion;
        fillHoles.checked = activeGroup.fill_holes ?? true;
        for (const part of activeGroup.parts) {
            const open = button(`${part.text} · Frame ${part.frame + 1}`, () => {
                resetDraft();
                seek(part.frame);
            }, "rp-flex rp-truncate");
            const edit = button("Edit", () => editPart(part));
            edit.setAttribute("aria-label", `Edit selection ${part.text} on frame ${part.frame + 1}`);
            edit.disabled = !timestamps.length;
            open.disabled = !timestamps.length;
            const check = button("Check", () => checkSaved(activeGroup, [part]));
            check.dataset.rpCheck = "";
            check.setAttribute("aria-label", `Check ${part.text} on the current frame`);
            check.title = "Detect this selection on the current frame";
            const remove = button("×", () => {
                activeGroup.parts.splice(activeGroup.parts.indexOf(part), 1);
                resetDraft();
                invalidateTracking();
                renderSettings();
                renderGroups();
            });
            remove.setAttribute("aria-label", `Remove selection ${part.text} on frame ${part.frame + 1}`);
            const actions = row(open, check, edit, remove);
            actions.classList.add("rp-part-actions");
            partList.append(actions);
        }
    }

    function renderAnchors() {
        anchors.replaceChildren();
        const frames = [...new Set(groups.flatMap(({ parts }) => parts.map(({ frame }) => frame)))].sort((a, b) => a - b);
        if (!frames.length) return;
        anchors.append(element("span", "Anchors"));
        for (const frame of frames) {
            const jump = button(`#${frame + 1}`, () => seek(frame));
            jump.setAttribute("aria-label", `Go to anchor frame ${frame + 1}`);
            jump.setAttribute("aria-pressed", String(frame === currentFrame));
            jump.disabled = !timestamps.length;
            const remove = button("", () => {
                for (const group of groups) group.parts = group.parts.filter((part) => part.frame !== frame);
                resetDraft();
                invalidateTracking();
                renderSettings();
                renderGroups();
            }, "rp-anchor-remove");
            remove.setAttribute("aria-label", `Remove anchor frame ${frame + 1}`);
            remove.title = "Remove all selections anchored to this frame";
            remove.disabled = !timestamps.length;
            const chip = element("span", "", "rp-anchor");
            chip.append(jump, remove);
            anchors.append(chip);
        }
    }

    function syncGroupCard(group) {
        const header = groupHeaders.get(group);
        if (!header) return;
        header.card.style.setProperty("--rp-group-color", group.color ?? colors[0]);
        const title = getName(group);
        header.open.textContent = group.parts.length
            ? `${title} · ${maps[group.preprocessor ?? "segmentation"]} · ${group.parts.length}` : title;
        header.remove.setAttribute("aria-label", `Remove group ${getName(group)}`);
    }

    function renderGroups() {
        groupList.replaceChildren();
        groupHeaders.clear();
        for (const group of groups) {
            const active = group === activeGroup;
            const card = element("section", "", "rp-group-card");
            const open = button("", () => {
                resetDraft();
                activeGroup = active ? null : group;
                name.hidden = true;
                renderSettings();
                renderGroups();
                draw();
            }, "rp-flex rp-truncate");
            open.setAttribute("aria-expanded", String(active));
            const remove = button("×", () => {
                const index = groups.indexOf(group);
                groups.splice(index, 1);
                if (group === activeGroup) {
                    activeGroup = groups[Math.min(index, groups.length - 1)] ?? null;
                    name.hidden = true;
                    resetDraft();
                }
                if (group.parts.length) invalidateTracking();
                renderSettings();
                renderGroups();
            });
            const heading = active ? row(open, renameButton, remove) : row(open, remove);
            card.append(heading);
            groupHeaders.set(group, { card, open, remove });
            syncGroupCard(group);
            if (active) {
                card.append(groupSettings, draft);
                if (group.parts.length) {
                    const check = button("Check group", () => checkSaved(group, group.parts, true));
                    check.dataset.rpCheck = "";
                    check.title = "Check this group on the current frame using its approved masks";
                    const savedHeader = row(element("h3", "Saved selections"), check);
                    savedHeader.classList.add("rp-between");
                    card.append(savedHeader, partList, checkSummary);
                }
            }
            groupList.append(card);
        }
        renderAnchors();
        updateButtons();
    }

    const previous = button("‹", () => seek(currentFrame - 1));
    const next = button("›", () => seek(currentFrame + 1));
    previous.setAttribute("aria-label", "Previous frame");
    next.setAttribute("aria-label", "Next frame");
    slider.addEventListener("input", () => {
        clearTimeout(scheduledSeek);
        frameInput.value = slider.valueAsNumber + 1;
        time.textContent = `${timestamps[slider.valueAsNumber].toFixed(2)} s`;
        scheduledSeek = setTimeout(() => seek(slider.valueAsNumber), 75);
    });
    slider.addEventListener("change", () => {
        clearTimeout(scheduledSeek);
        seek(slider.valueAsNumber);
    });
    frameInput.addEventListener("change", () => seek(frameInput.valueAsNumber - 1));
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
        activeGroup.name = name.value.trim();
        syncGroupCard(activeGroup);
    });
    name.addEventListener("blur", () => { name.hidden = true; });
    name.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            event.stopPropagation();
            name.value = activeGroup.name;
            name.blur();
        } else if (event.key === "Enter") {
            event.preventDefault();
            name.blur();
        }
    });
    mode.addEventListener("change", () => {
        activeGroup.preprocessor = mode.value;
        syncGroupCard(activeGroup);
    });
    color.addEventListener("input", () => {
        activeGroup.color = color.value;
        syncGroupCard(activeGroup);
        draw();
    });
    expansion.addEventListener("input", () => {
        const radius = expansion.valueAsNumber;
        if (Number.isInteger(radius) && radius >= 0 && radius <= 256) {
            activeGroup.expand = radius;
            draw();
        }
    });
    expansion.addEventListener("change", () => { expansion.value = activeGroup.expand; });
    fillHoles.addEventListener("change", () => {
        activeGroup.fill_holes = fillHoles.checked;
        draw();
    });

    const stage = element("div", "", "rp-stage");
    stage.append(canvas);
    const timeline = row(previous, slider, next, frameInput, time);
    timeline.classList.add("rp-timeline");
    const view = element("section", "", "rp-view");
    view.append(stage, timeline, anchors, row(track, clearCheckButton, viewStatus));

    const settings = element("details");
    settings.append(element("summary", "Detection confidence and tracking range"),
        row(label("Confidence", threshold)), row(label("From", from), label("To", to)));
    groupSettings.append(name, row(mode, color), row(label("Expand (px)", expansion), label("Fill holes", fillHoles)));
    const draft = element("section", "", "rp-group-content");
    draft.append(row(prompt, detect), status, candidateList, row(add, reset), settings);
    const renameButton = button("✎", () => {
        name.hidden = false;
        name.value = activeGroup.name;
        name.focus();
        name.select();
    });
    renameButton.title = "Rename group";
    renameButton.setAttribute("aria-label", "Rename group");
    const newButton = button("New group", () => {
        activeGroup = {
            name: "", preprocessor: "segmentation", color: colors[groups.length % colors.length],
            expand: defaultExpansion, fill_holes: true, parts: [],
        };
        groups.push(activeGroup);
        name.hidden = true;
        resetDraft();
        renderSettings();
        renderGroups();
        prompt.focus();
    });
    const groupHeader = row(element("h3", "Selection groups"), newButton);
    groupHeader.classList.add("rp-between");
    const side = element("aside");
    side.append(groupHeader, groupList);
    const layout = element("div", "", "rp-layout");
    layout.append(view, side);

    const header = element("header");
    const close = button("×", () => modal.close());
    close.setAttribute("aria-label", "Close editor");
    header.append(element("h2", "Video selections"), close);
    const footer = element("footer");
    footer.append(element("span", video, "rp-truncate"), row(button("Cancel", () => modal.close()),
        button("Apply selections", () => {
            const saved = groups.map((group) => ({
                ...group, name: group.parts.length ? getName(group) : group.name,
            }));
            const preview = ticket ? {
                ticket, frame: currentFrame,
                activeGroup: groups.filter(({ parts }) => parts.length).indexOf(activeGroup),
            } : null;
            onSave(saved, preview);
            modal.close();
        }, "rp-primary")));
    modal.append(element("style", editorStyle), header, layout, footer);
    modal.addEventListener("close", () => {
        life.abort();
        frameFetch?.abort();
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
    renderSettings();
    renderGroups();
    request(`/info?${new URLSearchParams({ video })}`, { signal: life.signal })
        .then((response) => response.json())
        .then((info) => {
            if (life.signal.aborted) return;
            timestamps = info.timestamps;
            sourceWidth = info.width;
            slider.max = timestamps.length - 1;
            frameInput.max = from.max = to.max = timestamps.length;
            renderSettings();
            renderGroups();
            seek(session.frame ?? activeGroup?.parts[0]?.frame ?? 0);
        })
        .catch((error) => { if (!life.signal.aborted) viewStatus.textContent = error.message; });
    return modal;
}

app.registerExtension({
    name: "RPNodes.SelectiveVideo",
    nodeCreated(node) {
        if (node.comfyClass !== "RPSelectiveVideoInput") return;
        const video = node.widgets.find((widget) => widget.name === "video");
        const selections = node.widgets.find((widget) => widget.name === "selections");
        if (!video || !selections) return;
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
            const { groups, trackingPreview } = readConfig(selections, filename);
            if (session.video !== filename) {
                session = {
                    video: filename, ticket: trackingPreview?.ticket, frame: trackingPreview?.frame,
                    signature: signature(groups),
                };
            }
            modal = openEditor(filename, groups, session, (savedGroups, preview) => {
                app.graph?.beforeChange?.();
                const data = { version: 2, video: filename, groups: savedGroups };
                if (preview) data.trackingPreview = preview;
                selections.value = JSON.stringify(data);
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
