import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const route = "/rp/selective-video";
const colors = ["#1678ff", "#ff6d32", "#58d58c", "#ec56ad", "#ffd650", "#9c83ff"];
const maps = {
    segmentation: "Color segmentation",
    depth: "Depth",
    canny: "Canny",
    openpose: "OpenPose",
};

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

async function request(path, options) {
    const response = await api.fetchApi(route + path, options);
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.error || `ComfyUI returned HTTP ${response.status}.`);
    }
    return response;
}

function readLayers(widget, video) {
    try {
        const data = JSON.parse(widget.value);
        return data.video === video && Array.isArray(data.layers) ? data.layers : [];
    } catch {
        return [];
    }
}

function openEditor(video, initialLayers, onSave) {
    const modal = element("dialog", "", "rp-selective-editor");
    modal.setAttribute("aria-label", "Selective Video · Selection editor");
    const style = element("style");
    style.textContent = `
        .rp-selective-editor { box-sizing:border-box; width:min(1120px,96vw); max-height:94dvh; overflow:auto;
            color:var(--fg-color,#ebedf1); background:var(--comfy-menu-bg,#23242a);
            border:1px solid var(--border-color,#545761); border-radius:14px; padding:20px;
            font:14px/1.4 system-ui,sans-serif; z-index:1000; }
        .rp-selective-editor::backdrop { background:#000a; }
        .rp-selective-editor * { box-sizing:border-box; }
        .rp-selective-editor h2,.rp-selective-editor h3 { margin:0; }
        .rp-selective-editor header,.rp-selective-editor footer,.rp-selective-editor .rp-row {
            display:flex; align-items:center; gap:8px; }
        .rp-selective-editor header,.rp-selective-editor footer { justify-content:space-between; }
        .rp-selective-editor footer { margin-top:14px; }
        .rp-selective-editor .rp-layout { display:grid; grid-template-columns:minmax(0,1fr) minmax(280px,325px);
            gap:18px; margin-top:14px; }
        .rp-selective-editor .rp-stage { display:grid; place-items:center; background:#101115; border-radius:9px;
            min-height:150px; overflow:hidden; }
        .rp-selective-editor canvas { width:100%; max-height:60dvh; object-fit:contain; display:block; }
        .rp-selective-editor .rp-timeline { margin-top:10px; }
        .rp-selective-editor .rp-timeline input[type=range] { flex:1; min-width:55px; }
        .rp-selective-editor .rp-frame { width:70px; }
        .rp-selective-editor .rp-time { min-width:60px; text-align:right; font-variant-numeric:tabular-nums; }
        .rp-selective-editor aside { display:flex; flex-direction:column; gap:9px; min-width:0; }
        .rp-selective-editor .rp-prompt { flex:1; min-width:0; }
        .rp-selective-editor select { flex:1; min-width:0; }
        .rp-selective-editor button,.rp-selective-editor select,.rp-selective-editor input:not([type=range]):not([type=checkbox]) {
            color:inherit; border:1px solid var(--border-color,#545761); background:var(--comfy-input-bg,#191b20);
            border-radius:6px; padding:6px 9px; font:inherit; }
        .rp-selective-editor button { cursor:pointer; white-space:nowrap; }
        .rp-selective-editor button:disabled { opacity:.5; cursor:default; }
        .rp-selective-editor button.rp-primary { background:#315fbc; color:#fff; border-color:#567fd1; }
        .rp-selective-editor button:focus-visible,.rp-selective-editor input:focus-visible,.rp-selective-editor select:focus-visible {
            outline:2px solid #83a8ff; outline-offset:2px; }
        .rp-selective-editor input[type=color] { width:42px; height:34px; }
        .rp-selective-editor details .rp-row { margin-top:6px; }
        .rp-selective-editor details label { display:flex; align-items:center; gap:6px; min-width:0; }
        .rp-selective-editor details input { width:75px; }
        .rp-selective-editor .rp-status { margin:0; min-height:20px; opacity:.87; }
        .rp-selective-editor .rp-layers,.rp-selective-editor .rp-candidates {
            max-height:155px; overflow:auto; display:grid; gap:4px; }
        .rp-selective-editor .rp-layer { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; text-align:left; }
        .rp-selective-editor .rp-dot { display:inline-block; height:10px; width:10px; flex:0 0 10px; border-radius:50%; }
        @media(max-width:730px) {
            .rp-selective-editor .rp-layout { grid-template-columns:minmax(0,1fr); }
            .rp-selective-editor canvas { max-height:40dvh; }
            .rp-selective-editor { padding:12px; }
        }
    `;

    const layers = initialLayers.map((layer) => ({ ...layer }));
    const canvas = element("canvas");
    const ctx = canvas.getContext("2d");
    const tintCanvas = element("canvas");
    const tint = tintCanvas.getContext("2d");
    const slider = input("range", 0, "Video frame timeline");
    slider.min = 0;
    slider.max = 0;
    slider.disabled = true;
    const frameInput = input("number", 1, "Frame number");
    frameInput.className = "rp-frame";
    frameInput.min = 1;
    frameInput.disabled = true;
    const time = element("span", "—", "rp-time");
    const status = element("p", "Reading video…", "rp-status");
    status.setAttribute("role", "status");
    const prompt = input("text", "", "Object description");
    prompt.className = "rp-prompt";
    prompt.placeholder = "e.g. woman in a red kimono";
    prompt.maxLength = 120;
    const threshold = input("number", 0.5, "Detection confidence");
    threshold.min = 0.01;
    threshold.max = 0.99;
    threshold.step = 0.05;
    const from = input("number", "", "First frame to track");
    const to = input("number", "", "Last frame to track");
    from.min = to.min = 1;
    from.placeholder = "Start";
    to.placeholder = "End";
    const mode = element("select");
    mode.setAttribute("aria-label", "Preprocessor");
    for (const [value, title] of Object.entries(maps)) mode.append(new Option(title, value));
    const color = input("color", colors[layers.length % colors.length], "Segmentation color");
    mode.addEventListener("change", () => { color.hidden = mode.value !== "segmentation"; });

    const candidatesList = element("div", "", "rp-candidates");
    const layersList = element("div", "", "rp-layers");
    const life = new AbortController();
    let frameFetch;
    let timestamps = [];
    let currentFrame = 0;
    let shownFrame = -1;
    let photo;
    let candidates = [];
    let checked = new Set();
    let verified = false;
    let detecting = false;
    let editIndex = -1;
    let detectionVersion = 0;

    function row(...contents) {
        const node = element("div", "", "rp-row");
        node.append(...contents);
        return node;
    }

    function label(text, control) {
        const node = element("label", text, "rp-row");
        node.append(control);
        return node;
    }

    function swatch(colorValue) {
        const dot = element("span", "", "rp-dot");
        dot.style.backgroundColor = colorValue;
        return dot;
    }

    function draw() {
        if (!photo) return;
        if (canvas.width !== photo.width || canvas.height !== photo.height) {
            canvas.width = tintCanvas.width = photo.width;
            canvas.height = tintCanvas.height = photo.height;
        }
        ctx.drawImage(photo, 0, 0);
        candidates.forEach((candidate, i) => {
            tint.clearRect(0, 0, tintCanvas.width, tintCanvas.height);
            tint.drawImage(candidate.mask, 0, 0);
            tint.globalCompositeOperation = "source-in";
            tint.fillStyle = colors[i % colors.length];
            tint.fillRect(0, 0, tintCanvas.width, tintCanvas.height);
            tint.globalCompositeOperation = "source-over";
            ctx.globalAlpha = checked.has(i) ? 0.57 : 0.18;
            ctx.drawImage(tintCanvas, 0, 0);
        });
        ctx.globalAlpha = 1;
    }

    function clearDetection() {
        detectionVersion += 1;
        candidates.forEach(({ mask }) => mask.close());
        candidates = [];
        checked.clear();
        verified = false;
        candidatesList.replaceChildren();
        updateButtons();
        draw();
    }

    function updateButtons() {
        detect.disabled = detecting || shownFrame !== currentFrame || !prompt.value.trim();
        add.disabled = !verified || !checked.size || shownFrame !== currentFrame;
    }

    async function seek(index) {
        if (!timestamps.length || !Number.isFinite(index)) return;
        const target = Math.min(timestamps.length - 1, Math.max(0, index));
        if (target !== currentFrame) clearDetection();
        currentFrame = target;
        frameInput.value = target + 1;
        slider.value = target;
        time.textContent = `${timestamps[target].toFixed(2)} s`;
        previous.disabled = !target;
        next.disabled = target === timestamps.length - 1;
        frameFetch?.abort();
        const controller = new AbortController();
        frameFetch = controller;
        shownFrame = -1;
        updateButtons();
        status.textContent = "Loading frame…";
        try {
            const query = new URLSearchParams({ video, frame: target });
            const response = await request(`/frame?${query}`, { signal: controller.signal });
            const image = await createImageBitmap(await response.blob());
            if (controller.signal.aborted) {
                image.close();
                return;
            }
            photo?.close();
            photo = image;
            shownFrame = target;
            draw();
            status.textContent = `Frame ${target + 1} of ${timestamps.length}`;
        } catch (error) {
            if (!controller.signal.aborted) status.textContent = error.message;
        } finally {
            updateButtons();
        }
    }

    const detect = button("Detect", async () => {
        const previousChoices = verified ? [...checked] : null;
        clearDetection();
        const version = detectionVersion;
        detecting = true;
        updateButtons();
        status.textContent = "Detecting with SAM 3.1…";
        try {
            const response = await request("/detect", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ video, frame: currentFrame, text: prompt.value.trim(), threshold: threshold.valueAsNumber }),
                signal: life.signal,
            });
            const result = await response.json();
            const previews = await Promise.all(result.candidates.map(async ({ score, mask }) => {
                const bytes = Uint8Array.from(atob(mask), (char) => char.charCodeAt(0));
                return { score, mask: await createImageBitmap(new Blob([bytes], { type: "image/png" })) };
            }));
            if (life.signal.aborted || version !== detectionVersion) {
                previews.forEach(({ mask }) => mask.close());
                return;
            }
            candidates = previews;
            checked = new Set(previousChoices?.filter((i) => i < previews.length) ?? previews.map((_, i) => i));
            verified = previews.length > 0;
            candidates.forEach(({ score }, i) => {
                const checkbox = input("checkbox", i, `Instance ${i + 1}`);
                checkbox.checked = checked.has(i);
                checkbox.addEventListener("change", () => {
                    if (checkbox.checked) checked.add(i);
                    else checked.delete(i);
                    updateButtons();
                    draw();
                });
                const item = element("label", "", "rp-row");
                item.append(checkbox, swatch(colors[i % colors.length]), `Instance ${i + 1} · ${Math.round(score * 100)}%`);
                candidatesList.append(item);
            });
            draw();
            status.textContent = verified ? "Choose the instances to track." : "No match. Try another description, frame or confidence.";
        } catch (error) {
            if (!life.signal.aborted && version === detectionVersion) status.textContent = error.message;
        } finally {
            detecting = false;
            updateButtons();
        }
    });

    const add = button("Add layer", () => {
        const start = from.value === "" ? 0 : from.valueAsNumber - 1;
        const end = to.value === "" ? null : to.valueAsNumber - 1;
        if (!Number.isInteger(start) || start < 0 || start > currentFrame ||
            (end !== null && (!Number.isInteger(end) || end < currentFrame || end >= timestamps.length))) {
            status.textContent = "The tracking range must contain the selected frame.";
            return;
        }
        const layer = {
            text: prompt.value.trim(),
            frame: currentFrame,
            threshold: threshold.valueAsNumber,
            preprocessor: mode.value,
            color: color.value,
            instances: [...checked].sort((a, b) => a - b),
        };
        if (start) layer.start = start;
        if (end !== null) layer.end = end;
        if (editIndex === -1) layers.push(layer);
        else layers[editIndex] = layer;
        resetDraft();
        renderLayers();
        status.textContent = "Layer ready. Apply selections to update the node.";
    });

    function resetDraft() {
        editIndex = -1;
        add.textContent = "Add layer";
        prompt.value = "";
        threshold.value = 0.5;
        mode.value = "segmentation";
        color.hidden = false;
        color.value = colors[layers.length % colors.length];
        from.value = to.value = "";
        clearDetection();
    }

    function renderLayers() {
        layersList.replaceChildren();
        if (!layers.length) layersList.append(element("span", "No layers"));
        layers.forEach((layer, i) => {
            const open = button(`${layer.text} · ${maps[layer.preprocessor ?? "segmentation"]} · #${layer.frame + 1}`, async () => {
                resetDraft();
                editIndex = i;
                add.textContent = "Update layer";
                prompt.value = layer.text;
                threshold.value = layer.threshold;
                mode.value = layer.preprocessor ?? "segmentation";
                color.value = layer.color ?? colors[i % colors.length];
                color.hidden = mode.value !== "segmentation";
                from.value = layer.start ? layer.start + 1 : "";
                to.value = layer.end == null ? "" : layer.end + 1;
                const pending = seek(layer.frame);
                if (layer.instances?.length) {
                    checked = new Set(layer.instances);
                    verified = true;
                }
                await pending;
            }, "rp-layer");
            open.disabled = !timestamps.length;
            const remove = button("×", () => {
                layers.splice(i, 1);
                resetDraft();
                renderLayers();
            });
            remove.setAttribute("aria-label", `Remove ${layer.text}`);
            layersList.append(row(swatch(layer.color ?? colors[i % colors.length]), open, remove));
        });
    }

    const previous = button("‹", () => seek(currentFrame - 1));
    const next = button("›", () => seek(currentFrame + 1));
    previous.setAttribute("aria-label", "Previous frame");
    next.setAttribute("aria-label", "Next frame");
    previous.disabled = next.disabled = true;
    let scheduledSeek;
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
    prompt.addEventListener("input", clearDetection);
    threshold.addEventListener("input", clearDetection);
    prompt.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            if (!detect.disabled) detect.click();
        }
    });

    const stage = element("div", "", "rp-stage");
    stage.append(canvas);
    const timeline = row(previous, slider, next, frameInput, time);
    timeline.classList.add("rp-timeline");
    const left = element("section");
    left.append(stage, timeline);
    const settings = element("details");
    settings.append(element("summary", "Detection confidence and tracking range"),
        row(label("Confidence", threshold)), row(label("From", from), label("To", to)));
    const right = element("aside");
    right.append(row(prompt, detect), status, candidatesList, row(mode, color, add), settings,
        element("h3", "Selection layers"), layersList);

    const layout = element("div", "", "rp-layout");
    layout.append(left, right);
    const header = element("header");
    const close = button("×", () => modal.close());
    close.setAttribute("aria-label", "Close editor");
    header.append(element("h2", "Video selections"), close);
    const footer = element("footer");
    footer.append(element("span", video), row(button("Cancel", () => modal.close()),
        button("Apply selections", () => { onSave(layers); modal.close(); }, "rp-primary")));
    modal.append(style, header, layout, footer);
    modal.addEventListener("close", () => {
        life.abort();
        frameFetch?.abort();
        clearTimeout(scheduledSeek);
        photo?.close();
        candidates.forEach(({ mask }) => mask.close());
        modal.remove();
    }, { once: true });

    document.body.append(modal);
    modal.showModal();
    renderLayers();
    updateButtons();
    request(`/info?${new URLSearchParams({ video })}`, { signal: life.signal })
        .then((response) => response.json())
        .then((info) => {
            if (life.signal.aborted) return;
            timestamps = info.timestamps;
            slider.max = timestamps.length - 1;
            frameInput.max = from.max = to.max = timestamps.length;
            slider.disabled = frameInput.disabled = false;
            renderLayers();
            seek(0);
        })
        .catch((error) => { if (!life.signal.aborted) status.textContent = error.message; });
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
        const editor = node.addWidget("button", "Edit video selections", null, () => {
            if (modal?.open || !video.value) return;
            const filename = video.value;
            modal = openEditor(filename, readLayers(selections, filename), (layers) => {
                app.graph?.beforeChange?.();
                selections.value = JSON.stringify({ video: filename, layers });
                selections.callback?.(selections.value);
                refresh();
                app.graph?.afterChange?.();
            });
        }, { serialize: false });
        const refresh = () => {
            const count = readLayers(selections, video.value).length;
            editor.label = count ? `Edit video selections (${count})` : "Edit video selections";
            node.setDirtyCanvas(true, true);
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
