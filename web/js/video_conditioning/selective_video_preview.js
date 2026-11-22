import { api } from "/scripts/api.js";

const route = "/rp/selective-video";
export const colors = ["#1678ff", "#ff6d32", "#58d58c", "#ec56ad", "#ffd650", "#9c83ff"];
export const defaultExpansion = 10;

export async function request(path, options) {
    const response = await api.fetchApi(route + path, options);
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        const failure = new Error(error.error || `ComfyUI returned HTTP ${response.status}.`);
        failure.status = response.status;
        throw failure;
    }
    return response;
}

export function decodeMask(encoded) {
    const bytes = Uint8Array.from(atob(encoded), (char) => char.charCodeAt(0));
    return createImageBitmap(new Blob([bytes], { type: "image/png" }));
}

function spreadMask(context, radius, horizontal) {
    const canvas = context.canvas;
    for (let step = 1; radius > 0; step *= 2) {
        const offset = Math.min(radius, step);
        const x = horizontal ? offset : 0;
        const y = horizontal ? 0 : offset;
        context.drawImage(canvas, -x, -y);
        context.drawImage(canvas, x, y);
        radius -= offset;
    }
}

export function createPainter(canvas) {
    const context = canvas.getContext("2d");
    const scratch = document.createElement("canvas");
    const tint = scratch.getContext("2d");

    return (photo, overlays) => {
        const { width, height } = photo;
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        if (scratch.width !== width || scratch.height !== height) {
            scratch.width = width;
            scratch.height = height;
        }

        context.drawImage(photo, 0, 0);
        for (const { bitmaps, color, opacity, radius = 0 } of overlays) {
            if (!bitmaps.length) continue;
            tint.clearRect(0, 0, width, height);
            for (const bitmap of bitmaps) tint.drawImage(bitmap, 0, 0, width, height);
            spreadMask(tint, radius, true);
            spreadMask(tint, radius, false);
            tint.globalCompositeOperation = "source-in";
            tint.fillStyle = color;
            tint.fillRect(0, 0, width, height);
            tint.globalCompositeOperation = "source-over";
            context.globalAlpha = opacity;
            context.drawImage(scratch, 0, 0);
        }
        context.globalAlpha = 1;
    };
}

export async function loadGroupOverlays(groups, frame, tracked, activeGroup, scale, readAnchor) {
    return Promise.all(groups.map(async (group, index) => {
        const owned = [];
        const load = async (encoded) => {
            const bitmap = await decodeMask(encoded).catch(() => null);
            if (bitmap) owned.push(bitmap);
            return bitmap;
        };
        const bitmaps = await Promise.all([
            ...group.parts.filter((part) => part.frame === frame && part.preview)
                .map((part) => (readAnchor ?? load)(part.preview)),
            tracked[index] ? load(tracked[index]) : null,
        ]);
        return {
            bitmaps: bitmaps.filter(Boolean),
            color: group.color ?? colors[0],
            opacity: index === activeGroup ? 0.55 : 0.38,
            radius: Math.round((group.expand ?? defaultExpansion) * scale),
            owned,
        };
    }));
}

export function releaseOverlays(overlays) {
    for (const { owned } of overlays) for (const bitmap of owned) bitmap.close();
}

export async function loadPhoto(video, frame, signal) {
    const query = new URLSearchParams({ video, frame });
    const response = await request(`/frame?${query}`, { signal });
    return createImageBitmap(await response.blob());
}

export async function loadTrackedMasks(ticket, frame, signal) {
    const query = new URLSearchParams({ ticket, frame });
    const response = await request(`/track-frame?${query}`, { signal });
    return (await response.json()).masks;
}

export function addNodePreview(node) {
    const container = document.createElement("div");
    container.className = "rp-selective-node-preview";
    container.setAttribute("aria-label", "Tracked video preview");
    Object.assign(container.style, { width: "100%", display: "none", overflow: "hidden", fontSize: "12px" });
    const canvas = document.createElement("canvas");
    Object.assign(canvas.style, { width: "100%", height: "100%", objectFit: "contain", display: "block" });
    const viewport = document.createElement("div");
    Object.assign(viewport.style, { width: "100%", background: "#111", borderRadius: "5px", overflow: "hidden" });
    viewport.append(canvas);
    const scrubber = document.createElement("input");
    scrubber.type = "range";
    scrubber.min = 0;
    scrubber.setAttribute("aria-label", "Tracked video frame");
    Object.assign(scrubber.style, { flex: "1", minWidth: "48px", margin: "0" });
    const frameLabel = document.createElement("span");
    frameLabel.setAttribute("role", "status");
    Object.assign(frameLabel.style, { overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "var(--fg-color, #dedee0)" });
    const bar = document.createElement("div");
    Object.assign(bar.style, { display: "flex", alignItems: "center", gap: "6px", minHeight: "28px" });
    bar.append(scrubber, frameLabel);
    container.append(viewport, bar);

    const paint = createPainter(canvas);
    let currentConfig;
    let lifetime;
    let frameRequest;
    let scheduled;
    let aspect = 16 / 9;

    const mediaHeight = (width) => Math.min(280, Math.max(90, (width - 20) / aspect));
    const widget = node.addDOMWidget("rp-tracked-video", "preview", container, {
        serialize: false, canvasOnly: true, hideOnZoom: false,
    });
    widget.serialize = false;
    widget.hidden = true;
    const height = (width) => currentConfig ? mediaHeight(width) + 30 + 2 * (widget.margin ?? 10) : 0;
    widget.computeLayoutSize = () => {
        const width = node.size[0];
        viewport.style.height = `${mediaHeight(width)}px`;
        return { minHeight: height(width), minWidth: 0 };
    };
    widget.computeSize = (width = node.size[0]) => {
        viewport.style.height = `${mediaHeight(width)}px`;
        return [width, currentConfig ? height(width) : -4];
    };

    const sourceVisibility = new WeakMap();
    function syncSourcePreview() {
        const source = node.widgets?.find((item) => item.name === "video-preview");
        if (!source) return;
        if (currentConfig && !sourceVisibility.has(source)) {
            sourceVisibility.set(source, source.hidden);
            source.hidden = true;
            source.element?.querySelector("video")?.pause();
        } else if (!currentConfig && sourceVisibility.has(source)) {
            source.hidden = sourceVisibility.get(source);
            sourceVisibility.delete(source);
        }
    }

    const addWidget = node.addDOMWidget;
    node.addDOMWidget = function (name, ...args) {
        const added = addWidget.call(this, name, ...args);
        if (name === "video-preview" && currentConfig) queueMicrotask(fitNode);
        return added;
    };

    function fitNode() {
        syncSourcePreview();
        viewport.style.height = `${mediaHeight(node.size[0])}px`;
        const size = node.computeSize();
        if (size[1] > 0) node.setSize([node.size[0], size[1]]);
        node.setDirtyCanvas(true, true);
    }

    async function showFrame(index) {
        const config = currentConfig;
        if (!config?.timestamps) return;
        const frame = Math.max(0, Math.min(config.timestamps.length - 1, index));
        frameRequest?.abort();
        const controller = new AbortController();
        frameRequest = controller;
        const { signal } = controller;
        scrubber.value = frame;
        frameLabel.textContent = `Loading #${frame + 1}…`;
        let photo;
        let overlays = [];
        try {
            const masksRequest = loadTrackedMasks(config.preview.ticket, frame, signal);
            const results = await Promise.allSettled([loadPhoto(config.video, frame, signal), masksRequest]);
            if (results[0].status === "fulfilled") photo = results[0].value;
            const failure = results.find((item) => item.status === "rejected");
            if (failure) throw failure.reason;
            if (signal.aborted) return;
            overlays = await loadGroupOverlays(config.groups, frame, results[1].value,
                config.preview.activeGroup, photo.width / config.sourceWidth);
            if (signal.aborted) return;
            paint(photo, overlays);
            canvas.style.visibility = "visible";
            frameLabel.title = "";
            frameLabel.textContent = `#${frame + 1} / ${config.timestamps.length}`;
        } catch (error) {
            if (signal.aborted) return;
            canvas.style.visibility = "hidden";
            frameLabel.title = error.message;
            frameLabel.textContent = error.status === 404 ? "Preview expired" : error.message;
            scrubber.disabled = true;
        } finally {
            photo?.close();
            releaseOverlays(overlays);
        }
    }

    scrubber.addEventListener("input", () => {
        clearTimeout(scheduled);
        scheduled = setTimeout(() => showFrame(scrubber.valueAsNumber), 65);
    });
    scrubber.addEventListener("change", () => {
        clearTimeout(scheduled);
        showFrame(scrubber.valueAsNumber);
    });

    function disconnect() {
        lifetime?.abort();
        frameRequest?.abort();
        clearTimeout(scheduled);
    }

    return {
        async update(video, groups, preview) {
            const key = preview?.ticket ? JSON.stringify([video, preview, groups]) : null;
            if (key === (currentConfig?.key ?? null)) return;
            disconnect();
            currentConfig = key ? { key, video, groups, preview } : null;
            container.style.display = key ? "block" : "none";
            widget.hidden = !key;
            fitNode();
            if (!key) return;
            const config = currentConfig;
            lifetime = new AbortController();
            const { signal } = lifetime;
            canvas.style.visibility = "hidden";
            scrubber.disabled = true;
            frameLabel.title = "";
            frameLabel.textContent = "Reading tracked video…";
            try {
                const query = new URLSearchParams({ video });
                const response = await request(`/info?${query}`, { signal });
                const { timestamps, width, height } = await response.json();
                if (signal.aborted) return;
                config.timestamps = timestamps;
                config.sourceWidth = width;
                aspect = width / height;
                scrubber.max = timestamps.length - 1;
                scrubber.disabled = false;
                fitNode();
                await showFrame(Number.isInteger(preview.frame) ? preview.frame : 0);
            } catch (error) {
                if (!signal.aborted) frameLabel.textContent = error.message;
            }
        },
        disconnect,
    };
}
