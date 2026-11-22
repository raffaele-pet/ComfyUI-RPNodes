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

function maskBytes(encoded) {
    return Uint8Array.fromBase64?.(encoded) ?? Uint8Array.from(atob(encoded), (byte) => byte.charCodeAt(0));
}

export function decodeMask(encoded) {
    return createImageBitmap(new Blob([maskBytes(encoded)], { type: "image/png" }));
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

function fillMaskHoles(context, work) {
    const { width, height } = context.canvas;
    const image = context.getImageData(0, 0, width, height);
    const rgba = image.data;
    const { visited, queue } = work;
    visited.fill(0);
    let head = 0;
    let tail = 0;
    function enqueue(index) {
        if (!visited[index] && rgba[index * 4 + 3] < 128) {
            visited[index] = 1;
            queue[tail++] = index;
        }
    }
    for (let x = 0; x < width; x += 1) {
        enqueue(x);
        enqueue(width * (height - 1) + x);
    }
    for (let y = 0; y < height; y += 1) {
        enqueue(y * width);
        enqueue(y * width + width - 1);
    }
    while (head < tail) {
        const index = queue[head++];
        const x = index % width;
        if (x > 0) enqueue(index - 1);
        if (x + 1 < width) enqueue(index + 1);
        if (index >= width) enqueue(index - width);
        if (index + width < visited.length) enqueue(index + width);
    }
    let changed = false;
    for (let index = 0; index < visited.length; index += 1) {
        const offset = index * 4;
        if (!visited[index] && rgba[offset + 3] < 128) {
            rgba.fill(255, offset, offset + 4);
            changed = true;
        }
    }
    if (changed) context.putImageData(image, 0, 0);
}

export function createPainter(canvas) {
    const context = canvas.getContext("2d");
    const scratch = document.createElement("canvas");
    const tint = scratch.getContext("2d", { willReadFrequently: true });
    let work;

    return (photo, overlays, width = photo.width, height = photo.height) => {
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        if (scratch.width !== width || scratch.height !== height) {
            scratch.width = width;
            scratch.height = height;
            work = null;
        }

        context.drawImage(photo, 0, 0, width, height);
        for (const { bitmaps, color, opacity, radius = 0, fillHoles = false } of overlays) {
            if (!bitmaps.length) continue;
            tint.clearRect(0, 0, width, height);
            for (const bitmap of bitmaps) tint.drawImage(bitmap, 0, 0, width, height);
            if (fillHoles) {
                work ??= { visited: new Uint8Array(width * height), queue: new Int32Array(width * height) };
                fillMaskHoles(tint, work);
            }
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
            fillHoles: group.fill_holes ?? true,
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

const playbackBlockSize = 24;

function sourceUrl(video) {
    const parts = video.replaceAll("\\", "/").split("/");
    const filename = parts.pop();
    const query = new URLSearchParams({ filename, subfolder: parts.join("/"), type: "input" });
    return api.apiURL(`/view?${query}`);
}

function frameAtTime(timestamps, time) {
    let low = 0;
    let high = timestamps.length;
    while (high - low > 1) {
        const mid = (low + high) >> 1;
        if (timestamps[mid] <= time + 0.001) low = mid;
        else high = mid;
    }
    return low;
}

function timecode(seconds) {
    const whole = Math.floor(seconds);
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

function packedOverlay(config, group, index, frame, scale) {
    if (!group || frame < group.start || frame >= group.start + group.count) return null;
    const width = group.stride * 8;
    const height = group.height;
    let surface = config.maskSurfaces[index];
    if (!surface || surface.canvas.width !== width || surface.canvas.height !== height) {
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext("2d");
        const image = context.createImageData(width, height);
        image.data.fill(255);
        surface = { canvas, context, image };
        config.maskSurfaces[index] = surface;
    }
    const base = (frame - group.start) * group.stride * height;
    const alpha = surface.image.data;
    let pixel = 3;
    for (let offset = base; offset < base + group.stride * height; offset += 1) {
        const value = group.bits[offset];
        for (let bit = 0; bit < 8; bit += 1) {
            alpha[pixel] = (value & (1 << bit)) ? 255 : 0;
            pixel += 4;
        }
    }
    surface.context.putImageData(surface.image, 0, 0);
    const style = config.groups[index];
    return {
        bitmaps: [surface.canvas],
        color: style.color ?? colors[0],
        opacity: index === config.preview.activeGroup ? 0.55 : 0.38,
        radius: Math.round((style.expand ?? defaultExpansion) * scale),
        fillHoles: style.fill_holes ?? true,
    };
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
    const playback = document.createElement("video");
    playback.preload = "none";
    playback.playsInline = true;
    playback.setAttribute("aria-hidden", "true");
    Object.assign(playback.style, { position: "absolute", opacity: "0", width: "1px", height: "1px", pointerEvents: "none" });
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.textContent = "▶";
    toggle.setAttribute("aria-label", "Play tracked video");
    Object.assign(toggle.style, {
        padding: "2px 5px", minWidth: "24px", borderRadius: "4px", border: "1px solid var(--border-color, #666)",
        background: "var(--comfy-input-bg, #242424)", color: "var(--fg-color, #dedee0)", cursor: "pointer",
    });
    const playbackTime = document.createElement("span");
    Object.assign(playbackTime.style, { whiteSpace: "nowrap", color: "var(--fg-color, #dedee0)" });

    const frameLabel = document.createElement("span");
    frameLabel.setAttribute("role", "status");
    Object.assign(frameLabel.style, { overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "var(--fg-color, #dedee0)" });
    const bar = document.createElement("div");
    Object.assign(bar.style, { display: "flex", alignItems: "center", gap: "6px", minHeight: "28px" });
    bar.append(toggle, playbackTime, scrubber, frameLabel);
    container.append(playback, viewport, bar);

    const paint = createPainter(canvas);
    let currentConfig;
    let lifetime;
    let frameRequest;
    let scheduled;
    let aspect = 16 / 9;
    let playing = false;
    let animation;
    let playVersion = 0;
    let paintedFrame = -1;

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
        updatePosition(config, frame);
        frameLabel.textContent = `Loading #${frame + 1}…`;
        getBlock(config, frame).promise.catch(() => {});
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
            toggle.disabled = true;
        } finally {
            photo?.close();
            releaseOverlays(overlays);
        }
    }

    function updatePosition(config, frame) {
        scrubber.value = frame;
        const timestamps = config.timestamps;
        const duration = Number.isFinite(playback.duration) ? playback.duration
            : timestamps.at(-1) + (timestamps[1] ?? 0);
        playbackTime.textContent = `${timecode(timestamps[frame])} / ${timecode(duration + 0.001)}`;
        frameLabel.textContent = `#${frame + 1} / ${timestamps.length}`;
    }

    function getBlock(config, frame) {
        const start = Math.floor(frame / playbackBlockSize) * playbackBlockSize;
        let entry = config.blocks.get(start);
        if (!entry) {
            entry = {};
            const query = new URLSearchParams({ ticket: config.preview.ticket, start });
            entry.promise = request(`/track-block?${query}`, { signal: config.signal })
                .then((response) => response.json())
                .then(({ groups }) => {
                    entry.groups = groups.map((group) => group && {
                        ...group, bits: maskBytes(group.bits),
                    });
                    return entry.groups;
                })
                .catch((error) => {
                    if (config.blocks.get(start) === entry) config.blocks.delete(start);
                    throw error;
                });
        }
        config.blocks.delete(start);
        config.blocks.set(start, entry);
        if (config.blocks.size > 4) config.blocks.delete(config.blocks.keys().next().value);
        return entry;
    }

    function cancelPlaybackFrame() {
        if (animation == null) return;
        if (playback.cancelVideoFrameCallback) playback.cancelVideoFrameCallback(animation);
        else cancelAnimationFrame(animation);
        animation = null;
    }

    function schedulePlaybackFrame(version) {
        if (!playing || version !== playVersion) return;
        animation = playback.requestVideoFrameCallback
            ? playback.requestVideoFrameCallback((_, { mediaTime }) => renderPlayback(version, mediaTime))
            : requestAnimationFrame(() => renderPlayback(version, playback.currentTime));
    }

    function stopPlayback() {
        playing = false;
        playVersion += 1;
        cancelPlaybackFrame();
        playback.pause();
        toggle.textContent = "▶";
        toggle.setAttribute("aria-label", "Play tracked video");
    }

    function playbackFailed(error, version) {
        if (version !== playVersion) return;
        stopPlayback();
        frameLabel.textContent = error.status === 404 ? "Preview expired" : "Playback unavailable";
        frameLabel.title = error.message;
    }

    function renderPlayback(version, mediaTime) {
        animation = null;
        const config = currentConfig;
        if (!playing || version !== playVersion) return;
        if (playback.readyState < 2) {
            schedulePlaybackFrame(version);
            return;
        }
        const frame = frameAtTime(config.timestamps, mediaTime);
        if (frame !== paintedFrame) {
            const entry = getBlock(config, frame);
            if (!entry.groups) {
                playback.pause();
                frameLabel.textContent = "Buffering masks…";
                entry.promise.then(() => {
                    if (version !== playVersion) return;
                    playback.currentTime = config.timestamps[frame];
                    playback.play().then(() => schedulePlaybackFrame(version), (error) => playbackFailed(error, version));
                }).catch((error) => playbackFailed(error, version));
                return;
            }
            const scale = Math.min(1, 640 / Math.max(playback.videoWidth, playback.videoHeight));
            const width = Math.max(1, Math.round(playback.videoWidth * scale));
            const height = Math.max(1, Math.round(playback.videoHeight * scale));
            const overlays = entry.groups.map((group, index) =>
                packedOverlay(config, group, index, frame, width / config.sourceWidth)).filter(Boolean);
            paint(playback, overlays, width, height);
            canvas.style.visibility = "visible";
            updatePosition(config, frame);
            paintedFrame = frame;

            const next = Math.floor(frame / playbackBlockSize + 1) * playbackBlockSize;
            if (next < config.timestamps.length && next - frame <= 12) {
                getBlock(config, next).promise.catch(() => {});
            }
        }
        schedulePlaybackFrame(version);
    }

    function startPlayback(frame) {
        const config = currentConfig;
        if (!config?.timestamps) return;
        const index = Math.max(0, Math.min(config.timestamps.length - 1, frame));
        const version = ++playVersion;
        cancelPlaybackFrame();
        frameRequest?.abort();
        playing = true;
        paintedFrame = -1;
        toggle.textContent = "⏸";
        toggle.setAttribute("aria-label", "Pause tracked video");
        frameLabel.title = "";
        updatePosition(config, index);
        playback.currentTime = config.timestamps[index];
        playback.play().then(() => schedulePlaybackFrame(version), (error) => playbackFailed(error, version));
    }

    toggle.addEventListener("click", () => {
        clearTimeout(scheduled);
        if (playing) {
            stopPlayback();
            showFrame(scrubber.valueAsNumber);
        } else {
            const frame = scrubber.valueAsNumber;
            startPlayback(frame === Number(scrubber.max) ? 0 : frame);
        }
    });
    playback.addEventListener("error", () => {
        if (playing) {
            playbackFailed(new Error(playback.error?.message || "The browser could not decode this video."), playVersion);
        }
    });
    playback.addEventListener("ended", () => {
        if (!playing) return;
        stopPlayback();
        showFrame(Number(scrubber.max));
    });

    function seekFromSlider() {
        if (playing) startPlayback(scrubber.valueAsNumber);
        else showFrame(scrubber.valueAsNumber);
    }

    scrubber.addEventListener("input", () => {
        clearTimeout(scheduled);
        scheduled = setTimeout(seekFromSlider, 65);
    });
    scrubber.addEventListener("change", () => {
        clearTimeout(scheduled);
        seekFromSlider();
    });

    function disconnect() {
        stopPlayback();
        lifetime?.abort();
        frameRequest?.abort();
        clearTimeout(scheduled);
        playback.removeAttribute("src");
        playback.load();
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
            config.signal = signal;
            config.blocks = new Map();
            config.maskSurfaces = [];
            playback.src = sourceUrl(video);
            canvas.style.visibility = "hidden";
            scrubber.disabled = true;
            toggle.disabled = true;
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
                toggle.disabled = false;
                fitNode();
                await showFrame(Number.isInteger(preview.frame) ? preview.frame : 0);
            } catch (error) {
                if (!signal.aborted) frameLabel.textContent = error.message;
            }
        },
        disconnect,
    };
}
