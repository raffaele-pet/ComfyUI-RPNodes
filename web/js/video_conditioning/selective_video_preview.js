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

export async function post(path, payload, signal) {
    const response = await request(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal,
    });
    return response.json();
}

export async function previewJob(path, payload, signal) {
    let query;
    try {
        const started = await post(path, payload, signal);
        query = new URLSearchParams({ job: started.job });
        while (!signal.aborted) {
            const response = await request(`/track-status?${query}`, { signal, cache: "no-store" });
            const result = await response.json();
            if (result.done) {
                if (result.error) throw new Error(result.error);
                return result;
            }
            await new Promise((resolve) => setTimeout(resolve, 450));
        }
        signal.throwIfAborted();
    } finally {
        if (signal.aborted && query) {
            request(`/track?${query}`, { method: "DELETE", keepalive: true }).catch(() => {});
        }
    }
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

    return (photo, overlays) => {
        const { width, height } = photo;
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

export function attachNativePreview(node, setStatus) {
    let key = null;
    let controller;
    let target;
    let replaced;
    const observer = new MutationObserver(show);

    function restore() {
        if (!replaced) return;
        const { player, original, url } = replaced;
        replaced = null;
        if (player.getAttribute("src") !== url) return;
        player.pause();
        if (original) player.setAttribute("src", original);
        else player.removeAttribute("src");
    }

    function show() {
        const player = node.videoContainer?.querySelector("video");
        if (!target || !player || player.getAttribute("src") === target.url) return;
        const { url, time } = target;
        replaced = { player, original: player.getAttribute("src"), url };
        player.addEventListener("loadedmetadata", () => {
            if (player.getAttribute("src") === url) player.currentTime = time;
        }, { once: true, signal: controller.signal });
        player.pause();
        player.setAttribute("src", url);
    }

    let container;
    function watch(element) {
        if (!element || element === container) return;
        observer.disconnect();
        container = element;
        observer.observe(container, { childList: true, subtree: true, attributes: true, attributeFilter: ["src"] });
        show();
    }

    const addDOMWidget = node.addDOMWidget;
    node.addDOMWidget = function (name, type, element, ...args) {
        const widget = addDOMWidget.call(this, name, type, element, ...args);
        if (name === "video-preview") watch(element);
        return widget;
    };
    watch(node.videoContainer);

    return {
        async update(video, groups, preview) {
            watch(node.videoContainer);
            const nextKey = preview?.ticket ? JSON.stringify([video, groups, preview]) : null;
            if (nextKey === key) return;
            key = nextKey;
            controller?.abort();
            target = null;
            restore();
            if (!key) {
                setStatus("");
                return;
            }

            controller = new AbortController();
            const { signal } = controller;
            setStatus("Preparing preview…");
            try {
                const result = await previewJob("/preview-video", { video, groups, ...preview }, signal);
                if (signal.aborted) return;
                const { time, ...params } = result.video;
                target = { url: api.apiURL(`/view?${new URLSearchParams(params)}`), time };
                show();
                setStatus("");
            } catch (error) {
                if (!signal.aborted) setStatus(error.message);
            }
        },
        disconnect() {
            controller?.abort();
            observer.disconnect();
            key = target = null;
            container = null;
            restore();
        },
    };
}
