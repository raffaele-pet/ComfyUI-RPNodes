import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";


const MAX_PREVIEW_HEIGHT = 160;


function chainCallback(object, property, callback) {
    const original = object?.[property];
    object[property] = function () {
        const result = original?.apply(this, arguments);
        return callback.apply(this, arguments) ?? result;
    };
}


function fitNodeHeight(node) {
    const computed = node.computeSize?.([node.size[0], node.size[1]]);
    if (computed?.[1] > 0) {
        node.setSize([node.size[0], computed[1]]);
    }
    node.graph?.setDirtyCanvas(true, true);
}


function previewHeight(node, aspectRatio) {
    const width = Math.max(1, node.size[0] - 20);
    const naturalHeight = aspectRatio > 0 ? width / aspectRatio : MAX_PREVIEW_HEIGHT;
    return Math.max(60, Math.min(MAX_PREVIEW_HEIGHT, naturalHeight));
}


function patchInputPreview(node, attempts = 180) {
    const widget = node.widgets?.find((item) => item.name === "video-preview");
    if (!widget) {
        if (attempts > 0) {
            requestAnimationFrame(() => patchInputPreview(node, attempts - 1));
        }
        return;
    }
    if (widget.rpPreviewPatched) return;
    widget.rpPreviewPatched = true;

    const applyMediaStyle = () => {
        const videos = node.videoContainer?.querySelectorAll?.("video") ?? [];
        for (const video of videos) {
            Object.assign(video.style, {
                width: "100%",
                height: "auto",
                maxHeight: `${MAX_PREVIEW_HEIGHT}px`,
                objectFit: "contain",
                display: "block",
            });
        }
        if (node.videoContainer) {
            node.videoContainer.style.maxHeight = `${MAX_PREVIEW_HEIGHT}px`;
            node.videoContainer.style.overflow = "hidden";
        }
    };

    widget.computeLayoutSize = () => {
        const video = node.videoContainer?.querySelector?.("video");
        const aspect = video?.videoWidth && video?.videoHeight
            ? video.videoWidth / video.videoHeight
            : 16 / 9;
        return { minHeight: previewHeight(node, aspect), minWidth: 0 };
    };
    widget.computeSize = (width) => {
        const video = node.videoContainer?.querySelector?.("video");
        const aspect = video?.videoWidth && video?.videoHeight
            ? video.videoWidth / video.videoHeight
            : 16 / 9;
        return [width, previewHeight(node, aspect)];
    };

    applyMediaStyle();
    const observer = new MutationObserver(() => {
        applyMediaStyle();
        fitNodeHeight(node);
    });
    if (node.videoContainer) {
        observer.observe(node.videoContainer, { childList: true, subtree: true });
    }
    const originalRemove = widget.onRemove;
    widget.onRemove = function () {
        observer.disconnect();
        return originalRemove?.apply(this, arguments);
    };
    fitNodeHeight(node);
}


function addOutputPreview(node) {
    const container = document.createElement("div");
    container.style.width = "100%";
    container.style.display = "none";
    container.style.overflow = "hidden";

    const video = document.createElement("video");
    video.controls = true;
    video.loop = true;
    video.playsInline = true;
    video.preload = "metadata";
    Object.assign(video.style, {
        width: "100%",
        height: "auto",
        maxHeight: `${MAX_PREVIEW_HEIGHT}px`,
        objectFit: "contain",
        display: "block",
    });
    container.appendChild(video);

    const widget = node.addDOMWidget("rp-video-preview", "preview", container, {
        serialize: false,
        canvasOnly: true,
        hideOnZoom: false,
    });
    widget.serialize = false;
    widget.computeLayoutSize = () => ({
        minHeight: container.style.display === "none"
            ? 0
            : previewHeight(node, video.videoWidth / video.videoHeight),
        minWidth: 0,
    });
    widget.computeSize = (width) => [
        width,
        container.style.display === "none"
            ? -4
            : previewHeight(node, video.videoWidth / video.videoHeight),
    ];

    video.addEventListener("loadedmetadata", () => fitNodeHeight(node));
    node.rpShowOutputVideo = (params) => {
        if (!params?.filename) return;
        const query = new URLSearchParams({ ...params, timestamp: Date.now() });
        video.src = api.apiURL(`/view?${query}`);
        container.style.display = "block";
        video.load();
        fitNodeHeight(node);
    };
}


app.registerExtension({
    name: "RPNodes.VideoPreviews",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "RPVideoToFrames") {
            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                const originalAddDOMWidget = this.addDOMWidget;
                this.addDOMWidget = function (name) {
                    const widget = originalAddDOMWidget.apply(this, arguments);
                    if (name === "video-preview") {
                        requestAnimationFrame(() => patchInputPreview(this));
                    }
                    return widget;
                };
                const videoWidget = this.widgets?.find((widget) => widget.name === "video");
                if (videoWidget) {
                    chainCallback(videoWidget, "callback", () => {
                        requestAnimationFrame(() => patchInputPreview(this));
                    });
                }
                requestAnimationFrame(() => patchInputPreview(this));
            });
        }

        if (nodeData.name === "RPFramesToVideo") {
            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                addOutputPreview(this);
            });
            chainCallback(nodeType.prototype, "onExecuted", function (message) {
                const preview = message?.rp_video?.[0];
                if (preview) this.rpShowOutputVideo?.(preview);
            });
        }
    },
});
