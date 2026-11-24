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


function addVideoPreview(node) {
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
        if (params?.filename) {
            const query = new URLSearchParams({ ...params, timestamp: Date.now() });
            video.src = api.apiURL(`/view?${query}`);
            container.style.display = "block";
        } else {
            video.removeAttribute("src");
            container.style.display = "none";
        }
        video.load();
        fitNodeHeight(node);
    };
    chainCallback(widget, "onRemove", () => {
        video.removeAttribute("src");
        video.load();
    });
}


function addStreamingVideoUpload(node) {
    const widget = node.widgets?.find((item) => item.name === "video");
    if (!widget) return;
    addVideoPreview(node);

    const updatePreview = () => {
        const parts = String(widget.value ?? "").replaceAll("\\", "/").split("/");
        const filename = parts.pop();
        node.rpShowOutputVideo({ filename, subfolder: parts.join("/"), type: "input" });
    };
    chainCallback(widget, "callback", updatePreview);
    chainCallback(node, "onConfigure", updatePreview);
    queueMicrotask(updatePreview);

    const picker = document.createElement("input");
    picker.type = "file";
    picker.accept = ".avi,.m2ts,.m4v,.mkv,.mov,.mp4,.mpeg,.mpg,.mts,.webm,.wmv";
    const upload = node.addWidget(
        "button", "choose video to upload", null, () => picker.click(), { serialize: false },
    );
    picker.addEventListener("change", async () => {
        const file = picker.files?.[0];
        picker.value = "";
        if (!file) return;

        upload.name = "uploading video…";
        node.graph?.setDirtyCanvas(true, true);
        picker.disabled = true;
        try {
            const query = new URLSearchParams({ filename: file.name });
            const response = await api.fetchApi(`/rp/video/upload?${query}`, { method: "POST", body: file });
            if (!response.ok) throw new Error(await response.text());
            const { video } = await response.json();
            const values = widget.options.values;
            if (!values.includes(video)) widget.options.values = [...values, video];
            widget.value = video;
            widget.callback?.(video);
        } catch (error) {
            app.extensionManager.toast.add({ severity: "error", summary: "RP video upload", detail: error.message });
        } finally {
            picker.disabled = false;
            upload.name = "choose video to upload";
            node.graph?.setDirtyCanvas(true, true);
        }
    });
}


app.registerExtension({
    name: "RPNodes.VideoPreviews",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "RPLoadVideo") {
            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                addStreamingVideoUpload(this);
            });
        }

        if (nodeData.name === "RPPreviewVideo") {
            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                addVideoPreview(this);
            });
            chainCallback(nodeType.prototype, "onExecuted", function (message) {
                const preview = message?.rp_video?.[0];
                if (preview) this.rpShowOutputVideo?.(preview);
            });
        }
    },
});
