import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";


const NODE_NAME = "RPPreviewImage";
const NATIVE_PREVIEW_WIDGET = "$$canvas-image-preview";
const INFO_HEIGHT = 24;


function imageUrl(data) {
    const query = new URLSearchParams({
        filename: data.filename,
        type: data.type,
        subfolder: data.subfolder ?? "",
    });
    return api.apiURL(
        `/view?${query}${app.getPreviewFormatParam?.() ?? ""}${app.getRandParam?.() ?? ""}`,
    );
}


function aspectRatio(width, height) {
    let a = width;
    let b = height;
    while (b !== 0) [a, b] = [b, a % b];
    return `${width / a}:${height / a}`;
}


function removeNativePreview(node) {
    const index = node.widgets?.findIndex((widget) => widget.name === NATIVE_PREVIEW_WIDGET) ?? -1;
    if (index < 0) return;
    node.widgets[index].onRemove?.();
    node.widgets.splice(index, 1);
}


class PreviewWidget {
    constructor(node) {
        this.name = "rp_preview_image";
        this.type = "custom";
        this.options = { serialize: false };
        this.value = null;
        this.node = node;
    }

    draw(ctx, node, width, y) {
        const images = node.imgs ?? [];
        const index = Math.min(node.imageIndex ?? 0, images.length - 1);
        const image = images[index];
        const previewHeight = Math.max(1, node.size[1] - y - INFO_HEIGHT);
        this.previewY = y;
        this.previewBottom = y + previewHeight;
        if (!image?.naturalWidth || !image?.naturalHeight) return;

        const scale = Math.min(width / image.naturalWidth, previewHeight / image.naturalHeight);
        const drawWidth = image.naturalWidth * scale;
        const drawHeight = image.naturalHeight * scale;
        ctx.drawImage(
            image,
            (width - drawWidth) / 2,
            y + (previewHeight - drawHeight) / 2,
            drawWidth,
            drawHeight,
        );

        ctx.save();
        ctx.fillStyle = "rgba(190, 190, 190, 0.9)";
        ctx.font = "14px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(
            `${image.naturalWidth} × ${image.naturalHeight} | `
                + aspectRatio(image.naturalWidth, image.naturalHeight),
            width / 2,
            node.size[1] - INFO_HEIGHT / 2,
        );

        if (images.length > 1) {
            ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
            ctx.font = "24px Arial";
            ctx.fillText("‹", 20, y + previewHeight / 2);
            ctx.fillText("›", width - 20, y + previewHeight / 2);
            ctx.font = "12px Arial";
            ctx.textAlign = "right";
            ctx.fillText(`${index + 1}/${images.length}`, width - 8, y + 12);
        }
        ctx.restore();
    }

    mouse(event, pos) {
        if (event.type !== "pointerdown" || (event.button != null && event.button !== 0)) {
            return false;
        }
        const images = this.node.imgs ?? [];
        if (images.length < 2 || pos[1] < this.previewY || pos[1] > this.previewBottom) {
            return false;
        }
        const direction = pos[0] <= 40 ? -1 : pos[0] >= this.node.size[0] - 40 ? 1 : 0;
        if (!direction) return false;
        this.node.imageIndex = ((this.node.imageIndex ?? 0) + direction + images.length)
            % images.length;
        this.node.setDirtyCanvas?.(true, true);
        return true;
    }

    computeSize(width) {
        return [width, 20];
    }
}


app.registerExtension({
    name: "RPNodes.PreviewImage",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            this.rpPreviewImages = this.imgs ?? [];
            this.rpPreviewControlsImages = false;
            Object.defineProperty(this, "imgs", {
                configurable: true,
                enumerable: true,
                get: () => this.rpPreviewImages,
                set: (images) => {
                    if (!this.rpPreviewControlsImages) this.rpPreviewImages = images ?? [];
                },
            });
            this.rpPreviewIndex = 0;
            Object.defineProperty(this, "imageIndex", {
                configurable: true,
                enumerable: true,
                get: () => this.rpPreviewIndex,
                set: (index) => {
                    this.rpPreviewIndex = Number.isInteger(index) && index >= 0 ? index : 0;
                },
            });
            const originalAddCustomWidget = this.addCustomWidget.bind(this);
            this.addCustomWidget = (widget) => widget?.name === NATIVE_PREVIEW_WIDGET
                ? widget
                : originalAddCustomWidget(widget);
            removeNativePreview(this);
            this.addCustomWidget(new PreviewWidget(this));
            const computed = this.computeSize?.() ?? this.size ?? [320, 360];
            this.setSize?.([Math.max(320, computed[0] ?? 0), Math.max(360, computed[1] ?? 0)]);
            return result;
        };

        const originalExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            const result = originalExecuted?.apply(this, arguments);
            this.rpPreviewControlsImages = true;
            this.imageIndex = 0;
            this.rpPreviewImages = (output.images ?? []).map((data) => {
                const image = new Image();
                image.onload = () => this.setDirtyCanvas?.(true, true);
                image.src = imageUrl(data);
                return image;
            });
            removeNativePreview(this);
            queueMicrotask(() => removeNativePreview(this));
            this.setDirtyCanvas?.(true, true);
            return result;
        };
    },
});
