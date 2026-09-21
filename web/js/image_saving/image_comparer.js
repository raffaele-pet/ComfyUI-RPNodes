import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";


const NODE_NAME = "RPImageComparer";
const NATIVE_PREVIEW_WIDGET = "$$canvas-image-preview";
const RESIZE_HANDLE_SIZE = 20;
const IMAGE_INFO_HEIGHT = 24;


function imageDataToUrl(data, preview = true) {
    const query = new URLSearchParams({
        filename: data.filename,
        type: data.type,
        subfolder: data.subfolder ?? "",
    });
    const previewParams = preview ? app.getPreviewFormatParam?.() ?? "" : "";
    return api.apiURL(`/view?${query}${previewParams}${app.getRandParam?.() ?? ""}`);
}


function removeNativeCanvasPreview(node) {
    const index = node.widgets?.findIndex((widget) => widget.name === NATIVE_PREVIEW_WIDGET) ?? -1;
    if (index < 0) return;
    node.widgets[index].onRemove?.();
    node.widgets.splice(index, 1);
}


function compareModeLabel(mode) {
    if (mode === "Click") return "Compare mode: Click";
    if (mode === "Side-by-side") return "Compare mode: Side-by-side";
    return "Compare mode: Wipe";
}


function isOverComparePreview(node, pos) {
    const previewY = node.rpComparerWidget?.previewY;
    const previewBottom = node.rpComparerWidget?.previewBottom;
    return Array.isArray(pos)
        && Number.isFinite(previewY)
        && Number.isFinite(previewBottom)
        && pos[1] >= previewY
        && pos[1] <= previewBottom
        && !isOverResizeHandle(node, pos);
}


function isOverResizeHandle(node, pos) {
    return Array.isArray(pos)
        && pos[1] >= node.size[1] - RESIZE_HANDLE_SIZE
        && (
            pos[0] <= RESIZE_HANDLE_SIZE
            || pos[0] >= node.size[0] - RESIZE_HANDLE_SIZE
        );
}


function setCompareCursor(event, canvas, cursor) {
    if (cursor == null) return;
    const element = canvas?.canvas ?? event?.currentTarget ?? event?.target;
    if (element?.style) element.style.cursor = cursor;
}


function aspectRatioLabel(width, height) {
    let a = Math.abs(Math.trunc(width));
    let b = Math.abs(Math.trunc(height));
    while (b !== 0) {
        [a, b] = [b, a % b];
    }
    const divisor = Math.max(1, a);
    return `${Math.trunc(width) / divisor}:${Math.trunc(height) / divisor}`;
}


class ImageComparerWidget {
    constructor(node) {
        this.name = "rp_image_comparer";
        this.type = "custom";
        this.options = {};
        this.node = node;
        this.selected = [];
        this.hitAreas = [];
        this._value = { images: [] };
    }

    set value(value) {
        let images = Array.isArray(value) ? value : value?.images ?? [];
        images = images.map((item, index) => {
            if (!item || typeof item === "string") {
                return {
                    url: item,
                    name: index === 0 ? "A" : "B",
                    selected: true,
                };
            }
            return item;
        });

        if (images.length > 2) {
            const hasAAndB = images.some((item) => item.name.startsWith("A"))
                && images.some((item) => item.name.startsWith("B"));
            if (!hasAAndB) images = [images[0], images[1]];
        }

        let selected = images.filter((item) => item.selected);
        if (!selected.length && images.length) images[0].selected = true;
        selected = images.filter((item) => item.selected);
        if (selected.length === 1 && images.length > 1) {
            images.find((item) => !item.selected).selected = true;
        }

        this._value = { images };
        this.setSelected(images.filter((item) => item.selected));
    }

    get value() {
        return this._value;
    }

    setSelected(selected) {
        this._value.images.forEach((item) => { item.selected = false; });
        this.node.imgs.length = 0;
        for (const item of selected.slice(0, 2)) {
            if (!item.img) {
                item.img = new Image();
                item.img.onload = () => this.node.setDirtyCanvas?.(true, true);
                item.img.src = item.url;
            }
            item.img.rpDownloadUrl = item.downloadUrl ?? item.url;
            item.selected = true;
            this.node.imgs.push(item.img);
        }
        this.selected = selected.slice(0, 2);
        this.node.setDirtyCanvas?.(true, true);
    }

    selectImage(item) {
        const selected = [...this.selected];
        if (item.name.startsWith("A")) selected[0] = item;
        else if (item.name.startsWith("B")) selected[1] = item;
        this.setSelected(selected.filter(Boolean));
    }

    draw(ctx, node, width, y) {
        this.hitAreas = [];
        if (this.value.images.length > 2) {
            ctx.save();
            ctx.textAlign = "left";
            ctx.textBaseline = "top";
            ctx.font = "14px Arial";
            const spacing = 5;
            const labels = this.value.images.map((image) => ({
                image,
                width: ctx.measureText(image.name).width,
            }));
            const totalWidth = labels.reduce((total, item) => total + item.width, 0)
                + spacing * (labels.length - 1);
            let x = (node.size[0] - totalWidth) / 2;
            for (const label of labels) {
                ctx.fillStyle = label.image.selected
                    ? "rgba(180, 180, 180, 1)"
                    : "rgba(180, 180, 180, 0.5)";
                ctx.fillText(label.image.name, x, y);
                this.hitAreas.push({
                    bounds: [x, y, label.width, 14],
                    image: label.image,
                });
                x += label.width + spacing;
            }
            ctx.restore();
            y += 20;
        }
        this.previewY = y;
        this.previewBottom = Math.max(y, node.size[1] - IMAGE_INFO_HEIGHT);
        let infoImage;

        if (node.properties?.comparer_mode === "Click") {
            infoImage = this.selected[node.isClickShowingAfter ? 1 : 0];
            this.drawImage(ctx, infoImage, y);
        } else if (node.properties?.comparer_mode === "Side-by-side") {
            this.drawSideBySide(ctx, y);
            infoImage = this.selected[node.imageIndex ?? 0];
        } else {
            this.drawImage(ctx, this.selected[0], y);
            if (node.isPointerOver) {
                this.drawImage(ctx, this.selected[1], y, node.pointerOverPos[0]);
            }
            infoImage = this.selected[node.isPointerOver ? node.imageIndex : 0];
        }
        this.drawImageInfo(ctx, infoImage);
    }

    drawSideBySide(ctx, y) {
        const [nodeWidth, nodeHeight] = this.node.size;
        const gap = 2;
        const halfWidth = (nodeWidth - gap) / 2;
        const height = Math.max(1, nodeHeight - y - IMAGE_INFO_HEIGHT);
        this.drawImageInBounds(ctx, this.selected[0], 0, y, halfWidth, height);
        this.drawImageInBounds(
            ctx,
            this.selected[1],
            halfWidth + gap,
            y,
            halfWidth,
            height,
        );

        ctx.save();
        ctx.fillStyle = "rgba(255, 255, 255, 0.55)";
        ctx.fillRect(halfWidth, y, gap, height);
        ctx.restore();
    }

    drawImageInfo(ctx, image) {
        const width = image?.img?.naturalWidth;
        const height = image?.img?.naturalHeight;
        if (!width || !height) return;

        ctx.save();
        ctx.fillStyle = "rgba(190, 190, 190, 0.9)";
        ctx.font = "14px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(
            `${width} × ${height} | ${aspectRatioLabel(width, height)}`,
            this.node.size[0] / 2,
            this.node.size[1] - IMAGE_INFO_HEIGHT / 2,
        );
        ctx.restore();
    }

    drawImageInBounds(ctx, image, x, y, width, height) {
        if (!image?.img?.naturalWidth || !image?.img?.naturalHeight) return;
        const scale = Math.min(
            width / image.img.naturalWidth,
            height / image.img.naturalHeight,
        );
        const drawWidth = image.img.naturalWidth * scale;
        const drawHeight = image.img.naturalHeight * scale;
        const drawX = x + (width - drawWidth) / 2;
        const drawY = y + (height - drawHeight) / 2;

        ctx.save();
        ctx.beginPath();
        ctx.rect(x, y, width, height);
        ctx.clip();
        ctx.drawImage(image.img, drawX, drawY, drawWidth, drawHeight);
        ctx.restore();
    }

    drawImage(ctx, image, y, cropX) {
        if (!image?.img?.naturalWidth || !image?.img?.naturalHeight) return;

        const [nodeWidth, nodeHeight] = this.node.size;
        const imageAspect = image.img.naturalWidth / image.img.naturalHeight;
        const availableHeight = Math.max(1, nodeHeight - y - IMAGE_INFO_HEIGHT);
        const widgetAspect = nodeWidth / availableHeight;
        let targetWidth;
        let targetHeight;
        let offsetX = 0;

        if (imageAspect > widgetAspect) {
            targetWidth = nodeWidth;
            targetHeight = nodeWidth / imageAspect;
        } else {
            targetHeight = availableHeight;
            targetWidth = availableHeight * imageAspect;
            offsetX = (nodeWidth - targetWidth) / 2;
        }

        const widthMultiplier = image.img.naturalWidth / targetWidth;
        const sourceWidth = cropX != null
            ? Math.max(0, (cropX - offsetX) * widthMultiplier)
            : image.img.naturalWidth;
        const destinationX = (nodeWidth - targetWidth) / 2;
        const destinationY = y + (availableHeight - targetHeight) / 2;
        const destinationWidth = cropX != null
            ? Math.max(0, cropX - offsetX)
            : targetWidth;

        ctx.save();
        ctx.beginPath();
        if (cropX != null) {
            ctx.rect(destinationX, destinationY, destinationWidth, targetHeight);
            ctx.clip();
        }
        ctx.drawImage(
            image.img,
            0,
            0,
            sourceWidth,
            image.img.naturalHeight,
            destinationX,
            destinationY,
            destinationWidth,
            targetHeight,
        );
        if (
            cropX != null
            && cropX >= destinationX
            && cropX <= destinationX + targetWidth
        ) {
            ctx.beginPath();
            ctx.moveTo(cropX, destinationY);
            ctx.lineTo(cropX, destinationY + targetHeight);
            ctx.globalCompositeOperation = "difference";
            ctx.strokeStyle = "rgba(255, 255, 255, 1)";
            ctx.stroke();
        }
        ctx.restore();
    }

    mouse(event, pos) {
        if (event.type !== "pointerdown") return false;
        for (const area of this.hitAreas) {
            const [x, y, width, height] = area.bounds;
            if (pos[0] >= x && pos[0] <= x + width && pos[1] >= y && pos[1] <= y + height) {
                this.selectImage(area.image);
                return true;
            }
        }
        return false;
    }

    computeSize(width) {
        return [width, 20];
    }

    serializeValue() {
        return {
            images: this._value.images.map((item) => {
                const result = { ...item };
                delete result.img;
                return result;
            }),
        };
    }
}


function comparerImages(output) {
    const aImages = output.a_images ?? [];
    const bImages = output.b_images ?? [];
    const images = [];
    const multiple = aImages.length + bImages.length > 2;

    for (const [index, data] of aImages.entries()) {
        images.push({
            name: aImages.length > 1 || multiple ? `A${index + 1}` : "A",
            selected: index === 0,
            url: imageDataToUrl(data),
            downloadUrl: imageDataToUrl(data, false),
        });
    }
    for (const [index, data] of bImages.entries()) {
        images.push({
            name: bImages.length > 1 || multiple ? `B${index + 1}` : "B",
            selected: index === 0,
            url: imageDataToUrl(data),
            downloadUrl: imageDataToUrl(data, false),
        });
    }
    return images;
}


app.registerExtension({
    name: "RPNodes.ImageComparer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        nodeType["@comparer_mode"] = {
            type: "combo",
            values: ["Slide", "Click", "Side-by-side"],
        };

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            this.properties ??= {};
            this.properties.comparer_mode ??= "Slide";
            this.imageIndex = 0;
            this.rpComparerImages = this.imgs ?? [];
            this.rpComparerControlsImages = false;
            Object.defineProperty(this, "imgs", {
                configurable: true,
                enumerable: true,
                get: () => this.rpComparerImages,
                set: (images) => {
                    if (!this.rpComparerControlsImages) {
                        this.rpComparerImages = images ?? [];
                    }
                },
            });
            this.isClickShowingAfter = false;
            this.isPointerOver = false;
            this.pointerOverPos = [0, 0];
            this.rpDownloadName = "#1";
            const originalAddCustomWidget = this.addCustomWidget.bind(this);
            this.addCustomWidget = (widget) => {
                if (widget?.name === NATIVE_PREVIEW_WIDGET) return widget;
                return originalAddCustomWidget(widget);
            };
            removeNativeCanvasPreview(this);
            this.rpCompareModeButton = this.addWidget(
                "button",
                compareModeLabel(this.properties.comparer_mode),
                null,
                () => {
                    this.properties.comparer_mode = this.properties.comparer_mode === "Click"
                        ? "Slide"
                        : "Click";
                    this.isClickShowingAfter = false;
                    this.imageIndex = 0;
                    this.rpCompareModeButton.name = compareModeLabel(
                        this.properties.comparer_mode,
                    );
                    this.setDirtyCanvas?.(true, true);
                },
                { serialize: false },
            );
            this.rpComparerWidget = this.addCustomWidget(new ImageComparerWidget(this));
            const computed = this.computeSize?.() ?? this.size ?? [320, 360];
            this.setSize?.([
                Math.max(320, computed[0] ?? 0),
                Math.max(360, computed[1] ?? 0),
            ]);
            this.setDirtyCanvas?.(true, true);
            return result;
        };

        const originalExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            const result = originalExecuted?.apply(this, arguments);
            this.rpDownloadName = output.download_name?.[0] ?? "#1";
            this.rpComparerControlsImages = true;
            this.rpComparerWidget.value = { images: comparerImages(output) };
            removeNativeCanvasPreview(this);
            queueMicrotask(() => removeNativeCanvasPreview(this));
            return result;
        };

        const originalMouseEnter = nodeType.prototype.onMouseEnter;
        nodeType.prototype.onMouseEnter = function () {
            const result = originalMouseEnter?.apply(this, arguments);
            this.isPointerOver = true;
            this.setDirtyCanvas?.(true, false);
            return result;
        };

        const originalMouseLeave = nodeType.prototype.onMouseLeave;
        nodeType.prototype.onMouseLeave = function (event, pos, canvas) {
            const result = originalMouseLeave?.apply(this, arguments);
            this.isPointerOver = false;
            setCompareCursor(event, canvas, "");
            this.setDirtyCanvas?.(true, false);
            return result;
        };

        const originalMouseMove = nodeType.prototype.onMouseMove;
        nodeType.prototype.onMouseMove = function (event, pos, canvas) {
            const result = originalMouseMove?.apply(this, arguments);
            if (!pos) return result;
            this.pointerOverPos = [...pos];
            this.imageIndex = this.properties.comparer_mode === "Click"
                ? (this.isClickShowingAfter ? 1 : 0)
                : (pos[0] > this.size[0] / 2 ? 1 : 0);
            const overPreview = isOverComparePreview(this, pos);
            const compareCursor = !overPreview
                ? null
                : this.properties.comparer_mode === "Click"
                    ? "pointer"
                    : this.properties.comparer_mode === "Slide"
                        ? "crosshair"
                        : null;
            setCompareCursor(event, canvas, compareCursor);
            this.setDirtyCanvas?.(true, false);
            return result;
        };

        const originalMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (event, pos) {
            const result = originalMouseDown?.apply(this, arguments);
            if (event?.button != null && event.button !== 0) return result;
            if (
                this.properties.comparer_mode === "Click"
                && isOverComparePreview(this, pos)
            ) {
                this.isClickShowingAfter = !this.isClickShowingAfter;
                this.imageIndex = this.isClickShowingAfter ? 1 : 0;
                this.setDirtyCanvas?.(true, false);
            }
            return result;
        };

    },
});
