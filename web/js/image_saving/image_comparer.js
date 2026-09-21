import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";


const NODE_NAME = "RPImageComparer";


function imageDataToUrl(data, preview = true) {
    const query = new URLSearchParams({
        filename: data.filename,
        type: data.type,
        subfolder: data.subfolder ?? "",
    });
    const previewParams = preview ? app.getPreviewFormatParam?.() ?? "" : "";
    return api.apiURL(`/view?${query}${previewParams}${app.getRandParam?.() ?? ""}`);
}


function menuLabel(option) {
    return typeof option?.content === "string" ? option.content : "";
}


function pngFilename(name) {
    const basename = String(name || "#1").replace(/\.(?:png|jpe?g|webp)$/i, "");
    return `${basename}.png`;
}


async function downloadSelectedImage(node) {
    const image = node.imgs?.[node.imageIndex ?? 0] ?? node.imgs?.[0];
    if (!image?.src) return;

    const sourceUrl = image.rpDownloadUrl ?? image.src;
    let href = sourceUrl;
    let objectUrl = null;
    try {
        const response = await fetch(sourceUrl);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        objectUrl = URL.createObjectURL(await response.blob());
        href = objectUrl;
    } catch (error) {
        console.warn("RP Image Comparer: using the preview URL for download.", error);
    }

    const link = document.createElement("a");
    link.href = href;
    link.download = pngFilename(node.rpDownloadName);
    document.body.appendChild(link);
    link.click();
    link.remove();
    if (objectUrl) setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
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

        if (node.properties?.comparer_mode === "Click") {
            this.drawImage(ctx, this.selected[node.isPointerDown ? 1 : 0], y);
        } else {
            this.drawImage(ctx, this.selected[0], y);
            if (node.isPointerOver) {
                this.drawImage(ctx, this.selected[1], y, node.pointerOverPos[0]);
            }
        }
    }

    drawImage(ctx, image, y, cropX) {
        if (!image?.img?.naturalWidth || !image?.img?.naturalHeight) return;

        const [nodeWidth, nodeHeight] = this.node.size;
        const imageAspect = image.img.naturalWidth / image.img.naturalHeight;
        const availableHeight = Math.max(1, nodeHeight - y);
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
            values: ["Slide", "Click"],
        };

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            this.properties ??= {};
            this.properties.comparer_mode ??= "Slide";
            this.imageIndex = 0;
            this.imgs = [];
            this.isPointerDown = false;
            this.isPointerOver = false;
            this.pointerOverPos = [0, 0];
            this.rpDownloadName = "#1";
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
            this.rpComparerWidget.value = { images: comparerImages(output) };
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
        nodeType.prototype.onMouseLeave = function () {
            const result = originalMouseLeave?.apply(this, arguments);
            this.isPointerOver = false;
            this.isPointerDown = false;
            this.setDirtyCanvas?.(true, false);
            return result;
        };

        const originalMouseMove = nodeType.prototype.onMouseMove;
        nodeType.prototype.onMouseMove = function (event, pos) {
            const result = originalMouseMove?.apply(this, arguments);
            if (!pos) return result;
            this.pointerOverPos = [...pos];
            this.imageIndex = pos[0] > this.size[0] / 2 ? 1 : 0;
            this.setDirtyCanvas?.(true, false);
            return result;
        };

        const originalMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (event) {
            const result = originalMouseDown?.apply(this, arguments);
            if (event?.button != null && event.button !== 0) return result;
            this.isPointerDown = true;
            this.imageIndex = 1;
            this.setDirtyCanvas?.(true, false);
            window.addEventListener("pointerup", () => {
                this.isPointerDown = false;
                this.setDirtyCanvas?.(true, false);
            }, { once: true });
            return result;
        };

        const originalMenu = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_canvas, options) {
            const result = originalMenu?.apply(this, arguments);
            const menu = Array.isArray(result) ? result : options;
            if (!Array.isArray(menu)) return result;

            const saveOption = {
                content: "Save Image",
                callback: () => downloadSelectedImage(this),
            };
            const existing = menu.findIndex((option) => /^Save Image(?:\s|$)/i.test(menuLabel(option)));
            if (existing >= 0) menu.splice(existing, 1, saveOption);
            else menu.unshift(saveOption);
            return result ?? menu;
        };
    },
});
