import { post } from "./selective_video_preview.js";

export const guideModes = {
    pose_guide: "OpenPose · Body, hands and face",
    face_guide: "OpenPose · Face only",
    rectangle_guide: "Tracked rectangle",
};

export const guideStyle = `
    .rp-selective-editor .rp-guide-layer { position:absolute; overflow:hidden; pointer-events:none; }
    .rp-selective-editor .rp-guide-box { position:absolute; outline:1px dashed #fff; box-shadow:0 0 0 1px #0008; }
    .rp-selective-editor .rp-guide-art { width:100%; height:100%; display:block; object-fit:fill; }
    .rp-selective-editor .rp-guide-art[hidden] { display:none; }
    .rp-selective-editor .rp-guide-box button { position:absolute; pointer-events:auto; touch-action:none;
        width:13px; height:13px; padding:0; border:1px solid #111; border-radius:3px; background:#fff;
        transform:translate(-50%,-50%); }
    .rp-selective-editor .rp-guide-box button[data-x="-1"] { left:0; }
    .rp-selective-editor .rp-guide-box button[data-x="1"] { left:100%; }
    .rp-selective-editor .rp-guide-box button[data-y="-1"] { top:0; }
    .rp-selective-editor .rp-guide-box button[data-y="1"] { top:100%; }
    .rp-selective-editor .rp-guide-status { font-size:13px; opacity:.85; }
    .rp-selective-editor .rp-guide-controls .rp-row { flex-wrap:wrap; }
    .rp-selective-editor .rp-guide-controls input[type=number] { width:60px; }
`;

const minScale = 0.1;
const maxScale = 4;
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

function element(tag, css, text) {
    const node = document.createElement(tag);
    if (css) node.className = css;
    if (text) node.textContent = text;
    return node;
}

function rectangleBounds(mask) {
    const { width, height } = mask;
    const pixels = mask.getContext("2d").getImageData(0, 0, width, height).data;
    let left = width, top = height, right = 0, bottom = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (pixels[(y * width + x) * 4 + 3] <= 127) continue;
            left = Math.min(left, x);
            top = Math.min(top, y);
            right = Math.max(right, x + 1);
            bottom = y + 1;
        }
    }
    return right ? [left, top, right, bottom] : null;
}

export function createGuideEditor(stage, video) {
    const layer = element("div", "rp-guide-layer");
    const box = element("div", "rp-guide-box");
    const image = element("img", "rp-guide-art");
    const rectangle = element("div", "rp-guide-art");
    image.alt = "";
    image.draggable = false;
    box.append(image, rectangle);
    layer.append(box);
    stage.append(layer);

    const controls = element("div", "rp-guide-controls");
    const row = element("div", "rp-row");
    const status = element("p", "rp-guide-status");
    status.setAttribute("role", "status");
    const hint = "Drag a corner. Hold Shift to keep proportions.";
    const reset = element("button", "", "Reset scale");
    const fields = ["Width", "Height"].map((title, index) => {
        const label = element("label", "", `${title} %`);
        const field = element("input", "rp-small-number");
        field.type = "number";
        field.min = minScale * 100;
        field.max = maxScale * 100;
        field.step = 1;
        field.addEventListener("change", () => {
            const scale = [...getScale()];
            if (Number.isFinite(field.valueAsNumber)) scale[index] = field.valueAsNumber / 100;
            setScale(scale);
        });
        label.append(field);
        row.append(label);
        return field;
    });
    row.append(reset);
    controls.append(row, status);
    controls.hidden = true;
    box.hidden = true;

    let group;
    let frameSize;
    let bounds;
    let pending;
    let requestedMask;
    let timer;
    let drag;

    function getScale() {
        return group?.guide_scale ?? [1, 1];
    }

    function position() {
        if (!bounds || !frameSize) return;
        const [frameWidth, frameHeight] = frameSize;
        const [left, top, right, bottom] = bounds;
        const [scaleX, scaleY] = getScale();
        const zoom = Math.min(stage.clientWidth / frameWidth, stage.clientHeight / frameHeight);
        const width = Math.max(1, Math.round((right - left) * scaleX));
        const height = Math.max(1, Math.round((bottom - top) * scaleY));
        Object.assign(layer.style, {
            width: `${frameWidth * zoom}px`, height: `${frameHeight * zoom}px`,
            left: `${(stage.clientWidth - frameWidth * zoom) / 2}px`,
            top: `${(stage.clientHeight - frameHeight * zoom) / 2}px`,
        });
        Object.assign(box.style, {
            left: `${Math.round((left + right - width) / 2) * zoom}px`,
            top: `${Math.round((top + bottom - height) / 2) * zoom}px`,
            width: `${width * zoom}px`, height: `${height * zoom}px`,
        });
    }

    function updateFields() {
        fields.forEach((field, index) => { field.value = Math.round(getScale()[index] * 100); });
    }

    function setScale(scale) {
        if (!group) return;
        group.guide_scale = scale.map((value) => clamp(value, minScale, maxScale));
        updateFields();
        position();
    }
    reset.onclick = () => setScale([1, 1]);

    for (const x of [-1, 1]) {
        for (const y of [-1, 1]) {
            const handle = element("button");
            Object.assign(handle.dataset, { x, y });
            handle.style.cursor = x === y ? "nwse-resize" : "nesw-resize";
            handle.tabIndex = -1;
            handle.setAttribute("aria-label", "Drag to resize guide");
            handle.title = hint;
            handle.onpointerdown = (event) => {
                if (event.button !== 0) return;
                event.preventDefault();
                handle.setPointerCapture(event.pointerId);
                const { width, height } = box.getBoundingClientRect();
                if (!width || !height) return;
                drag = { x: event.clientX, y: event.clientY, width, height, scale: getScale() };
            };
            handle.onpointermove = (event) => {
                if (!drag || !handle.hasPointerCapture(event.pointerId)) return;
                const dx = 2 * x * (event.clientX - drag.x) / drag.width;
                const dy = 2 * y * (event.clientY - drag.y) / drag.height;
                const [sx, sy] = drag.scale;
                if (event.shiftKey) {
                    const delta = Math.abs(dx) >= Math.abs(dy) ? dx : dy;
                    const factor = clamp(1 + delta, minScale / Math.min(sx, sy), maxScale / Math.max(sx, sy));
                    setScale([sx * factor, sy * factor]);
                } else {
                    setScale([sx * (1 + dx), sy * (1 + dy)]);
                }
            };
            handle.onlostpointercapture = () => { drag = null; };
            box.append(handle);
        }
    }

    const observer = new ResizeObserver(position);
    observer.observe(stage);

    function clear() {
        pending?.abort();
        clearTimeout(timer);
        requestedMask = null;
        drag = null;
        bounds = null;
        box.hidden = true;
    }

    return {
        controls,
        clear,
        disconnect() {
            clear();
            observer.disconnect();
        },
        update(activeGroup, frame, mask) {
            if (group !== activeGroup) {
                clear();
                group = activeGroup;
            }
            const kind = group?.preprocessor;
            controls.hidden = !Object.hasOwn(guideModes, kind);
            if (controls.hidden || !mask) {
                clear();
                status.textContent = controls.hidden ? "" : "Detect on this frame or run Track & preview to resize the guide.";
                updateFields();
                return;
            }
            frameSize = [mask.width, mask.height];
            if (kind === "rectangle_guide") {
                if (requestedMask) clear();
                bounds = rectangleBounds(mask);
                rectangle.style.background = group.color;
                rectangle.hidden = false;
                image.hidden = true;
                box.hidden = !bounds;
                updateFields();
                position();
                status.textContent = bounds ? hint : "No selection on this frame.";
                return;
            }
            const encoded = mask.toDataURL("image/png").slice("data:image/png;base64,".length);
            if (requestedMask?.frame === frame && requestedMask.kind === kind && requestedMask.encoded === encoded) return;
            clear();
            requestedMask = { frame, kind, encoded };
            const controller = new AbortController();
            pending = controller;
            status.textContent = "Finding pose…";
            updateFields();
            timer = setTimeout(async () => {
                try {
                    const result = await post("/guide-frame", {
                        video, frame, kind, mask: encoded,
                    }, controller.signal);
                    if (controller.signal.aborted) return;
                    bounds = result.bounds;
                    rectangle.hidden = true;
                    image.hidden = !bounds;
                    box.hidden = !bounds;
                    if (bounds) {
                        image.src = `data:image/png;base64,${result.image}`;
                        position();
                    }
                    status.textContent = bounds ? hint : "No matching pose on this frame.";
                } catch (error) {
                    if (!controller.signal.aborted) {
                        requestedMask = null;
                        status.textContent = error.message;
                    }
                }
            }, 120);
        },
    };
}
