import { post } from "./selective_video_preview.js";

export const guideModes = {
    pose_guide: "OpenPose",
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
    .rp-selective-editor .rp-guide-box .rp-guide-move { inset:0; width:100%; height:100%;
        border:0; border-radius:0; background:transparent; transform:none; cursor:move; }
    .rp-selective-editor .rp-guide-box .rp-guide-rotate { top:-25px; left:50%; width:16px; height:16px;
        border-radius:50%; cursor:grab; }
    .rp-selective-editor .rp-guide-rotate::after { content:""; position:absolute; top:100%; left:50%;
        width:1px; height:17px; background:#fff; pointer-events:none; }
    .rp-selective-editor .rp-guide-controls { position:absolute; z-index:1; left:10px; top:10px;
        display:grid; gap:8px; width:min(250px,calc(100% - 20px)); max-height:calc(100% - 20px);
        overflow:auto; padding:9px; background:var(--comfy-menu-bg,#202125);
        border:1px solid var(--border-color,#545761); border-radius:8px; box-shadow:0 2px 12px #0008; }
    .rp-selective-editor .rp-guide-controls[hidden] { display:none; }
    .rp-selective-editor .rp-guide-controls .rp-guide-panel-handle { display:flex; align-items:center; gap:6px;
        padding:0 2px 6px; color:inherit; background:transparent; border:0;
        border-bottom:1px solid var(--border-color,#545761); border-radius:0;
        cursor:grab; touch-action:none; font-weight:650; text-align:left; }
    .rp-selective-editor .rp-guide-controls .rp-guide-panel-handle:active { cursor:grabbing; }
    .rp-selective-editor .rp-guide-panel-grip { flex:none; width:10px; height:13px; opacity:.65;
        background:radial-gradient(circle,currentColor 1px,transparent 1.4px) 0 0 / 5px 5px; }
    .rp-selective-editor .rp-guide-fields,.rp-selective-editor .rp-guide-resets { display:grid; gap:6px; }
    .rp-selective-editor .rp-guide-fields { grid-template-columns:repeat(3,minmax(0,1fr)); }
    .rp-selective-editor .rp-guide-fields label { display:grid; gap:3px; font-size:12px; }
    .rp-selective-editor .rp-guide-controls input[type=number] { width:100%; min-width:0; padding:4px 5px; }
    .rp-selective-editor .rp-guide-resets { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .rp-selective-editor .rp-guide-resets button { white-space:normal; font-size:12px; padding:4px; }
    .rp-selective-editor .rp-guide-status { font-size:12px; opacity:.85; }
    .rp-selective-editor .rp-guide-status:empty { display:none; }
`;

const minScale = 0.1;
const maxScale = 4;
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const radians = (degrees) => degrees * Math.PI / 180;
const wrapAngle = (degrees) => ((degrees + 180) % 360 + 360) % 360 - 180;
const arrows = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };

function element(tag, css, text) {
    const node = document.createElement(tag);
    if (css) node.className = css;
    if (text) node.textContent = text;
    return node;
}

export function createGuideEditor(stage, video, guideVisible) {
    const layer = element("div", "rp-guide-layer");
    const box = element("div", "rp-guide-box");
    const image = element("img", "rp-guide-art");
    const rectangle = element("div", "rp-guide-art");
    const move = element("button", "rp-guide-move");
    const rotate = element("button", "rp-guide-rotate");
    move.setAttribute("aria-label", "Move guide");
    move.title = "Drag to move the guide";
    rotate.setAttribute("aria-label", "Rotate guide");
    rotate.title = "Drag to rotate the guide";
    rotate.tabIndex = -1;
    image.alt = "";
    image.draggable = false;
    box.append(image, rectangle, move, rotate);
    layer.append(box);
    stage.append(layer);

    const controls = element("section", "rp-guide-controls");
    controls.setAttribute("aria-label", "Guide transform");
    const fieldsRow = element("div", "rp-guide-fields");
    const panelHandle = element("button", "rp-guide-panel-handle");
    panelHandle.type = "button";
    panelHandle.setAttribute("aria-label", "Move guide panel");
    panelHandle.title = "Drag to move the panel. Use arrow keys to move it with the keyboard.";
    const grip = element("span", "rp-guide-panel-grip");
    grip.setAttribute("aria-hidden", "true");
    panelHandle.append(grip, "Guide transform");
    const status = element("p", "rp-guide-status");
    status.setAttribute("role", "status");
    const resetScale = element("button", "", "Reset scale");
    const resetPosition = element("button", "", "Reset position");
    resetPosition.title = "Reset offset and rotation";
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
        fieldsRow.append(label);
        return field;
    });
    const angleLabel = element("label", "", "Rotation °");
    const angleField = element("input", "rp-small-number");
    angleField.type = "number";
    angleField.min = -180;
    angleField.max = 180;
    angleField.step = 1;
    angleField.onchange = () => setRotation(Number.isFinite(angleField.valueAsNumber) ? angleField.valueAsNumber : getRotation());
    angleLabel.append(angleField);
    fieldsRow.append(angleLabel);
    const resets = element("div", "rp-guide-resets");
    resets.append(resetScale, resetPosition);
    controls.append(panelHandle, fieldsRow, resets, status);
    controls.hidden = true;
    box.hidden = true;
    stage.append(controls);

    let group;
    let frameSize;
    let geometry;
    let pending;
    let requestedMask;
    let timer;
    let drag;
    let panelDrag;

    function placePanel(x = controls.offsetLeft, y = controls.offsetTop) {
        if (controls.hidden) return;
        const margin = 10;
        const maxX = Math.max(margin, stage.clientWidth - controls.offsetWidth - margin);
        const maxY = Math.max(margin, stage.clientHeight - controls.offsetHeight - margin);
        controls.style.left = `${clamp(x, margin, maxX)}px`;
        controls.style.top = `${clamp(y, margin, maxY)}px`;
    }

    panelHandle.onpointerdown = (event) => {
        if (event.button !== 0) return;
        panelHandle.setPointerCapture(event.pointerId);
        panelDrag = [controls.offsetLeft - event.clientX, controls.offsetTop - event.clientY];
    };
    panelHandle.onpointermove = (event) => {
        if (!panelDrag || !panelHandle.hasPointerCapture(event.pointerId)) return;
        placePanel(event.clientX + panelDrag[0], event.clientY + panelDrag[1]);
    };
    panelHandle.onlostpointercapture = () => { panelDrag = null; };
    panelHandle.onkeydown = (event) => {
        const direction = arrows[event.key];
        if (!direction) return;
        event.preventDefault();
        placePanel(controls.offsetLeft + direction[0] * 12, controls.offsetTop + direction[1] * 12);
    };

    function updatePanel() {
        controls.hidden = !guideVisible() || !Object.hasOwn(guideModes, group?.preprocessor);
        placePanel();
    }

    function getScale() {
        return group?.guide_scale ?? [1, 1];
    }

    function getOffset() {
        return group?.guide_offset ?? [0, 0];
    }

    function getRotation() {
        return group?.guide_rotation ?? 0;
    }

    function position() {
        if (!geometry || !frameSize) return;
        const [frameWidth, frameHeight] = frameSize;
        const { center, size, angle } = geometry;
        const [scaleX, scaleY] = getScale();
        const [offsetX, offsetY] = getOffset();
        const zoom = Math.min(stage.clientWidth / frameWidth, stage.clientHeight / frameHeight);
        const width = Math.max(1, size[0] * scaleX);
        const height = Math.max(1, size[1] * scaleY);
        const cx = center[0] + offsetX * frameWidth;
        const cy = center[1] + offsetY * frameHeight;
        Object.assign(layer.style, {
            width: `${frameWidth * zoom}px`, height: `${frameHeight * zoom}px`,
            left: `${(stage.clientWidth - frameWidth * zoom) / 2}px`,
            top: `${(stage.clientHeight - frameHeight * zoom) / 2}px`,
        });
        Object.assign(box.style, {
            left: `${(cx - width / 2) * zoom}px`,
            top: `${(cy - height / 2) * zoom}px`,
            width: `${width * zoom}px`, height: `${height * zoom}px`,
            transform: `rotate(${angle + getRotation()}deg)`,
        });
    }

    function updateFields() {
        fields.forEach((field, index) => { field.value = Math.round(getScale()[index] * 100); });
        angleField.value = Math.round(getRotation());
    }

    function setScale(scale) {
        if (!group) return;
        group.guide_scale = scale.map((value) => clamp(value, minScale, maxScale));
        updateFields();
        position();
    }
    function setOffset(offset) {
        if (!group) return;
        group.guide_offset = offset.map((value) => clamp(value, -1, 1));
        position();
    }

    function setRotation(degrees) {
        if (!group) return;
        group.guide_rotation = wrapAngle(degrees);
        updateFields();
        position();
    }
    resetScale.onclick = () => setScale([1, 1]);
    resetPosition.onclick = () => {
        setOffset([0, 0]);
        setRotation(0);
    };

    function bindDrag(control, update) {
        control.onpointerdown = (event) => {
            if (event.button !== 0 || !geometry) return;
            event.preventDefault();
            control.setPointerCapture(event.pointerId);
            const { x, y, width, height } = box.getBoundingClientRect();
            const center = [x + width / 2, y + height / 2];
            const angle = radians(geometry.angle + getRotation());
            drag = {
                x: event.clientX, y: event.clientY, center,
                cos: Math.cos(angle), sin: Math.sin(angle),
                width: parseFloat(box.style.width), height: parseFloat(box.style.height),
                scale: getScale(), offset: getOffset(), rotation: getRotation(),
                frameWidth: layer.clientWidth, frameHeight: layer.clientHeight,
            };
        };
        control.onpointermove = (event) => {
            if (!drag || !control.hasPointerCapture(event.pointerId)) return;
            update(event, event.clientX - drag.x, event.clientY - drag.y);
        };
        control.onlostpointercapture = () => { drag = null; };
    }

    bindDrag(move, (_, dx, dy) => setOffset([
        drag.offset[0] + dx / drag.frameWidth,
        drag.offset[1] + dy / drag.frameHeight,
    ]));
    bindDrag(rotate, (event) => {
        const [cx, cy] = drag.center;
        const before = Math.atan2(drag.y - cy, drag.x - cx);
        const after = Math.atan2(event.clientY - cy, event.clientX - cx);
        setRotation(drag.rotation + (after - before) * 180 / Math.PI);
    });
    move.onkeydown = (event) => {
        const direction = arrows[event.key];
        if (!direction || !geometry) return;
        event.preventDefault();
        const step = event.shiftKey ? 10 : 1;
        setOffset(getOffset().map((offset, axis) => offset + direction[axis] * step / frameSize[axis]));
    };

    for (const x of [-1, 1]) {
        for (const y of [-1, 1]) {
            const handle = element("button");
            Object.assign(handle.dataset, { x, y });
            handle.style.cursor = x === y ? "nwse-resize" : "nesw-resize";
            handle.tabIndex = -1;
            handle.setAttribute("aria-label", "Drag to resize guide");
            handle.title = "Drag to resize. Hold Shift to resize freely.";
            bindDrag(handle, (event, screenX, screenY) => {
                const dx = 2 * x * (drag.cos * screenX + drag.sin * screenY) / drag.width;
                const dy = 2 * y * (-drag.sin * screenX + drag.cos * screenY) / drag.height;
                const [sx, sy] = drag.scale;
                if (event.shiftKey) {
                    setScale([sx * (1 + dx), sy * (1 + dy)]);
                } else {
                    const delta = Math.abs(dx) >= Math.abs(dy) ? dx : dy;
                    const factor = clamp(1 + delta, minScale / Math.min(sx, sy), maxScale / Math.max(sx, sy));
                    setScale([sx * factor, sy * factor]);
                }
            });
            box.append(handle);
        }
    }

    const observer = new ResizeObserver(() => {
        position();
        placePanel();
    });
    observer.observe(stage);
    observer.observe(controls);

    function clear() {
        pending?.abort();
        clearTimeout(timer);
        requestedMask = null;
        drag = null;
        geometry = null;
        box.hidden = true;
    }

    function showArtwork(artwork, isRectangle) {
        geometry = artwork?.geometry;
        rectangle.hidden = !isRectangle;
        image.hidden = isRectangle;
        box.hidden = !geometry;
        if (geometry) {
            if (isRectangle) rectangle.style.background = group.color;
            else image.src = `data:image/png;base64,${artwork.image}`;
            position();
        }
    }

    return {
        showPrepared(activeGroup, size, artwork) {
            clear();
            group = activeGroup;
            frameSize = size;
            updatePanel();
            updateFields();
            showArtwork(controls.hidden ? null : artwork, group?.preprocessor === "rectangle_guide");
            status.textContent = "";
        },
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
            updatePanel();
            if (controls.hidden || !mask) {
                clear();
                status.textContent = controls.hidden ? "" : "Detect on this frame or run Track & preview.";
                updateFields();
                return;
            }
            frameSize = [mask.width, mask.height];
            const isRectangle = kind === "rectangle_guide";
            if (isRectangle) rectangle.style.background = group.color;
            const encoded = mask.toDataURL("image/png").slice("data:image/png;base64,".length);
            if (requestedMask?.frame === frame && requestedMask.kind === kind && requestedMask.encoded === encoded) return;
            clear();
            requestedMask = { frame, kind, encoded };
            const controller = new AbortController();
            pending = controller;
            status.textContent = isRectangle ? "" : "Finding pose…";
            updateFields();
            timer = setTimeout(async () => {
                try {
                    const result = await post("/guide-frame", {
                        video, frame, kind, mask: encoded,
                    }, controller.signal);
                    if (controller.signal.aborted) return;
                    showArtwork(result, isRectangle);
                    status.textContent = geometry ? "" : isRectangle ? "No selection on this frame." : "No matching pose on this frame.";
                } catch (error) {
                    if (!controller.signal.aborted) {
                        requestedMask = null;
                        status.textContent = error.message;
                    }
                }
            }, isRectangle ? 0 : 120);
        },
    };
}
