import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";


const NODE_CLASSES = new Set(["SmartImageSize", "SmartImageResize"]);
let dataPromise;


function loadData() {
    dataPromise ??= api.fetchApi("/image-model-resolution-selector/data").then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
    });
    return dataPromise;
}


function formatDimension(item) {
    return `${item.ratio} (${item.label}) - ${item.width} x ${item.height}`;
}


function replaceComboValues(widget, values, preferred) {
    widget.options ??= {};
    widget.options.values = values;
    widget.value = values.includes(preferred) ? preferred : values[0];
}


function redraw(node) {
    const computed = node.computeSize();
    node.setSize([Math.max(node.size[0], computed[0]), computed[1]]);
    node.setDirtyCanvas(true, true);
    node.graph?.setDirtyCanvas(true, true);
}


function configureMenus(node, data) {
    const model = node.widgets?.find((widget) => widget.name === "model");
    const resolution = node.widgets?.find((widget) => widget.name === "resolution");
    const dimensions = node.widgets?.find((widget) => widget.name === "dimensions");
    const width = node.widgets?.find((widget) => widget.name === "width");
    const height = node.widgets?.find((widget) => widget.name === "height");
    const selectionMode = node.widgets?.find((widget) => widget.name === "selection_mode");
    if (!model || !resolution || !dimensions) return;

    function updateAutomaticState() {
        if (!selectionMode) return;
        const disabled = selectionMode.value === "automatic";
        for (const widget of [dimensions, width, height]) {
            if (!widget) continue;
            widget.disabled = disabled;
            widget.options ??= {};
            widget.options.disabled = disabled;
        }
        redraw(node);
    }

    function updateSizeWidgets(preferredWidth, preferredHeight) {
        if (!width || !height) return;
        const item = (data[model.value]?.[resolution.value] ?? []).find(
            (candidate) => formatDimension(candidate) === dimensions.value
        );
        if (!item) return;
        width.value = Number.isFinite(preferredWidth) ? preferredWidth : item.width;
        height.value = Number.isFinite(preferredHeight) ? preferredHeight : item.height;
    }

    function updateDimensions(preferred, preferredWidth, preferredHeight) {
        const items = data[model.value]?.[resolution.value] ?? [];
        replaceComboValues(dimensions, items.map(formatDimension), preferred);
        updateSizeWidgets(preferredWidth, preferredHeight);
        redraw(node);
    }

    function updateResolutions(preferredResolution, preferredDimensions, preferredWidth, preferredHeight) {
        const values = Object.keys(data[model.value] ?? {});
        replaceComboValues(resolution, values, preferredResolution);
        updateDimensions(preferredDimensions, preferredWidth, preferredHeight);
    }

    function synchronize() {
        const savedModel = model.value;
        const savedResolution = resolution.value;
        const savedDimensions = dimensions.value;
        const savedWidth = width?.value;
        const savedHeight = height?.value;
        replaceComboValues(model, Object.keys(data), savedModel);
        updateResolutions(savedResolution, savedDimensions, savedWidth, savedHeight);
        updateAutomaticState();
    }

    const modelCallback = model.callback;
    model.callback = function (value, ...args) {
        modelCallback?.call(this, value, ...args);
        updateResolutions(undefined, undefined, undefined, undefined);
    };

    const resolutionCallback = resolution.callback;
    resolution.callback = function (value, ...args) {
        resolutionCallback?.call(this, value, ...args);
        updateDimensions(undefined, undefined, undefined);
    };

    const dimensionsCallback = dimensions.callback;
    dimensions.callback = function (value, ...args) {
        dimensionsCallback?.call(this, value, ...args);
        updateSizeWidgets(undefined, undefined);
        redraw(node);
    };

    if (selectionMode) {
        const selectionModeCallback = selectionMode.callback;
        selectionMode.callback = function (value, ...args) {
            selectionModeCallback?.call(this, value, ...args);
            updateAutomaticState();
        };
    }

    node.__imageModelResolutionSync = synchronize;
    synchronize();
}


app.registerExtension({
    name: "RPNodes.SmartImage.DynamicMenus",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_CLASSES.has(nodeData.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = onNodeCreated?.apply(this, args);
            loadData()
                .then((data) => configureMenus(this, data))
                .catch((error) => console.error("Image Model Resolution Selector:", error));
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = onConfigure?.apply(this, args);
            setTimeout(() => this.__imageModelResolutionSync?.(), 0);
            return result;
        };
    },
});
