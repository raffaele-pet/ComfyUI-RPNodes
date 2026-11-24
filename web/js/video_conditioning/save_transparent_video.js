import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
    name: "RPNodes.SaveTransparentVideo",
    nodeCreated(node) {
        if (node.comfyClass !== "RPSaveTransparentVideo") return;

        const container = document.createElement("div");
        const link = document.createElement("a");
        const notice = document.createElement("span");
        container.hidden = true;
        container.append(link, notice);
        link.textContent = "Download MOV";
        link.style.color = "var(--input-text, #eaeaea)";
        notice.style.display = "block";
        Object.assign(container.style, {
            color: "var(--input-text, #eaeaea)", font: "13px/20px system-ui, sans-serif",
            textAlign: "center",
            alignContent: "center",
        });
        const widget = node.addDOMWidget("Download MOV", "link", container, { serialize: false });
        const height = () => notice.textContent ? 50 : 28;
        widget.computeSize = (width) => [width, container.hidden ? -4 : height()];
        widget.computeLayoutSize = () => ({ minWidth: 0, minHeight: container.hidden ? 0 : height() });

        const onExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            const result = onExecuted?.apply(this, arguments);
            const file = output.rp_transparent_export?.[0];
            if (file) {
                const params = new URLSearchParams({
                    filename: file.filename, type: file.type, subfolder: file.subfolder,
                });
                link.href = api.apiURL(`/view?${params}`);
                link.download = link.title = file.filename;
                notice.textContent = output.rp_transparent_empty?.[0] ? "No visible guide in this clip." : "";
                container.hidden = false;
                const [, height] = this.computeSize();
                if (height > this.size[1]) this.setSize([this.size[0], height]);
                this.setDirtyCanvas(true, true);
            }
            return result;
        };
    },
});
