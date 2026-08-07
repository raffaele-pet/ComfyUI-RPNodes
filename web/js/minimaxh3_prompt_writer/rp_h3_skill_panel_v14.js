import { app } from "/scripts/app.js";

const H3_COMMIT = "093f3129a3f7bd27c74928b1cd31a54fbdebe057";
const BASE = `https://github.com/MiniMax-AI/MiniMax-H3/blob/${H3_COMMIT}/assets/`;

const SKILLS = {
  "Auto — choose the best H3 skill": {
    summary: "Gemma selects one of eight independent creative overlays, or keeps the core contract when none clearly fits.",
  },
  "H3 Prompt Writing (Core)": {
    summary: "Strict H3 mode, reference, chronology, camera, dialogue, soundscape, and schema rules without an added style profile.",
  },
  "Minimalist Product Ad": {
    summary: "Independent product-focused overlay with restrained pacing, controlled light, and fact-safe copy.",
    url: `${BASE}minimalist-product-ad-generator.gif`,
  },
  "3D Animation Short": {
    summary: "Independent stylized-3D overlay for coherent identity, readable performance, and weighted motion.",
    url: `${BASE}3d-animation-short-generator.gif`,
  },
  "Papercraft Stop-Motion Explainer": {
    summary: "Independent papercraft overlay for layered dioramas, handmade motion, and tactile sound.",
    url: `${BASE}papercraft-stop-motion-explainer.gif`,
  },
  "Brand Promo Video": {
    summary: "Independent brand-promo overlay grounded only in user-supplied facts and demonstrable features.",
    url: `${BASE}brand-promo-video-generator.gif`,
  },
  "Music Video / Lyric Typography": {
    summary: "Independent audio-led overlay for timing, cuts, and readable requested lyric typography.",
    url: `${BASE}music-video-subtitle-generator.gif`,
  },
  "Co-op Game Intro": {
    summary: "Independent co-op overlay that keeps two-player identity, choices, sides, and UI ownership coherent.",
    url: `${BASE}co-op-game-intro-generator.gif`,
  },
  "Paper Collage Explainer": {
    summary: "Independent editorial-collage overlay for layered paper motion, contact shadows, and tactile sound.",
    url: `${BASE}paper-collage-explainer-generator.gif`,
  },
  "Hand-drawn + Live Action": {
    summary: "Independent mixed-media overlay for handmade drawing, live-action contact, and reactive camera behavior.",
    url: `${BASE}handdrawn-live-video-generator.gif`,
  },
};

const TARGETS = new Set(["RPH3I2VPromptWriter", "RPH3REF2VPromptWriter"]);
const STATIC_PANEL_HEIGHT = 112;
const STATIC_WIDGET_HEIGHT = 136;
const STATIC_NODE_HEIGHT = 730;

function makePanel() {
  const root = document.createElement("div");
  root.style.cssText = [
    "box-sizing:border-box",
    "width:100%",
    "max-width:100%",
    `height:${STATIC_PANEL_HEIGHT}px`,
    `min-height:${STATIC_PANEL_HEIGHT}px`,
    `max-height:${STATIC_PANEL_HEIGHT}px`,
    "padding:8px 10px",
    "border:1px solid rgba(255,255,255,.16)",
    "border-radius:7px",
    "background:rgba(0,0,0,.22)",
    "font:12px/1.35 system-ui,sans-serif",
    "color:var(--fg-color,#ddd)",
    "white-space:normal",
    "overflow:hidden",
    "overflow-wrap:anywhere",
  ].join(";");

  const title = document.createElement("div");
  title.style.cssText = "font-weight:650;margin-bottom:4px;white-space:normal;overflow-wrap:anywhere";
  const summary = document.createElement("div");
  summary.style.cssText = "opacity:.88;white-space:normal;overflow-wrap:anywhere";
  const link = document.createElement("a");
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = "Open the official GIF ↗";
  link.style.cssText = "display:none;margin-top:6px;color:#7fb6ff;text-decoration:none;white-space:normal;overflow-wrap:anywhere";
  link.addEventListener("pointerdown", (event) => event.stopPropagation());
  link.addEventListener("click", (event) => event.stopPropagation());
  root.append(title, summary, link);
  return { root, title, summary, link };
}

function updatePanel(panel, skillName) {
  const info = SKILLS[skillName] || SKILLS["H3 Prompt Writing (Core)"];
  panel.title.textContent = skillName || "H3 Prompt Writing (Core)";
  panel.summary.textContent = info.summary;
  if (info.url) {
    panel.link.href = info.url;
    panel.link.style.display = "inline-block";
  } else {
    panel.link.removeAttribute("href");
    panel.link.style.display = "none";
  }
}

app.registerExtension({
  name: "RP.H3PromptWriter.SkillPanel",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!TARGETS.has(nodeData.name)) return;

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated?.apply(this, arguments);
      if (typeof this.addDOMWidget !== "function") return result;

      const panel = makePanel();
      const panelWidget = this.addDOMWidget("rp_h3_skill_info", "div", panel.root, {
        serialize: false,
        hideOnZoom: true,
        getMinHeight: () => STATIC_WIDGET_HEIGHT,
      });
      panelWidget.serialize = false;
      panelWidget.computeSize = (width) => [width, STATIC_WIDGET_HEIGHT];

      const skillWidget = this.widgets?.find((widget) => widget.name === "skill");
      const formatDuration = () => {
        const durationWidget = this.widgets?.find((widget) => widget.name === "duration_seconds");
        if (durationWidget?.options) {
          // Keep the backend/UI step at one whole second, while displaying the
          // float contract explicitly as 1.0, 2.0, 3.0, and so on. ComfyUI can
          // recreate widget options while restoring a saved workflow, so this
          // is also applied by refresh() after onConfigure.
          durationWidget.options.precision = 1;
        }
      };
      const refresh = () => {
        formatDuration();
        updatePanel(panel, skillWidget?.value);
      };
      const applyStaticNodeSize = () => {
        this.setSize([this.size[0], STATIC_NODE_HEIGHT]);
        this.graph?.setDirtyCanvas(true, true);
      };
      if (skillWidget) {
        const originalCallback = skillWidget.callback;
        skillWidget.callback = function (value, ...args) {
          const callbackResult = originalCallback?.call(this, value, ...args);
          updatePanel(panel, value);
          return callbackResult;
        };
      }
      refresh();
      applyStaticNodeSize();

      const originalConfigure = this.onConfigure;
      this.onConfigure = function () {
        const configureResult = originalConfigure?.apply(this, arguments);
        queueMicrotask(() => {
          refresh();
          applyStaticNodeSize();
        });
        return configureResult;
      };
      return result;
    };
  },
});
