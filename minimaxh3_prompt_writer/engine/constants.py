"""Stable names, independent high-level profiles, and provenance metadata.

The profiles below use generic genre conventions. They are not copies,
translations, or detailed distillations of MiniMax documentation and do not
implement the upstream agentic workflows.
"""

from __future__ import annotations

from dataclasses import dataclass


BUNDLE_VERSION = "1.4.15+static-node-layout.2026-08-07"
H3_SKILLS_COMMIT = "093f3129a3f7bd27c74928b1cd31a54fbdebe057"
COMFYUI_REFERENCE_COMMIT = "7972b5ba7f1597f68261be33c912f5e5dba8b9c0"
FPS = 24
MAX_H3_PROMPT_CHARS = 7000

SKILL_AUTO = "Auto — choose the best H3 skill"
SKILL_CORE = "H3 Prompt Writing (Core)"
SKILL_PRODUCT = "Minimalist Product Ad"
SKILL_3D = "3D Animation Short"
SKILL_PAPERCRAFT = "Papercraft Stop-Motion Explainer"
SKILL_BRAND = "Brand Promo Video"
SKILL_MV = "Music Video / Lyric Typography"
SKILL_COOP = "Co-op Game Intro"
SKILL_COLLAGE = "Paper Collage Explainer"
SKILL_HANDDRAWN = "Hand-drawn + Live Action"

SKILL_CHOICES = (
    SKILL_AUTO,
    SKILL_CORE,
    SKILL_PRODUCT,
    SKILL_3D,
    SKILL_PAPERCRAFT,
    SKILL_BRAND,
    SKILL_MV,
    SKILL_COOP,
    SKILL_COLLAGE,
    SKILL_HANDDRAWN,
)


@dataclass(frozen=True)
class SkillProfile:
    identifier: str
    label: str
    use_when: str
    directives: str
    upstream_folder: str | None = None
    preview_filename: str | None = None

    @property
    def upstream_skill_url(self) -> str | None:
        if self.upstream_folder is None:
            return None
        return (
            "https://github.com/MiniMax-AI/MiniMax-H3/tree/"
            f"{H3_SKILLS_COMMIT}/skills/{self.upstream_folder}"
        )

    @property
    def upstream_preview_url(self) -> str | None:
        if self.preview_filename is None:
            return None
        return (
            "https://github.com/MiniMax-AI/MiniMax-H3/blob/"
            f"{H3_SKILLS_COMMIT}/assets/{self.preview_filename}"
        )


SKILL_PROFILES = {
    SKILL_CORE: SkillProfile(
        identifier="core-h3",
        label=SKILL_CORE,
        use_when="No specialized visual-production grammar is needed.",
        directives=(
            "Apply only the H3 mode, chronology, fidelity, camera, dialogue, "
            "soundscape, and output-schema contract. Do not impose a style that "
            "the user or connected media did not establish."
        ),
    ),
    SKILL_PRODUCT: SkillProfile(
        identifier="minimalist-product-ad-generator",
        label=SKILL_PRODUCT,
        use_when="A clean premium product short, launch shot, or e-commerce ad is requested.",
        upstream_folder="minimalist-product-ad-generator",
        preview_filename="minimalist-product-ad-generator.gif",
        directives=(
            "Use a product-centered composition, restrained pacing, controlled "
            "lighting, and legible motion. Preserve verified product geometry, "
            "materials, colors, and supplied text; do not invent product facts."
        ),
    ),
    SKILL_3D: SkillProfile(
        identifier="3d-animation-short-generator",
        label=SKILL_3D,
        use_when="A stylized 3D character animation or compact narrative short is requested.",
        upstream_folder="3d-animation-short-generator",
        preview_filename="3d-animation-short-generator.gif",
        directives=(
            "Use coherent stylized 3D animation, readable silhouettes, expressive "
            "performance, and physically weighted motion. Repeat only verified "
            "identity anchors and fit the action to the available duration."
        ),
    ),
    SKILL_PAPERCRAFT: SkillProfile(
        identifier="papercraft-stop-motion-explainer",
        label=SKILL_PAPERCRAFT,
        use_when="Knowledge or a concept should be explained as a tactile paper diorama.",
        upstream_folder="papercraft-stop-motion-explainer",
        preview_filename="papercraft-stop-motion-explainer.gif",
        directives=(
            "Use a tactile layered paper-diorama look, visibly handmade stepped "
            "motion, modest miniature-scale camera work, and synchronized paper "
            "sounds. Keep any educational content accurate to the task data."
        ),
    ),
    SKILL_BRAND: SkillProfile(
        identifier="brand-promo-video-generator",
        label=SKILL_BRAND,
        use_when="A factual launch, website, app, shop, or brand capability promo is requested.",
        upstream_folder="brand-promo-video-generator",
        preview_filename="brand-promo-video-generator.gif",
        directives=(
            "Build the promo around user-supplied brand facts and demonstrable "
            "features. Keep copy readable, pacing clear, and product or interface "
            "sounds precise; never fabricate claims, metrics, marks, or endorsements."
        ),
    ),
    SKILL_MV: SkillProfile(
        identifier="music-video-subtitle-generator",
        label=SKILL_MV,
        use_when="Lyrics, beat-reactive typography, or an audio-led music-video sequence is requested.",
        upstream_folder="mv-subtitle-skill-confirmed",
        preview_filename="music-video-subtitle-generator.gif",
        directives=(
            "When audio is connected, align motion, cuts, and requested lyric "
            "typography to its timing. Preserve supplied lyrics and reference "
            "roles, keep visible text readable, and mark uncertain words as unclear."
        ),
    ),
    SKILL_COOP: SkillProfile(
        identifier="co-op-game-intro-generator",
        label=SKILL_COOP,
        use_when="A two-player game menu, loadout confirmation, or cooperative opening is requested.",
        upstream_folder="co-op-game-intro-generator",
        preview_filename="co-op-game-intro-generator.gif",
        directives=(
            "Preserve both players' verified identities, sides, colors, names, "
            "selections, and interface ownership. Make menu-to-world progression "
            "legible and never swap player attributes or supplied UI text."
        ),
    ),
    SKILL_COLLAGE: SkillProfile(
        identifier="paper-collage-explainer-generator",
        label=SKILL_COLLAGE,
        use_when="An idea should become tactile editorial collage or social B-roll.",
        upstream_folder="paper-collage-explainer-generator",
        preview_filename="paper-collage-explainer-generator.gif",
        directives=(
            "Use tactile editorial paper collage, layered cut-outs, restrained "
            "assembly motion, contact shadows, and synchronized paper sounds. "
            "Respect connected keyframes instead of imposing a fixed opening."
        ),
    ),
    SKILL_HANDDRAWN: SkillProfile(
        identifier="handdrawn-live-video-generator",
        label=SKILL_HANDDRAWN,
        use_when="A rough glowing drawing should interact with and escape through a live-action space.",
        upstream_folder="handdrawn-live-video-generator",
        preview_filename="handdrawn-live-video-generator.gif",
        directives=(
            "Blend visibly handmade drawing with live action through clear "
            "physical contact, continuous transformation, and reactive camera "
            "behavior. Preserve rough stroke texture and fit the arc to duration."
        ),
    ),
}

SKILL_BY_ID = {profile.identifier: profile for profile in SKILL_PROFILES.values()}


def get_skill_profile(label: str) -> SkillProfile:
    """Return a known profile, falling back to the core contract."""

    return SKILL_PROFILES.get(label, SKILL_PROFILES[SKILL_CORE])


def skill_catalog_for_classifier() -> str:
    lines = []
    for label in SKILL_CHOICES:
        if label == SKILL_AUTO:
            continue
        profile = get_skill_profile(label)
        lines.append(f"- {profile.identifier}: {profile.use_when}")
    return "\n".join(lines)
