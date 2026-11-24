from .selective_video import RPSelectiveVideoEdgeFill, RPSelectiveVideoGuides, RPSelectiveVideoInput


NODE_CLASS_MAPPINGS = {
    "RPSelectiveVideoInput": RPSelectiveVideoInput,
    "RPSelectiveVideoEdgeFill": RPSelectiveVideoEdgeFill,
    "RPSelectiveVideoGuides": RPSelectiveVideoGuides,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RPSelectiveVideoInput": RPSelectiveVideoInput.title,
    "RPSelectiveVideoEdgeFill": RPSelectiveVideoEdgeFill.title,
    "RPSelectiveVideoGuides": RPSelectiveVideoGuides.title,
}
