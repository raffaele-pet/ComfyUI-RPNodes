from .selective_video import RPSelectiveVideoEdgeFill, RPSelectiveVideoInput


NODE_CLASS_MAPPINGS = {
    "RPSelectiveVideoInput": RPSelectiveVideoInput,
    "RPSelectiveVideoEdgeFill": RPSelectiveVideoEdgeFill,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RPSelectiveVideoInput": RPSelectiveVideoInput.title,
    "RPSelectiveVideoEdgeFill": RPSelectiveVideoEdgeFill.title,
}
