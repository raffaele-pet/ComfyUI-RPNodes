from .save_transparent_video import RPSaveTransparentVideo
from .selective_video import RPSelectiveVideoEdgeFill, RPSelectiveVideoGuides, RPSelectiveVideoInput


NODE_CLASS_MAPPINGS = {
    "RPSelectiveVideoInput": RPSelectiveVideoInput,
    "RPSelectiveVideoEdgeFill": RPSelectiveVideoEdgeFill,
    "RPSelectiveVideoGuides": RPSelectiveVideoGuides,
    "RPSaveTransparentVideo": RPSaveTransparentVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RPSelectiveVideoInput": RPSelectiveVideoInput.title,
    "RPSelectiveVideoEdgeFill": RPSelectiveVideoEdgeFill.title,
    "RPSelectiveVideoGuides": RPSelectiveVideoGuides.title,
    "RPSaveTransparentVideo": RPSaveTransparentVideo.title,
}
