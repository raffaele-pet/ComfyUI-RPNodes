from .save_transparent_video import RPSaveTransparentVideo
from .selective_video import RPSelectiveVideoEdgeFill, RPSelectiveVideoGuides


NODE_CLASS_MAPPINGS = {
    "RPSelectiveVideoEdgeFill": RPSelectiveVideoEdgeFill,
    "RPSelectiveVideoGuides": RPSelectiveVideoGuides,
    "RPSaveTransparentVideo": RPSaveTransparentVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RPSelectiveVideoEdgeFill": RPSelectiveVideoEdgeFill.title,
    "RPSelectiveVideoGuides": RPSelectiveVideoGuides.title,
    "RPSaveTransparentVideo": RPSaveTransparentVideo.title,
}
