from .streaming_video_nodes import RPLoadVideo, RPPreviewVideo
from .video_frames_workflow import (
    NODE_CLASS_MAPPINGS as FRAME_NODE_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as FRAME_NODE_NAMES,
)

NODE_CLASS_MAPPINGS = {
    **FRAME_NODE_CLASSES,
    "RPLoadVideo": RPLoadVideo,
    "RPPreviewVideo": RPPreviewVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **FRAME_NODE_NAMES,
    "RPLoadVideo": "RP Load Video (Upload)",
    "RPPreviewVideo": "RP Preview Video",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
