from .streaming_video_nodes import RPLoadVideo, RPPreviewVideo

NODE_CLASS_MAPPINGS = {
    "RPLoadVideo": RPLoadVideo,
    "RPPreviewVideo": RPPreviewVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RPLoadVideo": "RP Load Video",
    "RPPreviewVideo": "RP Preview Video",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
