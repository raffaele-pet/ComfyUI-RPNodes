"""RP H3 keyframe node registration."""

from .nodes import RPH3Keyframes


NODE_CLASS_MAPPINGS = {"RPH3Keyframes": RPH3Keyframes}
NODE_DISPLAY_NAME_MAPPINGS = {"RPH3Keyframes": "RP H3-Keyframes"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
