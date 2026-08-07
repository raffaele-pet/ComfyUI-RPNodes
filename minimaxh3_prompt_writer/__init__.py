"""MiniMax H3 prompt-writer node registration."""

from .nodes import RPH3I2VPromptWriter, RPH3REF2VPromptWriter


# ComfyUI V3 node classes also expose the V1-compatible interface used by the
# repository's shared root loader. Registering them here keeps all existing
# RPNodes available while preserving the tested V3 implementation.
NODE_CLASS_MAPPINGS = {
    "RPH3I2VPromptWriter": RPH3I2VPromptWriter,
    "RPH3REF2VPromptWriter": RPH3REF2VPromptWriter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RPH3I2VPromptWriter": "RP H3-I2V Prompt Writer",
    "RPH3REF2VPromptWriter": "RP H3-REF2V Prompt Writer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
