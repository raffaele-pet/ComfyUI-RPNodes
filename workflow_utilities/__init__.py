class RPImportWorkflow:
    """Editor control; importing is performed by the browser extension."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow": (["(select a saved workflow)"],),
                "placement": (["right", "below"],),
                "gap": ("INT", {"default": 160, "min": 0, "max": 10000}),
                "group_import": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "RP/workflow_utilities"
    DESCRIPTION = (
        "Import saved workflow JSON files into the current canvas, preserving "
        "their internal connections. Use the buttons on the node; no queue is needed."
    )

    def execute(self, workflow, placement="right", gap=160, group_import=True):
        return ()


class RPFixWorkflow:
    """Editor control for cleaning the current workflow."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "RP/workflow_utilities"
    DESCRIPTION = (
        "Remove stale links and references from the current workflow, including "
        "subgraphs. Click Fix workflow; no queue is needed."
    )

    def execute(self):
        return ()


class RPGroupsMuter:
    """Editor control for muting and enabling workflow groups."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "RP/workflow_utilities"
    DESCRIPTION = (
        "Mute or enable groups using toggles that reflect the actual node modes. "
        "Filter groups in the node properties; no queue is needed."
    )

    def execute(self):
        return ()


NODE_CLASS_MAPPINGS = {
    "RPImportWorkflow": RPImportWorkflow,
    "RPFixWorkflow": RPFixWorkflow,
    "RPGroupsMuter": RPGroupsMuter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RPImportWorkflow": "RP Import Workflow",
    "RPFixWorkflow": "RP Fix Workflow",
    "RPGroupsMuter": "RP Groups Muter",
}
