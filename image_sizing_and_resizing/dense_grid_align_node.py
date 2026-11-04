import numpy as np
import torch

from comfy_api.latest import io

from .dense_grid_align import estimate_deformation, grid_preview, warp_image


class RPDenseGridAlign(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RPDenseGridAlign", display_name="RP Dense Grid Align", category="RP/image_sizing_and_resizing",
            description="Locally deform the edited image toward the reference using a dense elastic grid, confidence checks and fixed anchors. White protect-mask areas stay unchanged. Keeps the edited resolution and RGB/RGBA detail.",
            inputs=[
                io.Image.Input("reference", tooltip="Original image used only to estimate geometry. It is resized to the edited resolution if needed."),
                io.Image.Input("image", tooltip="Edited image to realign. Its resolution and colors are preserved."),
                io.Mask.Input("protect_mask", optional=True, tooltip="White pixels must not move. This overrides automatic matching."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.05),
                io.Int.Input("grid_spacing", default=16, min=4, max=128, step=4, tooltip="Grid spacing in output pixels. Smaller values allow more local deformation."),
                io.Float.Input("max_shift", default=96, min=1.0, max=512.0, step=1.0, tooltip="Reject correspondences beyond this displacement in output pixels."),
                io.Float.Input("anchor_tolerance", default=1.5, min=0.0, max=8.0, step=0.25, tooltip="Suppress tiny motions and pin already aligned regions."),
                io.Float.Input("rigidity", default=0.5, min=0.05, max=4.0, step=0.05, advanced=True, tooltip="Higher values make the elastic grid smoother."),
                io.Int.Input("analysis_max_side", default=2048, min=256, max=4096, step=256, advanced=True),
                io.Float.Input("minimum_match", default=0.55, min=0.1, max=0.95, step=0.05, advanced=True, tooltip="Higher values accept fewer, more reliable local correspondences."),
            ],
            outputs=[
                io.Image.Output(display_name="aligned_image"),
                io.Image.Output(display_name="deformation_grid"),
                io.Mask.Output(display_name="correction_mask"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, reference, image, strength=1.0, grid_spacing=16, max_shift=96,
                anchor_tolerance=1.5, rigidity=0.5, analysis_max_side=2048,
                minimum_match=0.55, protect_mask=None):
        batch = image.shape[0]
        if reference.shape[0] not in (1, batch):
            raise ValueError("RP Dense Grid Align: reference must contain one image or match the edited batch size.")
        if protect_mask is not None and protect_mask.shape[0] not in (1, batch):
            raise ValueError("RP Dense Grid Align: protect_mask must contain one mask or match the edited batch size.")
        if image.shape[-1] not in (3, 4) or reference.shape[-1] not in (3, 4):
            raise ValueError("RP Dense Grid Align: use RGB or RGBA images.")
        edited = image.detach().to(device='cpu', dtype=torch.float32).numpy()
        original = reference.detach().to(device='cpu', dtype=torch.float32).numpy()
        masks = protect_mask.detach().to(device='cpu', dtype=torch.float32).numpy() if protect_mask is not None else None
        aligned, previews, corrections, reports = [], [], [], []
        for index in range(batch):
            ref = original[min(index, original.shape[0] - 1)]
            edited_frame = edited[index]
            mask = masks[min(index, masks.shape[0] - 1)] if masks is not None else None
            field, mesh, pinned, stats = estimate_deformation(
                ref, edited_frame, max_shift, grid_spacing, anchor_tolerance,
                rigidity, analysis_max_side, minimum_match, mask)
            result = warp_image(edited_frame, field, strength) if strength else edited_frame.copy()
            aligned.append(torch.from_numpy(result))
            previews.append(torch.from_numpy(grid_preview(ref, edited_frame, mesh, pinned, strength)))
            magnitude = np.linalg.norm(field * strength, axis=-1)
            corrections.append(torch.from_numpy(np.clip(magnitude / max(max_shift, 1), 0, 1)))
            reports.append(
                f'Image {index + 1}: {image.shape[2]} x {image.shape[1]}; '
                f'{stats["grid_columns"]} x {stats["grid_rows"]} grid ({stats["grid_points"]} points); '
                f'{stats["anchored_points"]} fixed anchors; '
                f'max correction {stats["max_displacement"] * strength:.2f} px; '
                f'unchanged pixels {np.mean(magnitude == 0):.1%}; '
                f'analysis {stats["analysis_size"][0]} x {stats["analysis_size"][1]}.'
            )
        return io.NodeOutput(
            torch.stack(aligned).to(device=image.device, dtype=image.dtype),
            torch.stack(previews).to(device=image.device, dtype=image.dtype),
            torch.stack(corrections).to(device=image.device, dtype=image.dtype),
            '\n'.join(reports),
        )


NODE_CLASS_MAPPINGS = {"RPDenseGridAlign": RPDenseGridAlign}
NODE_DISPLAY_NAME_MAPPINGS = {"RPDenseGridAlign": "RP Dense Grid Align"}
