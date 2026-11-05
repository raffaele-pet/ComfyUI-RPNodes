# RP Tile Merge Advanced

Located in `RP/image_sizing_and_resizing`. Connect decoded tile lists to `tiles`
and the original RP Tile Split `tile_context` directly to this node. It replaces
RP Tile Merge in the editing chain. RP Tile Split and the original Merge remain
unchanged.

## Alignment before compositing

The node reconstructs each exact original 1024×1024 reference tile from the
saved scaled image, padding and positions. It applies a confidence-tested
elastic grid **before any tile pixels are mixed**.
Already aligned points stay anchored. Padding is protected from matching and
excluded from color statistics and the final image.

All tile grids share the original image as their fixed coordinate system. This
keeps alignment tied to the source rather than accumulating drift by aligning
each tile only to its previous neighbor. Grid density, maximum shift, anchor
tolerance, rigidity, and matching threshold are adjustable. Matching is done at
the full 1024×1024 tile resolution. RGB and RGBA decoded tiles are accepted;
alpha does not drive matching and the final merged image is RGB.

## Structure-guided seams

After per-tile color matching, overlaps are cut along minimum-cost paths. The
cost considers disagreement between the two aligned tiles, strong edges, and
structural similarity to the original reference. Paths in left/top overlaps
are combined to handle four-tile intersections.

The default `seam_feather = 64` applies Laplacian-pyramid multiband blending
around the chosen seam. Color, illumination and texture/focus differences use
broad transitions. Where both tiles contain strong, conflicting contours, the
finest-scale transition narrows to four pixels on each side. This softens
differences in lighting and texture while limiting duplicated fine contours.
It does not blur the final image.

The feather value bounds the transition on each side of the seam. Pixels outside
this band retain the selected tile exactly; uncovered pixels always use the new
tile exactly. Empty canvas pixels never enter the pyramid as black image data.
Try 32–64 for visible seams, or up to 128 for broader tonal differences.
`seam_feather = 0` retains the original hard cut and selects one tile per pixel.
**Previously saved workflows keep their saved value, including 0: change it
manually to enable soft seams.** The integrated global color match is applied
after reconstruction and padding removal.

This reduces shifts and ghosting; it is not a guarantee of perfect geometry.
False correspondences, missing image content, newly invented objects, or two
genuinely different reconstructions can still leave a discontinuity. Hard-cut
mode prevents averaging-induced double exposure; adaptive soft blending limits
it but cannot ensure that every generated shape matches across every seam.

## Controls

| Input | Default | Effect |
| --- | --- | --- |
| color_method | reinhard_lab | Same native ComfyUI color matching as RP Tile Merge. |
| tile_color_strength | 0.75 | Match each aligned tile to its valid original region. |
| global_color_strength | 0.25 | Match the reconstructed image to the scaled original. |
| alignment_strength | 1 | 0 freezes tile geometry; 1 applies the full accepted grid deformation. |
| grid_spacing | 16 | Elastic grid spacing in tile pixels. |
| max_shift | 96 | Maximum accepted displacement, in pixels. |
| anchor_tolerance | 1 | Advanced: suppress insignificant motion and identify stable anchors. |
| seam_feather | 64 | Multiband transition radius, 1–128 px; 0 keeps the hard cut. |
| rigidity | 0.5 | Advanced: higher values smooth the elastic deformation more. |
| minimum_match | 0.55 | Advanced: higher values reject more ambiguous correspondences. |

## Outputs

- `image`: reconstructed RGB image without padding.
- `aligned_tiles`: actual aligned/color-matched RGB/RGBA tiles as an IMAGE list,
  useful for inspecting problems before compositing. Padded margins are retained
  here so each item remains 1024×1024.
- `deformation_grids`: contact sheet of each tile's reference/sampling grids;
  green points are fixed anchors, orange points are displaced.
- `correction_mask`: tile motion divided by `max_shift`, with diagnostic weights
  smoothly interpolated across a soft seam; not a per-frequency sampling map.
- `seam_mask`: boundaries between the selected source tiles, not confidence.
- `report`: total grid points, anchors, maximum shift and per-tile corrections.

One context supports a batch of source images. Keep decoded tiles in splitter
order, without dropping, duplicating, or changing their resolution. The node
receives the whole list in one call and returns one reconstructed IMAGE batch.

CPU OpenCV is sufficient. No alignment weights or model downloads are required.
See the RP Tile Split & RP Tile Merge Advanced group in
`RP_All_Nodes_Examples.json` for a model-free wiring example.
The blending follows the Gaussian/Laplacian pyramid approach described in the
[OpenCV image pyramids tutorial](https://docs.opencv.org/4.13.0/dc/dff/tutorial_py_pyramids.html).
