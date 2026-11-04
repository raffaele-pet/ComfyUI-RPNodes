# RP Dense Grid Align

Located in `RP/image_sizing_and_resizing`. Deforms an edited RGB/RGBA image
toward the original geometry while keeping the edited resolution and detail.
No reference pixels are composited into the result and no colors are matched.

Connect the original image to `reference` and the edited image to `image`.
The reference is resized to the edited dimensions when necessary. In the tiled
FLUX workflow, use `RP Tile Split.scaled_image` as reference and the output of
`RP Tile Merge` as image. Matching the already scaled reference avoids another
resampling. The node also works on corresponding tile lists before merging.

## How it works

1. Locally normalize luminance and suppress fine texture so sharpening and
   color changes are less likely to be mistaken for movement.
2. Estimate dense DIS optical flow in both directions. Reject correspondences
   outside the displacement bound, with weak local structural correlation, or
   inconsistent forward/backward matches. A candidate must improve alignment
   over the unchanged position.
3. Place accepted motions on a dense elastic grid. Already aligned regions
   supply fixed zero-motion anchors; uncertain regions have a weak zero-motion
   prior instead of being forced to match invented details.
4. Smooth the grid, suppress insignificant motion, and constrain neighboring
   vertices to prevent inverted cells. Interpolate a sampling map and warp the
   edited image once. All RGB/RGBA channels follow the same map.

The regular grid supplies coordinates, not semantic identity. It cannot ensure
that newly generated eyes, letters, repeated patterns or missing objects have
one unique correspondence. Hard automatic anchors are based on the matching
evidence and can be mistaken. Use `protect_mask` to explicitly freeze areas.
White protected pixels and areas with zero displacement retain the edited
pixels exactly, including alpha.

## Controls

| Input | Meaning |
| --- | --- |
| strength | Fraction of the estimated deformation to apply; 0 leaves the edit unchanged. |
| grid_spacing | Elastic grid spacing in output pixels; default 16. Smaller grids allow more local movement. |
| max_shift | Reject estimated motions larger than this many output pixels; default 96. |
| anchor_tolerance | Suppress tiny displacements and identify aligned regions; default 1.5 output pixels. |
| protect_mask | Optional white areas that must stay unchanged. Nearby grid vertices are pinned too. |
| rigidity | Advanced: larger values smooth the elastic deformation more strongly. |
| analysis_max_side | Advanced: maximum matching resolution; default 2048. Raise to 4096 for finer image evidence. |
| minimum_match | Advanced: local correlation threshold; raise it to reject more ambiguous matches. |

The matching resolution and the elastic grid density are separate controls.
A denser grid does not create image evidence beyond the matching resolution.
Repetitive or reconstructed content may need a higher threshold or protection.

## Outputs

- `aligned_image`: edited image after local geometric correction.
- `deformation_grid`: fixed reference grid beside its corresponding sampling
  grid on the edit. Green dots are fixed anchors; orange dots are displaced
  control points; gray dots have no significant motion. For readability the preview displays a
  subset of the full grid. The solver still uses all points.
- `correction_mask`: displacement magnitude divided by `max_shift`; black means
  exactly no warp. This is a motion visualization, not a confidence mask.
- `report`: actual grid size, anchor count, maximum applied displacement and
  fraction of pixels retaining zero displacement.

Single references/masks broadcast across IMAGE batches. Otherwise batch counts
must match. CPU OpenCV is sufficient; no weights, downloads, network requests
or CUDA extensions are needed. If OpenCV is absent, the node reports the missing
dependency when executed rather than preventing other RP nodes from loading.

The node can reduce local shifts, scale changes and elastic distortions. It
cannot recover information cropped away or unmix double contours already
blended from two different generated tiles. Aligning individual decoded tiles
to their original tiles **before** RP Tile Merge is worth testing when such
ghosting is the main issue.

See `example_workflows/RP_Dense_Grid_Align.json` for a model-free example.
