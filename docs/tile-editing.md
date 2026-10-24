# RP Tile Split / RP Tile Merge

Both nodes live in `RP/image_sizing_and_resizing`. They process RGB images;
alpha is discarded before splitting. The splitter outputs a real ComfyUI image
list, not one image batch, so standard editing nodes execute once per tile.

```text
Image → RP Tile Split.tiles → per-tile prompt / reference / sampler → VAE Decode
                  tile_context ────────────────────────────────→ RP Tile Merge
                                                               ↑ tiles
                                                          VAE Decode
```

The `tiles` list is ordered by source image, then row, then column. Connect it
to both the prompt expander's image input and the editing reference input.
Connect decoded tiles directly to Merge; no batch/list conversion is needed.
Merge also accepts one IMAGE batch containing all tiles in the same order.
Do not filter, shuffle, or change the number or dimensions of decoded tiles.

For FLUX.3, use `ReferenceMultiLatent` with **keep resolution** and its
`empty_latent` output for sampling. Every reference and decoded tile must remain
1024 × 1024. Merge detects incomplete lists or incorrect tile dimensions instead
of silently assembling them at incorrect positions.

## Geometry

For tile count N, overlap O is `16 × round((128 + 32 × sqrt(N)) / 16)`.
This density heuristic gives progressively wider transitions as the grid gains
seams; it is not a model-specific quality guarantee. For each factorization
`rows × columns = N`, the canvas dimensions are:

```text
stride = 1024 - O
canvas_width  = 1024 + (columns - 1) × stride
canvas_height = 1024 + (rows - 1) × stride
```

The selected grid minimizes the absolute log difference between canvas and
source aspect ratios. The image is proportionally fitted inside this canvas in
one resize, with dimensions rounded to the nearest integer pixel. Padding is
centered and fills only the unused canvas area by extending source edge pixels.
Tile origins are integer multiples of the common stride. No tile is resized
independently, and no source content is cropped.

`tile_context` carries these exact positions, padding, batch count, and the
scaled unpadded source. It belongs to this splitter execution; connect it to the
matching merge. Merge crops each tile to its real source area before color
matching and accumulation. Complementary cosine ramps feather overlapping
regions; a weight accumulator normalizes every output pixel, including corners
and external edges. Padding never appears in the assembled result.

## Color matching and output

Merge reuses native ComfyUI ColorTransfer (`reinhard_lab`, `mkl_lab`, or
`histogram`). Each processed tile is matched against its corresponding valid
source region, followed by an optional global match to the unpadded scaled
source. Strength 0 disables either stage. Defaults preserve the previous
workflow's 0.75 per-tile and 0.25 global strengths.

The result has the dimensions reported by Split's `layout`; it is not forced
back to the input resolution. Increasing the tile count usually increases
available detail, while very large inputs may be downsampled to the selected
budget. Integer dimensions can change the aspect ratio by less than one pixel
of rounding. Perfectly square tiles prevent geometry drift from independent
resizing, but cannot prevent the editing model from moving objects or inventing
details inside overlaps.

`RP_All_Nodes_Examples.json` includes a model-free Split → Merge round trip.
`FLUX.3 Klein Distilled - RP Tiled Upscale.json` includes the full per-tile
prompting and FLUX.3 editing chain.
