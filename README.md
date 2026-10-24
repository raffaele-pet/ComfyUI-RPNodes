# ComfyUI RPNodes

Utility nodes for ComfyUI, listed in the same order as the example workflow:

- [Image saving](#image-saving)
- [Image sizing and resizing](#image-sizing-and-resizing)
- [Workflow utilities](#workflow-utilities)
- [Face detection](#face-detection)
- [Image folder processing](#image-folder-processing)
- [Video frame processing](#video-frame-processing)
- [MiniMax H3 keyframes](#minimax-h3-keyframes)

## Image saving

Both nodes support a display name and a download filename template. Right-click
the preview and choose **Save Image** to download a PNG.

### RP Preview Image

Displays an image or image batch with its pixel dimensions and aspect ratio.
Use it to inspect processing results and download them with a chosen filename.
Its `images` output passes the input through unchanged.

### RP Image Comparer

Compares two images in one preview using a wipe, click, or side-by-side view.
Use it to check differences between an original image and a processed version.
Either image can be downloaded separately. The `images` output returns
`image_a` unchanged, including its full batch.

The default save-name template, `%display_name-date:yyyy-MM-dd_HHmmss%`, combines
the display name with a timestamp. Custom templates also support
`%display_name%` and `%date:...%`.

## Image sizing and resizing

The smart image nodes share model-specific resolution and aspect-ratio presets.
The catalog includes Boogu-Image-0.1, FireRed-Image-Edit-1.0, FLUX.2/3 Klein,
HiDream-O1-Image, Ideogram 4, Krea 2, MOSS-Image, Qwen-Image, SDXL, and
Z-Image-Turbo. New smart image nodes default to FLUX.3 Klein.

See [Smart image preset sources](./docs/smart-image-presets.md) for model
variants, supported tiers, pixel-grid rounding, and megapixel conventions.

### RP Smart Image Size

Provides width, height, aspect ratio, resolution, and megapixels from a model's
size presets. Use its outputs to configure image generation, latent dimensions,
or other nodes that need an explicit image size.

`aspect_ratio` reports the exact ratio of the selected dimensions;
`aspect_ratio_approx` identifies the preset ratio. For FLUX.3 Klein,
`megapixels` carries the selected target budget for downstream reference sizing.

### RP Smart Image Resize

Resizes an image, a mask, or both to a model's dimensions while keeping image
and mask aligned. Automatic mode selects the closest aspect-ratio preset;
manual mode lets you choose dimensions directly. It supports proportional
resizing, stretching, cropping, and padding.

`pad` uses a chosen color, `pad_edge` uses each edge's mean color, and
`pad_edge_pixel` extends its individual pixels. `pillarbox_blur` fills the
background with a blurred version of the image.

`resized_image_no_pad` returns a proportional resize without cropping or
padding, independently of the main output mode. The width, height, mask, and
ratio outputs describe the main result. The optional `resolution` input can
set its longer-side size; `megapixels` can drive downstream reference sizing.

### RP Image Minimum 1K

Enlarges images whose longer side is below 1024 pixels while preserving their
proportions. Images already at or above that size remain unchanged.

### RP Tile Split & RP Tile Merge

Model-independent image editing with exact **1024 × 1024** tiles. Choose a tile
count (4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, or 25); Split selects the grid
closest to the source proportions, scales the whole image once, and extends
edge pixels to pad the outer tiles. Overlap is automatic: 192–288 pixels,
increasing with the tile count. No per-tile stretching or resolution presets.

Connect Split's `tiles` **list** to the per-tile editing chain, decoded tiles to
Merge's `tiles`, and `tile_context` directly between the two nodes. Preserve the
list order. Merge applies normalized cosine feathering at the saved positions,
removes padding, and integrates native ComfyUI color matching per tile and
globally (default strengths 0.75 and 0.25). Padding is excluded from both color
statistics and the output. `scaled_image`, `tile_preview`, and `layout` show
the comparison reference, tile contact sheet, and computed dimensions.

Tile count controls the output pixel budget; very large inputs can be reduced.
Proportions are preserved to the nearest pixel, without cropping. Fixed tile
geometry prevents sizing drift; an editing model can still move scene details.
The nodes support IMAGE batches and require a current ComfyUI V3 API and its
native ColorTransfer node. See [tile editing](./docs/tile-editing.md) for wiring
and overlap calculations.

## Workflow utilities

These nodes control the editor directly, without running the queue.

### RP Bypasser

Enables or bypasses individual processing nodes connected to it. Each row
controls one source; **Bypass all / Enable all** controls all connected sources.
The controls follow the actual node modes, including manual changes.

### RP Groups Muter

Mutes or enables whole workflow groups. Each row controls one group, and
**Mute all / Unmute all** controls all listed groups together.

**Hide nested groups** hides child-group rows while their nodes remain
controlled by the outer group. The node properties can filter group names and
colors, change sorting, and show navigation arrows. Rows and buttons stay
synchronized with the actual node modes.

### RP Fix Workflow

Repairs broken links and invalid references in the current workflow, including
subgraphs. Existing nodes, settings, and valid connections are kept.
**Fix workflow and save** also saves under the current workflow name,
overwriting its saved file.

### RP Import Workflow

Adds one or more saved workflows to the current canvas while preserving
existing nodes and valid connections. Use it to combine workflow branches
without replacing the workflow already open. Imported nodes can be placed
beside or below the current content and wrapped in groups.

Supports saved workflow JSON v0.4 and v1. Source files are unchanged.
Importing is adapted from
[comfyui-workflow-merger](https://github.com/raffaele-pet/comfyui-workflow-merger).

## Face detection

### RP Face Detection

Detects faces and returns aligned crops, bounding boxes, and masks so each face
can be processed separately. The box can be resized, scaled around its original
center, or made square. **show_advanced** exposes the detection and crop settings.

`pad_outside_image` offers `none` (clip at image borders), `pad` (black),
`pad_edge` (mean edge colors), and `pad_edge_pixel` (extend edge pixels).
Padding preserves the full requested box without stretching image content.

Connect `original_images`, the processed `cropped_images`, and `bboxes` to
KJNodes **BatchUncrop** to return faces to their original positions. When padding
is enabled, `original_images` includes the expanded canvas.

- `bbox_masks` shows the detection rectangles.
- `image_area_mask` matches `original_images`: white for original image pixels,
  black for padding.
- `cropped_image_mask` shows the same valid image area within `cropped_images`.

Outputs are lists with one item per detected face. Requires **ComfyUI Impact
Pack**, **Impact Subpack**, and an installed detector model; the preferred model
is `bbox/face_yolov8m.pt`.

## Image folder processing

Processes folder contents one image at a time. The processing chain runs between
the loader and the loop saver, linked by `flow` and `image_context`.
Relative input folders use `ComfyUI/input`; relative output folders use
`ComfyUI/output`.

### RP Load Images from Folder

Loads images from a folder one at a time in natural filename order. It supplies
the current image, mask, and image context to the processing chain. Supports
BMP, JPEG, PNG, TIFF, and WebP.

### RP Save Images to Folder

Saves each processed result with its original filename and format, then advances
the loop to the next image. Place it at the end of the processing chain; one run
processes the folder. Use one loop saver per loader.

`overwrite` controls replacement of existing files. `clear_output_folder`
removes supported image files from the destination at the start of a run.

### RP Save Image to Folder (No Loop)

Saves an intermediate result and passes the image through unchanged. It does
not start or advance the loop, so it can save multiple processing stages before
the final loop saver.

### RP Load Prompt from File

Reads the prompt associated with the current image using `image_context`.
Use it to assign a different prompt to each image in a folder loop. Prompt
blocks are separated by a line containing `*` by default; the node can check
that prompt and image counts match.

## Video frame processing

Processes a video one frame at a time. The two nodes share `flow` and
`video_context`, with the image-processing chain between them. Extracted and
processed frames remain available in their output folders. FFmpeg is provided
through the package requirements.

### RP Video to Frames

Extracts a video into PNG frames and starts the frame-processing loop. It
preserves the source timing and audio information for rebuilding the video.

### RP Frames to Video

Saves the processed frames and rebuilds the video with its original frame rate
and optional source audio. It also provides an in-node preview. Existing videos
receive a numbered filename unless `overwrite` is enabled.

## MiniMax H3 keyframes

### RP H3-Keyframes

Uses an ordered sequence of still images to guide MiniMax H3 video generation.
The images are distributed automatically across the video duration, with up to
32 keyframes. Connecting an image reveals the next input.

The node needs H3 conditioning, its video VAE, and the AV latent. It returns
conditioning with the connected keyframes in order. Each image input accepts
one still image; images can be fitted by centered cropping or stretching.

## Installation

Open a terminal in `ComfyUI/custom_nodes` and run:

```bash
git clone https://github.com/raffaele-pet/ComfyUI-RPNodes.git
python -m pip install -r ComfyUI-RPNodes/requirements.txt
```

Restart ComfyUI and refresh the browser. Find the nodes by their RP names in
the node search.

## Example workflow

[`RP_All_Nodes_Examples.json`](./example_workflows/RP_All_Nodes_Examples.json)
contains all 17 RP nodes in 15 example groups, in the same order as this README.
Each group has a short English and Italian note describing its purpose.

Load the JSON through ComfyUI's workflow menu or drag it onto the canvas.
Choose your files and models before running the relevant groups. Workflow
utility buttons work without queue execution.
