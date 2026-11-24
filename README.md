# ComfyUI RPNodes

Utility nodes for ComfyUI:

- [Image saving](#image-saving)
- [Image sizing and resizing](#image-sizing-and-resizing)
- [Workflow utilities](#workflow-utilities)
- [Face detection](#face-detection)
- [Image folder processing](#image-folder-processing)
- [Video frame processing](#video-frame-processing)
- [Video conditioning](#video-conditioning)

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
statistics and the output. `scaled_image` and `layout` show the comparison
reference and computed dimensions. Connect `tiles` to an image preview to
inspect the individual tiles.

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

Adds saved workflows, external JSON files, or other open workflow tabs to the
current canvas while preserving existing nodes and valid connections. Use it
to combine workflow branches without replacing the workflow already open. Imported nodes can be placed
beside or below the current content and wrapped in groups.

Select a tab in **open_tab** and click **Import open tab** to copy its current
contents, including unsaved changes. The list follows the other open tabs;
the current tab is excluded. The source tab stays open and is not saved or modified.

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

`max_faces` selects the largest masks; `0` keeps all in detector order.
`face_count_policy` defaults to `all`. `single_or_error` requires one face;
`single_or_skip` emits `skip_reason` for zero or multiple faces, before applying
the limit. Connect it to the folder saver to skip the edit and continue.

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

Requires one result per source image. Connect `skip_reason` to skip without
saving; keep this saver as the only output after face detection.
`on_error=skip_image` skips invalid results and per-file save errors, logging
the filename and reason; the default is `stop`. Other processing errors still
stop execution. `image_count` includes skips; `last_image_path` is empty if the
last source was skipped.

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

FFmpeg is provided through the package requirements. The video nodes are under
`video/RPNodes` in the node menu.

### RP Load Video

Select a video from ComfyUI's input folder or upload it from the node. Set
`force_rate` to resample the video before skipping or selecting frames.
`skip_first_frames` skips that many frames at the chosen rate,
`select_every_nth` keeps every Nth frame, and `frame_load_cap` limits how many
selected frames are read. Zero leaves the rate or frame cap automatic. Set one
custom dimension to preserve the aspect ratio, or set both for an exact size.
Two zero dimensions keep the source size.

Used on its own, the node returns a standard `IMAGE` batch, its effective FPS
and frame count. `max_batch_mib` limits the standalone batch size. For a large
video, use the connected mode or set a smaller cap or resolution.

### RP Preview Video

Connect an `IMAGE` batch and set `fps` (0 defaults to 24). An optional
`AUDIO` input adds a soundtrack. The node previews a browser-playable H.264
MP4 in ComfyUI's temporary directory. Enable `save_video` to keep the MP4
under `ComfyUI/output` with the chosen `filename_prefix`.

#### Process a video one frame at a time

1. Connect the loader's `video_stream` directly to the preview's `video_stream`.
2. Connect the loader's `images` through an image-processing chain to the
   preview's `images`. The chain must return one image per frame at a fixed size.
3. Run the workflow once. With `fps` at 0, the preview uses the effective
   loader FPS. `keep_source_audio` keeps the source audio with matching timing.

The paired mode decodes and encodes through FFmpeg pipes and releases
frame-dependent workflow results between iterations. It reuses cached model
loads and other frame-independent results. When preserving audio, it stages
one compressed video on disk until the final duration is known, then removes
it after muxing. Keep the streaming image branch connected to the video preview
as its output. The workflow and its models must fit in memory for one frame.
Variable-rate sources are converted to a constant rate.

Import [RP Streaming Video](example_workflows/RP%20Streaming%20Video.json) and
choose a clip on the loader. The example processes each frame with Image Invert.

## Video conditioning

### RP H3-Keyframes

Uses an ordered sequence of still images to guide MiniMax H3 video generation.
The images are distributed automatically across the video duration, with up to
32 keyframes. Connecting an image reveals the next input.

The node needs H3 conditioning, its video VAE, and the AV latent. It returns
conditioning with the connected keyframes in order. Each image input accepts
one still image; images can be fitted by centered cropping or stretching.

### RP Selective Video Guides

Open **Edit video selections** and describe a subject in the first card.
Click **Detect** to save its anchor mask. **Add selection** opens another
card with its own preprocessor, color and tracking range. The color picker
offers a palette and manual color controls. **Select the guide** leaves the
selection available as a mask. Choose a guide to add a preprocessor.

Use the × on an anchor marker to remove the selections made on that frame.
**Fill holes** closes enclosed gaps before expansion, which defaults to
10 source pixels. Lower selections take precedence where their map guides
overlap.

**Track & preview** tracks every selection and prepares the selected
preprocessors at the node's output settings. Switch **Segmentation** and
**Guide** independently while scrubbing. **Apply selections** prepares pending
changes before closing the editor. The workflow reuses the prepared result
while it is cached for the same video, selections and output settings.

**OpenPose · Body, hands and face** and **OpenPose · Face only** render
landmarks for people matched to the selected region. **Tracked rectangle**
follows the tracked mask's main axis on each frame. Drag a guide to move it or
drag the round handle to rotate it. Drag a corner to resize proportionally
and hold **Shift** to resize each axis independently. Use **Reset scale** to
restore 100% or **Reset position** to clear the manual offset and rotation.

The loading controls work on the outputs. `force_rate` changes the sampling
rate before `skip_first_frames` and `select_every_nth`. `frame_load_cap` limits
the resulting frame count. A zero rate or frame cap keeps the original rate
or loads all remaining frames. Set one custom dimension to scale
proportionally, or set both to resize with a center crop. `format` aligns the
dimensions and frame count to a model's grid. The default, **None**, keeps the
source dimensions and count when the other controls are left at their
defaults. The selection editor continues to use source frame numbers.

`original_frames` gives the source RGB frames after sampling and resizing,
with the same shape and frame order as `MASK`.
`frame_count` reports the loaded frame count. `audio` follows the loaded
clip and is empty for silent sources. `video_info` connects to VHS Video Info
nodes. `MASK_IMAGE` gives the white-on-black mask as an RGB image batch.
`mask_video` gives the same frames as a silent **VIDEO** at the processed
clip's frame rate. `MASK` provides a binary batch for nodes that accept
ComfyUI masks directly. All three show the union of the selected regions,
including **Expand** and **Fill holes**. Connect `mask_video` to **Save Video**
to export the mask.

SAM 3.1 tracks each selection in both directions from its anchor.
`frames_map_guide` and `video` carry the source with the selected map guides
applied. `video` also carries the source audio and loaded frame rate.
`frames_transparent_guide` is an RGBA image batch with the transparent guides.
Connect it to **frames** on **RP Save Transparent Video** and connect
`video_info` to keep the loaded frame rate. The saver exports a ProRes 4444
MOV with alpha and plays a browser-compatible preview on a checkerboard.
ComfyUI's component video output uses the average frame rate when exporting
variable-rate sources.

Requires ComfyUI with native SAM 3.1 nodes and
[`sam3.1_multiplex_fp16.safetensors`](https://huggingface.co/Comfy-Org/sam3.1/tree/main/checkpoints)
in `ComfyUI/models/checkpoints`. Depth and OpenPose also use
[comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux)
and its Depth Anything V2 Large and OpenPose weights. Canny and color
segmentation use ComfyUI and SAM 3.1 alone.

The [Pika restoration workflow](./example_workflows/Pika%20H3%20Flex%20-%20Reference%20to%20Video%20Restoration%20%28Selective%20Depth%29.json)
applies depth to the samurai and his two sword hilts, and to the geisha in `Samurai.mp4`. It uses the original RGB
background as part of the H3 reference. It expects the source video, the
three reference images, and the H3 and Pika model files named by the workflow.
The reference branch repeats the final frame to fit H3's temporal grid so that
H3 keeps the source's final frames.

### RP Selective Video Edge Fill

This node uses SAM 3.1 tracking and the same loading controls and mask
outputs as **RP Selective Video Guides**. Each group has **Fill inward from
edges**, enabled by default. Every pixel inside the expanded selection takes
the color of the nearest pixel outside all selected regions. Pixels outside
the selections keep their original colors. Turning off the checkbox leaves
the group's selected area black. A selection that covers the entire frame
also turns black because it has no visible pixels to copy.

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
contains an example for each RP node.
Each group has a short English and Italian note describing its purpose.

Load the JSON through ComfyUI's workflow menu or drag it onto the canvas.
Choose your files and models before running the relevant groups. Workflow
utility buttons work without queue execution.
