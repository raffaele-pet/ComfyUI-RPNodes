# ComfyUI RPNodes

A collection of utility nodes for ComfyUI:

- **Image sizing and resizing:** `RP Smart Image Size`, `RP Smart Image Resize`,
  and `RP Image Minimum 1K`
- **Image folder processing:** `RP Load Images from Folder`,
  `RP Load Prompt from File`, `RP Save Image to Folder (No Loop)`, and
  `RP Save Images to Folder`
- **Image saving:** `RP Image Comparer` and `RP Preview Image`
- **Face detection:** `RP Face Detection`
- **Video frame processing:** `RP Video to Frames` and `RP Frames to Video`
- **MiniMax H3 keyframing:** `RP H3-Keyframes`
- **Workflow utilities:** `RP Import Workflow`, `RP Fix Workflow`,
  `RP Groups Muter`, and `RP Bypasser`

Python nodes are grouped in `image_sizing_and_resizing` and
`video_frame_processing`, the folder-processing nodes in
`image_folder_processing`, the keyframe node in `h3_keyframes`, and the workflow
utilities in `workflow_utilities`. Shared
browser extensions live under `web`. The complete example workflow is in
`example_workflows`.

## RP Face Detection

Available under `RP/face_detection`, with Python code in `face_detection`.
Combines **UltralyticsDetectorProvider → BBOX Detector (SEGS) →
SEGS to Mask List → Batch Crop From Mask**, with corrected crop geometry for
scaling and image borders. Requires **ComfyUI Impact Subpack** and
**ComfyUI Impact Pack**; install **ComfyUI KJNodes** to use **BatchUncrop**.
The model selector
uses Impact Subpack's installed models, preferring `bbox/face_yolov8m.pt`.
Models are loaded locally by the original provider.

- `width` and `height` set the white mask rectangle's pixel dimensions;
  **0** preserves the corresponding detected dimension.
- `scale` multiplies those dimensions, default **1.0**. Fractional values
  such as **1.1**, **1.2**, and **1.3** are supported.
- `square` uses the longer original mask side for both sides. Explicit size
  overrides then use the larger requested side, and scale is applied last.
  All sizes use the original detection's center and dimensions. Regions that
  extend beyond the image are clipped without moving their center. A requested
  square may therefore have only a rectangular portion visible at an edge.
- `show_advanced` reveals `threshold` **0.5**, `dilation` **10**,
  `crop_factor` **1.5**, `drop_size` **10**, `crop_size_mult` **3.0**, and
  `bbox_smooth_alpha` **0.5**. Hiding them preserves their saved values.

With width/height **0**, scale **1**, and square disabled, masks and crop sizing
use the original workflow settings. The original crop multiplier and smoothing
still affect the reference crop size; scale is applied to that original size.
The crop is an exact slice of the source
image: it is never resized or padded after clipping. `bboxes` describes that
same slice, so **BatchUncrop** restores alignment at any scale, pixel size, or
image border. The output `width`/`height` report the actual clipped crop size,
not the white mask rectangle or the size requested outside the image.
Very small crops retain at least two pixels per available axis to preserve
RGB channels through KJNodes' image conversion.

Outputs are `original_images`, `cropped_images`, `bboxes` (KJNodes BBOX
coordinates), `width`, `height`, and `bbox_masks` (white rectangles on black).
Connect the first three to **BatchUncrop** and `bbox_masks` to **MaskPreview**.
Outputs are lists, as in the original mask-list workflow: each detected face
gets its own crop, source image, coordinates, and mask. Image batches are
detected one image at a time. An image without detections reports a clear error.
See `example_workflows/RP_Face_Detection.json` for the supplied workflow
converted to this single node.

## Workflow utilities

All four nodes are available under `RP/workflow_utilities` and share the
`workflow_utilities` Python directory and `web/js/workflow_utilities` browser
extensions. Existing Import/Fix Workflow nodes keep their registered names.

### RP Bypasser

Connect an output from each node you want to control to RP Bypasser. A new
empty input appears after each connection. Each connected row shows the
source node's name and a **yes/no** toggle showing whether it is enabled.
Clicking the toggle switches that source between enabled and ComfyUI's native
bypass mode. **Bypass all** bypasses every connected source in one action.
Once all sources are bypassed, the button becomes **Enable all** and clicking
it enables them again. The label follows the actual modes, including manual
changes and reopening. There is no output or repeater node to connect.

Connecting, disconnecting, loading, and saving never change source node modes.
The controls read those modes directly, so manual changes and RP Groups Muter
stay in sync. Duplicate connections control the same source, and reroutes
resolve to their actual source node. The controller is editor-only and does
not execute in the queue. Native bypass passes compatible upstream data through;
it does not replace nodes that have no compatible input with generated data.

### RP Groups Muter

Shows one **Enable** toggle per group in the node's graph, with **yes/no**
indicating whether any node in that group is enabled. Click a row to mute or
enable all its nodes, including notes, labels, reroutes, and editor controls
such as RP Bypasser. The arrow navigates to the group. No node type is excluded.

The first toggle, **Hide nested groups**, is on by default: groups completely
contained in another group are hidden from the list, while their nodes remain
controlled by the outer group. Partially overlapping groups stay listed. Turn
it off to list all groups. This setting is saved with the workflow.

The rows always read the actual node modes, including after manual changes,
moving nodes, and reopening a saved workflow. Group toggle values are not saved as
a separate source of state. No queue execution or rgthree service is needed.

Use the node properties to configure `matchTitle` (text or regular expression),
`matchColors` (comma-separated hex colors or ComfyUI color names), `sort`
(`position` or `alphanumeric`), and `showNav`. Multiple muters stay synchronized,
and groups with identical names remain independent. Group membership uses the
center of each node; nested and overlapping groups reflect their shared nodes.

### RP Fix Workflow

Click **Fix workflow** to clean the current canvas, including its subgraphs.
It removes links to missing nodes, stale input/output link references, and
stale reroute references. Existing nodes, IDs, positions, groups, widget values,
and valid connections are retained.

A green notification reports the number of repaired links and references,
or **Workflow is already clean** when no changes are needed. The node has no
execution inputs or outputs and uses ComfyUI's standard layout.

**Fix workflow** changes only the canvas. **Fix workflow and save** also saves
directly under the current workflow name, overwriting its saved file without
asking for another name. It saves even when no fixes are needed.
The notification confirms saving only when it completes successfully.

### RP Import Workflow

Available under `RP/workflow_utilities`. Add the node to the main workflow canvas:

- Select a JSON from `workflow` and click **Import saved workflow**. The list
  reads the current ComfyUI user's `workflows` directory, including subfolders
  (normally `ComfyUI/user/default/workflows`). **Refresh saved workflows**
  updates the list after saving new workflows.
- **Choose file to import** opens the browser's file picker for JSON files in
  any folder. Multiple files can be selected and imported together.
- `placement` adds imports to the right or below the existing workflow;
  `gap` sets the spacing. `group_import` wraps each import in a named group.

Imports happen immediately without running the queue or opening another tab.
Existing node IDs and positions are retained; imported nodes, links, groups,
native reroutes, and subgraph definitions receive independent IDs. Internal
connections and widget values are preserved. Both Save-format v0.4 and v1 JSON
are supported, including mixed-format imports. API-format prompts are not
workflow files. Required custom nodes must be installed to run imported nodes.

Stale links to deleted nodes, missing link references, and stale reroute
references are discarded automatically so they do not block the import.
All available nodes and valid connections are imported; source files are unchanged.

Imports use ComfyUI's normal loading hooks for custom-node compatibility.
During graph loading, legacy connection callbacks can use string node IDs;
this also covers reopening the saved workflow.

The imported nodes are selected. Use Fit View to see the complete canvas
and save the combined workflow normally. The success notification shows only
the number of imported nodes; no status text is displayed inside the node. The
importer is an editor-only control with no execution inputs or outputs; it can
be removed after importing. Source files are read only and files selected in
the picker are processed in the browser.

The merge logic is adapted from
[`comfyui-workflow-merger`](https://github.com/raffaele-pet/comfyui-workflow-merger).
Regression tests:
`node --test tests/test_workflow_merge.mjs tests/test_workflow_load.mjs tests/test_workflow_fix.mjs tests/test_groups_muter.mjs tests/test_bypasser.mjs`.

## Image saving

### RP Image Comparer

Compares two images in the same node using the Image Comparer interaction from
rgthree. In the default `Slide` mode, move the pointer over the preview to
reveal either side. The `comparer_mode` property also provides `Click`, which
switches between the two images on each click, and `Side-by-side`, which
shows both images next to each other.

The `Display name` field identifies the image or model and defaults to `#1`.
The `Save name` field controls the browser-download filename and defaults to
`%display_name-date:yyyy-MM-dd_HHmmss%`. For example, a display name of
`qwen_image_2.1` produces a filename such as
`qwen_image_2.1-2026-09-21_155258.png`.

Right-click the left or right half of the preview and choose `Save Image` to
download the corresponding image. If either input contains a batch, use the
labels above the preview to select the two images to compare.

### RP Preview Image

Shows one image at a time, with its exact dimensions and aspect ratio below
the preview. `Display name` and `Save name` use the same defaults and filename
tokens as RP Image Comparer. Right-click and choose `Save Image` to download
the displayed PNG. For a batch, use the arrows on the preview to switch images.

## Image sizing and resizing

`RP Smart Image Size` and `RP Smart Image Resize` share the same model-aware
resolution database and dependent controls.

### Supported models

- Boogu-Image-0.1 Base / Edit
- Boogu-Image-0.1 Turbo
- FireRed-Image-Edit-1.0
- FLUX.2 Klein
- HiDream-O1-Image / Dev
- Ideogram 4
- Krea 2
- Qwen-Image-2.1
- Qwen-Image-2512
- Qwen-Image-Edit-2511
- SDXL
- Z-Image-Turbo

The resolution menus show the megapixel class, its K equivalent where
applicable, and the corresponding square dimensions. Each model only exposes
the resolution tiers it supports.

### RP Smart Image Size

Selects a model, a supported resolution class, and an aspect-ratio preset. It
is useful for configuring latent-image nodes, samplers, image generators, and
other nodes that require explicit width and height values.

#### Outputs

- `width` - selected width in pixels
- `height` - selected height in pixels
- `aspect_ratio_approx` - named preset ratio, such as `5:3`
- `aspect_ratio` - exact ratio of the selected pixel dimensions, such as `41:25` for `1312 × 800`
- `resolution` - numeric square-side resolution

### RP Smart Image Resize

Accepts an image, a mask, or both and adapts them to dimensions suitable for
the selected model. When only a mask is connected, the node also creates a
three-channel preview image from that mask.

The optional `resolution` input accepts an integer longer-side value from nodes
such as `ImageSize (LongerSide)`. When connected, RP Smart Image Resize preserves
that longer-side resolution while calculating the other side from the selected
aspect ratio.

#### Selection modes

- `automatic` - selects the available preset whose aspect ratio is closest to
  the connected image or mask and uses its exact catalog width and height. The
  dimensions, width, and height controls are disabled in the interface.
- `manual` - allows direct preset selection and editable width and height
  values.

#### Outputs

- `IMAGE`
- `width`
- `height`
- `aspect_ratio_approx` - retains the former aspect-ratio output behavior
- `aspect_ratio` - exact ratio calculated from the output image width and height
- `resolution`
- `mask`
- `resized_image_no_pad` - the source image resized proportionally to fit the
  selected target, exactly as if `keep_proportion` were `resize`, regardless
  of the selected mode. Uses the same preset, dimension override, and
  interpolation; preserves the source content without padding or cropping.

Connect `resized_image_no_pad` to **Get Image Size & Count**, then send its
width/height to a downstream crop node to remove the generated image's padding.
The existing `IMAGE`, width/height, ratio, resolution, and mask outputs still
describe the primary result using the chosen `keep_proportion` mode.

### RP Image Minimum 1K

Upscales images whose longer side is below 1024 px while preserving their
aspect ratio. Images already at or above 1024 px remain unchanged.

## Video frame processing

Paired nodes for extracting, processing, and rebuilding videos frame by frame.

### RP Video to Frames

Extracts a video to persistent PNG frames and starts the integrated processing
loop.

### RP Frames to Video

Saves the processed frames and rebuilds the MP4, with optional source audio and
an in-node preview.

When `overwrite` is disabled, existing output videos are kept and the next
available numbered name is used, such as `processed_video_1.mp4`,
`processed_video_2.mp4`. Enabling `overwrite` replaces the requested file.

### Connecting the pair

- `flow` to `flow`
- `video_context` to `video_context`
- `image` through the processing nodes to `processed_image`

Source and processed frames remain accessible under `ComfyUI/output`. FFmpeg is
provided through the package requirements.

## Image folder processing

The image-folder nodes provide an integrated, one-item-at-a-time flow without
extracting or rebuilding a video.

### RP Load Images from Folder

Reads supported images from one folder in natural filename order and sends one
`IMAGE` and `MASK` through the workflow at a time. Relative source paths are
resolved from `ComfyUI/input`. Supported formats are BMP, JPEG, PNG, TIFF, and
WebP.

### RP Save Images to Folder

Saves each processed image to the selected destination, preserving its original
filename and format, then advances the integrated flow to the next source image.
Relative destination paths are resolved from `ComfyUI/output`.

### Connecting the pair

- `flow` to `flow`
- `image_context` to `image_context`
- `image` through the processing nodes to `processed_image`

Source and destination folders must be different. `clear_output_folder` removes
supported image files from the destination at the start of the run, while
`overwrite` controls whether matching destination filenames may be replaced.

### RP Load Prompt from File

Reads one prompt block for the current image by using the index stored in
`image_context`. Prompt blocks are separated by a line containing the selected
separator, `*` by default. When `require_matching_count` is enabled, execution
stops if the number of prompt blocks differs from the number of source images.

### RP Save Image to Folder (No Loop)

Saves an intermediate image and returns the same image unchanged so processing
can continue downstream. This node never advances or starts the folder loop and
is intended for one or more intermediate saves inside the processing chain.

Only one `RP Save Images to Folder` should be connected to a given
`RP Load Images from Folder`. Place it at the end of the complete chain as the
loop controller. Use `RP Save Image to Folder (No Loop)` for every intermediate
save. The loop controller reports an error if another loop controller is
connected to the same loader.

## MiniMax H3 keyframes

### RP H3-Keyframes

Adds an ordered sequence of still images to MiniMax H3 conditioning without
manual position controls. The connected images are distributed automatically
across the exact length of the H3 AV latent.

The image inputs grow automatically. The node initially shows
`frame_1`; connecting it reveals `frame_2`, and the sequence
continues up to 32 images. Each input accepts one still image rather than an
image batch. The `crop` option either stretches each image to the target canvas
or applies a centered cover crop before VAE encoding.

Typical wiring is:

1. Connect the H3 conditioning and AV latent to `conditioning` and `latent`.
2. Connect the MiniMax H3 video VAE to `vae`.
3. Connect the still images in their intended order, then send the output
   conditioning to the sampler path.

The node replaces the conditioning's keyframe list with the connected images
in input order.

## Installation

Open a terminal in `ComfyUI/custom_nodes` and run:

```bash
git clone https://github.com/raffaele-pet/ComfyUI-RPNodes.git
python -m pip install -r ComfyUI-RPNodes/requirements.txt
```

Restart ComfyUI and refresh the browser. The image-sizing nodes are available
under `image/resolution`; the image-folder nodes are available under
`image/RPNodes`; the image comparer is available under `image/saving`; the
video-processing nodes are available under `video/RPNodes`; the MiniMax H3
keyframe node is available under `RP/MiniMax H3`; the workflow utilities are
available under `RP/workflow_utilities`.

## Example workflow

[`RP_All_Nodes_Examples.json`](./example_workflows/RP_All_Nodes_Examples.json)
contains all 16 RP nodes in 14 example groups, with English and Italian notes.
It covers image sizing, resizing, folder processing, previews and comparison,
video frame processing, workflow utilities, and a complete FastH3 keyframe chain.

Drag the JSON onto the ComfyUI canvas or load it through the workflow menu.
Enable only the example group you want to run and choose your input files,
folders, and models as described in its note. The H3 example also needs the H3
models and auxiliary nodes used by its sampling chain. Workflow utility buttons
work directly on the canvas without running the queue.

## Notes

- The resolution database includes both manufacturer-published presets and
  practical model-aware dimensions for additional aspect ratios.
- `aspect_ratio_approx` is the named preset when the output canvas uses its
  catalog dimensions. Pixel-grid rounding can make the mathematical ratio of
  those dimensions differ slightly from that name. In Resize, modes that retain
  the input shape and manually overridden dimensions retain the former behavior
  of reporting the actual pixel ratio in `aspect_ratio_approx`. The new
  `aspect_ratio` output always reports the exact ratio of the output pixels.
- Sources and estimation rules for each model are documented in
  [Smart image preset sources](./docs/smart-image-presets.md).
- Qwen-Image-2.1 keeps the manufacturer-published native 2K sizes. Its other
  presets are calculated from the selected square pixel budget on a 32-pixel
  grid, matching the model's official ComfyUI workflow.
- Very wide or tall formats may be less stable than a model's native training
  ratios.
