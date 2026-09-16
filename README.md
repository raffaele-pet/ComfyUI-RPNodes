# ComfyUI RPNodes

A collection of utility nodes for ComfyUI:

- **Image sizing and resizing:** `RP Smart Image Size`, `RP Smart Image Resize`,
  and `RP Image Minimum 1K`
- **Image folder processing:** `RP Load Images from Folder`,
  `RP Load Prompt from File`, `RP Save Image to Folder (No Loop)`, and
  `RP Save Images to Folder`
- **Video frame processing:** `RP Video to Frames` and `RP Frames to Video`
- **MiniMax H3 keyframing:** `RP H3-Keyframes`

Python nodes are grouped in `image_sizing_and_resizing` and
`video_frame_processing`, the folder-processing nodes in
`image_folder_processing`, and the keyframe node in `h3_keyframes`. Shared
browser extensions live under `web`, and example workflows remain together in
`example_workflows`.

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
- Qwen-Image-2512
- Qwen-Image-Edit-2511
- SDXL
- Z-Image-Turbo

### RP Smart Image Size

![RP Smart Image Size](./image_sizing_and_resizing/images/smart-image-size.jpg)

Selects a model, a supported resolution class, and an aspect-ratio preset. It
is useful for configuring latent-image nodes, samplers, image generators, and
other nodes that require explicit width and height values.

#### Outputs

- `width` - selected width in pixels
- `height` - selected height in pixels
- `aspect_ratio` - selected ratio, such as `16:9`
- `resolution` - numeric square-side resolution

### RP Smart Image Resize

![RP Smart Image Resize](./image_sizing_and_resizing/images/smart-image-resize.png)

Accepts an image, a mask, or both and adapts them to dimensions suitable for
the selected model. When only a mask is connected, the node also creates a
three-channel preview image from that mask.

The optional `resolution` input accepts an integer longer-side value from nodes
such as `ImageSize (LongerSide)`. When connected, RP Smart Image Resize preserves
that longer-side resolution while calculating the other side from the selected
aspect ratio.

#### Selection modes

- `automatic` - selects the available preset whose aspect ratio is closest to
  the connected image or mask. The dimensions, width, and height controls are
  disabled in the interface.
- `manual` - allows direct preset selection and editable width and height
  values.

#### Outputs

- `IMAGE`
- `width`
- `height`
- `aspect_ratio`
- `resolution`
- `mask`

### RP Image Minimum 1K

Upscales images whose longer side is below 1024 px while preserving their
aspect ratio. Images already at or above 1024 px remain unchanged.

## Video frame processing

Paired nodes for extracting, processing, and rebuilding videos frame by frame.

![RP Video to Frames to Video workflow](./video_frame_processing/images/rp-video-to-frames-to-video.png)

### RP Video to Frames

Extracts a video to persistent PNG frames and starts the integrated processing
loop.

### RP Frames to Video

Saves the processed frames and rebuilds the MP4, with optional source audio and
an in-node preview.

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
`image/RPNodes`; the video-processing nodes are available under `video/RPNodes`;
the MiniMax H3 keyframe node is available under `RP/MiniMax H3`.

## Example workflows

The [`example_workflows`](./example_workflows) directory contains ready-to-use
workflows for each node category:

- [`smart-image-size-resize.json`](./example_workflows/smart-image-size-resize.json)
- [`video-frames-process-video.json`](./example_workflows/video-frames-process-video.json)

Drag a JSON file onto the ComfyUI canvas or load it through the workflow menu.

## Notes

- The resolution database includes both manufacturer-published presets and
  practical model-aware dimensions for additional aspect ratios.
- Very wide or tall formats may be less stable than a model's native training
  ratios.
