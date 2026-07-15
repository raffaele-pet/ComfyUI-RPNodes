# ComfyUI RPNodes

A focused ComfyUI custom-node package for selecting model-aware image
resolutions and adapting images or masks to suitable dimensions for a chosen
image-generation model.

The package provides four nodes:

- **Smart Image Size**
- **Smart Image Resize**
- **RP Video to Frames**
- **RP Frames to Video**

The two image nodes share the same resolution database. Model, resolution, and
dimensions menus are dependent: changing the model refreshes the available
resolution classes, and changing the resolution refreshes the available aspect
ratios and pixel dimensions.

## Supported models

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

## Smart Image Size

![Smart Image Size](./images/smart-image-size.jpg)

Selects a model, a supported resolution class, and an aspect-ratio preset. It
is useful for configuring latent-image nodes, samplers, image generators, and
other nodes that require explicit width and height values.

### Outputs

- `width` - selected width in pixels
- `height` - selected height in pixels
- `aspect_ratio` - selected ratio, such as `16:9`
- `resolution` - numeric square-side resolution

## Smart Image Resize

![Smart Image Resize](./images/smart-image-resize.png)

Accepts an image, a mask, or both and adapts them to dimensions suitable for
the selected model. When only a mask is connected, the node also creates a
three-channel preview image from that mask.

The optional `resolution` input accepts an integer longer-side value from nodes
such as `ImageSize (LongerSide)`. When connected, Smart Image Resize preserves
that longer-side resolution while calculating the other side from the selected
aspect ratio.

### Selection modes

- `automatic` - selects the available preset whose aspect ratio is closest to
  the connected image or mask. The dimensions, width, and height controls are
  disabled in the interface.
- `manual` - allows direct preset selection and editable width and height
  values.

### Outputs

- `IMAGE`
- `width`
- `height`
- `aspect_ratio`
- `resolution`
- `mask`

## Installation

Open a terminal in `ComfyUI/custom_nodes` and run:

```bash
git clone https://github.com/raffaele-pet/ComfyUI-RPNodes.git
```

Restart ComfyUI and refresh the browser. Both nodes are available under the
`image/resolution` category.

## Example workflow

The [`example_workflows`](./example_workflows) directory contains a ready-to-use
workflow demonstrating both nodes:

- [`smart-image-size-resize.json`](./example_workflows/smart-image-size-resize.json)
- [`video-frames-process-video.json`](./example_workflows/video-frames-process-video.json)

Drag the JSON file onto the ComfyUI canvas or load it through the workflow
menu.

## Video frame workflow

`RP Video to Frames` uploads/selects a video, extracts every frame as PNG and
creates a second persistent folder for processed frames. Relative folder names
are created below `ComfyUI/output`.

Folder values beginning with `output/` or `ComfyUI/output/` are normalized to
the same ComfyUI output directory instead of being appended twice. Use a
dedicated processed-frame subfolder, such as `video_frames/processed`; the
output root itself cannot be cleared for safety.

The two video nodes contain the complete frame loop. Connect them as follows:

- `flow` directly to `flow`.
- `video_context` directly to `video_context`.
- `image` through any image-processing nodes and then to `processed_image`.

`RP Video to Frames` loads one source frame at a time. `RP Frames to Video`
saves the processed frame, advances the integrated loop, and builds the MP4
after the final iteration. Easy Use loop nodes and a separate `SaveImage` node
are not required.

Both nodes show an aspect-ratio-aware video preview capped at 160 pixels high.
The source preview updates when a video is selected, and the output preview is
displayed after the processed MP4 has been assembled.

The rebuilt MP4 is saved in the processed-frames folder. When
`copy_original_audio` is enabled, the original audio track is included in the
new video. `replace_existing_frames` and `clear_processed_frames` are disabled
by default so that manually edited frame files are not deleted unexpectedly.

The video nodes require FFmpeg. Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

On supported Windows, macOS, and Linux systems, `imageio-ffmpeg` supplies the
FFmpeg executable. The nodes use FFprobe when it is available and otherwise
read the required FPS and dimensions directly from FFmpeg.

## Notes

- The resolution database includes both manufacturer-published presets and
  practical model-aware dimensions for additional aspect ratios.
- Very wide or tall formats may be less stable than a model's native training
  ratios.
