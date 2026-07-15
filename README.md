# ComfyUI RPNodes

A collection of ComfyUI utility nodes organized into two independent toolsets:

- **Image sizing and resizing:** `Smart Image Size` and `Smart Image Resize`
- **Video frame processing:** `RP Video to Frames` and `RP Frames to Video`

Each pair was designed to work in the same field and can be used together in a
single workflow. The two toolsets remain independent from one another.

## Image sizing and resizing

`Smart Image Size` and `Smart Image Resize` were created together for
model-aware image dimensions. They share the same resolution database and use
dependent model, resolution, aspect-ratio, and dimensions controls.

Changing the model refreshes the available resolution classes. Changing the
resolution refreshes the available aspect ratios and pixel dimensions.

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

### Smart Image Size

![Smart Image Size](./images/smart-image-size.jpg)

Selects a model, a supported resolution class, and an aspect-ratio preset. It
is useful for configuring latent-image nodes, samplers, image generators, and
other nodes that require explicit width and height values.

#### Outputs

- `width` - selected width in pixels
- `height` - selected height in pixels
- `aspect_ratio` - selected ratio, such as `16:9`
- `resolution` - numeric square-side resolution

### Smart Image Resize

![Smart Image Resize](./images/smart-image-resize.png)

Accepts an image, a mask, or both and adapts them to dimensions suitable for
the selected model. When only a mask is connected, the node also creates a
three-channel preview image from that mask.

The optional `resolution` input accepts an integer longer-side value from nodes
such as `ImageSize (LongerSide)`. When connected, Smart Image Resize preserves
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

## Video frame processing

`RP Video to Frames` and `RP Frames to Video` were created together to automate
per-frame video processing. The first node starts the integrated frame loop;
the second node saves each processed frame, completes the loop, and rebuilds
the video.

![RP Video to Frames to Video workflow](./images/rp-video-to-frames-to-video.png)

### RP Video to Frames

Uploads or selects a video, extracts every frame as PNG, and stores the source
frames in a persistent user-selected folder. It loads one source frame at a
time and sends it through the connected image-processing nodes.

The node also creates the persistent folder that will receive the processed
frames. Its source-video preview automatically preserves the video aspect ratio
and is capped at 160 pixels high.

### RP Frames to Video

Receives the processed image, saves it to the processed-frames folder, and
advances the integrated loop. After the final frame, it rebuilds the MP4 and
shows the completed video directly inside the node.

When `copy_original_audio` is enabled, the original audio track is included in
the rebuilt video. A separate loop node and `SaveImage` node are not required.

### Connecting the pair

- Connect `flow` directly to `flow`.
- Connect `video_context` directly to `video_context`.
- Send `image` through any image-processing nodes and connect the result to
  `processed_image`.

### Frame folders and video output

Relative folder names are created below `ComfyUI/output`. Folder values
beginning with `output/` or `ComfyUI/output/` are normalized to the same ComfyUI
output directory instead of being appended twice.

Use a dedicated processed-frame subfolder, such as `video_frames/processed`;
the output root itself cannot be cleared for safety. The rebuilt MP4 is saved
inside the processed-frames folder.

`replace_existing_frames` and `clear_processed_frames` are disabled by default
so that manually edited frame files are not deleted unexpectedly.

### FFmpeg dependency

The video nodes require FFmpeg. On supported Windows, macOS, and Linux systems,
`imageio-ffmpeg` supplies the FFmpeg executable. The nodes use FFprobe when it
is available and otherwise read the required FPS and dimensions directly from
FFmpeg.

## Installation

Open a terminal in `ComfyUI/custom_nodes` and run:

```bash
git clone https://github.com/raffaele-pet/ComfyUI-RPNodes.git
python -m pip install -r ComfyUI-RPNodes/requirements.txt
```

Restart ComfyUI and refresh the browser. The image-sizing nodes are available
under `image/resolution`; the video-processing nodes are available under
`video/RPNodes`.

## Example workflows

The [`example_workflows`](./example_workflows) directory contains one ready-to-use
workflow for each pair:

- [`smart-image-size-resize.json`](./example_workflows/smart-image-size-resize.json)
- [`video-frames-process-video.json`](./example_workflows/video-frames-process-video.json)

Drag a JSON file onto the ComfyUI canvas or load it through the workflow menu.

## Notes

- The resolution database includes both manufacturer-published presets and
  practical model-aware dimensions for additional aspect ratios.
- Very wide or tall formats may be less stable than a model's native training
  ratios.
