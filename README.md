# ComfyUI Raffaele Pet Nodes

A focused ComfyUI custom-node package for selecting model-aware image
resolutions and adapting images or masks to the most suitable dimensions for a
chosen image-generation model.

The package currently provides two nodes:

- **Image Smart Resolution**
- **Image Smart Resize**

Both nodes share the same resolution database. Model, resolution, and
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

`FLUX.2 Klein` is the default model in both nodes.

## Image Smart Resolution

Selects a model, a supported resolution class, and an aspect-ratio preset. It
is useful for configuring latent-image nodes, samplers, image generators, and
other nodes that require explicit width and height values.

### Outputs

- `width` — selected width in pixels
- `height` — selected height in pixels
- `aspect_ratio` — selected ratio, such as `16:9`
- `resolution` — numeric square-side resolution; for example, `1K`, `1.5K`,
  and `2K` menu entries output `1024`, `1536`, and `2048`

## Image Smart Resize

Accepts an image, a mask, or both and adapts them to dimensions suitable for
the selected model. When only a mask is connected, the node also creates a
three-channel preview image from that mask.

### Selection modes

- `automatic` — default mode; selects the available preset whose aspect ratio
  is closest to the connected image or mask. The dimensions, width, and height
  controls are disabled in the interface.
- `manual` — allows direct preset selection and editable width and height
  values.

### Resize modes

- `stretch` — resizes directly to the target width and height.
- `resize` — preserves the source aspect ratio and fits inside the target
  bounds without padding.
- `pad` — fits the source inside the target and fills the remaining area with
  `pad_color`.
- `pad_edge` — fills the remaining area with averaged edge colors.
- `pad_edge_pixel` — default mode; extends the exact border pixels into the
  padded area.
- `crop` — crops the source to the target aspect ratio and then resizes it.
- `pillarbox_blur` — creates a darkened blurred background behind the fitted
  source.
- `total_pixels` — preserves the source aspect ratio while matching the total
  pixel area of the selected target dimensions.

### Additional controls

- `upscale_method` — `lanczos` (default), `bicubic`, `bilinear`, `area`, or
  `nearest-exact`
- `pad_color` — RGB padding color, such as `0, 0, 0`
- `crop_position` — `center`, `top`, `bottom`, `left`, or `right`

Image resizing runs on the CPU for stable Lanczos support. Masks always use
nearest-exact interpolation to preserve hard edges.

### Outputs

- `IMAGE`
- `width`
- `height`
- `aspect_ratio`
- `resolution`
- `mask`

## Installation

### ComfyUI Manager

After this repository is listed in ComfyUI Manager, search for
`ComfyUI-raffaele-pet-nodes` and install it from the Manager interface.

### Manual installation

Open a terminal in `ComfyUI/custom_nodes` and run:

```bash
git clone https://github.com/raffaele-pet/ComfyUI-raffaele-pet-nodes.git
```

Restart ComfyUI and refresh the browser. Both nodes are available under the
`image/resolution` category.

## Example workflow

The [`example_workflows`](./example_workflows) directory contains a ready-to-use
workflow demonstrating both nodes:

- [`image-resolution-resize.json`](./example_workflows/image-resolution-resize.json)

Drag the JSON file onto the ComfyUI canvas or load it through the workflow
menu.

## Project structure

```text
ComfyUI-raffaele-pet-nodes/
├── example_workflows/
│   └── image-resolution-resize.json
├── web/js/
│   └── image_model_resolution_selector.js
├── __init__.py
├── aware_resize.py
├── nodes.py
├── resolutions.json
└── README.md
```

## Notes

- The resolution database includes both manufacturer-published presets and
  practical model-aware dimensions for additional aspect ratios.
- Very wide or tall formats may be less stable than a model's native training
  ratios.
- Existing workflow compatibility is preserved through stable internal node
  identifiers.
