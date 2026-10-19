# Smart image preset sources

Initial audit: 2026-09-29; FLUX.3 Klein added and checked on 2026-10-19.
Both smart image nodes read the same `resolutions.json`.
The resolution tier names use a square image as a reference; non-square presets
can have a different pixel count. A named aspect ratio is approximate when the
pixel dimensions have been rounded to a model-compatible grid.

| Model | Published reference | How the catalog uses it |
| --- | --- | --- |
| Boogu-Image-0.1 Base / Edit; Turbo | [Boogu model card](https://huggingface.co/Boogu/Boogu-Image-0.1-Base) | Base and Edit: 1K, 1.5K and 2K; Turbo: 1K. Nine ratios are named by the maker. Other ratios are estimates. |
| FireRed-Image-Edit-1.0 | [FireRed project](https://github.com/FireRedTeam/FireRed-Image-Edit) | The project uses roughly 1K image inputs. The exact ratio list is estimated, not a published bucket table. |
| FLUX.2 Klein | [BFL endpoint reference](https://github.com/black-forest-labs/skills/blob/master/skills/bfl-api/references/endpoints.md) | Width and height are selectable on a 16-pixel grid, with a published maximum around 4 MP. The 1K, 1.5K and 2K ratio buckets are estimates. |
| FLUX.3 Klein | [BFL specifications](https://docs.bfl.ml/flux_3/flux3_image_klein), [Klein API size tiers](https://docs.bfl.ml/api-reference/utility/generate-an-image-with-flux-3-klein), [official demo source](https://huggingface.co/spaces/black-forest-labs/flux-3-image-klein/blob/a3f1a096a9cfd68587da22044390990a425aeeb0/app.py) | The distilled Klein supports up to 4K. The local catalog uses 768, 1024, 1536, 2048 and 4096 square references. Dimensions follow the demo's 16-pixel rounding and 4096-pixel side cap. Extra ratios and the intermediate 768/1536 sizes are derived using the same sizing rule. |
| HiDream-O1-Image / Dev | [HiDream model card](https://huggingface.co/HiDream-ai/HiDream-O1-Image) and [official demo presets](https://huggingface.co/spaces/HiDream-ai/HiDream-O1-Image/blob/main/app.py) | The 2K entries for 1:1, 4:3, 3:4, 16:9, 9:16, 3:2, 2:3, 21:9, 9:7 and 7:9 follow the demo. Other ratios and the 1K/1.5K buckets are estimates. |
| Ideogram 4 | [Ideogram model specification](https://ideogram.ai/blog/ideogram-4.0/) and [inference guide](https://github.com/ideogram-oss/ideogram4/blob/main/docs/inference.md) | The open-weight model accepts multiples of 16 from 256 to 2048 pixels per side and ratios up to 6:1. Catalog dimensions within those limits are estimates, including the intermediate 1.5K tier. The hosted API exposes 1K and 2K resolution options. |
| Krea 2 | [Krea official demo](https://huggingface.co/spaces/krea/Krea-2/blob/main/app.py) | Its demo publishes 1024 square, 832×1216 portrait (13:19), 1216×832 landscape (19:13) and 2048 square. These portrait and landscape sizes have their own ratio entries rather than being labeled 2:3 and 3:2. The remaining entries are estimates. The 2K estimates stay within the demo's 2048-pixel-per-side limit. 2K may need more GPU memory. |
| Qwen-Image-2.1 | [Qwen model card](https://huggingface.co/Qwen/Qwen-Image-2.1) | Seven 2K dimensions come directly from the model card. The other 2K ratios and all 1K, 2 MP and 3 MP entries are 32-pixel-grid estimates using each square preset's pixel budget. |
| Qwen-Image-2512 | [Qwen model card](https://huggingface.co/Qwen/Qwen-Image-2512) | Seven 1328-based dimensions are published. The remaining ratios are estimates. |
| Qwen-Image-Edit-2511 | [Qwen model card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | The 1K bucket follows common editing use; the extended ratios are estimates. |
| SDXL | [Stability AI SDXL documentation](https://stability.ai/sdxl-aws-documentation) | 1024 square and several landscape/portrait buckets are published; additional ratios are estimates. |
| Z-Image-Turbo | [Tongyi-MAI official demo](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/app.py) and [maintainer guidance](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/28) | The 1K, 1280 and 1536 tiers contain the demo's ten published ratios per tier. The remaining ratios are estimates. The former 2K tier was removed because the official demo does not list it and a maintainer recommends the demonstrated resolution grids. |

The catalog is a local ComfyUI sizing aid, not a promise that a hosted service
will accept every width and height. Memory demand and image quality may vary,
especially for estimated wide or tall presets.

## FLUX.3 Klein sizing

The current official demo offers 1K, 2K and 4K for distilled Klein. BFL also
publishes `768sq` and `1.5k` API tiers; this local catalog represents those with
768 and 1536 square references. It does not claim an exact copy of the hosted
API's bucket table, which permits sides up to 6256 pixels.

For each ratio, start from the square reference's pixel area, fit both sides
within 4096, then round each side to the nearest multiple of 16, as in the demo's
`dimensions_for_aspect`. Its seven named ratios are 1:1, 4:3, 3:4, 3:2, 2:3,
16:9 and 9:16. The other fourteen catalog ratios are estimates under that rule.
For example, 1K 16:9 is 1360×768; 2K 16:9 is 2736×1536; 4K 16:9 is
4096×2304. The 4K square is 4096×4096, while non-square 4K presets have less
area because of the side cap.

This profile targets distilled Klein. The demo's 1K-only Base menu is a GPU
resource limit; BFL's model specifications list up to 4K for both checkpoints.
