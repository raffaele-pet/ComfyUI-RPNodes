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
| FLUX.3 Klein | [BFL specifications](https://docs.bfl.ml/flux_3/flux3_image_klein), [Klein API size tiers](https://docs.bfl.ai/api-reference/utility/generate-an-image-with-flux-3-klein), [ComfyUI reference resize](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_edit_model.py) | The local catalog uses 768, 1024, 1536, 2048 and 4096 square references. Presets use Set Reference Multi Latent's target budgets and the VAE's 16-pixel crop, with stable canvas dimensions when the budget is connected downstream. These are derived sizing presets, not published optimal buckets. |
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

The target budgets are 0.5625, 1.0, 2.25, 4.0 and 16.0 in ComfyUI units
(1024² pixels per unit). The 768-square tier was previously labeled 0.59 MP
using a decimal approximation; it now shows 0.5625 in the same units as the
other tiers. Old resolution and dimension labels still resolve by square side
and nominal ratio.

For each ratio, start at the selected target area, round both sides to integer
pixels as Set Reference Multi Latent does, then crop to the VAE's 16-pixel grid.
If applying the same budget would change that grid again, repeat the sizing
step until the canvas is stable. The catalog stores the resulting dimensions
and the target budget separately. For example, 1 MP 4:5 is 912×1136 with
`megapixels = 1.0`; the actual pixel area is slightly below that target.
The reference node still resamples the pixels before the VAE crops them.

The previous 4096-pixel side cap has been removed to preserve the 16 MP target
for non-square 4K presets: 16:9 is 5456×3072 and 21:9 is 6256×2672.
The API explicitly names 15 ratios; the extra 3:5, 5:8, 7:9, 9:7, 8:5 and
5:3 entries are estimates. No preset is presented as an optimal training bucket.

This profile targets distilled Klein. BFL's specifications list up to 4K for
both checkpoints. Native reference preparation varies by implementation and
by the number of reference images; this catalog aligns with ComfyUI's
Set Reference Multi Latent sizing behavior.
