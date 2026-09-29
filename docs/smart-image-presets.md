# Smart image preset sources

Checked on 2026-09-29. Both smart image nodes read the same `resolutions.json`.
The resolution tier names use a square image as a reference; non-square presets
can have a different pixel count. A named aspect ratio is approximate when the
pixel dimensions have been rounded to a model-compatible grid.

| Model | Published reference | How the catalog uses it |
| --- | --- | --- |
| Boogu-Image-0.1 Base / Edit; Turbo | [Boogu model card](https://huggingface.co/Boogu/Boogu-Image-0.1-Base) | Base and Edit: 1K, 1.5K and 2K; Turbo: 1K. Nine ratios are named by the maker. Other ratios are estimates. |
| FireRed-Image-Edit-1.0 | [FireRed project](https://github.com/FireRedTeam/FireRed-Image-Edit) | The project uses roughly 1K image inputs. The exact ratio list is estimated, not a published bucket table. |
| FLUX.2 Klein | [BFL endpoint reference](https://github.com/black-forest-labs/skills/blob/master/skills/bfl-api/references/endpoints.md) | Width and height are selectable on a 16-pixel grid, with a published maximum around 4 MP. The 1K, 1.5K and 2K ratio buckets are estimates. |
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
