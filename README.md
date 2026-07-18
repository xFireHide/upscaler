## ⚠️ Compatibility (avoids ~90% of errors)

- **Use Python 3.10 or 3.11.** No `basicsr`/`gfpgan` wheels exist for 3.12+.
- **`numpy < 2.0`** (already in `requirements.txt`). The script auto-installs the
  `torchvision.transforms.functional_tensor` shim, so don't patch the library.

## Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# NVIDIA (CUDA 12.1) — before requirements; CPU/Apple Silicon skip this line:
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Weights (`RealESRGAN_x4plus.pth`, `GFPGANv1.4.pth`) download automatically on first run.

## Usage

```bash
# images in ./input  →  results in ./output
python topaz_killer_upscaler.py

# with options:
python topaz_killer_upscaler.py -i input -o output --width 5460 --height 3072 --format png
```

All flags: `python topaz_killer_upscaler.py --help`

## Key flags

| Flag | Purpose |
|---|---|
| `--tile 256` | less VRAM (`512` default; `128` for CPU; `0` disables). OOM auto-shrinks the tile |
| `--detail 0.8` | less "plastic" look (blends AI × Lanczos; `1.0` = 100% AI) |
| `--denoise 5` / `--auto-denoise` | clean noise/compression before upscaling |
| `--sharpen 0.5` | finishing sharpness |
| `--fidelity 0.5` | face weight in GFPGAN (`--no-face` disables) |
| `-m weights/4x-UltraSharp.pth` | community weights (better microtexture; `*_anime_6B` → `--num-block 6`) |

**Output looks "plasticky"?** lower `--detail` or use `4x-UltraSharp`.
**Noisy?** raise `--denoise` or use `--auto-denoise`.

## Notes

- **Exact resolution:** if output exceeds the target, it downscales with Lanczos
  without losing detail. `--resize-mode exact` forces `W×H`; `--width 0 --height 0`
  keeps native 4x.
- **Fault tolerance:** a failing image is logged to `upscaler.log` and the batch
  continues. Re-running skips what already exists in `output/` (`--overwrite` redoes).
