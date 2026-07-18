# Topaz-Killer Upscaler

Super-resolução em lote (Real-ESRGAN + GFPGAN): ultra-nitidez, microtextura
realista, remoção de artefatos de compressão e saídas até 8K.

## ⚠️ Compatibilidade (evita ~90% dos erros)

- **Use Python 3.10 ou 3.11.** Em 3.12+ não há wheels de `basicsr`/`gfpgan`.
- **`numpy < 2.0`** (já no `requirements.txt`). O script instala sozinho o shim
  do `torchvision.transforms.functional_tensor`, então não edite a lib.

## Instalação

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# NVIDIA (CUDA 12.1) — antes do requirements; CPU/Apple Silicon pula esta linha:
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Os pesos (`RealESRGAN_x4plus.pth`, `GFPGANv1.4.pth`) baixam sozinhos na 1ª execução.

## Uso

```bash
# imagens em ./input  →  resultado em ./output
python topaz_killer_upscaler.py

# com opções:
python topaz_killer_upscaler.py -i input -o output --width 5460 --height 3072 --format png
```

Todas as flags: `python topaz_killer_upscaler.py --help`

## Flags principais

| Flag | Para quê |
|---|---|
| `--tile 256` | menos VRAM (`512` padrão; `128` p/ CPU; `0` desliga). OOM reduz o tile sozinho |
| `--detail 0.8` | menos aspecto "plástico" (mistura IA × Lanczos; `1.0` = 100% IA) |
| `--denoise 5` / `--auto-denoise` | limpa ruído/compressão antes do upscale |
| `--sharpen 0.5` | nitidez no acabamento |
| `--fidelity 0.5` | peso do rosto no GFPGAN (`--no-face` desliga) |
| `-m weights/4x-UltraSharp.pth` | pesos da comunidade (melhor microtextura; `*_anime_6B` → `--num-block 6`) |

**Saída "plastificada"?** baixe `--detail` ou use `4x-UltraSharp`.
**Ruidosa?** suba `--denoise` ou `--auto-denoise`.

## Notas

- **Resolução exata:** ultrapassando o alvo, reduz com Lanczos sem perder detalhe.
  `--resize-mode exact` força `W×H`; `--width 0 --height 0` mantém o 4x nativo.
- **Tolerância a falhas:** imagem que falha vai pro `upscaler.log` e o lote segue.
  Reexecutar pula o que já existe em `output/` (`--overwrite` refaz).
