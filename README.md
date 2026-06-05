# Topaz-Killer Upscaler

Pipeline de super-resolução em lote de nível comercial (Real-ESRGAN + GFPGAN),
focado em **ultra-nitidez**, **microtextura realista** (grama, pele, tecidos),
**remoção de artefatos de compressão** e saídas em **5460×3072 / 4x / 8K**.

---

## ⚠️ Leia primeiro (compatibilidade — economiza horas de debug)

Este stack (`basicsr` / `realesrgan` / `gfpgan`) é poderoso mas **antigo e
sensível a versões**. Dois pontos são a causa de ~90% das falhas de instalação:

1. **Python 3.10 ou 3.11.** Não use 3.12/3.13/3.14 — não há wheels do
   `basicsr`/`gfpgan` e o `torch` antigo não compila. Você tem 3.14 instalado;
   crie um venv dedicado com 3.11.
2. **`numpy < 2.0`** e **`torchvision` moderno**: o `basicsr` importa
   `torchvision.transforms.functional_tensor`, removido no torchvision ≥ 0.17.
   O script já instala automaticamente um *shim* no topo
   (`_install_torchvision_shim`), então **não precisa editar a biblioteca**.

---

## Instalação

```bash
# 1) venv com Python 3.11 (ajuste o caminho do seu python3.11)
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2a) NVIDIA (CUDA 12.1): instale torch com CUDA ANTES do requirements
pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 2b) CPU ou Apple Silicon (MPS): pule o passo acima
# 3) restante das dependências
pip install -r requirements.txt
```

Os pesos (`RealESRGAN_x4plus.pth`, `GFPGANv1.4.pth`) são **baixados
automaticamente** na primeira execução para a pasta `weights/`.

---

## Uso rápido

```bash
# coloque suas imagens em ./input  (jpg, jpeg, png, webp, bmp, tiff)
python topaz_killer_upscaler.py
# resultado em ./output
```

Tudo é configurável no topo do script **ou** via CLI:

```bash
python topaz_killer_upscaler.py \
    -i fotos_originais -o fotos_4k \
    --width 5460 --height 3072 \
    --fidelity 0.5 \
    --format png
```

Veja todas as opções: `python topaz_killer_upscaler.py --help`

---

## Pipeline adaptativo (o que aproxima do Topaz em hardware leve)

Sem GPU forte, difusão (SUPIR) está fora. Em vez disso o pipeline ficou
**adaptativo** — tudo em CPU/MPS, sem dependências novas:

| Recurso | Flag | O que resolve |
|---|---|---|
| **Tiling com feather** | `--overlap 32` | elimina linhas de emenda em céu/pele/gradientes (artefato nº1 que "denuncia" não-Topaz) |
| **Análise de conteúdo** | (auto) `--no-analyze` p/ desligar | loga blur (Laplaciano) e ruído (Immerkaer) de cada imagem |
| **Pré-denoise** | `--denoise 5` ou `--auto-denoise` | limpa ruído/artefatos de compressão **antes** do upscale |
| **Blend de detalhe** | `--detail 0.8` | mistura IA × Lanczos: reduz o efeito "plástico/pintura a óleo" e recupera fidelidade (`1.0`=100% IA) |
| **Unsharp final** | `--sharpen 0.5` | nitidez percebida no acabamento |
| **Retry de OOM** | (auto) | se faltar memória, reduz o tile pela metade e tenta de novo, sozinho |

Exemplos práticos:

```bash
# Foto antiga ruidosa, evitando aspecto plástico:
python topaz_killer_upscaler.py --auto-denoise --detail 0.85 --sharpen 0.4

# Textura realista máxima (grama/tecido), sem suavizar:
python topaz_killer_upscaler.py -m weights/4x-UltraSharp.pth --detail 1.0
```

> **Dica:** se a saída parecer "plastificada", baixe `--detail` para `0.8`.
> Se parecer ruidosa/artefatada, suba `--denoise` ou use `--auto-denoise`.

---

## Ajustando o Tiling (VRAM) — leia se der "Out of Memory"

Saídas 5.4K+ consomem muita memória. O **tiling** fatia a imagem em blocos.
Quanto menor o tile, menos VRAM (porém um pouco mais lento). Ajuste `--tile`
(ou `TILE_SIZE` no script):

| Hardware                         | `--tile` sugerido |
|----------------------------------|-------------------|
| GPU forte (≥12 GB) ou sem limite | `1024` ou `0` (desliga) |
| GPU média (6–8 GB)               | `512` (padrão)    |
| GPU fraca (4 GB) / Apple MPS     | `256`             |
| CPU / muito pouca memória        | `128`             |

```bash
python topaz_killer_upscaler.py --tile 256
```

- `--overlap` controla a sobreposição usada no **feather** (fusão suave entre
  tiles). O padrão `32` já elimina seams; se ainda notar emenda em superfícies
  muito lisas, suba para `48`/`64` (custa um pouco mais de tempo).
- Se faltar memória, o script **reduz o tile automaticamente** e tenta de novo
  (retry de OOM) — você não precisa intervir.
- O cache do acelerador é **limpo após cada imagem** (`empty_cache`), então o
  lote inteiro roda sem acumular memória.

---

## Pesos customizados da comunidade (4x-UltraSharp, NMKD-Siax…)

Coloque o `.pth` em `weights/` e aponte para ele:

```bash
python topaz_killer_upscaler.py -m weights/4x-UltraSharp.pth
```

Esses modelos usam a mesma arquitetura RRDBNet x4 (23 blocos), então funcionam
direto. Para os modelos leves `*_anime_6B`, adicione `--num-block 6`.

> **Dica anti-"plastificado":** `4x-UltraSharp` e `4x_NMKD-Siax_200k` preservam
> microtextura melhor que o `x4plus` padrão em fotos reais. Comece por eles se
> notar efeito de "pintura a óleo".

---

## Restauração facial (Face Recovery)

- Ativada por padrão. O **GFPGAN detecta rostos automaticamente**, restaura
  apenas a região facial e faz o blend com o fundo ampliado pelo Real-ESRGAN.
  Sem rostos na imagem, só o Real-ESRGAN é aplicado.
- `--fidelity 0.5`: peso de fidelidade do rosto (0.0–1.0). Valores **mais
  altos** geram mais detalhe; **mais baixos** ficam mais fiéis/suaves.
- Desligar: `--no-face`.
- **CodeFormer** (alternativa): instale `pip install codeformer-pip`. A
  semântica do peso é inversa à do GFPGAN (no CodeFormer, menor = mais detalhe).

---

## Resolução exata (downsample Lanczos)

A IA amplia em fator fixo (4x). Se a saída ultrapassar o alvo
(`--width/--height`), o script reduz com **Lanczos** (Pillow) para cravar a
resolução **sem perder os detalhes gerados pela IA**.

- `--resize-mode fit` (padrão): mantém o aspect ratio dentro da caixa alvo.
- `--resize-mode exact`: força exatamente `W×H` (pode distorcer).
- Para manter a saída nativa 4x, use `--width 0 --height 0`.

---

## Tolerância a falhas

Imagem corrompida ou que falhe é **registrada em `upscaler.log`**, ignorada, e
o lote continua. Ao final, um resumo lista total / sucesso / pulados / falhas.
Reexecutar pula o que já existe em `output/` (use `--overwrite` para refazer).

---

## Estrutura

```
upscaler/
├── topaz_killer_upscaler.py   # pipeline completo (modular)
├── requirements.txt
├── README.md
├── input/                     # suas imagens
├── output/                    # resultados
├── weights/                   # pesos (.pth) — baixados automaticamente
└── upscaler.log               # log detalhado
```
