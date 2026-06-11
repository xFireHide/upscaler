#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topaz_killer_upscaler.py
========================

Pipeline de super-resolução em lote de nível comercial, projetado para se
aproximar da qualidade do Topaz Photo AI em hardware modesto (CPU / MPS / GPU
de pouca memória), SEM modelos de difusão pesados.

Pipeline adaptativo (ordem de execução por imagem):
    1. ANÁLISE de conteúdo (blur via Laplaciano + ruído via Immerkaer).
    2. PRÉ-DENOISE opcional (OpenCV) p/ ruído e artefatos de compressão.
    3. SUPER-RESOLUÇÃO generativa (Real-ESRGAN RRDBNet x4, pesos customizáveis).
       - Tiling com FEATHER GAUSSIANO (overlap) -> sem linhas de emenda.
       - Retry automático com tile menor em caso de Out-of-Memory.
    4. RESTAURAÇÃO FACIAL automática (GFPGAN) com blend e fidelidade ajustável.
    5. BLEND DE DETALHE (IA x Lanczos) p/ controlar o efeito "plastificado".
    6. UNSHARP finishing opcional (nitidez percebida).
    7. REDIMENSIONAMENTO exato pós-IA com Lanczos (Pillow).
    8. Detecção automática de hardware: CUDA -> MPS -> CPU.

Autor: Principal CV/DL Engineer
Licença: MIT
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# COMPATIBILITY SHIM  (DEVE rodar ANTES de importar basicsr/gfpgan/realesrgan)
# ---------------------------------------------------------------------------
# basicsr importa `torchvision.transforms.functional_tensor`, removido no
# torchvision >= 0.17. Recriamos o módulo apontando p/ a API nova, sem editar
# a biblioteca.
import sys
import types
import warnings

# O torchvision 0.15/0.16 emite um UserWarning de depreciação ao importar
# functional_tensor (inofensivo aqui — o shim abaixo cobre a remoção futura).
warnings.filterwarnings(
    "ignore", message=".*functional_tensor.*", category=UserWarning
)


def _install_torchvision_shim() -> None:
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return
    except ImportError:
        try:
            from torchvision.transforms.functional import rgb_to_grayscale
        except ImportError:
            return
        shim = types.ModuleType("torchvision.transforms.functional_tensor")
        shim.rgb_to_grayscale = rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = shim


_install_torchvision_shim()

# ---------------------------------------------------------------------------
# Imports padrão
# ---------------------------------------------------------------------------
import argparse
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer

try:
    from gfpgan import GFPGANer

    _GFPGAN_AVAILABLE = True
except Exception:  # pragma: no cover
    GFPGANer = None  # type: ignore
    _GFPGAN_AVAILABLE = False


# ===========================================================================
# 1. CONFIGURAÇÃO GLOBAL  (ajuste aqui)
# ===========================================================================

# --- Diretórios -------------------------------------------------------------
INPUT_DIR = "input"
OUTPUT_DIR = "output"
WEIGHTS_DIR = "weights"
LOG_FILE = "upscaler.log"

# --- Modelo de Super-Resolução ---------------------------------------------
MODEL_NAME = "RealESRGAN_x4plus"
MODEL_PATH: str | None = None       # ex.: "weights/4x-UltraSharp.pth"
NETSCALE = 4
NUM_BLOCK = 23                      # 23 (x4plus/UltraSharp/Siax) | 6 (*_anime_6B)

# --- Tiling com feather (GERENCIAMENTO CRÍTICO DE VRAM) --------------------
#   GPU forte (>=12GB): 1024 ou 0 (desliga) | média (6-8GB): 512
#   GPU fraca / MPS: 256 | CPU / pouca memória: 128
TILE_SIZE = 512
TILE_OVERLAP = 32      # sobreposição entre tiles (em px de ENTRADA) p/ o feather
PRE_PAD = 0

# --- Análise de conteúdo & pré-denoise -------------------------------------
AUTO_ANALYZE = True        # estima blur/ruído e loga; com AUTO_DENOISE, decide
AUTO_DENOISE = False       # liga denoise automaticamente se ruído for alto
DENOISE_STRENGTH = 0       # 0 = off. 3-5 leve, 7-10 forte (luminância h do NLM)
NOISE_SIGMA_THRESHOLD = 4.0  # acima disso (escala 0-255) consideramos "ruidoso"

# --- Restauração Facial -----------------------------------------------------
ENABLE_FACE_ENHANCE = True
FACE_FIDELITY_WEIGHT = 0.5   # 0.0-1.0; maior = mais detalhe gerado no rosto
GFPGAN_MODEL = "GFPGANv1.4"
ONLY_CENTER_FACE = False

# --- Anti-plástico / fidelidade & nitidez ----------------------------------
# DETAIL_STRENGTH: 1.0 = saída 100% IA. <1.0 mistura com o Lanczos do original
# (recupera microtextura/fidelidade e reduz o aspecto "pintura a óleo").
DETAIL_STRENGTH = 1.0
SHARPEN_AMOUNT = 0.0       # 0 = off. 0.3-0.8 = unsharp mask sutil no final

# --- Alvo de resolução exata (downsample Lanczos pós-IA) -------------------
TARGET_WIDTH: int | None = 5460
TARGET_HEIGHT: int | None = 3072
RESIZE_MODE = "fit"        # "fit" (preserva aspect) | "exact" (força WxH)

# --- Saída ------------------------------------------------------------------
OUTPUT_FORMAT = "png"
JPEG_QUALITY = 100
PNG_COMPRESSION = 1

# --- Performance ------------------------------------------------------------
USE_FP16: bool | None = None
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

_WEIGHT_URLS = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "GFPGANv1.4": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "GFPGANv1.3": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
}


# ===========================================================================
# Logging
# ===========================================================================
def setup_logging(log_file: str = LOG_FILE) -> logging.Logger:
    logger = logging.getLogger("topaz_killer")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ===========================================================================
# 8. Detecção de Hardware
# ===========================================================================
@dataclass
class Device:
    torch_device: torch.device
    kind: str
    use_fp16: bool

    @classmethod
    def auto(cls, force_fp16: bool | None = None) -> "Device":
        if torch.cuda.is_available():
            return cls(torch.device("cuda"), "cuda",
                       True if force_fp16 is None else force_fp16)
        if torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return cls(torch.device("mps"), "mps",
                       False if force_fp16 is None else force_fp16)
        return cls(torch.device("cpu"), "cpu", False)

    def empty_cache(self) -> None:
        if self.kind == "cuda":
            torch.cuda.empty_cache()
        elif self.kind == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def describe(self) -> str:
        if self.kind == "cuda":
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"CUDA ({name}, {mem:.1f} GB) | fp16={self.use_fp16}"
        return f"{self.kind.upper()} | fp16={self.use_fp16}"


# ===========================================================================
# 3. Tiling com FEATHER GAUSSIANO (drop-in para Real-ESRGAN / GFPGAN)
# ===========================================================================
class FeatheredUpscaler:
    """
    Envolve um RealESRGANer e processa a imagem em tiles SOBREPOSTOS, fundindo-os
    com uma máscara de peso suave (feather). Isso elimina as linhas de emenda
    típicas do tiling padrão em superfícies lisas (céu, pele, gradientes).

    Expõe `enhance(img, outscale)` compatível com a API do RealESRGANer, podendo
    ser usado diretamente como `bg_upsampler` do GFPGAN.
    """

    def __init__(self, upsampler: RealESRGANer, scale: int,
                 tile: int, overlap: int, logger: logging.Logger):
        self.upsampler = upsampler  # RealESRGANer com tiling interno DESLIGADO
        self.scale = scale
        self.tile = tile
        self.overlap = overlap
        self.logger = logger

    @staticmethod
    def _feather_mask(th: int, tw: int, feather: int) -> np.ndarray:
        """Máscara 2D separável: rampa suave nas bordas, 1.0 no interior."""
        wy = np.ones(th, np.float32)
        wx = np.ones(tw, np.float32)
        fy = min(feather, th // 2)
        fx = min(feather, tw // 2)
        if fy > 0:
            ramp = (np.arange(1, fy + 1, dtype=np.float32)) / (fy + 1)
            wy[:fy] = ramp
            wy[-fy:] = ramp[::-1]
        if fx > 0:
            ramp = (np.arange(1, fx + 1, dtype=np.float32)) / (fx + 1)
            wx[:fx] = ramp
            wx[-fx:] = ramp[::-1]
        return np.outer(wy, wx)[..., None]

    def enhance(self, img: np.ndarray, outscale: int | None = None):
        s = outscale or self.scale
        h, w = img.shape[:2]

        # Imagem pequena ou tiling desligado -> caminho direto.
        if self.tile <= 0 or (h <= self.tile and w <= self.tile):
            out, _ = self.upsampler.enhance(img, outscale=s)
            return out, None

        out_h, out_w = h * s, w * s
        acc = np.zeros((out_h, out_w, 3), np.float32)
        wsum = np.zeros((out_h, out_w, 1), np.float32)
        step = max(1, self.tile - self.overlap)
        feather_out = self.overlap * s

        ys = list(range(0, h, step))
        xs = list(range(0, w, step))
        for y in ys:
            for x in xs:
                y1 = min(y + self.tile, h)
                x1 = min(x + self.tile, w)
                y0 = max(0, y1 - self.tile)   # alinha à borda no último tile
                x0 = max(0, x1 - self.tile)

                tile_in = img[y0:y1, x0:x1]
                tile_out, _ = self.upsampler.enhance(tile_in, outscale=s)
                th, tw = tile_out.shape[:2]

                mask = self._feather_mask(th, tw, feather_out)
                oy, ox = y0 * s, x0 * s
                acc[oy:oy + th, ox:ox + tw] += tile_out.astype(np.float32) * mask
                wsum[oy:oy + th, ox:ox + tw] += mask

        wsum[wsum == 0] = 1.0
        maxv = 65535 if img.dtype == np.uint16 else 255
        return (acc / wsum).clip(0, maxv).astype(img.dtype), None


# ===========================================================================
# Resolução / download de pesos
# ===========================================================================
def resolve_weight(name_or_path: str | None, default_key: str,
                   logger: logging.Logger) -> str:
    Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)

    if name_or_path and Path(name_or_path).is_file():
        logger.info(f"  Pesos customizados: {name_or_path}")
        return str(Path(name_or_path).resolve())

    key = default_key
    if name_or_path and name_or_path in _WEIGHT_URLS:
        key = name_or_path

    local = Path(WEIGHTS_DIR) / f"{key}.pth"
    if local.is_file():
        return str(local.resolve())

    if key not in _WEIGHT_URLS:
        raise FileNotFoundError(
            f"Pesos '{name_or_path}' não encontrados e sem URL conhecida. "
            f"Baixe manualmente para '{WEIGHTS_DIR}/'."
        )

    logger.info(f"  Baixando pesos '{key}' (primeira execução)...")
    return load_file_from_url(
        url=_WEIGHT_URLS[key], model_dir=WEIGHTS_DIR,
        progress=True, file_name=f"{key}.pth",
    )


# ===========================================================================
# 1+4. Pipeline (Real-ESRGAN feathered + GFPGAN)
# ===========================================================================
@dataclass
class UpscalePipeline:
    device: Device
    upsampler: FeatheredUpscaler
    face_enhancer: object | None = None
    netscale: int = NETSCALE

    @classmethod
    def build(
        cls,
        device: Device,
        logger: logging.Logger,
        model_path: str | None = MODEL_PATH,
        model_name: str = MODEL_NAME,
        netscale: int = NETSCALE,
        num_block: int = NUM_BLOCK,
        tile: int = TILE_SIZE,
        overlap: int = TILE_OVERLAP,
        pre_pad: int = PRE_PAD,
        enable_face: bool = ENABLE_FACE_ENHANCE,
    ) -> "UpscalePipeline":
        rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                       num_block=num_block, num_grow_ch=32, scale=netscale)
        weight_path = resolve_weight(model_path or model_name, model_name, logger)

        # Tiling INTERNO do Real-ESRGAN desligado (tile=0): nós fazemos o nosso
        # com feather. pre_pad continua valendo por tile.
        raw = RealESRGANer(
            scale=netscale, model_path=weight_path, dni_weight=None, model=rrdb,
            tile=0, tile_pad=0, pre_pad=pre_pad,
            half=device.use_fp16, device=device.torch_device, gpu_id=None,
        )
        upsampler = FeatheredUpscaler(raw, netscale, tile, overlap, logger)
        logger.info(f"  Real-ESRGAN pronto (tile={tile}, overlap={overlap}, "
                    f"feather=on).")

        face_enhancer = None
        if enable_face:
            if not _GFPGAN_AVAILABLE:
                logger.warning("  GFPGAN não instalado -> face recovery OFF.")
            else:
                gfp_path = resolve_weight(GFPGAN_MODEL, GFPGAN_MODEL, logger)
                face_enhancer = GFPGANer(
                    model_path=gfp_path, upscale=netscale, arch="clean",
                    channel_multiplier=2, bg_upsampler=upsampler,  # feathered!
                    device=device.torch_device,
                )
                logger.info("  GFPGAN pronto (face recovery + blend automático).")

        return cls(device, upsampler, face_enhancer, netscale)

    def set_tile(self, tile: int) -> None:
        """Reduz o tile em runtime (usado no retry de OOM)."""
        self.upsampler.tile = tile

    def enhance(self, img_bgr: np.ndarray, fidelity: float) -> np.ndarray:
        if self.face_enhancer is not None:
            _, _, output = self.face_enhancer.enhance(
                img_bgr, has_aligned=False, only_center_face=ONLY_CENTER_FACE,
                paste_back=True, weight=fidelity,
            )
            return output
        output, _ = self.upsampler.enhance(img_bgr, outscale=self.netscale)
        return output


# ===========================================================================
# 1/2. Análise de conteúdo + pré/pós-processamento (CPU, OpenCV)
# ===========================================================================
def analyze_image(img_bgr: np.ndarray) -> tuple[float, float]:
    """
    Retorna (blur_var, noise_sigma).
      - blur_var: variância do Laplaciano (baixo = imagem mole/desfocada).
      - noise_sigma: estimativa de ruído de Immerkaer (escala 0-255).
    Operações baratas (numpy/cv2), seguras em CPU.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    h, w = gray.shape
    if h > 2 and w > 2:
        kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
        conv = cv2.filter2D(gray.astype(np.float64), -1, kernel)
        sigma = float(np.sqrt(np.pi / 2) / (6 * (w - 2) * (h - 2))
                      * np.sum(np.abs(conv)))
    else:
        sigma = 0.0
    return blur_var, sigma


def denoise(img_bgr: np.ndarray, strength: int) -> np.ndarray:
    """Denoise rápido (Non-Local Means colorido) aplicado ANTES do upscale."""
    if strength <= 0:
        return img_bgr
    return cv2.fastNlMeansDenoisingColored(
        img_bgr, None, h=strength, hColor=strength,
        templateWindowSize=7, searchWindowSize=21,
    )


def lanczos_resize(img_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize Lanczos de alta fidelidade via Pillow (entrada/saída BGR)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((w, h), Image.Resampling.LANCZOS)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def blend_detail(ai_bgr: np.ndarray, original_bgr: np.ndarray,
                 detail: float) -> np.ndarray:
    """
    Mistura a saída da IA com o upscale Lanczos do ORIGINAL.
    detail=1.0 -> 100% IA; detail<1.0 recupera fidelidade e reduz o "plástico".
    """
    if detail >= 0.999:
        return ai_bgr
    h, w = ai_bgr.shape[:2]
    base = lanczos_resize(original_bgr, w, h)
    detail = max(0.0, min(1.0, detail))
    return cv2.addWeighted(ai_bgr, detail, base, 1.0 - detail, 0.0)


def unsharp(img_bgr: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp mask sutil para nitidez percebida final."""
    if amount <= 0:
        return img_bgr
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=3.0)
    return cv2.addWeighted(img_bgr, 1.0 + amount, blur, -amount, 0.0)


# ===========================================================================
# 7. Redimensionamento exato com Lanczos
# ===========================================================================
def fit_to_target(img_bgr: np.ndarray, target_w: int | None,
                  target_h: int | None, mode: str = RESIZE_MODE) -> np.ndarray:
    if target_w is None and target_h is None:
        return img_bgr
    h, w = img_bgr.shape[:2]

    if mode == "exact" and target_w and target_h:
        new_w, new_h = target_w, target_h
    else:
        tw, th = target_w or w, target_h or h
        scale = min(tw / w, th / h)
        if scale >= 1.0:
            return img_bgr  # não amplia via Lanczos
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))

    if (new_w, new_h) == (w, h):
        return img_bgr
    return lanczos_resize(img_bgr, new_w, new_h)


# ===========================================================================
# I/O de imagem
# ===========================================================================
_EXIF_ORIENTATION_TAG = 274


def _apply_exif_orientation(img: np.ndarray, path: Path) -> np.ndarray:
    """
    cv2.imread(IMREAD_UNCHANGED) ignora a tag de orientação EXIF, então fotos
    de celular em retrato sairiam deitadas. Lê a tag via Pillow (só metadados,
    sem decodificar pixels) e aplica a transformação equivalente.
    """
    try:
        with Image.open(path) as pil:
            orientation = pil.getexif().get(_EXIF_ORIENTATION_TAG, 1)
    except Exception:
        return img
    if orientation == 2:
        return cv2.flip(img, 1)
    if orientation == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(img, 0)
    if orientation == 5:
        return cv2.transpose(img)
    if orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.transpose(img), -1)
    if orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Arquivo ilegível ou corrompido: {path.name}")
    img = _apply_exif_orientation(img, path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def save_image(img_bgr: np.ndarray, out_path: Path, fmt: str) -> None:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("jpg", "jpeg"):
        pil.save(out_path, format="JPEG", quality=JPEG_QUALITY,
                 subsampling=0, optimize=True)
    else:
        pil.save(out_path, format="PNG", compress_level=PNG_COMPRESSION)


# ===========================================================================
# Batch
# ===========================================================================
@dataclass
class ProcessOptions:
    fidelity: float = FACE_FIDELITY_WEIGHT
    denoise_strength: int = DENOISE_STRENGTH
    auto_analyze: bool = AUTO_ANALYZE
    auto_denoise: bool = AUTO_DENOISE
    detail: float = DETAIL_STRENGTH
    sharpen: float = SHARPEN_AMOUNT
    target_w: int | None = TARGET_WIDTH
    target_h: int | None = TARGET_HEIGHT
    resize_mode: str = RESIZE_MODE
    out_fmt: str = OUTPUT_FORMAT
    overwrite: bool = False


@dataclass
class BatchStats:
    total: int = 0
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)


def collect_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def _enhance_with_oom_retry(pipeline: UpscalePipeline, img: np.ndarray,
                            opts: ProcessOptions, logger: logging.Logger,
                            src_name: str) -> np.ndarray:
    """Tenta o upscale; em OOM, reduz o tile pela metade e tenta de novo."""
    original_tile = pipeline.upsampler.tile
    tile = original_tile if original_tile > 0 else 512
    while True:
        try:
            return pipeline.enhance(img, fidelity=opts.fidelity)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:  # type: ignore[attr-defined]
            msg = str(e).lower()
            is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or \
                "out of memory" in msg or "alloc" in msg
            if not is_oom or tile <= 64:
                raise
            tile = max(64, tile // 2)
            pipeline.set_tile(tile)
            pipeline.device.empty_cache()
            logger.warning(f"  OOM em {src_name}: reduzindo tile -> {tile} e "
                           f"tentando novamente.")


def process_one(pipeline: UpscalePipeline, src: Path, out_path: Path,
                opts: ProcessOptions, logger: logging.Logger) -> str:
    """Processa uma imagem e retorna uma string de status p/ log."""
    img = read_image(src)
    in_h, in_w = img.shape[:2]
    notes = []

    # 1. Análise de conteúdo
    if opts.auto_analyze:
        blur_var, sigma = analyze_image(img)
        quality = "nítida" if blur_var > 150 else \
                  ("média" if blur_var > 40 else "mole/desfocada")
        notes.append(f"blur={blur_var:.0f}({quality}) noise_sigma={sigma:.1f}")
        # 2. Denoise automático se ruidoso e usuário não fixou força
        if opts.auto_denoise and opts.denoise_strength == 0 \
                and sigma > NOISE_SIGMA_THRESHOLD:
            opts = ProcessOptions(**{**opts.__dict__,
                                     "denoise_strength": int(min(10, sigma))})
            notes.append(f"auto-denoise h={opts.denoise_strength}")

    # 2. Pré-denoise
    work = denoise(img, opts.denoise_strength) if opts.denoise_strength > 0 else img

    # 3+4. Upscale (feather) + face recovery, com retry de OOM
    output = _enhance_with_oom_retry(pipeline, work, opts, logger, src.name)

    # 5. Blend de detalhe (anti-plástico / fidelidade)
    output = blend_detail(output, img, opts.detail)

    # 6. Unsharp finishing
    output = unsharp(output, opts.sharpen)

    # 7. Resolução exata
    output = fit_to_target(output, opts.target_w, opts.target_h,
                           mode=opts.resize_mode)

    save_image(output, out_path, opts.out_fmt)
    out_h, out_w = output.shape[:2]
    return f"{in_w}x{in_h} -> {out_w}x{out_h} | {' | '.join(notes)}"


def run_batch(pipeline: UpscalePipeline, input_dir: Path, output_dir: Path,
              opts: ProcessOptions, logger: logging.Logger) -> BatchStats:
    images = collect_images(input_dir)
    stats = BatchStats(total=len(images))
    if not images:
        logger.warning(f"Nenhuma imagem suportada em '{input_dir}'.")
        return stats

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\nProcessando {len(images)} imagem(ns) -> '{output_dir}'\n")
    base_tile = pipeline.upsampler.tile

    pbar = tqdm(images, unit="img", dynamic_ncols=True)
    for src in pbar:
        pbar.set_description(f"{src.name[:40]:<40}")
        out_path = output_dir / f"{src.stem}.{opts.out_fmt}"

        if out_path.exists() and not opts.overwrite:
            stats.skipped += 1
            logger.debug(f"SKIP (já existe): {src.name}")
            continue

        t0 = time.perf_counter()
        try:
            info = process_one(pipeline, src, out_path, opts, logger)
            logger.debug(f"OK {src.name}: {info} em {time.perf_counter()-t0:.1f}s")
            stats.ok += 1
        except Exception as e:
            stats.failed += 1
            stats.failures.append((src.name, str(e)))
            logger.error(f"FALHA em {src.name}: {e}")
            logger.debug(traceback.format_exc())
        finally:
            pipeline.set_tile(base_tile)   # restaura tile após possível retry
            pipeline.device.empty_cache()

    pbar.close()
    return stats


def print_summary(stats: BatchStats, logger: logging.Logger) -> None:
    logger.info("\n" + "=" * 56)
    logger.info("RESUMO DO LOTE")
    logger.info("=" * 56)
    logger.info(f"  Total:    {stats.total}")
    logger.info(f"  Sucesso:  {stats.ok}")
    logger.info(f"  Pulados:  {stats.skipped}")
    logger.info(f"  Falhas:   {stats.failed}")
    if stats.failures:
        logger.info("\n  Falhas (ver upscaler.log):")
        for name, reason in stats.failures:
            logger.info(f"    - {name}: {reason[:80]}")
    logger.info("=" * 56)


# ===========================================================================
# CLI
# ===========================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Topaz-Killer: SR adaptativa em lote (Real-ESRGAN + GFPGAN).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", default=INPUT_DIR)
    p.add_argument("-o", "--output", default=OUTPUT_DIR)
    p.add_argument("-m", "--model-path", default=MODEL_PATH,
                   help="Caminho p/ .pth customizado (4x-UltraSharp etc.)")
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--num-block", type=int, default=NUM_BLOCK,
                   help="23 padrão; 6 p/ modelos *_anime_6B")
    p.add_argument("--netscale", type=int, default=NETSCALE)
    p.add_argument("--tile", type=int, default=TILE_SIZE,
                   help="Tile (0 desliga; reduza p/ menos VRAM)")
    p.add_argument("--overlap", type=int, default=TILE_OVERLAP,
                   help="Sobreposição p/ o feather (px de entrada)")
    p.add_argument("--no-face", action="store_true")
    p.add_argument("--fidelity", type=float, default=FACE_FIDELITY_WEIGHT)
    p.add_argument("--denoise", type=int, default=DENOISE_STRENGTH,
                   help="Força do pré-denoise (0=off, 3-5 leve, 7-10 forte)")
    p.add_argument("--auto-denoise", action="store_true", default=AUTO_DENOISE,
                   help="Liga denoise automaticamente em imagens ruidosas")
    p.add_argument("--no-analyze", dest="analyze", action="store_false",
                   default=AUTO_ANALYZE)
    p.add_argument("--detail", type=float, default=DETAIL_STRENGTH,
                   help="1.0=100%% IA; <1.0 mistura Lanczos (anti-plástico)")
    p.add_argument("--sharpen", type=float, default=SHARPEN_AMOUNT,
                   help="Unsharp final (0=off; 0.3-0.8 sutil)")
    p.add_argument("--width", type=int, default=TARGET_WIDTH)
    p.add_argument("--height", type=int, default=TARGET_HEIGHT)
    p.add_argument("--resize-mode", choices=["fit", "exact"], default=RESIZE_MODE)
    p.add_argument("--format", choices=["png", "jpg"], default=OUTPUT_FORMAT)
    p.add_argument("--fp16", dest="fp16", action="store_true", default=None)
    p.add_argument("--fp32", dest="fp16", action="store_false")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    logger = setup_logging()

    logger.info("=" * 56)
    logger.info("  TOPAZ-KILLER UPSCALER  |  pipeline adaptativo")
    logger.info("=" * 56)

    input_dir, output_dir = Path(args.input), Path(args.output)
    if not input_dir.is_dir():
        logger.error(f"Diretório de entrada inexistente: {input_dir}")
        return 1

    device = Device.auto(force_fp16=args.fp16 if args.fp16 is not None else USE_FP16)
    logger.info(f"Dispositivo: {device.describe()}")

    target_w = args.width if args.width and args.width > 0 else None
    target_h = args.height if args.height and args.height > 0 else None

    logger.info("Carregando modelos...")
    try:
        pipeline = UpscalePipeline.build(
            device=device, logger=logger, model_path=args.model_path,
            model_name=args.model_name, netscale=args.netscale,
            num_block=args.num_block, tile=args.tile, overlap=args.overlap,
            enable_face=not args.no_face,
        )
    except Exception as e:
        logger.error(f"Falha ao construir o pipeline: {e}")
        logger.debug(traceback.format_exc())
        return 1

    opts = ProcessOptions(
        fidelity=args.fidelity, denoise_strength=args.denoise,
        auto_analyze=args.analyze, auto_denoise=args.auto_denoise,
        detail=args.detail, sharpen=args.sharpen,
        target_w=target_w, target_h=target_h,
        resize_mode=args.resize_mode,
        out_fmt=args.format, overwrite=args.overwrite,
    )

    stats = run_batch(pipeline, input_dir, output_dir, opts, logger)
    print_summary(stats, logger)
    return 0 if stats.failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
