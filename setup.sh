#!/usr/bin/env bash
# ============================================================================
#  setup.sh — cria o ambiente do Topaz-Killer Upscaler
#  Requer Python 3.10 ou 3.11 (o stack basicsr/gfpgan NÃO suporta 3.12+).
# ============================================================================
set -euo pipefail

# Procura um interpretador 3.11 ou 3.10 (nessa ordem).
PYBIN=""
for cand in python3.11 python3.10; do
    if command -v "$cand" >/dev/null 2>&1; then PYBIN="$cand"; break; fi
done
if [ -z "$PYBIN" ]; then
    echo "ERRO: Python 3.11 ou 3.10 não encontrado."
    echo "Instale um deles (ex.: 'brew install python@3.11') e rode de novo."
    exit 1
fi
echo ">> Usando: $($PYBIN --version)"

# Cria o venv
$PYBIN -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

# GPU NVIDIA? Descomente o bloco abaixo ANTES do requirements para CUDA 12.1.
# pip install torch==2.1.2 torchvision==0.16.2 \
#     --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# Estrutura de pastas
mkdir -p input output weights

echo
echo ">> Pronto. Para usar:"
echo "   source .venv/bin/activate"
echo "   # coloque imagens em ./input"
echo "   python topaz_killer_upscaler.py"
