#!/usr/bin/env bash
# Abilita Whisper sul Neural Engine (Core ML) per pywhispercpp.
#
# Il wheel di default (da `uv sync`) gira su Metal/GPU. Questo script:
#   1. scarica l'encoder Core ML GIÀ compilato da Hugging Face (niente Xcode/coremlc);
#   2. ricompila pywhispercpp dai sorgenti con WHISPER_COREML=1 (basta i Command Line Tools);
#   3. corregge l'rpath del modulo .so (bug noto: punta alla dir di build, ora sparita).
#
# Rieseguilo dopo aver ricreato .venv o aggiornato pywhispercpp.
# Uso:  ./scripts/setup_coreml.sh
set -euo pipefail

MODEL="large-v3-turbo"
MLMODELC_ZIP="ggml-${MODEL}-encoder.mlmodelc.zip"
HF_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${MLMODELC_ZIP}"

echo "==> Individuo la cartella modelli di pywhispercpp"
MODELS_DIR=$(uv run --no-sync python -c "import pywhispercpp.constants as c; print(c.MODELS_DIR)")
echo "    $MODELS_DIR"
mkdir -p "$MODELS_DIR"

if [ ! -d "$MODELS_DIR/ggml-${MODEL}-encoder.mlmodelc" ]; then
  echo "==> Scarico l'encoder Core ML precompilato (~1 GB)"
  curl -L -o "$MODELS_DIR/$MLMODELC_ZIP" "$HF_URL"
  ( cd "$MODELS_DIR" && unzip -oq "$MLMODELC_ZIP" && rm -f "$MLMODELC_ZIP" && rm -rf __MACOSX )
else
  echo "==> Encoder Core ML già presente, salto il download"
fi

echo "==> Ricompilo pywhispercpp con Core ML (WHISPER_COREML=1)"
WHISPER_COREML=1 uv pip install --reinstall --no-binary pywhispercpp --no-cache-dir pywhispercpp

echo "==> Correggo l'rpath del modulo .so (aggiungo @loader_path)"
SO=$(uv run --no-sync python -c "import _pywhispercpp, pathlib; print(_pywhispercpp.__file__)")
if ! otool -l "$SO" | grep -A2 LC_RPATH | grep -q "@loader_path"; then
  install_name_tool -add_rpath @loader_path "$SO"
  echo "    @loader_path aggiunto"
else
  echo "    @loader_path già presente"
fi

echo "==> Verifica"
uv run --no-sync python -c "from pywhispercpp.model import Model; print('pywhispercpp Core ML OK')"
echo "Fatto. Al primo uso l'ANE compila il modello per il device (lento la prima volta)."
