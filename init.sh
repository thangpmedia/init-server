#!/usr/bin/env bash
# Minimal installer for:
# - Custom node: Detect_Strongest_Colors.py -> workspace/ComfyUI/custom_nodes
# - Model files (parallel downloads) into workspace/ComfyUI/models/...
# - screen + updog install, then auto-run updog in detached screen
#
# Env overrides:
#   JOBS   = parallel download workers (default 3)
#   DIR    = directory to serve with updog (default workspace/ComfyUI)
#   PORT   = updog port (default 8000)
#   PASS   = updog password (default 123123aA)
#   HF_TOKEN = Hugging Face token for gated assets (optional)

set -Eeuo pipefail

JOBS="${JOBS:-3}"
UPDOG_DIR="${DIR:-workspace/ComfyUI}"
UPDOG_PORT="${PORT:-8000}"
UPDOG_PASS="${PASS:-123123aA}"
UPDOG_SESSION="updog"

# ---- Paths (fixed to match your layout) ----
BASE="workspace/ComfyUI"
TEXT_ENCODERS="${BASE}/models/text_encoders"
DIFFUSION="${BASE}/models/diffusion_models"
VAE="${BASE}/models/vae"
CUSTOM_NODES="${BASE}/custom_nodes"

# ---- Targets ----
CUSTOM_NODE_URL="https://raw.githubusercontent.com/thangpmedia/init-server/refs/heads/main/Detect_Strongest_Colors.py"

MAP=(
  "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors|${TEXT_ENCODERS}"
  "https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors|${DIFFUSION}"
  "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors|${VAE}"
)

# ---- Utils ----
need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' is required." >&2; exit 1; }; }

ensure_dirs() {
  mkdir -p "${TEXT_ENCODERS}" "${DIFFUSION}" "${VAE}" "${CUSTOM_NODES}"
}

# curl downloader → tmp file then atomic mv; supports HF token
dl_one() {
  local url="$1" dest_dir="$2"
  local filename; filename="$(basename "${url%%\?*}")"
  mkdir -p "$dest_dir"
  local final="${dest_dir}/${filename}"
  local temp="${final}.part"

  if [[ -f "$final" ]]; then
    echo "✓ Exists, skip: $final"
    return 0
  fi

  local args=(-fSL --retry 5 --retry-delay 2 --connect-timeout 30 -o "$temp")
  # Resume if part file exists
  [[ -f "$temp" ]] && args+=(-C -)
  # HF token for huggingface
  if [[ -n "${HF_TOKEN:-}" && "$url" =~ ^https://([a-zA-Z0-9_-]+\.)?huggingface\.co(/|$|\?) ]]; then
    args+=(-H "Authorization: Bearer $HF_TOKEN")
  fi

  echo "→ Downloading: $filename"
  if curl "${args[@]}" "$url"; then
    mv -f "$temp" "$final"
    echo "✓ Saved: $final"
  else
    echo "✗ Failed: $url" >&2
    rm -f "$temp" || true
    return 1
  fi
}

download_custom_node() {
  echo "==> Installing custom node"
  dl_one "$CUSTOM_NODE_URL" "$CUSTOM_NODES"
}

download_models_parallel() {
  echo "==> Parallel model downloads ($JOBS workers)"
  local -i running=0
  for entry in "${MAP[@]}"; do
    IFS='|' read -r url dest <<<"$entry"
    ( dl_one "$url" "$dest" ) &
    running=$((running+1))
    if (( running >= JOBS )); then
      wait -n || true
      running=$((running-1))
    fi
  done
  wait || true
  echo "==> Model downloads complete"
}

install_updog_and_screen() {
  echo "==> Installing screen + python3-pip + updog (Ubuntu/Debian)"
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get not found — please install screen & updog manually." >&2
  else
    local SUDO=""
    if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi
    export DEBIAN_FRONTEND=noninteractive
    $SUDO apt-get update -y
    $SUDO apt-get install -y --no-install-recommends screen python3-pip ca-certificates curl
    python3 -m pip install --upgrade pip updog
  fi
  need screen
  need updog
}

start_updog() {
  echo "==> Starting Updog in detached screen"
  mkdir -p "$UPDOG_DIR"
  # Kill existing session if any
  if screen -ls | grep -q "[.]${UPDOG_SESSION}"; then
    screen -S "${UPDOG_SESSION}" -X quit || true
    sleep 0.2
  fi
  screen -S "${UPDOG_SESSION}" -dm bash -lc \
    "updog -p '${UPDOG_PORT}' -d '${UPDOG_DIR}' --password '${UPDOG_PASS}'"

  echo "Updog running."
  echo "  Session: screen -r ${UPDOG_SESSION}"
  echo "  Stop:    screen -S ${UPDOG_SESSION} -X quit"
  echo "  URL:     http://<server-ip>:${UPDOG_PORT}  (pass: ${UPDOG_PASS})"
}

main() {
  need curl
  ensure_dirs
  download_custom_node
  download_models_parallel
  install_updog_and_screen
  start_updog
  echo "All done."
}

main "$@"
