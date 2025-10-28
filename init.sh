#!/usr/bin/env bash

# ===================== User/Env Setup =====================
# Uses your original provisioning script structure, then:
# 1) Downloads 3 extra models in parallel to specific folders
# 2) Installs screen + updog
# 3) Starts updog in a detached screen session with password 123123aA

set -Eeuo pipefail

# ---- Your original vars ----
source /venv/main/bin/activate || true
: "${WORKSPACE:=$PWD}"
COMFYUI_DIR="${WORKSPACE}/ComfyUI"

APT_PACKAGES=(
    # add apt packages here if needed
)

PIP_PACKAGES=(
    # add pip packages here if needed
)

NODES=(
    # "https://github.com/ltdrdata/ComfyUI-Manager"
    # "https://github.com/cubiq/ComfyUI_essentials"
)

WORKFLOWS=(
    "https://gist.githubusercontent.com/robballantyne/f8cb692bdcd89c96c0bd1ec0c969d905/raw/2d969f732d7873f0e1ee23b2625b50f201c722a5/flux_dev_example.json"
)

CLIP_MODELS=(
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"
)

UNET_MODELS=(
)

VAE_MODELS=(
)

# ---- New: parallelism for extra downloads & updog config ----
JOBS="${JOBS:-3}"
UPDOG_DIR="${DIR:-${WORKSPACE}/ComfyUI}"
UPDOG_PORT="${PORT:-8000}"
UPDOG_PASS="123123aA"
UPDOG_SESSION="updog"

# Extra files to fetch in parallel (dest dirs are relative to COMFYUI_DIR)
# Format: "URL|relative_dest_dir"
PARALLEL_MAP=(
  "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors|models/text_encoders"
  "https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors|models/diffusion_models"
  "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors|models/vae"
)

# ===================== Provisioning Core =====================

function provisioning_start() {
    provisioning_print_header
    provisioning_get_apt_packages
    provisioning_get_nodes
    provisioning_get_pip_packages

    workflows_dir="${COMFYUI_DIR}/user/default/workflows"
    mkdir -p "${workflows_dir}"
    provisioning_get_files \
        "${workflows_dir}" \
        "${WORKFLOWS[@]}"

    # Get licensed models if HF_TOKEN set & valid
    if provisioning_has_valid_hf_token; then
        UNET_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors")
        VAE_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors")
    else
        UNET_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors")
        VAE_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors")
        sed -i 's/flux1-dev\.safetensors/flux1-schnell.safetensors/g' "${workflows_dir}/flux_dev_example.json" || true
    fi

    provisioning_get_files \
        "${COMFYUI_DIR}/models/unet" \
        "${UNET_MODELS[@]}"
    provisioning_get_files \
        "${COMFYUI_DIR}/models/vae" \
        "${VAE_MODELS[@]}"
    provisioning_get_files \
        "${COMFYUI_DIR}/models/clip" \
        "${CLIP_MODELS[@]}"

    # ---- New: parallel downloads for your 3 specific files ----
    provisioning_get_files_parallel "${COMFYUI_DIR}" "${PARALLEL_MAP[@]}"
    provisioning_download \
      "https://raw.githubusercontent.com/thangpmedia/init-server/refs/heads/main/Detect_Strongest_Colors.py" \
      "${COMFYUI_DIR}/custom_nodes"
    provisioning_install_updog_and_screen
    provisioning_start_updog

    provisioning_print_end
}

function provisioning_get_apt_packages() {
    if [[ ${#APT_PACKAGES[@]} -gt 0 ]]; then
        provisioning_ensure_apt
        sudo apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"
    fi
}

function provisioning_get_pip_packages() {
    if [[ ${#PIP_PACKAGES[@]} -gt 0 ]]; then
        pip install --no-cache-dir "${PIP_PACKAGES[@]}"
    fi
}

function provisioning_get_nodes() {
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="${COMFYUI_DIR}/custom_nodes/${dir}"
        requirements="${path}/requirements.txt"
        if [[ -d $path ]]; then
            if [[ ${AUTO_UPDATE:-true,,} != "false" ]]; then
                printf "Updating node: %s...\n" "${repo}"
                ( cd "$path" && git pull )
                if [[ -e $requirements ]]; then
                   pip install --no-cache-dir -r "$requirements"
                fi
            fi
        else
            printf "Downloading node: %s...\n" "${repo}"
            git clone "${repo}" "${path}" --recursive
            if [[ -e $requirements ]]; then
                pip install --no-cache-dir -r "${requirements}"
            fi
        fi
    done
}

function provisioning_get_files() {
    [[ -z "${2:-}" ]] && return 1
    dir="$1"; shift
    arr=("$@")
    mkdir -p "$dir"
    printf "Downloading %s file(s) to %s...\n" "${#arr[@]}" "$dir"
    for url in "${arr[@]}"; do
        printf "Downloading: %s\n" "${url}"
        provisioning_download "${url}" "${dir}"
        printf "\n"
    done
}

function provisioning_print_header() {
    printf "\n##############################################\n#                                            #\n#          Provisioning container            #\n#                                            #\n#         This will take some time           #\n#                                            #\n# Your container will be ready on completion #\n#                                            #\n##############################################\n\n"
}

function provisioning_print_end() {
    printf "\nProvisioning complete: Application will start now\n\n"
}

function provisioning_has_valid_hf_token() {
    [[ -n "${HF_TOKEN:-}" ]] || return 1
    local url="https://huggingface.co/api/whoami-v2"
    local response
    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $HF_TOKEN" \
        -H "Content-Type: application/json")
    [[ "$response" -eq 200 ]]
}

function provisioning_has_valid_civitai_token() {
    [[ -n "${CIVITAI_TOKEN:-}" ]] || return 1
    local url="https://civitai.com/api/v1/models?hidden=1&limit=1"
    local response
    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $CIVITAI_TOKEN" \
        -H "Content-Type: application/json")
    [[ "$response" -eq 200 ]]
}

# Download from $1 URL to $2 directory path (keeps original filename)
function provisioning_download() {
    local url="$1" dest="$2" auth_token=""
    if [[ -n "${HF_TOKEN:-}" && $url =~ ^https://([a-zA-Z0-9_-]+\.)?huggingface\.co(/|$|\?) ]]; then
        auth_token="$HF_TOKEN"
    elif [[ -n "${CIVITAI_TOKEN:-}" && $url =~ ^https://([a-zA-Z0-9_-]+\.)?civitai\.com(/|$|\?) ]]; then
        auth_token="$CIVITAI_TOKEN"
    fi
    if [[ -n $auth_token ]]; then
        wget --header="Authorization: Bearer $auth_token" \
             -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" \
             -P "$dest" "$url"
    else
        wget -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" \
             -P "$dest" "$url"
    fi
}

# ===================== New: Parallel Downloader =====================

# Takes base_dir + list of "URL|relative_dest"
function provisioning_get_files_parallel() {
    local base="$1"; shift
    local entries=("$@")
    local -i running=0
    local -i limit="${JOBS}"

    echo "Parallel downloading (${limit} workers) to ${base} ..."
    for entry in "${entries[@]}"; do
        IFS='|' read -r url rel <<<"$entry"
        local_dir="${base}/${rel}"
        mkdir -p "${local_dir}"

        (
            echo "→ ${url}"
            provisioning_download "${url}" "${local_dir}"
            echo "✓ Saved to ${local_dir}"
        ) &

        running=$((running+1))
        if (( running >= limit )); then
            wait -n || true
            running=$((running-1))
        fi
    done
    wait || true
    echo "Parallel downloads completed."
}

# ===================== New: Updog & Screen =====================

function provisioning_ensure_apt() {
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "apt-get not found. This script targets Ubuntu/Debian." >&2
        exit 1
    fi
    sudo true >/dev/null 2>&1 || true
    export DEBIAN_FRONTEND=noninteractive
    sudo apt-get update -y
}

function provisioning_install_updog_and_screen() {
    echo "Installing screen + python3-pip + updog ..."
    provisioning_ensure_apt
    sudo apt-get install -y --no-install-recommends screen python3-pip curl ca-certificates
    python3 -m pip install --upgrade pip updog
}

function provisioning_start_updog() {
    local session="${UPDOG_SESSION}"
    local dir="${UPDOG_DIR}"
    local port="${UPDOG_PORT}"
    local pass="${UPDOG_PASS}"

    mkdir -p "${dir}"
    # Kill existing session if present
    if screen -ls | grep -q "[.]${session}"; then
        screen -S "${session}" -X quit || true
        sleep 0.2
    fi

    # Resolve updog binary
    local updog_bin
    updog_bin="$(command -v updog || true)"
    if [[ -z "$updog_bin" ]]; then
        echo "ERROR: updog not found after installation." >&2
        return 1
    fi

    echo "Starting Updog in detached screen: session='${session}', dir='${dir}', port='${port}'"
    screen -S "${session}" -dm bash -lc \
      "'$updog_bin' -p '${port}' -d '${dir}' --password '${pass}'"

    echo "Updog is running."
    echo "Attach:  screen -r ${session}"
    echo "Stop:    screen -S ${session} -X quit"
    echo "URL:     http://<server-ip>:${port}   (password: ${pass})"
}

# ===================== Entry Point =====================

# Allow user to disable provisioning if they started with a script they didn't want
if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi
