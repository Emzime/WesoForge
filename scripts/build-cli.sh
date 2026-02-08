#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

prereq_banner() {
  echo "" >&2
  echo "=== Prérequis: build CLI WesoForge ===" >&2
  echo "Composants requis :" >&2
  echo "  - Rust toolchain (rustup + cargo)" >&2
  echo "  - Git (recommandé)" >&2
  echo "" >&2
  echo "Si vous l'autorisez, le script tente d'installer automatiquement les composants manquants." >&2
  echo "Forcer le comportement via: BBR_PREREQ_AUTO=1 (installer) ou BBR_PREREQ_AUTO=0 (ne pas installer)." >&2
  echo "" >&2
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

pkg_mgr() {
  if have_cmd apt-get; then echo "apt"; return; fi
  if have_cmd dnf; then echo "dnf"; return; fi
  if have_cmd yum; then echo "yum"; return; fi
  if have_cmd pacman; then echo "pacman"; return; fi
  if have_cmd zypper; then echo "zypper"; return; fi
  if have_cmd brew; then echo "brew"; return; fi
  echo ""
}

ask_consent() {
  local missing=($@)
  if [[ ${#missing[@]} -eq 0 ]]; then
    return 1
  fi

  if [[ -n "${BBR_PREREQ_AUTO:-}" ]]; then
    case "${BBR_PREREQ_AUTO,,}" in
      1|true|yes|y) return 0 ;;
      0|false|no|n) return 1 ;;
    esac
  fi

  echo "Composants manquants :" >&2
  for m in "${missing[@]}"; do
    echo "  - $m" >&2
  done
  echo "" >&2
  read -r -p "Autoriser l'installation automatique de TOUS les composants manquants ? (y/N) " ans
  [[ "${ans,,}" == "y" || "${ans,,}" == "yes" ]]
}

install_rustup() {
  local pm="$1"
  if have_cmd rustup && have_cmd cargo && have_cmd rustc; then return 0; fi
  case "$pm" in
    apt)
      sudo apt-get update
      sudo apt-get install -y curl ca-certificates build-essential pkg-config
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
      ;;
    dnf|yum)
      sudo "$pm" install -y curl ca-certificates gcc gcc-c++ make pkgconfig
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
      ;;
    pacman)
      sudo pacman -Sy --noconfirm curl ca-certificates base-devel pkgconf
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
      ;;
    zypper)
      sudo zypper refresh
      sudo zypper install -y curl ca-certificates gcc gcc-c++ make pkg-config
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
      ;;
    brew)
      brew install rustup-init
      rustup-init -y
      ;;
    *)
      echo "error: impossible d'installer rustup automatiquement (gestionnaire de paquets inconnu)." >&2
      return 1
      ;;
  esac
  if [[ -d "$HOME/.cargo/bin" ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
  fi
}

install_git() {
  local pm="$1"
  if have_cmd git; then return 0; fi
  case "$pm" in
    apt) sudo apt-get update && sudo apt-get install -y git ;;
    dnf|yum) sudo "$pm" install -y git ;;
    pacman) sudo pacman -Sy --noconfirm git ;;
    zypper) sudo zypper install -y git ;;
    brew) brew install git ;;
    *) return 1 ;;
  esac
}

ensure_prereqs_cli() {
  prereq_banner
  local pm
  pm="$(pkg_mgr)"

  local missing=()
  (have_cmd cargo && have_cmd rustc) || missing+=("Rust toolchain")
  have_cmd git || missing+=("Git")

  if [[ ${#missing[@]} -eq 0 ]]; then
    return 0
  fi

  if [[ -z "$pm" ]]; then
    echo "error: aucun gestionnaire de paquets détecté (apt/dnf/yum/pacman/zypper/brew)." >&2
    echo "Installez les prérequis manuellement puis relancez ce script." >&2
    return 1
  fi

  if ! ask_consent "${missing[@]}"; then
    echo "error: prérequis manquants et installation automatique non autorisée." >&2
    return 1
  fi

  install_git "$pm" || true
  install_rustup "$pm"
}

ensure_prereqs_cli

DIST_DIR="${DIST_DIR:-$ROOT/dist}"
mkdir -p "$DIST_DIR"

workspace_version() {
  awk '
    /^\[workspace\.package\]/ { in_pkg = 1; next }
    in_pkg && /^\[/ { in_pkg = 0 }
    in_pkg && match($0, /^version[[:space:]]*=[[:space:]]*"/) {
      rest = substr($0, RSTART + RLENGTH)
      if (match(rest, /[^"]*/)) {
        print substr(rest, RSTART, RLENGTH)
        exit
      }
    }
  ' Cargo.toml
}

platform_arch() {
  local arch
  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64) echo "amd64" ;;
    aarch64|arm64) echo "arm64" ;;
    *) echo "$arch" ;;
  esac
}

platform_os() {
  local os
  os="$(uname -s)"
  case "$os" in
    Linux) echo "Linux" ;;
    Darwin) echo "macOS" ;;
    *) echo "$os" ;;
  esac
}

TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
mkdir -p "$TARGET_DIR"
if ! : >"$TARGET_DIR/.wesoforge-write-test" 2>/dev/null; then
  echo "warning: CARGO_TARGET_DIR is not writable ($TARGET_DIR); using $ROOT/target instead" >&2
  TARGET_DIR="$ROOT/target"
  mkdir -p "$TARGET_DIR"
fi
rm -f "$TARGET_DIR/.wesoforge-write-test" >/dev/null 2>&1 || true
export CARGO_TARGET_DIR="$TARGET_DIR"

if [[ "${BBR_SKIP_CARGO_BUILD:-0}" != "1" ]]; then
  echo "Building WesoForge CLI (wesoforge)..." >&2
  CARGO_ARGS=(build -p bbr-client --release --features prod-backend)
  if [[ "${CARGO_LOCKED:-0}" == "1" ]]; then
    CARGO_ARGS+=(--locked)
  fi
  if [[ "${CARGO_OFFLINE:-0}" == "1" ]]; then
    CARGO_ARGS+=(--offline)
  fi
  cargo "${CARGO_ARGS[@]}"
fi

VERSION="$(workspace_version)"
if [[ -z "${VERSION:-}" ]]; then
  echo "error: failed to determine workspace version from Cargo.toml" >&2
  exit 1
fi
ARCH="$(platform_arch)"
OS="$(platform_os)"

BIN_SRC="$TARGET_DIR/release/wesoforge"
BIN_DST="$DIST_DIR/WesoForge-cli_${OS}_${VERSION}_${ARCH}"

if [[ ! -f "$BIN_SRC" ]]; then
  echo "error: expected binary not found at: $BIN_SRC" >&2
  exit 1
fi

install -m 0755 "$BIN_SRC" "$BIN_DST"
echo "Wrote: $BIN_DST" >&2
