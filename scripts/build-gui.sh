#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

prereq_banner() {
  echo "" >&2
  echo "=== Prérequis: build GUI WesoForge ===" >&2
  echo "Composants requis :" >&2
  echo "  - Rust toolchain (rustup + cargo)" >&2
  echo "  - Node.js" >&2
  echo "  - pnpm (via corepack ou npm)" >&2
  echo "  - Tauri CLI (cargo tauri)" >&2
  echo "  - Git" >&2
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

  # Ensure cargo is on PATH for this session.
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

install_node() {
  local pm="$1"
  if have_cmd node; then return 0; fi
  case "$pm" in
    apt) sudo apt-get update && sudo apt-get install -y nodejs npm ;;
    dnf|yum) sudo "$pm" install -y nodejs npm ;;
    pacman) sudo pacman -Sy --noconfirm nodejs npm ;;
    zypper) sudo zypper install -y nodejs npm ;;
    brew) brew install node ;;
    *) return 1 ;;
  esac
}

ensure_pnpm() {
  if have_cmd pnpm; then return 0; fi
  if have_cmd corepack; then
    corepack enable
    corepack prepare pnpm@latest --activate
    return 0
  fi
  if have_cmd npm; then
    npm i -g pnpm
    return 0
  fi
  return 1
}

ensure_tauri_cli() {
  if have_cmd cargo; then
    if cargo tauri --version >/dev/null 2>&1; then
      return 0
    fi
    cargo install tauri-cli --locked
    return 0
  fi
  return 1
}

install_tauri_system_deps_linux() {
  # Best-effort packages for Tauri on Linux.
  # Package names vary by distro; this covers the common case.
  local pm="$1"
  case "$pm" in
    apt)
      sudo apt-get update
      sudo apt-get install -y \
        libgtk-3-dev \
        libayatana-appindicator3-dev \
        libwebkit2gtk-4.1-dev \
        librsvg2-dev \
        libssl-dev \
        pkg-config
      ;;
    dnf|yum)
      sudo "$pm" install -y \
        gtk3-devel \
        webkit2gtk4.1-devel \
        openssl-devel \
        pkgconf-pkg-config \
        librsvg2-devel
      ;;
    pacman)
      sudo pacman -Sy --noconfirm \
        gtk3 \
        webkit2gtk-4.1 \
        openssl \
        pkgconf \
        librsvg
      ;;
    zypper)
      sudo zypper install -y \
        gtk3-devel \
        webkit2gtk3-devel \
        libopenssl-devel \
        pkg-config \
        librsvg-devel
      ;;
    brew)
      # macOS uses system frameworks; no equivalent here.
      ;;
  esac
}

ensure_prereqs_gui() {
  prereq_banner

  local pm
  pm="$(pkg_mgr)"

  local missing=()
  have_cmd git || missing+=("Git")
  (have_cmd cargo && have_cmd rustc) || missing+=("Rust toolchain")
  have_cmd node || missing+=("Node.js")
  (have_cmd pnpm || have_cmd corepack || have_cmd npm) || missing+=("pnpm")
  if ! (have_cmd cargo && cargo tauri --version >/dev/null 2>&1); then
    missing+=("Tauri CLI")
  fi

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

  # Install order matters.
  install_git "$pm" || true
  install_rustup "$pm"
  install_node "$pm"
  ensure_pnpm

  # Linux-only native deps for Tauri.
  if [[ "$(uname -s)" == "Linux" ]]; then
    install_tauri_system_deps_linux "$pm" || true
  fi

  ensure_tauri_cli
}

ensure_prereqs_gui

UNAME="$(uname -s)"
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

TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
mkdir -p "$TARGET_DIR"
if ! : >"$TARGET_DIR/.wesoforge-write-test" 2>/dev/null; then
  echo "warning: CARGO_TARGET_DIR is not writable ($TARGET_DIR); using $ROOT/target instead" >&2
  TARGET_DIR="$ROOT/target"
  mkdir -p "$TARGET_DIR"
fi
rm -f "$TARGET_DIR/.wesoforge-write-test" >/dev/null 2>&1 || true
export CARGO_TARGET_DIR="$TARGET_DIR"

if ! command -v pnpm >/dev/null 2>&1; then
  echo "error: pnpm not found (needed to build the Svelte frontend)." >&2
  exit 1
fi

if [[ ! -d ui/node_modules ]]; then
  pnpm -C ui install
fi

if ! cargo tauri --help >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: `cargo tauri` not found.

Install the Tauri CLI (cargo subcommand) first, for example:
  cargo install tauri-cli
EOF
  exit 1
fi

DIST_DIR="${DIST_DIR:-$ROOT/dist}"
mkdir -p "$DIST_DIR"

SUPPORT_DEVTOOLS="${SUPPORT_DEVTOOLS:-0}"
# Build the GUI with GPU probing/selection code enabled.
# Runtime usage still depends on user selection/env and successful dynamic loading.
FEATURES="prod-backend,gpu"
OUT_PREFIX="WesoForge-gui"
if [[ "$SUPPORT_DEVTOOLS" == "1" ]]; then
  FEATURES="$FEATURES,support-devtools"
  OUT_PREFIX="WesoForge-gui-support"
fi

VERSION="$(workspace_version)"
if [[ -z "${VERSION:-}" ]]; then
  echo "error: failed to determine workspace version from Cargo.toml" >&2
  exit 1
fi
ARCH="$(platform_arch)"

CARGO_ARGS=()
if [[ "${CARGO_LOCKED:-0}" == "1" ]]; then
  CARGO_ARGS+=(--locked)
fi
if [[ "${CARGO_OFFLINE:-0}" == "1" ]]; then
  CARGO_ARGS+=(--offline)
fi

fix_linux_appimage_diricon() {
  local appimage="$1"
  if [[ ! -f "$appimage" ]]; then
    return 0
  fi
  if ! command -v mksquashfs >/dev/null 2>&1; then
    echo "warning: mksquashfs not found; skipping AppImage icon metadata normalization." >&2
    return 0
  fi

  local workdir
  workdir="$(mktemp -d)"
  if ! (cd "$workdir" && APPIMAGE_EXTRACT_AND_RUN=1 "$appimage" --appimage-extract >/dev/null 2>&1); then
    echo "warning: failed to extract AppImage for icon normalization; keeping original bundle." >&2
    rm -rf "$workdir"
    return 0
  fi

  local diricon="$workdir/squashfs-root/.DirIcon"
  local diricon_target=""
  if [[ -L "$diricon" ]]; then
    diricon_target="$(readlink "$diricon" || true)"
  fi
  if [[ -z "$diricon_target" || "$diricon_target" != /* ]]; then
    rm -rf "$workdir"
    return 0
  fi

  local icon_name=""
  if [[ -f "$workdir/squashfs-root/WesoForge.png" ]]; then
    icon_name="WesoForge.png"
  elif [[ -f "$workdir/squashfs-root/bbr-client-gui.png" ]]; then
    icon_name="bbr-client-gui.png"
  else
    echo "warning: AppImage extracted but no icon payload found; skipping icon normalization." >&2
    rm -rf "$workdir"
    return 0
  fi

  ln -snf "$icon_name" "$diricon"

  local offset=""
  offset="$(APPIMAGE_EXTRACT_AND_RUN=1 "$appimage" --appimage-offset 2>/dev/null || true)"
  if [[ ! "$offset" =~ ^[0-9]+$ ]]; then
    echo "warning: failed to read AppImage runtime offset; keeping original bundle." >&2
    rm -rf "$workdir"
    return 0
  fi

  dd if="$appimage" of="$workdir/runtime" bs=1 count="$offset" status=none
  mksquashfs "$workdir/squashfs-root" "$workdir/new.squashfs" -root-owned -noappend -quiet >/dev/null
  cat "$workdir/runtime" "$workdir/new.squashfs" > "$workdir/fixed.AppImage"
  chmod +x "$workdir/fixed.AppImage"
  cp -f "$workdir/fixed.AppImage" "$appimage"
  rm -rf "$workdir"
}

if [[ "$UNAME" == "Linux" ]]; then
  if [[ "${BBR_SKIP_CARGO_BUILD:-0}" != "1" ]]; then
    echo "Building WesoForge GUI AppImage (features: $FEATURES)..." >&2
    (
      cd crates/client-gui
      export NO_STRIP=1
      if [[ "${#CARGO_ARGS[@]}" -gt 0 ]]; then
        cargo tauri build --features "$FEATURES" --bundles appimage -- "${CARGO_ARGS[@]}"
      else
        cargo tauri build --features "$FEATURES" --bundles appimage
      fi
    )
  fi
  APPIMAGE_DIR="$TARGET_DIR/release/bundle/appimage"
  APPIMAGE_SRC="$(ls -1t "$APPIMAGE_DIR"/*.AppImage 2>/dev/null | head -n 1 || true)"
  if [[ -z "$APPIMAGE_SRC" ]]; then
    echo "error: no AppImage found under: $APPIMAGE_DIR" >&2
    exit 1
  fi
  fix_linux_appimage_diricon "$APPIMAGE_SRC"
  APPIMAGE_DST="$DIST_DIR/${OUT_PREFIX}_Linux_${VERSION}_${ARCH}.AppImage"
  install -m 0755 "$APPIMAGE_SRC" "$APPIMAGE_DST"
  echo "Wrote: $APPIMAGE_DST" >&2
elif [[ "$UNAME" == "Darwin" ]]; then
  if [[ "${BBR_SKIP_CARGO_BUILD:-0}" != "1" ]]; then
    echo "Building WesoForge GUI (macOS DMG, features: $FEATURES)..." >&2
    (
      cd crates/client-gui
      if [[ "${#CARGO_ARGS[@]}" -gt 0 ]]; then
        cargo tauri build --features "$FEATURES" --bundles dmg -- "${CARGO_ARGS[@]}"
      else
        cargo tauri build --features "$FEATURES" --bundles dmg
      fi
    )
  fi
  DMG_DIR="$TARGET_DIR/release/bundle/dmg"
  DMG_SRC="$(ls -1t "$DMG_DIR"/*.dmg 2>/dev/null | head -n 1 || true)"
  if [[ -z "$DMG_SRC" ]]; then
    echo "error: no DMG found under: $DMG_DIR" >&2
    exit 1
  fi
  DMG_DST="$DIST_DIR/${OUT_PREFIX}_macOS_${VERSION}_${ARCH}.dmg"
  cp "$DMG_SRC" "$DMG_DST"
  echo "Wrote: $DMG_DST" >&2
else
  echo "error: GUI build is only supported on Linux and macOS (got: $UNAME)" >&2
  exit 1
fi
