#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec cargo run -p bbr-client -- --backend-url "http://127.0.0.1:8080" "$@"

