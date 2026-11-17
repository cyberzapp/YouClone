#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$PWD/.voxm5/tools:/opt/homebrew/bin:$PATH"
if [[ "$(uname -s -m)" != "Darwin arm64" ]]; then
  echo "YouClone requires native Apple Silicon macOS." >&2
  exit 1
fi
required=(ffmpeg node)
if [[ "${1:-}" != "--offline" ]]; then
  required+=(uv npm)
fi
missing=()
for app in "${required[@]}"; do
  command -v "$app" >/dev/null || missing+=("$app")
done
if (( ${#missing[@]} )); then
  printf 'Missing tools: %s\nInstall them with Homebrew, then rerun setup.\n' "${missing[*]}" >&2
  exit 1
fi
export HF_HOME="$PWD/.voxm5/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export NEXT_TELEMETRY_DISABLED=1
if [[ "${1:-}" != "--offline" ]]; then
  if command -v pnpm >/dev/null; then
    pnpm install --frozen-lockfile
  elif [[ ! -x node_modules/.bin/next || ! -x node_modules/.bin/tsc ||
          ! -x node_modules/.bin/electron-rebuild || ! -f node_modules/electron/cli.js ]]; then
    npm exec --yes --package=pnpm@11.25.0 -- pnpm install --frozen-lockfile
  else
    echo "Using the JavaScript dependencies already installed in this folder."
  fi
  uv sync --locked --python 3.12
  .venv/bin/python -m backend.download
  npm run rebuild
  npm run build
fi
if [[ ! -x .venv/bin/python || ! -f out/index.html || ! -f node_modules/electron/cli.js ]]; then
  echo "Run online setup before using --offline." >&2
  exit 1
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec node node_modules/electron/cli.js .
