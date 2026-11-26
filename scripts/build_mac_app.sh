#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_DEST="/Applications/YouClone.app"
ELECTRON_APP="$ROOT/node_modules/electron/dist/Electron.app"
ICON_ICNS="$ROOT/desktop/assets/YouClone.icns"

if [[ ! -d "$ELECTRON_APP" ]]; then
  echo "Electron.app not found at $ELECTRON_APP" >&2
  exit 1
fi
if [[ ! -f "$ICON_ICNS" ]]; then
  echo "YouClone.icns not found at $ICON_ICNS" >&2
  exit 1
fi

echo "Building standalone macOS bundle -> $APP_DEST"
rm -rf "$APP_DEST"
ditto "$ELECTRON_APP" "$APP_DEST"

# Rename main executable to YouClone
mv "$APP_DEST/Contents/MacOS/Electron" "$APP_DEST/Contents/MacOS/YouClone"

# Install custom macOS .icns icon
rm -f "$APP_DEST/Contents/Resources/electron.icns"
cp "$ICON_ICNS" "$APP_DEST/Contents/Resources/YouClone.icns"

# Patch Info.plist cleanly with PlistBuddy
PLIST="$APP_DEST/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleName YouClone" "$PLIST"
/usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName YouClone" "$PLIST"
/usr/libexec/PlistBuddy -c "Set :CFBundleExecutable YouClone" "$PLIST"
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier com.suman.youclone" "$PLIST"
/usr/libexec/PlistBuddy -c "Set :CFBundleIconFile YouClone.icns" "$PLIST"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion 2.0.0" "$PLIST"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString 2.0.0" "$PLIST"
/usr/libexec/PlistBuddy -c "Delete :NSMicrophoneUsageDescription" "$PLIST" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Add :NSMicrophoneUsageDescription string 'YouClone uses your microphone to record voice samples for Instant and Professional Voice Cloning.'" "$PLIST"

# Remove default Electron demo app.asar and install clean YouClone app bootstrapper
rm -rf "$APP_DEST/Contents/Resources/default_app.asar" "$APP_DEST/Contents/Resources/app"
mkdir -p "$APP_DEST/Contents/Resources/app"

cat > "$APP_DEST/Contents/Resources/app/package.json" <<EOF
{
  "name": "youclone",
  "version": "2.0.0",
  "main": "launcher.cjs"
}
EOF

cat > "$APP_DEST/Contents/Resources/app/launcher.cjs" <<EOF
const path = require("node:path");
const root = "$ROOT";
process.env.YOUCLONE_ROOT = root;
process.env.HF_HOME = path.join(root, ".voxm5", "huggingface");
process.env.HF_HUB_OFFLINE = "1";
process.env.TRANSFORMERS_OFFLINE = "1";
process.env.PATH = path.join(root, ".voxm5", "tools") + ":/opt/homebrew/bin:/usr/local/bin:" + (process.env.PATH || "");
require(path.join(root, "desktop", "main.cjs"));
EOF

# Ad-hoc sign so macOS launches cleanly without broken-signature errors
codesign --force --deep --sign - "$APP_DEST" >/dev/null 2>&1 || true
touch "$APP_DEST"

# Refresh LaunchServices icon & Spotlight registration
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP_DEST" 2>/dev/null || true

echo "Successfully installed $APP_DEST"
ls -lh "$APP_DEST/Contents/MacOS/YouClone" "$APP_DEST/Contents/Resources/YouClone.icns"
