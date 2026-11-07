#!/bin/sh
# Check dependencies, build, connect to Claude Code and Claude Desktop, start.
# Run through `make install` (or `make update`, which pulls first).
#
# Anything missing is reported with the exact command that fixes it. In a
# terminal you are asked before that command runs; nothing is installed
# without a yes.
set -eu

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BIN="$ROOT/target/release/text-to-voice-local-mcp"
NAME="text-to-voice"
MODELS="$HOME/Library/Application Support/text-to-voice-local-mcp/models"
KOKORO_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
ok() { printf '  \033[32m✓\033[0m %s\n' "$*"; }
bad() { printf '  \033[31m✗\033[0m %s\n' "$*"; }

# Offer to run a fix. Returns 0 if it ran and worked.
offer() {
    what="$1"
    fix="$2"
    bad "$what is missing. To fix it, run:"
    printf '\n      %s\n\n' "$fix"
    if [ -t 0 ]; then
        printf '    Run it now? [y/N] '
        read -r answer
        case "$answer" in
            y | Y | yes) sh -c "$fix" && return 0 ;;
        esac
    fi
    return 1
}

need() {
    # need "<what>" "<check command>" "<fix command>"
    if sh -c "$2" >/dev/null 2>&1; then
        ok "$1"
    elif offer "$1" "$3" && sh -c "$2" >/dev/null 2>&1; then
        ok "$1 (installed)"
    else
        echo
        bold "Stopped: $1 is needed. Fix it and run make install again."
        exit 1
    fi
}

bold "Checking what is needed"
[ "$(uname -s)" = "Darwin" ] || {
    bad "This tool is for macOS."
    exit 1
}
ok "macOS $(sw_vers -productVersion) on $(uname -m)"
need "Xcode Command Line Tools (compiler)" "xcode-select -p && /usr/bin/clang --version" "xcode-select --install"
need "Homebrew" "command -v brew" '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
need "CMake (builds espeak-ng)" "command -v cmake" "brew install cmake"

# Rust: cargo on PATH, or Homebrew's rustup (which does not put cargo on PATH).
if ! command -v cargo >/dev/null 2>&1; then
    if command -v rustup >/dev/null 2>&1 && rustup which cargo >/dev/null 2>&1; then
        PATH="$(dirname "$(rustup which cargo)"):$PATH"
        export PATH
    fi
fi
need "Rust (cargo)" "command -v cargo" "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && . \"\$HOME/.cargo/env\""
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env" 2>/dev/null || true

echo
bold "Building (the first build takes a few minutes)"
cargo build --release
/usr/bin/strip "$BIN" 2>/dev/null || true
ok "built $BIN ($(du -h "$BIN" | cut -f1))"

echo
bold "Connecting apps"
if command -v claude >/dev/null 2>&1; then
    claude mcp remove --scope user "$NAME" >/dev/null 2>&1 || true
    claude mcp add --scope user "$NAME" -- "$BIN" >/dev/null
    ok "Claude Code: \"$NAME\" (every project)"
else
    bad "Claude Code not found; skipped. Later: claude mcp add --scope user $NAME -- \"$BIN\""
fi
DESKTOP="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
if [ -d "$(dirname "$DESKTOP")" ]; then
    # A running Claude Desktop rewrites this file from memory and drops the
    # new entry, so it has to be closed while the file is edited.
    while pgrep -x Claude >/dev/null 2>&1; do
        if [ -t 0 ]; then
            printf '    Claude Desktop is open and would undo this change. Quit it (Cmd+Q), then press Enter. '
            read -r _
        else
            bad "Claude Desktop is open; quit it and run make install again to connect it."
            break
        fi
    done
    /usr/bin/python3 - "$DESKTOP" "$NAME" "$BIN" <<'EOF'
import json, os, sys
path, name, binary = sys.argv[1:]
config = json.load(open(path)) if os.path.exists(path) else {}
config.setdefault("mcpServers", {})[name] = {"command": binary}
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
os.replace(tmp, path)
EOF
    ok "Claude Desktop: \"$NAME\" (open Claude Desktop again to load it)"
else
    bad "Claude Desktop not found; skipped."
fi

echo
bold "Starting"
# Replaces a running background copy from an older build, then lists models.
if ! "$BIN" models --json | /usr/bin/python3 -c 'import json,sys; m=[x for x in json.load(sys.stdin)["models"] if x["engine"]!="macos"]; sys.exit(0 if m else 1)'; then
    if offer "A neural voice model (only the robotic macOS voices were found)" \
        "mkdir -p \"$MODELS\" && curl -fL --progress-bar -o \"$MODELS/kokoro-v1.0.onnx\" $KOKORO_URL/kokoro-v1.0.onnx && curl -fL --progress-bar -o \"$MODELS/voices-v1.0.bin\" $KOKORO_URL/voices-v1.0.bin"; then
        ok "Kokoro v1.0 downloaded (355 MB) to $MODELS"
    else
        bad "Continuing with the macOS voices only."
    fi
fi
"$BIN" models | grep -vE '^ |^Searched' | sed 's/^/  /'
"$BIN" speak "Text to voice is installed and ready." >/dev/null 2>&1 && ok "spoke a test line" || bad "could not play the test line (is the sound on?)"

echo
bold "Done. Ask any Claude chat to \"say something out loud\"."
