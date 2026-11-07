#!/bin/sh
# Build text-to-voice-local-mcp and connect it to Claude Code and Claude Desktop.
#
#   ./install.sh            build, then register with both apps
#   ./install.sh --build    build only
#
# Nothing is installed globally: the apps run the binary straight from this
# folder's target/release/. Rebuilding updates them (the shared background
# copy notices a new build and restarts itself).
set -eu

cd "$(dirname "$0")"
ROOT="$(pwd)"
BIN="$ROOT/target/release/text-to-voice-local-mcp"
NAME="text-to-voice"

# Homebrew's rustup does not put cargo on PATH; go through rustup then.
if command -v cargo >/dev/null 2>&1; then
    CARGO="cargo"
elif command -v rustup >/dev/null 2>&1; then
    PATH="$(dirname "$(rustup which cargo)"):$PATH"
    export PATH
    CARGO="cargo"
else
    echo "Rust is needed: https://rustup.rs" >&2
    exit 1
fi
$CARGO build --release
/usr/bin/strip "$BIN" 2>/dev/null || true
echo "built $BIN ($(du -h "$BIN" | cut -f1))"

[ "${1:-}" = "--build" ] && exit 0

# Claude Code (user scope: every project).
if command -v claude >/dev/null 2>&1; then
    claude mcp remove --scope user "$NAME" >/dev/null 2>&1 || true
    claude mcp add --scope user "$NAME" -- "$BIN"
    echo "added to Claude Code as \"$NAME\""
fi

# Claude Desktop: add one entry to mcpServers, leaving everything else as is.
DESKTOP="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
if [ -d "$(dirname "$DESKTOP")" ]; then
    /usr/bin/python3 - "$DESKTOP" "$NAME" "$BIN" <<'EOF'
import json, os, sys
path, name, binary = sys.argv[1:]
config = json.load(open(path)) if os.path.exists(path) else {}
config.setdefault("mcpServers", {})[name] = {"command": binary, "args": []}
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
os.replace(tmp, path)
EOF
    echo "added to Claude Desktop as \"$NAME\" (restart Claude Desktop to load it)"
fi

"$BIN" models | head -1
