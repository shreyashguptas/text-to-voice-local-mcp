# make install    check dependencies, build, connect Claude Code + Desktop, start
# make update     pull the latest code, then install
# make uninstall  disconnect from Claude Code + Desktop and stop
# make test       run the tests
# make status     is the voice in use, which model is loaded, memory

BIN := target/release/text-to-voice-local-mcp
NAME := text-to-voice

.PHONY: install update uninstall test status

install:
	@sh scripts/install.sh

update:
	@git pull --ff-only
	@sh scripts/install.sh

uninstall:
	-@claude mcp remove --scope user $(NAME) >/dev/null 2>&1 && echo "removed from Claude Code"
	@/usr/bin/python3 -c 'import json,os,sys; p=os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json"); c=json.load(open(p)) if os.path.exists(p) else {}; c.get("mcpServers",{}).pop("$(NAME)",None) is not None and (json.dump(c,open(p,"w"),indent=2), print("removed from Claude Desktop"))'
	-@pkill -f "$(CURDIR)/$(BIN) daemon" && echo "stopped the background copy"
	@echo "The binary is still in $(BIN); delete this folder to remove it completely."

test:
	@PATH="$$(dirname "$$(rustup which cargo 2>/dev/null || command -v cargo)"):$$PATH" cargo test --release

status:
	@$(BIN) status
