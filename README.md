# text-to-voice-local-mcp

Natural-sounding text to speech for every app on your Mac, using the voice
models already on it. One small Rust binary. Nothing is downloaded, nothing
leaves the machine.

- **Finds your models.** Kokoro, Piper voices, and the macOS system voices,
  wherever they already live (Hugging Face cache, project folders, Downloads).
- **Speaks, or saves a file.** Plays through the Mac's speakers, or writes
  WAV, M4A, AIFF, FLAC or CAF.
- **One copy for the whole Mac.** Every app shares a single background copy
  of the model. While it is speaking for one chat, other chats get an instant
  "busy" answer naming the app that has it.
- **Light.** 11 MB while idle. The model loads on first use (0.2 s) and the
  background copy exits after 5 minutes of quiet, which hands all of its
  memory back.

## Install

Needs Rust ([rustup.rs](https://rustup.rs)).

```sh
git clone https://github.com/shreyashguptas/text-to-voice-local-mcp.git
cd text-to-voice-local-mcp
./install.sh
```

`install.sh` builds the binary and registers it as an MCP server called
`text-to-voice` with Claude Code (user scope) and Claude Desktop. Restart
Claude Desktop afterwards. For any other MCP app, point it at the binary with
no arguments:

```json
{ "mcpServers": { "text-to-voice": { "command": "/path/to/target/release/text-to-voice-local-mcp" } } }
```

## How it works

```
Claude Code     ──▶ relay ──┐
Claude Desktop  ──▶ relay ──┼──▶  one shared background copy
any MCP app     ──▶ relay ──┤      (the model, the speaker)
terminal CLI    ────────────┘      unix socket in /tmp
```

Each app starts the binary as its MCP server. That process is only a relay: it
passes messages to one shared background copy over a Unix socket that only
your user can open. The first relay to need it starts it. When it has been idle
it exits, and the next request starts it again; the relay replays the app's
MCP handshake, so the app never notices.

Text becomes phonemes first. English goes through
[misaki](https://github.com/hexgrad/misaki) (ported to Rust), the converter
Kokoro was trained with. Every other language, and unknown English words, go
through [espeak-ng](https://github.com/espeak-ng/espeak-ng), compiled into the
binary with its data. The model runs on ONNX Runtime on the CPU, a sentence at
a time, so playback starts after the first sentence instead of after the whole
text.

## MCP tools

| Tool | What it does |
|---|---|
| `speak` | Says text out loud. Returns when speech starts (or ends, with `wait: true`). |
| `synthesize` | Makes an audio file and returns its path; can also return the audio itself (`return_audio`). |
| `list_models` | Installed models, best first: what each is, languages, voice count, default voice, speed range, size, memory, license. |
| `list_voices` | Voices, filterable by `model`, `language` (`fr`, `en-GB`, `French`) or `gender`. |
| `stop` | Cuts off whatever is being said, whichever app started it. |
| `status` | Who is using the voice, which model is loaded, memory use, connected apps. |

`model`, `voice`, `language` and `speed` are optional everywhere. With none,
you get the best installed model and its default voice. `speed` runs from 0.5
to 2.0. `speak` takes `interrupt: true` to cut off another chat instead of
getting "busy".

## Command line

The same things from a terminal, through the same shared copy. The binary is
`target/release/text-to-voice-local-mcp`; put it on your `PATH` or alias it.

```sh
text-to-voice-local-mcp models                       # what is installed, and where it looked
text-to-voice-local-mcp voices -l fr                 # French voices
text-to-voice-local-mcp speak "Hello there"          # say it
text-to-voice-local-mcp speak -v bm_george -s 1.2 "Cheerio"
echo "Long text" | text-to-voice-local-mcp speak     # from standard input
text-to-voice-local-mcp save "Hi" -o hi.m4a          # format from the extension
text-to-voice-local-mcp stop
text-to-voice-local-mcp status
```

## Models it understands

| Engine | Files | Notes |
|---|---|---|
| **Kokoro** 82M | `kokoro-v1.0.onnx` (or `.fp16` / `.int8`) + `voices-v1.0.bin`, from [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx/releases); or Hugging Face `onnx-community/Kokoro-82M-v1.0-ONNX` (`onnx/model*.onnx` + `voices/`) | 54 voices in 9 languages: US and UK English, Spanish, French, Hindi, Italian, Japanese, Brazilian Portuguese, Mandarin. Apache-2.0. |
| **Piper** | `voice.onnx` + `voice.onnx.json`, from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) | Small and fast, dozens of languages. |
| **macOS** | built in | Used only when you ask for it or no neural model speaks the language. Premium voices can be added in System Settings → Accessibility → Spoken Content. |

It looks in `~/.cache/huggingface/hub`, `~/.cache`, `~/.local/share`,
`~/Library/Application Support`, `~/models`, `~/Developer`, `~/Documents` and
`~/Downloads`, and in
`~/Library/Application Support/text-to-voice-local-mcp/models` if you make
it. Add more folders with `TTV_MODEL_PATHS`. New models show up on the next
`list_models` with `refresh: true` (or `models` in the terminal).

Adding a better model type later means adding one file in `src/engine/`.

## Settings

Environment variables, set in the app's MCP config or your shell:

| Variable | Default | Meaning |
|---|---|---|
| `TTV_DEFAULT_MODEL` | best found | Model id to use when none is given. |
| `TTV_DEFAULT_VOICE` | model's default | Voice id to use when none is given. |
| `TTV_MODEL_PATHS` | | Extra folders to search, `:`-separated. |
| `TTV_IDLE_UNLOAD_SECS` | `300` | Quiet seconds before the background copy exits and frees its memory. |
| `TTV_DAEMON_IDLE_SECS` | `600` | Seconds with no app connected before it exits (when no model was loaded). |
| `TTV_THREADS` | half the cores, max 8 | CPU threads per inference. |

Settings are read by the background copy when it starts, so they come from
whichever app starts it first.

## Measured on an M4 Max (Kokoro v1.0, full precision)

| | |
|---|---|
| Idle background copy | 11 MB |
| Each app's relay | ~3 MB |
| Model loaded | ~540 MB |
| While reading long text | up to ~1.26 GB |
| Load time | 0.2 s |
| Speed | 26 s of speech made in ~3 s (about 8× faster than real time) |
| Time to first sound | ~0.5 s |

Two settings matter most. The `ort` crate by default lets ONNX Runtime pick
hardware "for efficiency", which on a Mac also hands the model to Core ML. That
keeps a second copy of the weights and spends 2 s compiling on every load, so
this tool forces the CPU. Long text is also read in pieces of at most 250
phonemes, cut at sentence ends, because the model's scratch memory grows with
piece length.

## Files it writes

Everything goes in `/tmp/text-to-voice-local-mcp-<uid>/` (mode 700): the
socket, a log, the unpacked espeak-ng data (8.7 MB), and audio files saved
without an `output_path`. macOS clears `/tmp` on restart.

## License

GPL-3.0-or-later, because espeak-ng (GPL-3.0) is compiled in. Kokoro's weights
are Apache-2.0; misaki-rs, espeak-rs and the Kokoro vocabulary (from
kokoro-onnx) are MIT; ort, rodio and rmcp are MIT/Apache-2.0.
