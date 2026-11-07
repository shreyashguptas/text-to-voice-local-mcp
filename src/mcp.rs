//! The MCP tools. They run inside the shared background copy (the daemon);
//! each app talks to it through a small relay (see `daemon.rs`).

use crate::audio::{self, Format, PlayOutcome};
use crate::speech::{BusyInfo, Voice};
use crate::engine::{ModelInfo, VoiceInfo, language_matches};
use crate::registry::Registry;
use crate::speech;
use base64::Engine as _;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerConfig};
use rmcp::{Json, Peer, RoleServer, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

const INSTRUCTIONS: &str = "\
Local text-to-speech on this Mac. Everything runs on the machine; nothing is sent anywhere.

Typical use:
- `speak` says text out loud through the Mac's speakers. It returns as soon as speech starts \
  (pass `wait: true` to return when it ends).
- `synthesize` makes an audio file (wav, m4a, aiff, flac, caf) and returns its path, and the audio \
  itself if `return_audio` is true.
- `list_models` and `list_voices` show what is installed: models, voices, languages, speeds.
- `stop` cuts off whatever is being said.
- `status` shows whether the voice is in use, and by which app.

All of `model`, `voice`, `language` and `speed` are optional. With none, the best installed model \
and its default voice are used. Give `language` (\"fr\", \"British English\", \"ja\") to get the best \
voice for it, or `voice` (\"af_heart\", \"bm_george\") to pick one. Speed 1.0 is normal; 0.5 to 2.0.

One model is shared by every app on the Mac, and it does one thing at a time. If another chat is \
using it, `speak` and `synthesize` return at once with a \"Busy\" message naming that app; try again \
in a moment rather than retrying in a loop. Write text the way it should be said: spell out symbols \
and code, and keep sentences short for the most natural reading.";

/// State shared by every connected app.
pub struct Shared {
    pub registry: Arc<Registry>,
    pub voice: Arc<Voice>,
    pub clients: AtomicUsize,
}

impl Shared {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { registry: Arc::new(Registry::new()), voice: Arc::default(), clients: AtomicUsize::new(0) })
    }
}

fn client_name(peer: &Peer<RoleServer>) -> String {
    peer.peer_info()
        .map(|i| i.client_info.name.clone())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "an unnamed app".into())
}

#[derive(Clone)]
pub struct TtsServer {
    shared: Arc<Shared>,
    tool_router: ToolRouter<Self>,
}

// ---- inputs ----------------------------------------------------------------------

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ListModelsArgs {
    /// Search the disk again for models added since the last look. Default false.
    #[serde(default)]
    pub refresh: bool,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ListVoicesArgs {
    /// Only voices of this model id (from list_models).
    #[serde(default)]
    pub model: Option<String>,
    /// Only voices for this language: a code (`en`, `en-GB`, `fr`, `ja`) or a name (`French`).
    #[serde(default)]
    pub language: Option<String>,
    /// Only `female` or `male` voices.
    #[serde(default)]
    pub gender: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SpeakArgs {
    /// What to say. Plain text, any length; long text is read sentence by sentence.
    pub text: String,
    /// Voice id or name from list_voices, e.g. `af_heart`. Default: the model's default voice.
    #[serde(default)]
    pub voice: Option<String>,
    /// Model id from list_models. Default: the best installed model.
    #[serde(default)]
    pub model: Option<String>,
    /// Language to speak in when no voice is given: `en-US`, `en-GB`, `fr`, `es`, `ja`, `hi`, ...
    #[serde(default)]
    pub language: Option<String>,
    /// Speaking speed. 1.0 is normal, 0.5 is half speed, 2.0 is double. Default 1.0.
    #[serde(default)]
    pub speed: Option<f32>,
    /// Loudness from 0.0 to 2.0. Default 1.0.
    #[serde(default)]
    pub volume: Option<f32>,
    /// Return only after speech has finished. Default false: return as soon as it starts.
    #[serde(default)]
    pub wait: bool,
    /// If another chat is using the voice, cut it off and speak instead of returning "Busy".
    /// Default false.
    #[serde(default)]
    pub interrupt: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SynthesizeArgs {
    /// What to say.
    pub text: String,
    /// Voice id or name from list_voices. Default: the model's default voice.
    #[serde(default)]
    pub voice: Option<String>,
    /// Model id from list_models. Default: the best installed model.
    #[serde(default)]
    pub model: Option<String>,
    /// Language to speak in when no voice is given.
    #[serde(default)]
    pub language: Option<String>,
    /// Speaking speed, 0.5 to 2.0. Default 1.0.
    #[serde(default)]
    pub speed: Option<f32>,
    /// File format. Default: from `output_path`'s extension, else `wav`.
    #[serde(default)]
    pub format: Option<Format>,
    /// Where to save the file: an absolute path (`~` is allowed), or a folder to save into.
    /// Default: a new file in /tmp/text-to-voice-local-mcp-<uid>/, which macOS clears on restart.
    #[serde(default)]
    pub output_path: Option<String>,
    /// Also return the audio itself, base64-encoded, in the result. Default false.
    /// About 64 KB per second of speech as WAV, 6 KB as m4a.
    #[serde(default)]
    pub return_audio: bool,
}

// ---- outputs ---------------------------------------------------------------------

#[derive(Debug, Serialize, JsonSchema)]
pub struct ModelsOut {
    /// Installed models, best first. The first is used by default.
    pub models: Vec<ModelInfo>,
    /// Folders searched for models, and how many levels deep.
    pub searched: Vec<crate::discovery::Root>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct VoicesOut {
    pub count: usize,
    pub voices: Vec<VoiceInfo>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct SpeakOut {
    /// `playing` (returned early), `finished` or `stopped`.
    pub status: String,
    pub model: String,
    pub voice: String,
    /// Length of the speech in seconds. Present when `wait` was true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct SynthesizeOut {
    /// Absolute path of the audio file.
    pub path: String,
    pub format: Format,
    pub mime_type: String,
    pub duration_seconds: f64,
    pub sample_rate: u32,
    pub bytes: u64,
    pub model: String,
    pub voice: String,
    /// Seconds spent making the audio (after the model was loaded).
    pub synth_seconds: f64,
    /// Seconds spent loading the model (0 when it was already loaded).
    pub load_seconds: f64,
    /// How many pieces the text was read in (long text is split at sentences).
    pub pieces: usize,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct StatusOut {
    /// Who is using the voice right now; empty when it is free.
    pub busy: Option<BusyInfo>,
    /// Apps connected to the shared server right now.
    pub connected_apps: usize,
    /// Model currently in memory, if any.
    pub loaded_model: Option<String>,
    /// Memory this server process is using, in megabytes.
    pub memory_mb: f64,
    /// Seconds a model stays in memory after its last use.
    pub idle_unload_seconds: u64,
    pub version: String,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct StopOut {
    /// What was stopped; empty when nothing was playing.
    pub stopped: Option<BusyInfo>,
}

// ---- tools -----------------------------------------------------------------------

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

#[tool_router]
impl TtsServer {
    pub fn new(shared: Arc<Shared>) -> Self {
        Self { shared, tool_router: Self::tool_router() }
    }

    /// List the text-to-speech models installed on this Mac, best first, with what each one is,
    /// its languages, voice count, default voice, speed range, size, memory use and whether it is
    /// loaded. The first model is the default. Also lists the folders that were searched.
    #[tool(annotations(title = "List voice models", read_only_hint = true, open_world_hint = false))]
    async fn list_models(&self, Parameters(args): Parameters<ListModelsArgs>) -> Result<Json<ModelsOut>, String> {
        let reg = self.shared.registry.clone();
        let models = tokio::task::spawn_blocking(move || reg.model_infos(args.refresh)).await.map_err(err)?;
        Ok(Json(ModelsOut { models, searched: crate::discovery::roots() }))
    }

    /// List the voices you can pick, optionally only for one model, language or gender. Each voice
    /// has an id to pass as `voice`, its language and gender, and notes on quality.
    #[tool(annotations(title = "List voices", read_only_hint = true, open_world_hint = false))]
    async fn list_voices(&self, Parameters(args): Parameters<ListVoicesArgs>) -> Result<Json<VoicesOut>, String> {
        let reg = self.shared.registry.clone();
        let voices = tokio::task::spawn_blocking(move || voices_matching(&reg, &args)).await.map_err(err)??;
        Ok(Json(VoicesOut { count: voices.len(), voices }))
    }

    /// Say text out loud on this Mac's speakers with a local neural voice. Returns as soon as
    /// speech starts, unless `wait` is true. If another chat is using the voice, returns at once
    /// with a "Busy" message saying which app.
    #[tool(annotations(title = "Speak out loud", read_only_hint = false, destructive_hint = false, idempotent_hint = false, open_world_hint = false))]
    async fn speak(&self, peer: Peer<RoleServer>, Parameters(args): Parameters<SpeakArgs>) -> Result<Json<SpeakOut>, String> {
        let shared = self.shared.clone();
        let client = client_name(&peer);
        tokio::task::spawn_blocking(move || -> Result<SpeakOut, String> {
            let reg = shared.registry.clone();
            let choice =
                reg.choose(args.model.as_deref(), args.voice.as_deref(), args.language.as_deref()).map_err(err)?;
            let guard = match shared.voice.try_take(&client, "speaking", &args.text) {
                Ok(g) => g,
                Err(_) if args.interrupt => {
                    shared.voice.stop();
                    take_after_stop(&shared.voice, &client, &args.text)?
                }
                Err(busy) => return Err(busy.message()),
            };
            let (model, voice) = (choice.model.info().id.clone(), choice.voice.id.clone());
            let speaking =
                speech::start(reg, choice, args.text, args.speed.unwrap_or(1.0), args.volume.unwrap_or(1.0), guard);
            speaking.started().map_err(|e| format!("{e:#}"))?;
            if !args.wait {
                return Ok(SpeakOut { status: "playing".into(), model, voice, duration_seconds: None });
            }
            let done = speaking.finish().map_err(|e| format!("{e:#}"))?;
            Ok(SpeakOut {
                status: match done.outcome {
                    PlayOutcome::Finished => "finished".into(),
                    PlayOutcome::Stopped => "stopped".into(),
                },
                model,
                voice,
                duration_seconds: Some((done.stats.duration() * 100.0).round() / 100.0),
            })
        })
        .await
        .map_err(err)?
        .map(Json)
    }

    /// Turn text into an audio file with a local neural voice, without playing it. Returns the
    /// file's path, length and format, and the audio itself when `return_audio` is true.
    #[tool(annotations(title = "Make an audio file", read_only_hint = false, destructive_hint = false, idempotent_hint = true, open_world_hint = false))]
    async fn synthesize(&self, peer: Peer<RoleServer>, Parameters(args): Parameters<SynthesizeArgs>) -> Result<CallToolResult, String> {
        let shared = self.shared.clone();
        let client = client_name(&peer);
        let return_audio = args.return_audio;
        let out = tokio::task::spawn_blocking(move || {
            let _guard = shared.voice.try_take(&client, "making a file of", &args.text).map_err(|b| b.message())?;
            synthesize_to_file(&shared.registry, args)
        })
        .await
        .map_err(err)??;
        let mut content = vec![ContentBlock::json(&out).map_err(|e| e.message.to_string())?];
        if return_audio {
            let bytes = std::fs::read(&out.path).map_err(err)?;
            let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
            content.push(ContentBlock::audio(b64, out.mime_type.clone()));
        }
        let mut result = CallToolResult::success(content);
        result.structured_content = serde_json::to_value(&out).ok();
        Ok(result)
    }

    /// Stop speaking right away, whichever app started it.
    #[tool(annotations(title = "Stop speaking", read_only_hint = false, destructive_hint = false, idempotent_hint = true, open_world_hint = false))]
    async fn stop(&self) -> Result<Json<StopOut>, String> {
        Ok(Json(StopOut { stopped: self.shared.voice.stop() }))
    }

    /// Show whether the voice is in use and by which app, which model is in memory, how much
    /// memory the shared server uses, and how many apps are connected.
    #[tool(annotations(title = "Server status", read_only_hint = true, open_world_hint = false))]
    async fn status(&self) -> Result<Json<StatusOut>, String> {
        Ok(Json(StatusOut {
            busy: self.shared.voice.busy(),
            connected_apps: self.shared.clients.load(Ordering::SeqCst),
            loaded_model: self.shared.registry.loaded_id(),
            memory_mb: crate::memory_mb(),
            idle_unload_seconds: std::env::var("TTV_IDLE_UNLOAD_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(300),
            version: env!("CARGO_PKG_VERSION").into(),
        }))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for TtsServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new("text-to-voice-local-mcp", env!("CARGO_PKG_VERSION"))
                    .with_title("Local text to voice"),
            )
            .with_instructions(INSTRUCTIONS)
    }
}

// ---- shared with the CLI -----------------------------------------------------------

pub fn voices_matching(reg: &Registry, args: &ListVoicesArgs) -> Result<Vec<VoiceInfo>, String> {
    let models = reg.models(false);
    let wanted_model = args.model.as_deref().map(str::trim).filter(|s| !s.is_empty());
    if let Some(m) = wanted_model {
        if !models.iter().any(|x| x.info().id.eq_ignore_ascii_case(m) || x.info().engine == m) {
            let ids: Vec<String> = models.iter().map(|x| x.info().id.clone()).collect();
            return Err(format!("no model `{m}`. Available: {}", ids.join(", ")));
        }
    }
    Ok(models
        .iter()
        .filter(|m| wanted_model.is_none_or(|w| m.info().id.eq_ignore_ascii_case(w) || m.info().engine == w))
        .flat_map(|m| m.voices().iter().cloned())
        .filter(|v| args.language.as_deref().is_none_or(|l| language_matches(&v.language, &v.language_name, l)))
        .filter(|v| args.gender.as_deref().is_none_or(|g| v.gender.eq_ignore_ascii_case(g.trim())))
        .collect())
}

pub fn synthesize_to_file(reg: &Registry, args: SynthesizeArgs) -> Result<SynthesizeOut, String> {
    let choice = reg.choose(args.model.as_deref(), args.voice.as_deref(), args.language.as_deref()).map_err(err)?;
    let requested = args.output_path.as_deref().map(str::trim).filter(|s| !s.is_empty()).map(expand_home);
    let format = args
        .format
        .or_else(|| requested.as_deref().and_then(Format::from_path))
        .unwrap_or(Format::Wav);
    let path = match requested {
        Some(p) if p.is_dir() => p.join(default_name(format)),
        Some(p) if p.is_absolute() => {
            if Format::from_path(&p).is_none() { p.with_extension(format.extension()) } else { p }
        }
        Some(p) => return Err(format!("output_path must be absolute, got `{}`", p.display())),
        None => audio::scratch_dir().map_err(err)?.join(default_name(format)),
    };
    let mut samples = Vec::new();
    let stats = reg
        .synthesize(&choice, &args.text, args.speed.unwrap_or(1.0), |piece| {
            samples.extend_from_slice(&piece);
            true
        })
        .map_err(|e| format!("{e:#}"))?;
    let bytes = audio::save(&samples, stats.sample_rate, format, &path).map_err(|e| format!("{e:#}"))?;
    Ok(SynthesizeOut {
        path: path.display().to_string(),
        format,
        mime_type: format.mime().into(),
        duration_seconds: (stats.duration() * 100.0).round() / 100.0,
        sample_rate: stats.sample_rate,
        bytes,
        model: stats.model,
        voice: stats.voice,
        synth_seconds: (stats.synth_seconds * 1000.0).round() / 1000.0,
        load_seconds: (stats.load_seconds * 1000.0).round() / 1000.0,
        pieces: stats.chunks,
    })
}

fn default_name(format: Format) -> String {
    format!("speech-{}.{}", audio::unique_stamp(), format.extension())
}

fn expand_home(p: &str) -> PathBuf {
    match (p.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ if p == "~" => std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default(),
        _ => PathBuf::from(p),
    }
}

/// `interrupt`: wait briefly for the stopped speech to let go of the voice.
fn take_after_stop(voice: &Arc<Voice>, client: &str, text: &str) -> Result<speech::Guard, String> {
    for _ in 0..60 {
        if let Ok(g) = voice.try_take(client, "speaking", text) {
            return Ok(g);
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    Err(voice.busy().map(|b| b.message()).unwrap_or_else(|| "the voice did not stop in time".into()))
}
