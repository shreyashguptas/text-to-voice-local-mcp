//! Text-to-speech engines.
//!
//! An *engine* knows one family of model files (Kokoro, Piper, the macOS
//! system voices). Discovery turns files on disk into [`ModelInfo`]s, which
//! are cheap metadata. A model is only loaded into memory (`LoadedModel`) when
//! someone asks it to speak, and the registry drops it again after it has been
//! idle for a while.

pub mod kokoro;
pub mod piper;
pub mod system;

use serde::Serialize;
use std::path::PathBuf;

/// Everything a caller can know about a model without loading it.
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct ModelInfo {
    /// Stable id to pass as `model` to the other tools, e.g. `kokoro-v1.0`.
    pub id: String,
    /// Engine family: `kokoro`, `piper` or `macos`.
    pub engine: String,
    /// Human-readable name.
    pub name: String,
    /// What the model is and what it is good at.
    pub description: String,
    /// Rough quality, best first: `neural-high`, `neural`, `system`.
    pub quality: String,
    /// The model file on disk (the voices folder for macOS).
    pub path: String,
    /// Size of the model files on disk, in megabytes.
    pub size_mb: f64,
    /// Rough memory the model takes while loaded, in megabytes.
    pub approx_memory_mb: f64,
    /// Samples per second of the audio it produces.
    pub sample_rate: u32,
    /// Languages it speaks, as BCP 47 codes such as `en-US`.
    pub languages: Vec<String>,
    /// How many voices it has. `list_voices` lists them.
    pub voice_count: usize,
    /// The voice used when the caller does not pick one.
    pub default_voice: String,
    /// Lowest and highest `speed` it accepts (1.0 is normal).
    pub speed_range: [f32; 2],
    /// License of the model weights, when known.
    pub license: String,
    /// True when this model is currently loaded in memory.
    pub loaded: bool,
    /// True when this is the model used when the caller does not pick one.
    pub is_default: bool,
}

/// One voice of one model.
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct VoiceInfo {
    /// Id to pass as `voice`, e.g. `af_heart`.
    pub id: String,
    /// Model this voice belongs to.
    pub model: String,
    /// Human-readable name, e.g. `Heart`.
    pub name: String,
    /// BCP 47 language code, e.g. `en-US`.
    pub language: String,
    /// Language in words, e.g. `English (US)`.
    pub language_name: String,
    /// `female`, `male`, or `unknown`.
    pub gender: String,
    /// Anything worth knowing about the voice (quality grade, accent).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub notes: String,
}

/// What a caller asks for, after defaults have been filled in.
#[derive(Debug, Clone)]
pub struct SynthRequest {
    pub text: String,
    pub voice: String,
    pub speed: f32,
}

/// A model that has been found on disk and can be loaded.
pub trait Model: Send + Sync {
    fn info(&self) -> &ModelInfo;
    fn voices(&self) -> &[VoiceInfo];
    /// Load the weights. Called on first use, on a blocking thread.
    fn load(&self) -> anyhow::Result<Box<dyn LoadedModel>>;
}

/// A model in memory, ready to speak.
pub trait LoadedModel: Send {
    fn sample_rate(&self) -> u32;
    /// Split text into pieces the model can say in one pass. Each piece is
    /// synthesized on its own, so playback can start after the first one.
    fn plan(&mut self, req: &SynthRequest) -> anyhow::Result<Vec<Chunk>>;
    /// Synthesize one planned piece into mono samples in -1..1.
    fn render(&mut self, req: &SynthRequest, chunk: &Chunk) -> anyhow::Result<Vec<f32>>;
}

/// One piece of a planned utterance.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Engine-specific payload: phonemes for neural models, text for `say`.
    pub payload: String,
    /// Seconds of silence to put after this piece.
    pub pause_after: f32,
}

/// Turn a file name like `kokoro-v1.0.int8` into an id like `kokoro-v1.0-int8`.
pub fn slug(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut dash = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '.' {
            out.push(c.to_ascii_lowercase());
            dash = false;
        } else if !dash && !out.is_empty() {
            out.push('-');
            dash = true;
        }
    }
    while out.ends_with('-') || out.ends_with('.') {
        out.pop();
    }
    out
}

pub fn file_size_mb(paths: &[PathBuf]) -> f64 {
    let bytes: u64 = paths
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| if m.is_dir() { 0 } else { m.len() })
        .sum();
    (bytes as f64 / 1_048_576.0 * 10.0).round() / 10.0
}

/// Language names for the codes the engines use.
pub fn language_name(code: &str) -> String {
    let lower = code.to_ascii_lowercase().replace('_', "-");
    let name = match lower.as_str() {
        "en-us" => "English (US)",
        "en-gb" => "English (UK)",
        "en-au" => "English (Australia)",
        "en-in" => "English (India)",
        "en-ie" => "English (Ireland)",
        "en-za" => "English (South Africa)",
        "en-scotland" => "English (Scotland)",
        "es" | "es-es" => "Spanish (Spain)",
        "es-mx" => "Spanish (Mexico)",
        "es-419" => "Spanish (Latin America)",
        "fr" | "fr-fr" => "French",
        "fr-ca" => "French (Canada)",
        "hi" | "hi-in" => "Hindi",
        "it" | "it-it" => "Italian",
        "pt-br" => "Portuguese (Brazil)",
        "pt" | "pt-pt" => "Portuguese (Portugal)",
        "ja" | "ja-jp" => "Japanese",
        "zh" | "zh-cn" | "cmn" => "Chinese (Mandarin)",
        "zh-tw" => "Chinese (Taiwan)",
        "zh-hk" => "Chinese (Cantonese)",
        "de" | "de-de" => "German",
        "nl" | "nl-nl" => "Dutch",
        "ru" | "ru-ru" => "Russian",
        "ko" | "ko-kr" => "Korean",
        "ar" | "ar-001" => "Arabic",
        "sv" | "sv-se" => "Swedish",
        "pl" | "pl-pl" => "Polish",
        "tr" | "tr-tr" => "Turkish",
        _ => "",
    };
    if !name.is_empty() {
        return name.to_string();
    }
    // Fall back to the bare language part.
    let base = lower.split('-').next().unwrap_or("");
    match base {
        "en" => "English",
        "es" => "Spanish",
        "fr" => "French",
        "de" => "German",
        "pt" => "Portuguese",
        "zh" => "Chinese",
        _ => code,
    }
    .to_string()
}

/// Normalise a language code to BCP 47 style: `en_US` -> `en-US`.
pub fn bcp47(code: &str) -> String {
    let mut parts = code.split(['_', '-']);
    let lang = parts.next().unwrap_or("").to_ascii_lowercase();
    let rest: Vec<String> = parts
        .map(|p| if p.len() == 2 { p.to_ascii_uppercase() } else { p.to_string() })
        .collect();
    if rest.is_empty() { lang } else { format!("{lang}-{}", rest.join("-")) }
}

/// Does a voice's language match what the caller asked for? Accepts codes
/// (`en`, `en-GB`, `fr`) and names (`French`, `english`).
pub fn language_matches(voice_lang: &str, voice_lang_name: &str, wanted: &str) -> bool {
    let w = wanted.trim().to_ascii_lowercase().replace('_', "-");
    if w.is_empty() {
        return true;
    }
    let v = voice_lang.to_ascii_lowercase();
    if v == w || v.starts_with(&format!("{w}-")) {
        return true;
    }
    voice_lang_name.to_ascii_lowercase().contains(&w)
}

/// Shared ONNX Runtime setup for the neural engines.
pub mod onnx {
    use anyhow::{Result, anyhow};
    use ort::environment::Environment;
    use ort::session::Session;
    use ort::session::builder::GraphOptimizationLevel;
    use std::path::Path;
    use std::sync::OnceLock;

    fn env() -> Result<&'static Environment> {
        static ENV: OnceLock<Environment> = OnceLock::new();
        if let Some(e) = ENV.get() {
            return Ok(e);
        }
        let e = ort::init()
            .with_name("text-to-voice-local-mcp")
            .build()
            .map_err(|e| anyhow!("could not start ONNX Runtime: {e}"))?;
        Ok(ENV.get_or_init(|| e))
    }

    /// Threads for one inference. `TTV_THREADS` overrides; the default leaves
    /// cores free for whatever else the Mac is doing.
    pub fn threads() -> usize {
        if let Some(n) = std::env::var("TTV_THREADS").ok().and_then(|v| v.parse().ok()) {
            return n;
        }
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        (cores / 2).clamp(1, 8)
    }

    /// Run options that hand the arena's scratch memory back after each
    /// piece, so a long text does not leave the server holding its peak.
    pub fn run_options() -> Result<ort::session::RunOptions> {
        let mut o = ort::session::RunOptions::new().map_err(|e| anyhow!("{e}"))?;
        o.set("memory.enable_memory_arena_shrinkage", "cpu:0").map_err(|e| anyhow!("{e}"))?;
        Ok(o)
    }

    pub fn session(path: &Path) -> Result<Session> {
        let fail = |e: ort::Error<ort::session::builder::SessionBuilder>| anyhow!("{e}");
        // The ort crate defaults to letting ONNX Runtime pick devices for
        // "max efficiency", which on a Mac also hands the graph to Core ML:
        // a second copy of the weights (820 MB loaded instead of 442 MB) and
        // 2 s of Core ML compiling on every load. CPU only is as fast.
        // Prepacking off saves another ~40 MB.
        Session::builder(env()?)
            .map_err(|e| anyhow!("{e}"))?
            .with_auto_device(ort::session::builder::AutoDevicePolicy::PreferCPU)
            .map_err(fail)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(fail)?
            .with_prepacking(false)
            .map_err(fail)?
            .with_intra_threads(threads())
            .map_err(fail)?
            .with_inter_threads(1)
            .map_err(fail)?
            .commit_from_file(path)
            .map_err(|e| anyhow!("could not load {}: {e}", path.display()))
    }
}
