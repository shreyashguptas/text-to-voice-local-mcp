//! Keeps the list of models, picks the right model and voice for a request,
//! and holds at most one model in memory at a time.
//!
//! Memory policy: nothing is loaded at startup. The first request loads its
//! model; asking for a different model swaps it; and once a model has not
//! been used for `TTV_IDLE_UNLOAD_SECS` (default 300) the daemon exits, which
//! is the only way to give all of ONNX Runtime's memory back to macOS.

use crate::engine::{LoadedModel, Model, ModelInfo, SynthRequest, VoiceInfo, language_matches};
use anyhow::{Result, anyhow, bail};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

pub struct Registry {
    models: RwLock<Option<Vec<Arc<dyn Model>>>>,
    loaded: Mutex<Option<Loaded>>,
    /// Id of the loaded model, readable while a synthesis holds `loaded`.
    loaded_name: RwLock<Option<String>>,
    idle_unload: Duration,
}

struct Loaded {
    id: String,
    model: Box<dyn LoadedModel>,
    last_used: Instant,
}

/// The model and voice a request resolved to.
#[derive(Clone)]
pub struct Choice {
    pub model: Arc<dyn Model>,
    pub voice: VoiceInfo,
}

/// Result of a finished synthesis.
#[derive(Debug, Clone)]
pub struct SynthStats {
    pub model: String,
    pub voice: String,
    pub sample_rate: u32,
    pub samples: usize,
    pub chunks: usize,
    pub load_seconds: f64,
    pub synth_seconds: f64,
}

impl SynthStats {
    pub fn duration(&self) -> f64 {
        self.samples as f64 / self.sample_rate as f64
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    pub fn new() -> Self {
        let secs = std::env::var("TTV_IDLE_UNLOAD_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
        Self { models: RwLock::new(None), loaded: Mutex::new(None), loaded_name: RwLock::new(None), idle_unload: Duration::from_secs(secs) }
    }

    /// All models, scanning the disk on first use (or when `refresh`).
    pub fn models(&self, refresh: bool) -> Vec<Arc<dyn Model>> {
        if !refresh {
            if let Some(m) = self.models.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
                return m.clone();
            }
        }
        let found = crate::discovery::scan();
        *self.models.write().unwrap_or_else(|e| e.into_inner()) = Some(found.clone());
        found
    }

    pub fn default_model(&self) -> Option<Arc<dyn Model>> {
        let models = self.models(false);
        if let Ok(id) = std::env::var("TTV_DEFAULT_MODEL") {
            if let Some(m) = models.iter().find(|m| m.info().id.eq_ignore_ascii_case(&id)) {
                return Some(m.clone());
            }
        }
        models.first().cloned()
    }

    /// Model metadata with `loaded` / `is_default` filled in.
    pub fn model_infos(&self, refresh: bool) -> Vec<ModelInfo> {
        let models = self.models(refresh);
        let default = self.default_model().map(|m| m.info().id.clone());
        let loaded = self.loaded_id();
        models
            .iter()
            .map(|m| {
                let mut i = m.info().clone();
                i.loaded = loaded.as_deref() == Some(i.id.as_str());
                i.is_default = default.as_deref() == Some(i.id.as_str());
                i
            })
            .collect()
    }

    pub fn loaded_id(&self) -> Option<String> {
        self.loaded_name.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn find_model(&self, wanted: &str) -> Result<Arc<dyn Model>> {
        let lookup = |models: &[Arc<dyn Model>]| -> Option<Arc<dyn Model>> {
            let w = wanted.to_ascii_lowercase();
            models
                .iter()
                .find(|m| m.info().id == w)
                .or_else(|| models.iter().find(|m| m.info().id.starts_with(&w)))
                .or_else(|| models.iter().find(|m| m.info().engine == w))
                .cloned()
        };
        if let Some(m) = lookup(&self.models(false)) {
            return Ok(m);
        }
        // It may have been added since the last scan.
        let models = self.models(true);
        lookup(&models).ok_or_else(|| {
            let ids: Vec<String> = models.iter().map(|m| m.info().id.clone()).collect();
            anyhow!("no model `{wanted}`. Available: {}", ids.join(", "))
        })
    }

    /// Pick the model and voice for a request. Every argument is optional:
    ///
    /// * `voice` alone searches every model, best model first.
    /// * `language` alone picks the best voice for that language.
    /// * nothing picks the default model and its default voice.
    pub fn choose(&self, model: Option<&str>, voice: Option<&str>, language: Option<&str>) -> Result<Choice> {
        let candidates: Vec<Arc<dyn Model>> = match model.filter(|s| !s.trim().is_empty()) {
            Some(m) => vec![self.find_model(m.trim())?],
            None => {
                let mut all = self.models(false);
                if let Some(d) = self.default_model() {
                    all.retain(|m| m.info().id != d.info().id);
                    all.insert(0, d);
                }
                all
            }
        };
        if candidates.is_empty() {
            bail!("no voice models found on this Mac. Run `text-to-voice-local-mcp models` to see where the tool looks.");
        }
        let env_voice = std::env::var("TTV_DEFAULT_VOICE").ok();
        let voice = voice.filter(|s| !s.trim().is_empty()).map(str::trim);
        let language = language.filter(|s| !s.trim().is_empty()).map(str::trim);

        if let Some(v) = voice {
            let matches = |vi: &VoiceInfo| vi.id.eq_ignore_ascii_case(v) || vi.name.eq_ignore_ascii_case(v);
            for m in &candidates {
                if let Some(vi) = m.voices().iter().find(|vi| matches(vi)) {
                    return Ok(Choice { model: m.clone(), voice: vi.clone() });
                }
            }
            // Partial match ("heart" -> af_heart, "Ava" -> "Ava (Premium)").
            let lower = v.to_ascii_lowercase();
            for m in &candidates {
                if let Some(vi) = m.voices().iter().find(|vi| {
                    vi.id.to_ascii_lowercase().contains(&lower) || vi.name.to_ascii_lowercase().contains(&lower)
                }) {
                    return Ok(Choice { model: m.clone(), voice: vi.clone() });
                }
            }
            bail!("no voice `{v}`. Call list_voices to see the voices you have.");
        }

        if let Some(lang) = language {
            for m in &candidates {
                let voices = m.voices();
                let default = &m.info().default_voice;
                let pick = voices
                    .iter()
                    .find(|vi| &vi.id == default && language_matches(&vi.language, &vi.language_name, lang))
                    .or_else(|| voices.iter().find(|vi| language_matches(&vi.language, &vi.language_name, lang)));
                if let Some(vi) = pick {
                    return Ok(Choice { model: m.clone(), voice: vi.clone() });
                }
            }
            bail!("no installed voice speaks `{lang}`. Call list_models to see the languages you have.");
        }

        let m = &candidates[0];
        let voice_id = env_voice
            .filter(|v| m.voices().iter().any(|vi| &vi.id == v))
            .unwrap_or_else(|| m.info().default_voice.clone());
        let vi = m
            .voices()
            .iter()
            .find(|vi| vi.id == voice_id)
            .or(m.voices().first())
            .cloned()
            .ok_or_else(|| anyhow!("model `{}` has no voices", m.info().id))?;
        Ok(Choice { model: m.clone(), voice: vi })
    }

    /// Synthesize `text`, handing each piece of audio to `sink` as soon as it
    /// is ready. `sink` returns false to stop early. Blocking.
    pub fn synthesize(
        &self,
        choice: &Choice,
        text: &str,
        speed: f32,
        mut sink: impl FnMut(Vec<f32>) -> bool,
    ) -> Result<SynthStats> {
        let info = choice.model.info();
        let [lo, hi] = info.speed_range;
        if !(lo..=hi).contains(&speed) {
            bail!("speed must be between {lo} and {hi}, got {speed}");
        }
        if text.trim().is_empty() {
            bail!("there is no text to speak");
        }
        let req = SynthRequest { text: text.to_string(), voice: choice.voice.id.clone(), speed };

        let mut guard = self.loaded.lock().unwrap_or_else(|e| e.into_inner());
        let t0 = Instant::now();
        if guard.as_ref().map(|l| l.id != info.id).unwrap_or(true) {
            // Free the old model before loading the new one.
            *guard = None;
            *self.loaded_name.write().unwrap_or_else(|e| e.into_inner()) = None;
            let model = choice.model.load()?;
            *guard = Some(Loaded { id: info.id.clone(), model, last_used: Instant::now() });
            *self.loaded_name.write().unwrap_or_else(|e| e.into_inner()) = Some(info.id.clone());
        }
        let load_seconds = t0.elapsed().as_secs_f64();
        let loaded = guard.as_mut().expect("just loaded");
        loaded.last_used = Instant::now();

        let t1 = Instant::now();
        let chunks = loaded.model.plan(&req)?;
        if chunks.is_empty() {
            bail!("the text has nothing the voice can say");
        }
        let sample_rate = loaded.model.sample_rate();
        let mut samples = 0;
        for chunk in &chunks {
            let audio = loaded.model.render(&req, chunk)?;
            samples += audio.len();
            if !sink(audio) {
                break;
            }
        }
        loaded.last_used = Instant::now();
        Ok(SynthStats {
            model: info.id.clone(),
            voice: choice.voice.id.clone(),
            sample_rate,
            samples,
            chunks: chunks.len(),
            load_seconds,
            synth_seconds: t1.elapsed().as_secs_f64(),
        })
    }

    /// Drop the loaded model if it has been idle too long. Returns true when
    /// it did. Call periodically.
    pub fn unload_if_idle(&self) -> bool {
        if let Ok(mut guard) = self.loaded.try_lock() {
            if guard.as_ref().is_some_and(|l| l.last_used.elapsed() >= self.idle_unload) {
                *guard = None;
                *self.loaded_name.write().unwrap_or_else(|e| e.into_inner()) = None;
                return true;
            }
        }
        false
    }
}
