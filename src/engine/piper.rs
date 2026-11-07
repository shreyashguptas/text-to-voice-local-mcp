//! Piper voices (rhasspy/piper, OHF-Voice/piper1-gpl): a `name.onnx` model with
//! its `name.onnx.json` config beside it. Piper voices are small (5-120 MB),
//! fast, and exist for dozens of languages.

use super::{Chunk, LoadedModel, Model, ModelInfo, SynthRequest, VoiceInfo, bcp47, file_size_mb, language_name, slug};
use crate::phonemize;
use anyhow::{Context, Result, bail};
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
struct Config {
    audio: Audio,
    espeak: Option<Espeak>,
    #[serde(default)]
    inference: Inference,
    phoneme_id_map: HashMap<String, Vec<i64>>,
    #[serde(default)]
    num_speakers: usize,
    #[serde(default)]
    speaker_id_map: HashMap<String, i64>,
    language: Option<Language>,
    dataset: Option<String>,
    #[serde(default)]
    phoneme_type: Option<String>,
}
#[derive(Debug, Clone, Deserialize)]
struct Audio {
    sample_rate: u32,
    quality: Option<String>,
}
#[derive(Debug, Clone, Deserialize)]
struct Espeak {
    voice: String,
}
#[derive(Debug, Clone, Deserialize)]
struct Inference {
    #[serde(default = "d_noise")]
    noise_scale: f32,
    #[serde(default = "d_length")]
    length_scale: f32,
    #[serde(default = "d_noise_w")]
    noise_w: f32,
}
impl Default for Inference {
    fn default() -> Self {
        Self { noise_scale: d_noise(), length_scale: d_length(), noise_w: d_noise_w() }
    }
}
fn d_noise() -> f32 {
    0.667
}
fn d_length() -> f32 {
    1.0
}
fn d_noise_w() -> f32 {
    0.8
}
#[derive(Debug, Clone, Deserialize)]
struct Language {
    code: Option<String>,
}

pub struct PiperModel {
    info: ModelInfo,
    voices: Vec<VoiceInfo>,
    model_path: PathBuf,
    config: Config,
}

/// `voice.onnx` + `voice.onnx.json` (or `voice.json`) is a Piper voice.
pub fn detect(onnx: &Path) -> Option<PiperModel> {
    let json = [
        PathBuf::from(format!("{}.json", onnx.display())),
        onnx.with_extension("json"),
    ]
    .into_iter()
    .find(|p| p.is_file())?;
    let config: Config = serde_json::from_str(&std::fs::read_to_string(&json).ok()?).ok()?;
    if config.phoneme_type.as_deref().is_some_and(|t| t != "espeak") {
        return None; // text- or pinyin-based exports need their own reader
    }
    let espeak_voice = config.espeak.as_ref()?.voice.clone();
    let id = slug(&onnx.file_stem()?.to_string_lossy());
    let lang = bcp47(
        config
            .language
            .as_ref()
            .and_then(|l| l.code.clone())
            .as_deref()
            .unwrap_or(&espeak_voice),
    );
    let lang_name = language_name(&lang);
    let quality = config.audio.quality.clone().unwrap_or_else(|| "medium".into());
    let dataset = config.dataset.clone().unwrap_or_else(|| id.clone());

    let mut voices: Vec<VoiceInfo> = if config.num_speakers > 1 && !config.speaker_id_map.is_empty() {
        let mut s: Vec<(&String, &i64)> = config.speaker_id_map.iter().collect();
        s.sort_by_key(|(_, i)| **i);
        s.into_iter()
            .map(|(name, _)| VoiceInfo {
                id: name.clone(),
                model: id.clone(),
                name: name.clone(),
                language: lang.clone(),
                language_name: lang_name.clone(),
                gender: "unknown".into(),
                notes: format!("Speaker of the {dataset} dataset."),
            })
            .collect()
    } else {
        vec![VoiceInfo {
            id: id.clone(),
            model: id.clone(),
            name: dataset.clone(),
            language: lang.clone(),
            language_name: lang_name.clone(),
            gender: "unknown".into(),
            notes: format!("Piper `{quality}` quality."),
        }]
    };
    voices.dedup_by(|a, b| a.id == b.id);
    let size_mb = file_size_mb(&[onnx.to_path_buf()]);
    let info = ModelInfo {
        id: id.clone(),
        engine: "piper".into(),
        name: format!("Piper {dataset} ({lang_name}, {quality})"),
        description: format!(
            "Piper: a small, fast neural voice (VITS) from the Rhasspy / Open Home Foundation project. \
             {lang_name}, `{quality}` quality, {} voice(s). Text is read with espeak-ng.",
            voices.len()
        ),
        quality: "neural".into(),
        path: onnx.display().to_string(),
        size_mb,
        approx_memory_mb: (size_mb * 1.3 + 40.0).round(),
        sample_rate: config.audio.sample_rate,
        languages: vec![lang],
        voice_count: voices.len(),
        default_voice: voices[0].id.clone(),
        speed_range: [0.5, 2.0],
        license: "See the voice's MODEL_CARD (often MIT or CC-BY).".into(),
        loaded: false,
        is_default: false,
    };
    Some(PiperModel { info, voices, model_path: onnx.to_path_buf(), config })
}

impl Model for PiperModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }
    fn voices(&self) -> &[VoiceInfo] {
        &self.voices
    }
    fn load(&self) -> Result<Box<dyn LoadedModel>> {
        let session = super::onnx::session(&self.model_path)?;
        let has_sid = session.inputs().iter().any(|i| i.name() == "sid");
        Ok(Box::new(LoadedPiper { session, run_options: super::onnx::run_options()?, config: self.config.clone(), has_sid }))
    }
}

struct LoadedPiper {
    session: Session,
    run_options: ort::session::RunOptions,
    config: Config,
    has_sid: bool,
}

impl LoadedModel for LoadedPiper {
    fn sample_rate(&self) -> u32 {
        self.config.audio.sample_rate
    }

    fn plan(&mut self, req: &SynthRequest) -> Result<Vec<Chunk>> {
        let voice = &self.config.espeak.as_ref().context("voice has no espeak setting")?.voice;
        let phonemes = phonemize::espeak(&req.text, voice)?;
        // Piper is trained one sentence at a time.
        let pieces = phonemize::split(&phonemes, 400, true);
        let n = pieces.len();
        Ok(pieces
            .into_iter()
            .enumerate()
            .map(|(i, p)| Chunk {
                pause_after: if i + 1 < n { phonemize::pause_after(&p).max(0.2) } else { 0.0 },
                payload: p,
            })
            .collect())
    }

    fn render(&mut self, req: &SynthRequest, chunk: &Chunk) -> Result<Vec<f32>> {
        let map = &self.config.phoneme_id_map;
        let id = |s: &str| map.get(s).and_then(|v| v.first()).copied();
        let (bos, eos, pad) = (id("^").unwrap_or(1), id("$").unwrap_or(2), id("_").unwrap_or(0));
        let mut ids = vec![bos, pad];
        let mut buf = [0u8; 4];
        for c in chunk.payload.chars() {
            if let Some(v) = map.get(c.encode_utf8(&mut buf) as &str) {
                ids.extend_from_slice(v);
                ids.push(pad);
            }
        }
        ids.push(eos);
        if ids.len() <= 3 {
            return Ok(vec![]);
        }
        let inf = &self.config.inference;
        let len = ids.len();
        let mut inputs: Vec<(&str, ort::value::DynValue)> = vec![
            ("input", Tensor::from_array(([1usize, len], ids))?.into_dyn()),
            ("input_lengths", Tensor::from_array(([1usize], vec![len as i64]))?.into_dyn()),
            (
                "scales",
                Tensor::from_array(([3usize], vec![inf.noise_scale, inf.length_scale / req.speed, inf.noise_w]))?
                    .into_dyn(),
            ),
        ];
        if self.has_sid {
            let sid = self.config.speaker_id_map.get(&req.voice).copied().unwrap_or(0);
            inputs.push(("sid", Tensor::from_array(([1usize], vec![sid]))?.into_dyn()));
        }
        let outputs = self.session.run_with_options(inputs, &self.run_options)?;
        let (_, audio) = outputs[0].try_extract_tensor::<f32>()?;
        if audio.is_empty() {
            bail!("the voice produced no audio");
        }
        let mut audio = phonemize::trim_silence(audio).to_vec();
        let sr = self.config.audio.sample_rate as f32;
        audio.extend(std::iter::repeat_n(0.0, (chunk.pause_after * sr) as usize));
        Ok(audio)
    }
}
