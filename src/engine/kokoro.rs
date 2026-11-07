//! Kokoro (hexgrad/Kokoro-82M), exported to ONNX.
//!
//! Recognised layouts:
//!
//! * kokoro-onnx releases: `kokoro-v1.0.onnx` (or `.fp16` / `.int8`) next to
//!   `voices-v1.0.bin`, a numpy `.npz` holding every voice.
//! * Hugging Face `onnx-community/Kokoro-82M-*-ONNX`: `onnx/model*.onnx` with a
//!   `voices/` folder of raw `*.bin` voice files beside `onnx/`.
//!
//! Each voice is a table of 510 style vectors (one per input length) of 256
//! floats. Voices are read from disk only when first used.

use super::{Chunk, LoadedModel, Model, ModelInfo, SynthRequest, VoiceInfo, file_size_mb, slug};
use crate::phonemize;
use anyhow::{Context, Result, anyhow, bail};
use ort::session::Session;
use ort::value::{Tensor, TensorElementType, ValueType};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

pub const SAMPLE_RATE: u32 = 24_000;
/// Longest piece read in one pass. The model takes up to 510, but its scratch
/// memory grows with length: 250 keeps a busy server near 1.26 GB instead of
/// 1.85 GB, still cuts at sentence ends, and is ~8x faster than real time.
const PIECE_PHONEMES: usize = 250;
const STYLE_DIM: usize = 256;
const VOCAB_JSON: &str = include_str!("../../assets/kokoro_vocab.json");

#[derive(Debug, Clone)]
enum VoiceSource {
    /// One `.npz` file holding `<voice>.npy` entries.
    Npz(PathBuf),
    /// A folder of raw little-endian f32 `<voice>.bin` files.
    Dir(PathBuf),
}

pub struct KokoroModel {
    info: ModelInfo,
    voices: Vec<VoiceInfo>,
    model_path: PathBuf,
    source: VoiceSource,
}

/// Look at an `.onnx` file and, if it is a Kokoro model with voices beside
/// it, describe it.
pub fn detect(onnx: &Path) -> Option<KokoroModel> {
    let source = find_voices(onnx)?;
    let names = match &source {
        VoiceSource::Npz(p) => npz_names(p).ok()?,
        VoiceSource::Dir(d) => {
            let mut v: Vec<String> = std::fs::read_dir(d)
                .ok()?
                .flatten()
                .filter_map(|e| {
                    let p = e.path();
                    (p.extension()? == "bin").then(|| p.file_stem()?.to_str().map(String::from))?
                })
                .collect();
            v.sort();
            v
        }
    };
    if names.is_empty() {
        return None;
    }
    let id = model_id(onnx);
    let voices: Vec<VoiceInfo> = names.iter().map(|n| voice_info(n, &id)).collect();
    let mut languages: Vec<String> = voices.iter().map(|v| v.language.clone()).collect();
    languages.sort();
    languages.dedup();
    let default_voice = ["af_heart", "af_bella", "am_michael"]
        .iter()
        .find(|v| names.iter().any(|n| n == *v))
        .map(|v| v.to_string())
        .unwrap_or_else(|| names[0].clone());

    let stem = onnx.file_stem()?.to_string_lossy().to_lowercase();
    let quantized = ["int8", "q8", "q4", "uint8", "quant"].iter().any(|q| stem.contains(q));
    let half = stem.contains("fp16") || stem.contains("f16");
    let precision = if quantized {
        "8-bit or smaller (quantized: smaller and lighter, slightly lower quality)"
    } else if half {
        "16-bit (half precision)"
    } else {
        "32-bit (full precision, best quality)"
    };
    let mut files = vec![onnx.to_path_buf()];
    if let VoiceSource::Npz(p) = &source {
        files.push(p.clone());
    }
    let size_mb = file_size_mb(&files);
    let model_mb = file_size_mb(&files[..1]);

    let info = ModelInfo {
        id,
        engine: "kokoro".into(),
        name: format!("Kokoro 82M ({})", onnx.file_name()?.to_string_lossy()),
        description: format!(
            "Kokoro: an 82-million-parameter open neural voice model from hexgrad, among the most \
             natural-sounding small models. {} voices in {} languages. Weights: {precision}. English \
             is read with misaki, the converter Kokoro was trained with; other languages with espeak-ng.",
            voices.len(),
            languages.len()
        ),
        quality: if quantized { "neural".into() } else { "neural-high".into() },
        path: onnx.display().to_string(),
        size_mb,
        // Measured on Apple Silicon (CPU only): weights plus the English
        // dictionaries and runtime buffers, ~540 MB for Kokoro v1.0.
        approx_memory_mb: (model_mb * 1.1 + 170.0).round(),
        sample_rate: SAMPLE_RATE,
        languages,
        voice_count: voices.len(),
        default_voice,
        speed_range: [0.5, 2.0],
        license: "Apache-2.0".into(),
        loaded: false,
        is_default: false,
    };
    Some(KokoroModel { info, voices, model_path: onnx.to_path_buf(), source })
}

fn find_voices(onnx: &Path) -> Option<VoiceSource> {
    let dir = onnx.parent()?;
    // A voices .npz next to the model.
    let mut npz: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_lowercase();
            name.contains("voice") && (name.ends_with(".bin") || name.ends_with(".npz")) && p.is_file()
        })
        .collect();
    npz.sort();
    // Prefer the voices file whose version matches the model's.
    let stem = onnx.file_stem()?.to_string_lossy().to_lowercase();
    if let Some(p) = npz
        .iter()
        .find(|p| {
            let n = p.file_stem().unwrap_or_default().to_string_lossy().to_lowercase();
            n.split('-').skip(1).any(|ver| stem.contains(ver))
        })
        .or(npz.first())
    {
        if is_zip(p) {
            return Some(VoiceSource::Npz(p.clone()));
        }
    }
    // A voices/ folder next to the model or next to its onnx/ folder.
    for base in [Some(dir), dir.parent()].into_iter().flatten() {
        let d = base.join("voices");
        if d.is_dir() {
            return Some(VoiceSource::Dir(d));
        }
    }
    None
}

fn is_zip(p: &Path) -> bool {
    let mut magic = [0u8; 4];
    std::fs::File::open(p)
        .and_then(|mut f| f.read_exact(&mut magic))
        .map(|_| magic == *b"PK\x03\x04")
        .unwrap_or(false)
}

/// `kokoro-v1.0.onnx` -> `kokoro-v1.0`; `.../Kokoro-82M-v1.0-ONNX/onnx/model_q8f16.onnx`
/// -> `kokoro-82m-v1.0-onnx-q8f16`.
fn model_id(onnx: &Path) -> String {
    let stem = onnx.file_stem().unwrap_or_default().to_string_lossy().to_string();
    if stem.to_lowercase().contains("kokoro") {
        return slug(&stem);
    }
    let family = onnx
        .ancestors()
        .filter_map(|a| a.file_name()?.to_str())
        .find(|n| n.to_lowercase().contains("kokoro"))
        .map(|n| n.rsplit("--").next().unwrap_or(n).to_string())
        .unwrap_or_else(|| "kokoro".into());
    let variant = stem.trim_start_matches("model").trim_start_matches(['_', '-']);
    if variant.is_empty() { slug(&family) } else { slug(&format!("{family}-{variant}")) }
}

/// Kokoro voice ids encode language and gender: `af_heart` is American
/// English, female.
fn voice_info(id: &str, model: &str) -> VoiceInfo {
    let mut chars = id.chars();
    let lang = chars.next().unwrap_or('a');
    let gender = match chars.next() {
        Some('f') => "female",
        Some('m') => "male",
        _ => "unknown",
    };
    let (code, lang_name) = match lang {
        'a' => ("en-US", "English (US)"),
        'b' => ("en-GB", "English (UK)"),
        'e' => ("es", "Spanish"),
        'f' => ("fr-FR", "French"),
        'h' => ("hi", "Hindi"),
        'i' => ("it", "Italian"),
        'j' => ("ja", "Japanese"),
        'p' => ("pt-BR", "Portuguese (Brazil)"),
        'z' => ("zh", "Chinese (Mandarin)"),
        _ => ("und", "Unknown"),
    };
    let base = id.split_once('_').map(|(_, n)| n).unwrap_or(id);
    let name: String = base
        .split('_')
        .map(|w| {
            let mut c = w.chars();
            c.next().map(|f| f.to_uppercase().chain(c).collect::<String>()).unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ");
    let notes = match id {
        "af_heart" => "Best overall quality. The default.",
        "af_bella" => "Very high quality.",
        "af_nicole" => "Soft, close-up, whispery delivery.",
        "am_santa" | "em_santa" | "pm_santa" => "Novelty Santa voice.",
        _ if matches!(lang, 'j' | 'z') => "Read with espeak-ng phonemes; pronunciation is rougher than English.",
        _ => "",
    };
    VoiceInfo {
        id: id.to_string(),
        model: model.to_string(),
        name,
        language: code.into(),
        language_name: lang_name.into(),
        gender: gender.into(),
        notes: notes.into(),
    }
}

/// Which converter reads text for a voice, by its language letter.
enum Reader {
    English { british: bool },
    Espeak(&'static str),
}

fn reader_for(voice: &str) -> Reader {
    match voice.chars().next() {
        Some('b') => Reader::English { british: true },
        Some('e') => Reader::Espeak("es"),
        Some('f') => Reader::Espeak("fr"),
        Some('h') => Reader::Espeak("hi"),
        Some('i') => Reader::Espeak("it"),
        Some('j') => Reader::Espeak("ja"),
        Some('p') => Reader::Espeak("pt-br"),
        Some('z') => Reader::Espeak("cmn"),
        _ => Reader::English { british: false },
    }
}

impl Model for KokoroModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }
    fn voices(&self) -> &[VoiceInfo] {
        &self.voices
    }
    fn load(&self) -> Result<Box<dyn LoadedModel>> {
        let session = super::onnx::session(&self.model_path)?;
        let tokens_input = if session.inputs().iter().any(|i| i.name() == "input_ids") {
            "input_ids"
        } else {
            "tokens"
        };
        let speed_type = session
            .inputs()
            .iter()
            .find(|i| i.name() == "speed")
            .and_then(|i| match i.dtype() {
                ValueType::Tensor { ty, .. } => Some(*ty),
                _ => None,
            })
            .unwrap_or(TensorElementType::Float32);
        let vocab: serde_json::Value = serde_json::from_str(VOCAB_JSON)?;
        let vocab: HashMap<char, i64> = vocab["vocab"]
            .as_object()
            .context("bad vocab")?
            .iter()
            .filter_map(|(k, v)| Some((k.chars().next()?, v.as_i64()?)))
            .collect();
        Ok(Box::new(LoadedKokoro {
            session,
            run_options: super::onnx::run_options()?,
            tokens_input,
            speed_type,
            vocab,
            source: self.source.clone(),
            styles: HashMap::new(),
        }))
    }
}

struct LoadedKokoro {
    session: Session,
    run_options: ort::session::RunOptions,
    tokens_input: &'static str,
    speed_type: TensorElementType,
    vocab: HashMap<char, i64>,
    source: VoiceSource,
    /// Voices read so far, each 510 x 256 floats (about 0.5 MB).
    styles: HashMap<String, Vec<f32>>,
}

impl LoadedKokoro {
    fn style(&mut self, voice: &str) -> Result<&Vec<f32>> {
        if !self.styles.contains_key(voice) {
            let data = match &self.source {
                VoiceSource::Npz(p) => read_npz_voice(p, voice)?,
                VoiceSource::Dir(d) => {
                    let bytes = std::fs::read(d.join(format!("{voice}.bin")))
                        .with_context(|| format!("voice `{voice}` not found"))?;
                    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
                }
            };
            if data.len() < STYLE_DIM || data.len() % STYLE_DIM != 0 {
                bail!("voice `{voice}` has an unexpected size ({} floats)", data.len());
            }
            // Keep at most a handful of voices in memory.
            if self.styles.len() >= 8 {
                self.styles.clear();
            }
            self.styles.insert(voice.to_string(), data);
        }
        Ok(&self.styles[voice])
    }
}

impl LoadedModel for LoadedKokoro {
    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn plan(&mut self, req: &SynthRequest) -> Result<Vec<Chunk>> {
        let phonemes = match reader_for(&req.voice) {
            Reader::English { british } => phonemize::english(&req.text, british)?,
            Reader::Espeak(lang) => phonemize::espeak(&req.text, lang)?,
        };
        let known: String = phonemes
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .filter(|c| self.vocab.contains_key(c))
            .collect();
        let pieces = phonemize::split(&known, PIECE_PHONEMES, true);
        let n = pieces.len();
        Ok(pieces
            .into_iter()
            .enumerate()
            .map(|(i, p)| Chunk {
                pause_after: if i + 1 < n { phonemize::pause_after(&p) } else { 0.0 },
                payload: p,
            })
            .collect())
    }

    fn render(&mut self, req: &SynthRequest, chunk: &Chunk) -> Result<Vec<f32>> {
        let mut tokens: Vec<i64> = Vec::with_capacity(chunk.payload.len() + 2);
        tokens.push(0);
        tokens.extend(chunk.payload.chars().filter_map(|c| self.vocab.get(&c).copied()));
        tokens.push(0);
        let n = tokens.len() - 2;
        if n == 0 {
            return Ok(vec![]);
        }
        let table = self.style(&req.voice)?;
        let rows = table.len() / STYLE_DIM;
        let row = n.min(rows) - 1;
        let style = table[row * STYLE_DIM..(row + 1) * STYLE_DIM].to_vec();

        let tokens = Tensor::from_array(([1usize, tokens.len()], tokens))?;
        let style = Tensor::from_array(([1usize, STYLE_DIM], style))?;
        let speed: ort::value::DynValue = match self.speed_type {
            TensorElementType::Int32 => Tensor::from_array(([1usize], vec![req.speed.round().max(1.0) as i32]))?.into_dyn(),
            TensorElementType::Int64 => Tensor::from_array(([1usize], vec![req.speed.round().max(1.0) as i64]))?.into_dyn(),
            _ => Tensor::from_array(([1usize], vec![req.speed]))?.into_dyn(),
        };
        let outputs = self.session.run_with_options(ort::inputs![
            self.tokens_input => tokens,
            "style" => style,
            "speed" => speed,
        ], &self.run_options)?;
        let (_, audio) = outputs[0].try_extract_tensor::<f32>()?;
        let mut audio = phonemize::trim_silence(audio).to_vec();
        audio.extend(std::iter::repeat_n(0.0, (chunk.pause_after * SAMPLE_RATE as f32) as usize));
        Ok(audio)
    }
}

// ---- numpy .npz / .npy -----------------------------------------------------

fn npz_names(p: &Path) -> Result<Vec<String>> {
    let zip = zip::ZipArchive::new(std::fs::File::open(p)?)?;
    let mut names: Vec<String> = zip
        .file_names()
        .filter_map(|n| n.ok()?.strip_suffix(".npy").map(String::from))
        .collect();
    names.sort();
    Ok(names)
}

fn read_npz_voice(p: &Path, voice: &str) -> Result<Vec<f32>> {
    let mut zip = zip::ZipArchive::new(std::fs::File::open(p)?)?;
    let mut entry = zip
        .by_name(&format!("{voice}.npy"))
        .map_err(|_| anyhow!("voice `{voice}` is not in {}", p.display()))?;
    let mut bytes = Vec::with_capacity(entry.size() as usize);
    entry.read_to_end(&mut bytes)?;
    parse_npy_f32(&bytes)
}

/// Minimal `.npy` reader for little-endian float32 arrays.
fn parse_npy_f32(b: &[u8]) -> Result<Vec<f32>> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        bail!("not a numpy array");
    }
    let (header_len, start) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&b[start..start + header_len])?;
    if !header.contains("'<f4'") {
        bail!("voice array is not float32 ({header})");
    }
    if header.contains("'fortran_order': True") {
        bail!("voice array is in Fortran order");
    }
    let data = &b[start + header_len..];
    Ok(data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids() {
        assert_eq!(model_id(Path::new("/x/kokoro-v1.0.onnx")), "kokoro-v1.0");
        assert_eq!(model_id(Path::new("/x/kokoro-v1.0.int8.onnx")), "kokoro-v1.0.int8");
        assert_eq!(
            model_id(Path::new("/h/models--onnx-community--Kokoro-82M-v1.0-ONNX/snapshots/abc/onnx/model_q8f16.onnx")),
            "kokoro-82m-v1.0-onnx-q8f16"
        );
    }

    #[test]
    fn voices() {
        let v = voice_info("bm_george", "m");
        assert_eq!((v.language.as_str(), v.gender.as_str(), v.name.as_str()), ("en-GB", "male", "George"));
        let v = voice_info("zf_xiaoxiao", "m");
        assert_eq!(v.language, "zh");
    }
}
